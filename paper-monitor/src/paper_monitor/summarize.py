"""Summary layer for cn-ai-monitor.

Reads selected papers from data/processed/selected_papers_YYYY-MM-DD.json and writes:

- data/processed/summaries_YYYY-MM-DD.json
- data/processed/summaries_YYYY-MM-DD.md

Default behavior is deterministic template summarization.

Set USE_DEEPSEEK_SUMMARY=true and DEEPSEEK_API_KEY to enable DeepSeek summaries.
The DeepSeek mode still falls back to template summaries per paper if an API call fails.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any,Optional

import requests


DEEPSEEK_API_URL=os.getenv("DEEPSEEK_API_URL","https://api.deepseek.com/chat/completions")
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL=os.getenv("DEEPSEEK_MODEL","deepseek-chat")
USE_DEEPSEEK_SUMMARY=os.getenv("USE_DEEPSEEK_SUMMARY","false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEEPSEEK_TEMPERATURE=float(os.getenv("DEEPSEEK_TEMPERATURE","0.2"))
DEEPSEEK_MAX_TOKENS=int(os.getenv("DEEPSEEK_MAX_TOKENS","900"))
DEEPSEEK_RETRIES=int(os.getenv("DEEPSEEK_RETRIES","3"))
DEEPSEEK_RETRY_BASE_SECONDS=int(os.getenv("DEEPSEEK_RETRY_BASE_SECONDS","10"))
DEEPSEEK_SLEEP_SECONDS=float(os.getenv("DEEPSEEK_SLEEP_SECONDS","0.5"))
DEEPSEEK_TIMEOUT_SECONDS=int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS","120"))

_SENTENCE_SPLIT_RE=re.compile(r"(?<=[.!?])\s+")
_DATE_TAG_RE=re.compile(r"(\d{4}-\d{2}-\d{2})")
_WS_RE=re.compile(r"\s+")

CORE_TOPIC_RULES:tuple[tuple[str,tuple[str,...]],...]=(
    (
        "large language models",
        (
            "large language model",
            "llm",
            "language model",
            "reasoning model",
            "instruction tuning",
            "alignment",
            "pretraining",
        ),
    ),
    (
        "multimodal AI",
        (
            "multimodal",
            "vision-language",
            "vision language",
            "vlm",
            "image generation",
            "video generation",
            "text-to-video",
            "text to video",
        ),
    ),
    (
        "AI agents and tool use",
        (
            "agentic",
            "ai agent",
            "tool use",
            "workflow",
            "planning",
            "plan-execute",
        ),
    ),
    (
        "benchmarks and datasets",
        (
            "benchmark",
            "dataset",
            "evaluation",
            "leaderboard",
        ),
    ),
    (
        "retrieval and recommendation",
        (
            "retrieval",
            "rag",
            "recommendation",
            "recommender",
            "personalized recommendation",
        ),
    ),
    (
        "model architecture or training",
        (
            "transformer",
            "mixture of experts",
            "moe",
            "diffusion",
            "reinforcement learning",
            "scaling",
        ),
    ),
)


def _safe_str(value:Any)->str:
    if value is None:
        return ""

    if isinstance(value,list):
        return " ".join(str(item) for item in value)

    return str(value)


def _safe_list(value:Any)->list[Any]:
    if isinstance(value,list):
        return value

    return []


def _clean_text(text:Any)->str:
    raw=_safe_str(text)
    return _WS_RE.sub(" ",raw).strip()


def _truncate(text:str,max_chars:int=260)->str:
    text=_clean_text(text)

    if len(text)<=max_chars:
        return text

    return text[:max_chars-1].rstrip()+"…"


def _first_sentences(text:str,limit:int=2,max_chars:int=520)->str:
    text=_clean_text(text)

    if not text:
        return ""

    sentences=[
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(text)
        if sentence.strip()
    ]

    if not sentences:
        return _truncate(text,max_chars)

    selected=" ".join(sentences[:limit])
    return _truncate(selected,max_chars)


def _get_original(paper:dict[str,Any])->dict[str,Any]:
    original=paper.get("original")

    if isinstance(original,dict):
        return original

    return {}


def _get_summary_text(paper:dict[str,Any])->str:
    original=_get_original(paper)

    for key in ("summary","abstract","tldr"):
        value=paper.get(key)
        if isinstance(value,str) and value.strip():
            return _clean_text(value)

        value=original.get(key)
        if isinstance(value,str) and value.strip():
            return _clean_text(value)

    return ""


def _get_title(paper:dict[str,Any])->str:
    return _clean_text(paper.get("title")) or "(untitled)"


def _get_url(paper:dict[str,Any])->str|None:
    for key in ("url","landing_page_url","pdf_url"):
        value=paper.get(key)
        if isinstance(value,str) and value.strip():
            return value.strip()

    original=_get_original(paper)
    for key in ("url","landing_page_url","pdf_url"):
        value=original.get(key)
        if isinstance(value,str) and value.strip():
            return value.strip()

    return None


def _get_institutions(paper:dict[str,Any])->list[str]:
    institutions=_safe_list(paper.get("institutions"))
    return [str(item) for item in institutions if str(item).strip()]


def _get_countries(paper:dict[str,Any])->list[str]:
    countries=_safe_list(paper.get("countries"))
    return [str(item).upper() for item in countries if str(item).strip()]


def _detect_topic(title:str,summary:str)->str:
    text=(title+" "+summary).lower()

    for topic,terms in CORE_TOPIC_RULES:
        for term in terms:
            if term.lower() in text:
                return topic

    return "AI research"


def _format_institution_phrase(institutions:list[str])->str:
    if not institutions:
        return "unspecified institutions"

    if len(institutions)==1:
        return institutions[0]

    if len(institutions)==2:
        return institutions[0]+" and "+institutions[1]

    return ", ".join(institutions[:2])+", and others"


def _extract_key_signals(paper:dict[str,Any])->list[str]:
    signals=[]

    china_methods=_safe_list(paper.get("china_match_methods"))
    significance_reasons=_safe_list(paper.get("significance_reasons"))
    ai_reasons=_safe_list(paper.get("ai_relevance_reasons"))

    for value in china_methods+significance_reasons+ai_reasons:
        if not isinstance(value,str):
            continue

        if value not in signals:
            signals.append(value)

    return signals[:8]


def _make_risk_flags(paper:dict[str,Any])->list[str]:
    flags=[]

    if paper.get("needs_review"):
        flags.append("needs manual review")

    if paper.get("china_affiliation_level")=="review":
        flags.append("China affiliation is review-level, not strong")

    if paper.get("openalex_match_accepted") is False:
        flags.append("OpenAlex match was rejected")

    if not _get_institutions(paper):
        flags.append("no verified institution metadata")

    if not _get_summary_text(paper):
        flags.append("no abstract/summary text available")

    return flags


def _selection_bucket(paper:dict[str,Any])->str:
    level=_safe_str(paper.get("digest_selection_level"))

    if level=="strong":
        return "verified_cn"

    if level=="review":
        return "review_candidate"

    china_level=_safe_str(paper.get("china_affiliation_level"))

    if china_level=="strong":
        return "verified_cn"

    if china_level=="review":
        return "review_candidate"

    return "other"


def template_summary(paper:dict[str,Any],rank:int)->dict[str,Any]:
    title=_get_title(paper)
    summary_text=_get_summary_text(paper)
    institutions=_get_institutions(paper)
    countries=_get_countries(paper)

    topic=_detect_topic(title,summary_text)
    institution_phrase=_format_institution_phrase(institutions)
    abstract_digest=_first_sentences(summary_text,limit=2,max_chars=520)

    significance_score=paper.get("significance_score")
    ai_relevance_score=paper.get("ai_relevance_score")
    china_level=paper.get("china_affiliation_level") or "unknown"
    risk_flags=_make_risk_flags(paper)

    bullets=[]

    if abstract_digest:
        bullets.append("Core idea: "+abstract_digest)
    else:
        bullets.append("Core idea: The paper appears to be related to "+topic+" based on title and metadata, but no abstract text was available.")

    bullets.append("Affiliation signal: "+institution_phrase+"; country metadata: "+(", ".join(countries) if countries else "not available")+".")

    bullets.append(
        "Selection signal: significance score="
        +str(significance_score)
        +", AI relevance score="
        +str(ai_relevance_score)
        +", China-affiliation level="
        +str(china_level)
        +"."
    )

    if china_level=="strong":
        why_it_matters=(
            "This paper is worth tracking because it passed the current digest filter for "
            +topic
            +" and has a verified or high-confidence China-affiliation signal."
        )
    else:
        why_it_matters=(
            "This paper is worth tracking as a review-level candidate because it scored highly on AI relevance and significance, "
            +"but its China-affiliation signal needs manual verification."
        )

    confidence="high"
    if paper.get("needs_review") or china_level!="strong":
        confidence="medium"
    if paper.get("openalex_match_accepted") is False:
        confidence="low"

    return {
        "rank":rank,
        "title":title,
        "source":paper.get("source"),
        "published":paper.get("published"),
        "arxiv_id":paper.get("arxiv_id"),
        "openalex_id":paper.get("openalex_id"),
        "url":_get_url(paper),
        "institutions":institutions,
        "countries":countries,
        "topic":topic,
        "summary_bullets":bullets,
        "technical_contribution":abstract_digest or "Not available from current metadata.",
        "why_it_matters":why_it_matters,
        "china_affiliation_assessment":(
            "Verified China-affiliated paper."
            if china_level=="strong"
            else "Review-level China-affiliation candidate; manual verification required."
        ),
        "policy_or_market_relevance":"Not assessed in template mode.",
        "recommended_action":(
            "include"
            if confidence=="high"
            else "manual_review"
        ),
        "significance_score":significance_score,
        "ai_relevance_score":ai_relevance_score,
        "china_affiliation_level":china_level,
        "digest_selection_level":paper.get("digest_selection_level"),
        "digest_selection_reason":paper.get("digest_selection_reason"),
        "selection_bucket":_selection_bucket(paper),
        "core_ai_digest_signal":paper.get("core_ai_digest_signal"),
        "openalex_match_method":paper.get("openalex_match_method"),
        "openalex_match_score":paper.get("openalex_match_score"),
        "openalex_match_accepted":paper.get("openalex_match_accepted"),
        "key_signals":_extract_key_signals(paper),
        "risk_flags":risk_flags,
        "confidence":confidence,
        "summary_mode":"template",
        "llm_error":None,
        "original":paper,
    }


def _paper_prompt_payload(paper:dict[str,Any],draft:dict[str,Any])->dict[str,Any]:
    return {
        "title":draft.get("title"),
        "abstract":_get_summary_text(paper),
        "arxiv_id":paper.get("arxiv_id"),
        "url":draft.get("url"),
        "published":paper.get("published"),
        "institutions":draft.get("institutions"),
        "countries":draft.get("countries"),
        "topic_guess":draft.get("topic"),
        "scores":{
            "significance_score":paper.get("significance_score"),
            "ai_relevance_score":paper.get("ai_relevance_score"),
        },
        "affiliation":{
            "china_affiliation_level":paper.get("china_affiliation_level"),
            "china_match_methods":paper.get("china_match_methods"),
            "china_match_terms":paper.get("china_match_terms"),
            "needs_review":paper.get("needs_review"),
            "digest_selection_level":paper.get("digest_selection_level"),
            "digest_selection_reason":paper.get("digest_selection_reason"),
            "openalex_match_accepted":paper.get("openalex_match_accepted"),
            "openalex_match_method":paper.get("openalex_match_method"),
            "openalex_match_score":paper.get("openalex_match_score"),
        },
        "risk_flags":draft.get("risk_flags"),
    }


def _extract_json_object(text:str)->dict[str,Any]:
    try:
        data=json.loads(text)
        if isinstance(data,dict):
            return data
    except json.JSONDecodeError:
        pass

    start=text.find("{")
    end=text.rfind("}")

    if start!=-1 and end!=-1 and end>start:
        data=json.loads(text[start:end+1])
        if isinstance(data,dict):
            return data

    raise ValueError("DeepSeek response did not contain a valid JSON object.")


def call_deepseek_summary(paper:dict[str,Any],draft:dict[str,Any])->dict[str,Any]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY.")

    system_prompt=(
        "You are an AI research analyst. Return only valid json. "
        "Your job is to summarize AI research papers for an international China AI monitoring digest. "
        "Be careful: if China affiliation is review-level or metadata is missing, explicitly state that verification is required. "
        "Do not invent institutions, authors, results, benchmarks, or claims that are not in the provided metadata."
    )

    schema_example={
        "topic":"large language models | multimodal AI | AI agents and tool use | benchmarks and datasets | AI research",
        "summary_bullets":[
            "One concise bullet explaining the core technical idea.",
            "One concise bullet explaining method/data/evaluation if available.",
            "One concise bullet explaining what is new or uncertain."
        ],
        "technical_contribution":"2-3 sentence technical contribution, grounded only in the abstract.",
        "why_it_matters":"1-2 sentence significance for China AI monitoring.",
        "china_affiliation_assessment":"State whether this is verified China-affiliated or only a review-level candidate.",
        "policy_or_market_relevance":"1 sentence on policy/market/strategic relevance, or say not clear from metadata.",
        "recommended_action":"include | manual_review | exclude",
        "confidence":"high | medium | low"
    }

    user_payload={
        "task":"Return a json object following the schema. Do not use markdown.",
        "schema":schema_example,
        "paper":_paper_prompt_payload(paper,draft),
    }

    payload={
        "model":DEEPSEEK_MODEL,
        "messages":[
            {"role":"system","content":system_prompt},
            {"role":"user","content":json.dumps(user_payload,ensure_ascii=False)},
        ],
        "temperature":DEEPSEEK_TEMPERATURE,
        "max_tokens":DEEPSEEK_MAX_TOKENS,
        "response_format":{"type":"json_object"},
    }

    headers={
        "Authorization":"Bearer "+DEEPSEEK_API_KEY,
        "Content-Type":"application/json",
    }

    last_error=None

    for attempt in range(1,DEEPSEEK_RETRIES+1):
        try:
            response=requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=(10,DEEPSEEK_TIMEOUT_SECONDS),
            )

            if response.status_code in (429,500,502,503,504):
                retry_after=response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_seconds=int(retry_after)
                else:
                    wait_seconds=DEEPSEEK_RETRY_BASE_SECONDS*attempt

                print(
                    "DeepSeek transient status "
                    +str(response.status_code)
                    +" for "
                    +str(draft.get("arxiv_id") or draft.get("title"))
                    +"; retrying in "
                    +str(wait_seconds)
                    +"s."
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            data=response.json()
            content=data["choices"][0]["message"]["content"]
            parsed=_extract_json_object(content)

            merged=dict(draft)
            merged.update(
                {
                    "topic":_clean_text(parsed.get("topic")) or draft.get("topic"),
                    "summary_bullets":[
                        _clean_text(item)
                        for item in _safe_list(parsed.get("summary_bullets"))
                        if _clean_text(item)
                    ][:5] or draft.get("summary_bullets"),
                    "technical_contribution":_clean_text(parsed.get("technical_contribution")) or draft.get("technical_contribution"),
                    "why_it_matters":_clean_text(parsed.get("why_it_matters")) or draft.get("why_it_matters"),
                    "china_affiliation_assessment":_clean_text(parsed.get("china_affiliation_assessment")) or draft.get("china_affiliation_assessment"),
                    "policy_or_market_relevance":_clean_text(parsed.get("policy_or_market_relevance")) or draft.get("policy_or_market_relevance"),
                    "recommended_action":_clean_text(parsed.get("recommended_action")) or draft.get("recommended_action"),
                    "confidence":_clean_text(parsed.get("confidence")) or draft.get("confidence"),
                    "summary_mode":"deepseek",
                    "llm_error":None,
                }
            )

            return merged

        except Exception as exc:
            last_error=exc
            if attempt<DEEPSEEK_RETRIES:
                wait_seconds=DEEPSEEK_RETRY_BASE_SECONDS*attempt
                print(
                    "DeepSeek summary failed for "
                    +str(draft.get("arxiv_id") or draft.get("title"))
                    +" attempt "
                    +str(attempt)
                    +"/"
                    +str(DEEPSEEK_RETRIES)
                    +": "
                    +str(exc)
                )
                print("Waiting "+str(wait_seconds)+"s before retry...")
                time.sleep(wait_seconds)

    fallback=dict(draft)
    fallback["summary_mode"]="template_fallback"
    fallback["llm_error"]=str(last_error)
    return fallback


def summarize_paper(paper:dict[str,Any],rank:int)->dict[str,Any]:
    draft=template_summary(paper,rank)

    if USE_DEEPSEEK_SUMMARY:
        summary=call_deepseek_summary(paper,draft)
        if DEEPSEEK_SLEEP_SECONDS>0:
            time.sleep(DEEPSEEK_SLEEP_SECONDS)
        return summary

    return draft


def find_project_root(start:Optional[Path]=None)->Path:
    candidates=[]

    if start is not None:
        candidates.append(Path(start).resolve())
    else:
        candidates.append(Path.cwd().resolve())

    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass

    for candidate in candidates:
        for parent in [candidate]+list(candidate.parents):
            if (parent/"pyproject.toml").exists():
                return parent

    return candidates[0]


def find_latest_selected_file(project_root:Path)->Optional[Path]:
    processed_dir=project_root/"data"/"processed"

    if not processed_dir.exists():
        return None

    candidates=sorted(processed_dir.glob("selected_papers_*.json"))

    if not candidates:
        return None

    return candidates[-1]


def extract_date_tag(path:Path)->str:
    match=_DATE_TAG_RE.search(path.name)

    if match:
        return match.group(1)

    return datetime.now().strftime("%Y-%m-%d")


def load_json_list(path:Path)->list[dict[str,Any]]:
    with path.open("r",encoding="utf-8") as f:
        data=json.load(f)

    if not isinstance(data,list):
        raise ValueError("Expected JSON list at "+str(path))

    return [item for item in data if isinstance(item,dict)]


def write_json(data:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)

    with path.open("w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)


def _group_summaries(summaries:list[dict[str,Any]])->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    verified=[]
    review=[]
    other=[]

    for item in summaries:
        bucket=item.get("selection_bucket")
        if bucket=="verified_cn":
            verified.append(item)
        elif bucket=="review_candidate":
            review.append(item)
        else:
            other.append(item)

    return verified,review,other


def _write_markdown_section(lines:list[str],title:str,items:list[dict[str,Any]])->None:
    lines.append("## "+title)
    lines.append("")

    if not items:
        lines.append("No papers in this section.")
        lines.append("")
        return

    for item in items:
        lines.append("### "+str(item["rank"])+". "+str(item.get("title") or "(untitled)"))
        lines.append("")
        lines.append("- **Topic:** "+str(item.get("topic") or "AI research"))
        lines.append("- **Institutions:** "+(", ".join(item.get("institutions") or []) or "not available"))
        lines.append("- **Countries:** "+(", ".join(item.get("countries") or []) or "not available"))
        lines.append("- **Scores:** significance="+str(item.get("significance_score"))+", AI="+str(item.get("ai_relevance_score")))
        lines.append("- **Confidence:** "+str(item.get("confidence")))
        lines.append("- **Summary mode:** "+str(item.get("summary_mode")))
        if item.get("arxiv_id"):
            lines.append("- **arXiv:** "+str(item.get("arxiv_id")))
        if item.get("openalex_id"):
            lines.append("- **OpenAlex:** "+str(item.get("openalex_id")))
        if item.get("url"):
            lines.append("- **URL:** "+str(item.get("url")))
        lines.append("")
        lines.append("**Summary**")
        lines.append("")
        for bullet in item.get("summary_bullets") or []:
            lines.append("- "+str(bullet))
        lines.append("")
        lines.append("**Technical contribution:** "+str(item.get("technical_contribution") or ""))
        lines.append("")
        lines.append("**Why it matters:** "+str(item.get("why_it_matters") or ""))
        lines.append("")
        lines.append("**China-affiliation assessment:** "+str(item.get("china_affiliation_assessment") or ""))
        lines.append("")
        lines.append("**Policy/market relevance:** "+str(item.get("policy_or_market_relevance") or ""))
        lines.append("")
        lines.append("**Recommended action:** "+str(item.get("recommended_action") or ""))
        lines.append("")
        risk_flags=item.get("risk_flags") or []
        if risk_flags:
            lines.append("**Risk flags:** "+", ".join(str(flag) for flag in risk_flags))
            lines.append("")
        if item.get("llm_error"):
            lines.append("**LLM error:** "+str(item.get("llm_error")))
            lines.append("")


def write_markdown(summaries:list[dict[str,Any]],path:Path,date_tag:str)->None:
    lines=[]
    verified,review,other=_group_summaries(summaries)

    lines.append("# CN AI Monitor Summaries — "+date_tag)
    lines.append("")
    lines.append("Generated by cn-ai-monitor.")
    lines.append("")
    lines.append("**Counts:** verified CN papers="+str(len(verified))+", review-level candidates="+str(len(review))+", other="+str(len(other))+".")
    lines.append("")
    lines.append("Review-level candidates are not presented as fully verified China-affiliated papers; they require manual or Claude verification.")
    lines.append("")

    if not summaries:
        lines.append("No selected papers passed the current digest filter.")
        lines.append("")
    else:
        _write_markdown_section(lines,"Verified China-affiliated papers",verified)
        _write_markdown_section(lines,"Review-level China-affiliation candidates",review)
        if other:
            _write_markdown_section(lines,"Other selected papers",other)

    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(lines),encoding="utf-8")


def _ensure_utf8_stdout()->None:
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower()!="utf-8":
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main(argv:Optional[list[str]]=None)->int:
    _ensure_utf8_stdout()

    parser=argparse.ArgumentParser(
        description="Create template or DeepSeek summaries for selected CN AI papers."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to selected_papers_YYYY-MM-DD.json. Defaults to latest data/processed/selected_papers_*.json.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <project>/data/processed.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Only summarize the top N papers. 0 means summarize all selected papers.",
    )

    parser.add_argument(
        "--use-deepseek",
        action="store_true",
        help="Enable DeepSeek summaries for this run, overriding USE_DEEPSEEK_SUMMARY.",
    )

    args=parser.parse_args(argv)
    project_root=find_project_root()

    if args.use_deepseek:
        global USE_DEEPSEEK_SUMMARY
        USE_DEEPSEEK_SUMMARY=True

    if args.input:
        input_path=Path(args.input).resolve()
    else:
        found=find_latest_selected_file(project_root)
        if found is None:
            print("Error: no selected_papers_*.json found in "+str(project_root/"data"/"processed"),file=sys.stderr)
            return 1
        input_path=found

    if not input_path.exists():
        print("Error: input file not found: "+str(input_path),file=sys.stderr)
        return 1

    if args.output_dir:
        output_dir=Path(args.output_dir).resolve()
    else:
        output_dir=project_root/"data"/"processed"

    try:
        papers=load_json_list(input_path)
    except Exception as exc:
        print("Error: failed to load "+str(input_path)+": "+str(exc),file=sys.stderr)
        return 1

    if args.top and args.top>0:
        papers=papers[:args.top]

    print("DeepSeek mode: "+str(USE_DEEPSEEK_SUMMARY))
    print("DeepSeek key detected: "+str(bool(DEEPSEEK_API_KEY)))
    print("DeepSeek model: "+DEEPSEEK_MODEL)

    if USE_DEEPSEEK_SUMMARY and not DEEPSEEK_API_KEY:
        print("Error: USE_DEEPSEEK_SUMMARY is enabled but DEEPSEEK_API_KEY is missing.",file=sys.stderr)
        return 1

    summaries=[]

    for index,paper in enumerate(papers,start=1):
        title=_get_title(paper)
        print("Summarizing "+str(index)+"/"+str(len(papers))+": "+title)
        summaries.append(summarize_paper(paper,rank=index))

    date_tag=extract_date_tag(input_path)
    summaries_json_path=output_dir/("summaries_"+date_tag+".json")
    summaries_md_path=output_dir/("summaries_"+date_tag+".md")

    write_json(summaries,summaries_json_path)
    write_markdown(summaries,summaries_md_path,date_tag)

    verified,review,other=_group_summaries(summaries)
    deepseek_count=sum(1 for item in summaries if item.get("summary_mode")=="deepseek")
    fallback_count=sum(1 for item in summaries if item.get("summary_mode")=="template_fallback")
    template_count=sum(1 for item in summaries if item.get("summary_mode")=="template")

    print("input:          "+str(input_path))
    print("summaries json: "+str(summaries_json_path))
    print("summaries md:   "+str(summaries_md_path))
    print("papers loaded:  "+str(len(papers)))
    print("summaries made: "+str(len(summaries)))
    print("verified CN:    "+str(len(verified)))
    print("review-level:   "+str(len(review)))
    print("other:          "+str(len(other)))
    print("deepseek:       "+str(deepseek_count))
    print("template:       "+str(template_count))
    print("fallback:       "+str(fallback_count))

    if summaries:
        print("")
        print("Top summaries:")
        for item in summaries[:min(5,len(summaries))]:
            print("- "+str(item.get("title"))+" | confidence="+str(item.get("confidence"))+" | mode="+str(item.get("summary_mode")))
    else:
        print("")
        print("No selected papers to summarize. Empty summary files were still written.")

    return 0


if __name__=="__main__":
    sys.exit(main())
