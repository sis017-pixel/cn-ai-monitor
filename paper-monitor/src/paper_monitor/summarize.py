"""Template summary layer for cn-ai-monitor.

Reads selected papers from data/processed/selected_papers_YYYY-MM-DD.json and writes:

- data/processed/summaries_YYYY-MM-DD.json
- data/processed/summaries_YYYY-MM-DD.md

This module performs no network access. It creates deterministic summaries first; LLM
summarization can be added after the pipeline is stable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any,Optional


_SENTENCE_SPLIT_RE=re.compile(r"(?<=[.!?])\s+")
_DATE_TAG_RE=re.compile(r"(\d{4}-\d{2}-\d{2})")
_WS_RE=re.compile(r"\s+")

CORE_TOPIC_RULES:tuple[tuple[str,tuple[str,...]],...]=(
    ("large language models",("large language model","llm","language model","reasoning model","instruction tuning","alignment","pretraining")),
    ("multimodal AI",("multimodal","vision-language","vision language","vlm","image generation","video generation","text-to-video","text to video")),
    ("AI agents and tool use",("agentic","ai agent","tool use","workflow","planning","plan-execute")),
    ("benchmarks and datasets",("benchmark","dataset","evaluation","leaderboard")),
    ("retrieval and recommendation",("retrieval","rag","recommendation","recommender","personalized recommendation")),
    ("model architecture or training",("transformer","mixture of experts","moe","diffusion","reinforcement learning","scaling")),
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


def summarize_paper(paper:dict[str,Any],rank:int)->dict[str,Any]:
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

    why_it_matters=(
        "This paper is worth tracking because it passed the current digest filter for "
        +topic
        +" and has a verified or high-confidence China-affiliation signal."
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
        "why_it_matters":why_it_matters,
        "significance_score":significance_score,
        "ai_relevance_score":ai_relevance_score,
        "china_affiliation_level":china_level,
        "core_ai_digest_signal":paper.get("core_ai_digest_signal"),
        "openalex_match_method":paper.get("openalex_match_method"),
        "openalex_match_score":paper.get("openalex_match_score"),
        "openalex_match_accepted":paper.get("openalex_match_accepted"),
        "key_signals":_extract_key_signals(paper),
        "risk_flags":_make_risk_flags(paper),
        "confidence":confidence,
        "original":paper,
    }


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


def write_markdown(summaries:list[dict[str,Any]],path:Path,date_tag:str)->None:
    lines=[]
    lines.append("# CN AI Monitor Summaries — "+date_tag)
    lines.append("")

    if not summaries:
        lines.append("No selected papers passed the current digest filter.")
        lines.append("")
    else:
        for item in summaries:
            lines.append("## "+str(item["rank"])+". "+item["title"])
            lines.append("")
            lines.append("- **Topic:** "+str(item.get("topic") or "AI research"))
            lines.append("- **Institutions:** "+(", ".join(item.get("institutions") or []) or "not available"))
            lines.append("- **Countries:** "+(", ".join(item.get("countries") or []) or "not available"))
            lines.append("- **Scores:** significance="+str(item.get("significance_score"))+", AI="+str(item.get("ai_relevance_score")))
            lines.append("- **Confidence:** "+str(item.get("confidence")))
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
            lines.append("**Why it matters:** "+str(item.get("why_it_matters") or ""))
            lines.append("")
            risk_flags=item.get("risk_flags") or []
            if risk_flags:
                lines.append("**Risk flags:** "+", ".join(str(flag) for flag in risk_flags))
                lines.append("")

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
        description="Create template summaries for selected CN AI papers."
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

    args=parser.parse_args(argv)
    project_root=find_project_root()

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

    summaries=[
        summarize_paper(paper,rank=index)
        for index,paper in enumerate(papers,start=1)
    ]

    date_tag=extract_date_tag(input_path)
    summaries_json_path=output_dir/("summaries_"+date_tag+".json")
    summaries_md_path=output_dir/("summaries_"+date_tag+".md")

    write_json(summaries,summaries_json_path)
    write_markdown(summaries,summaries_md_path,date_tag)

    print("input:          "+str(input_path))
    print("summaries json: "+str(summaries_json_path))
    print("summaries md:   "+str(summaries_md_path))
    print("papers loaded:  "+str(len(papers)))
    print("summaries made: "+str(len(summaries)))

    if summaries:
        print("")
        print("Top summaries:")
        for item in summaries[:min(5,len(summaries))]:
            print("- "+str(item.get("title"))+" | confidence="+str(item.get("confidence"))+" | topic="+str(item.get("topic")))
    else:
        print("")
        print("No selected papers to summarize. Empty summary files were still written.")

    return 0


if __name__=="__main__":
    sys.exit(main())
