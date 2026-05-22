"""
src/paper_monitor/publish.py

Final deterministic publishing layer for cn-ai-monitor.

Combines selected papers, DeepSeek summaries, and Claude verification results
into an evidence-aware digest.

Inputs:
  - data/processed/selected_papers_YYYY-MM-DD.json
  - data/processed/summaries_YYYY-MM-DD.json
  - data/processed/verification_results_YYYY-MM-DD.json

Outputs:
  - data/processed/final_digest_YYYY-MM-DD.json
  - data/processed/final_digest_YYYY-MM-DD.md

No API calls. No LLM calls. This is deterministic assembly only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any,Optional


LOGGER=logging.getLogger("publish")

_DATE_RE=re.compile(r"(\d{4}-\d{2}-\d{2})")
_WS_RE=re.compile(r"\s+")

VERIFIED_VERDICTS={"verified_cn","likely_cn"}
ECOSYSTEM_VERDICTS={"weak_cn_signal"}
INSUFFICIENT_VERDICTS={"insufficient_evidence","not_cn_affiliated"}

BUCKET_LABELS={
    "verified_cn":"Verified China-Affiliated AI Papers",
    "ecosystem_signal":"China Ecosystem Signals",
    "insufficient_evidence":"Insufficient Evidence / Audit Appendix",
}


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _utc_now_iso()->str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _clean_text(value:Any)->str:
    return _WS_RE.sub(" ",_safe_str(value)).strip()


def _normalize_title(value:Any)->str:
    text=_clean_text(value).lower()
    text=re.sub(r"[^a-z0-9]+"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text


def _clean_arxiv_id(value:Any)->str:
    text=_clean_text(value).lower()
    if not text:
        return ""
    if "/abs/" in text:
        text=text.split("/abs/")[-1]
    text=re.sub(r"^arxiv:","",text)
    return text.strip()


def _clean_arxiv_id_no_version(value:Any)->str:
    return re.sub(r"v\d+$","",_clean_arxiv_id(value))


def _get_original(record:dict[str,Any])->dict[str,Any]:
    original=record.get("original")
    if isinstance(original,dict):
        return original
    return {}


def _get_field(record:dict[str,Any],key:str)->Any:
    if key in record:
        return record.get(key)
    return _get_original(record).get(key)


def _title(record:dict[str,Any])->str:
    title=_clean_text(record.get("title"))
    if title:
        return title
    title=_clean_text(_get_original(record).get("title"))
    return title or "(untitled)"


def _arxiv_id(record:dict[str,Any])->str:
    for key in ("arxiv_id","id","paper_id"):
        value=_clean_text(record.get(key))
        if value:
            return value
    original=_get_original(record)
    for key in ("arxiv_id","id","paper_id"):
        value=_clean_text(original.get(key))
        if value:
            return value
    return ""


def _url(record:dict[str,Any])->str:
    for key in ("url","landing_page_url","pdf_url"):
        value=_clean_text(record.get(key))
        if value:
            return value
    original=_get_original(record)
    for key in ("url","landing_page_url","pdf_url"):
        value=_clean_text(original.get(key))
        if value:
            return value
    return ""


def _authors(record:dict[str,Any])->list[str]:
    values=_safe_list(record.get("authors"))
    if values:
        return [str(item) for item in values if str(item).strip()]
    values=_safe_list(_get_original(record).get("authors"))
    return [str(item) for item in values if str(item).strip()]


def _institutions(record:dict[str,Any])->list[str]:
    for key in ("institutions","institution_names","affiliations","raw_affiliations"):
        values=_safe_list(record.get(key))
        if values:
            return [str(item) for item in values if str(item).strip()]
    original=_get_original(record)
    for key in ("institutions","institution_names","affiliations","raw_affiliations"):
        values=_safe_list(original.get(key))
        if values:
            return [str(item) for item in values if str(item).strip()]
    return []


def _countries(record:dict[str,Any])->list[str]:
    for key in ("countries","country_codes"):
        values=_safe_list(record.get(key))
        if values:
            return [str(item).upper() for item in values if str(item).strip()]
    original=_get_original(record)
    for key in ("countries","country_codes"):
        values=_safe_list(original.get(key))
        if values:
            return [str(item).upper() for item in values if str(item).strip()]
    return []


def _summary_text(record:dict[str,Any])->str:
    for key in ("summary","abstract","technical_contribution","why_it_matters"):
        value=_clean_text(record.get(key))
        if value:
            return value
    original=_get_original(record)
    for key in ("summary","abstract","technical_contribution","why_it_matters"):
        value=_clean_text(original.get(key))
        if value:
            return value
    return ""


def _score(record:dict[str,Any],key:str)->float:
    value=_get_field(record,key)
    try:
        return float(value)
    except (TypeError,ValueError):
        return 0.0


def _record_keys(record:dict[str,Any])->list[str]:
    keys=[]

    def add_keys(arxiv:Any,title:Any,url:Any)->None:
        cleaned_arxiv=_clean_arxiv_id(arxiv)
        cleaned_arxiv_no_version=_clean_arxiv_id_no_version(arxiv)
        normalized_title=_normalize_title(title)
        cleaned_url=_clean_text(url).lower()
        if cleaned_arxiv:
            keys.append("arxiv:"+cleaned_arxiv)
        if cleaned_arxiv_no_version:
            keys.append("arxiv_no_version:"+cleaned_arxiv_no_version)
        if normalized_title:
            keys.append("title:"+normalized_title)
        if cleaned_url:
            keys.append("url:"+cleaned_url)

    add_keys(_arxiv_id(record),_title(record),_url(record))

    original=_get_original(record)
    if original:
        add_keys(original.get("arxiv_id"),original.get("title"),original.get("url"))

    candidate_payload=record.get("candidate_payload")
    if isinstance(candidate_payload,dict):
        add_keys(
            candidate_payload.get("arxiv_id"),
            candidate_payload.get("title"),
            candidate_payload.get("url"),
        )

    seen=set()
    deduped=[]
    for key in keys:
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

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


def extract_date_tag(path:Path)->str:
    match=_DATE_RE.search(path.name)
    if match:
        return match.group(1)
    return dt.date.today().isoformat()


def find_latest_file(project_root:Path,pattern:str)->Optional[Path]:
    processed_dir=project_root/"data"/"processed"
    if not processed_dir.exists():
        return None

    candidates=list(processed_dir.glob(pattern))
    if not candidates:
        return None

    def sort_key(path:Path)->tuple[str,float]:
        match=_DATE_RE.search(path.name)
        date_value=match.group(1) if match else ""
        try:
            mtime=path.stat().st_mtime
        except OSError:
            mtime=0.0
        return (date_value,mtime)

    return sorted(candidates,key=sort_key)[-1]


def load_json(path:Path)->Any:
    with path.open("r",encoding="utf-8") as f:
        return json.load(f)


def load_json_list(path:Path)->list[dict[str,Any]]:
    data=load_json(path)
    if isinstance(data,list):
        return [item for item in data if isinstance(item,dict)]
    if isinstance(data,dict):
        for key in ("papers","selected","items","summaries","results"):
            values=data.get(key)
            if isinstance(values,list):
                return [item for item in values if isinstance(item,dict)]
    raise ValueError("Expected JSON list or object containing a list at "+str(path))


def write_json(data:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,default=str)


def build_lookup(records:list[dict[str,Any]])->dict[str,dict[str,Any]]:
    lookup={}
    for record in records:
        for key in _record_keys(record):
            lookup[key]=record
    return lookup


def lookup_record(record:dict[str,Any],lookup:dict[str,dict[str,Any]])->Optional[dict[str,Any]]:
    for key in _record_keys(record):
        if key in lookup:
            return lookup[key]
    return None


# ---------------------------------------------------------------------------
# Bucket logic
# ---------------------------------------------------------------------------

def verification_from_result(result:Optional[dict[str,Any]])->dict[str,Any]:
    if not result:
        return {}
    verification=result.get("verification")
    if isinstance(verification,dict):
        return verification
    return {}


def get_verdict(result:Optional[dict[str,Any]])->str:
    verification=verification_from_result(result)
    return _clean_text(verification.get("verdict"))


def is_strong_selected(record:dict[str,Any])->bool:
    digest_level=_clean_text(_get_field(record,"digest_selection_level"))
    china_level=_clean_text(_get_field(record,"china_affiliation_level"))
    countries=_countries(record)
    institutions=_institutions(record)

    if digest_level=="strong":
        return True
    if china_level in {"strong","verified"}:
        return True
    if "CN" in countries and institutions:
        return True
    return False


def is_review_selected(record:dict[str,Any])->bool:
    digest_level=_clean_text(_get_field(record,"digest_selection_level"))
    china_level=_clean_text(_get_field(record,"china_affiliation_level"))

    if digest_level=="review":
        return True
    if china_level=="review":
        return True
    if bool(_get_field(record,"selected_via_review_gate")):
        return True
    if bool(_get_field(record,"needs_review")):
        return True
    return False


def determine_bucket(record:dict[str,Any],verification_result:Optional[dict[str,Any]])->tuple[str,str]:
    verdict=get_verdict(verification_result)

    if is_strong_selected(record):
        return "verified_cn","strong_classification_signal"

    if verdict in VERIFIED_VERDICTS:
        return "verified_cn","claude_promoted_"+verdict

    if verdict in ECOSYSTEM_VERDICTS:
        return "ecosystem_signal","claude_weak_cn_signal"

    if verdict in INSUFFICIENT_VERDICTS:
        return "insufficient_evidence","claude_"+verdict

    if is_review_selected(record):
        return "insufficient_evidence","review_candidate_without_claude_verdict"

    return "insufficient_evidence","not_selected_or_no_evidence"


# ---------------------------------------------------------------------------
# Digest item assembly
# ---------------------------------------------------------------------------

def summary_from_record(summary_record:Optional[dict[str,Any]],selected_record:dict[str,Any])->dict[str,Any]:
    if summary_record:
        return {
            "summary_mode":summary_record.get("summary_mode"),
            "topic":summary_record.get("topic"),
            "summary_bullets":summary_record.get("summary_bullets") or [],
            "technical_contribution":summary_record.get("technical_contribution"),
            "why_it_matters":summary_record.get("why_it_matters"),
            "china_affiliation_assessment":summary_record.get("china_affiliation_assessment"),
            "policy_or_market_relevance":summary_record.get("policy_or_market_relevance"),
            "confidence":summary_record.get("confidence"),
            "risk_flags":summary_record.get("risk_flags") or [],
            "llm_error":summary_record.get("llm_error"),
        }

    summary_text=_summary_text(selected_record)
    bullets=[]
    if summary_text:
        bullets.append(summary_text[:500]+("…" if len(summary_text)>500 else ""))

    return {
        "summary_mode":"metadata_only",
        "topic":"AI research",
        "summary_bullets":bullets,
        "technical_contribution":None,
        "why_it_matters":None,
        "china_affiliation_assessment":None,
        "policy_or_market_relevance":None,
        "confidence":None,
        "risk_flags":[],
        "llm_error":None,
    }


def build_digest_item(
    selected_record:dict[str,Any],
    summary_record:Optional[dict[str,Any]],
    verification_result:Optional[dict[str,Any]],
    rank:int,
)->dict[str,Any]:
    bucket,bucket_reason=determine_bucket(selected_record,verification_result)
    verification=verification_from_result(verification_result)
    summary=summary_from_record(summary_record,selected_record)

    return {
        "rank":rank,
        "bucket":bucket,
        "bucket_label":BUCKET_LABELS[bucket],
        "bucket_reason":bucket_reason,
        "title":_title(selected_record),
        "authors":_authors(selected_record),
        "arxiv_id":_arxiv_id(selected_record),
        "url":_url(selected_record),
        "published":_get_field(selected_record,"published"),
        "institutions":_institutions(selected_record),
        "countries":_countries(selected_record),
        "significance_score":_score(selected_record,"significance_score"),
        "ai_relevance_score":_score(selected_record,"ai_relevance_score"),
        "china_affiliation_level":_clean_text(_get_field(selected_record,"china_affiliation_level")),
        "digest_selection_level":_clean_text(_get_field(selected_record,"digest_selection_level")),
        "digest_selection_reason":_get_field(selected_record,"digest_selection_reason"),
        "selected_via_review_gate":bool(_get_field(selected_record,"selected_via_review_gate")),
        "needs_review":bool(_get_field(selected_record,"needs_review")),
        "openalex_id":_get_field(selected_record,"openalex_id"),
        "openalex_match_accepted":_get_field(selected_record,"openalex_match_accepted"),
        "openalex_match_method":_get_field(selected_record,"openalex_match_method"),
        "openalex_match_score":_get_field(selected_record,"openalex_match_score"),
        "summary":summary,
        "claude_verification":verification,
        "claude_verdict":verification.get("verdict"),
        "claude_confidence":verification.get("confidence"),
        "evidence_for_cn":verification.get("evidence_for_cn") or [],
        "evidence_against_or_missing":verification.get("evidence_against_or_missing") or [],
        "verified_institutions":verification.get("verified_institutions") or [],
        "followup_checks":verification.get("followup_checks") or [],
        "original":selected_record,
    }


def sort_digest_items(items:list[dict[str,Any]])->list[dict[str,Any]]:
    bucket_order={"verified_cn":0,"ecosystem_signal":1,"insufficient_evidence":2}
    return sorted(
        items,
        key=lambda item:(
            bucket_order.get(item["bucket"],99),
            -float(item.get("significance_score") or 0),
            -float(item.get("ai_relevance_score") or 0),
            str(item.get("title") or ""),
        ),
    )


def assemble_digest(
    selected:list[dict[str,Any]],
    summaries:list[dict[str,Any]],
    verification_results:list[dict[str,Any]],
    date_tag:str,
    input_paths:dict[str,Optional[str]],
)->dict[str,Any]:
    summary_lookup=build_lookup(summaries)
    verification_lookup=build_lookup(verification_results)
    items=[]

    for selected_record in selected:
        summary_record=lookup_record(selected_record,summary_lookup)
        verification_result=lookup_record(selected_record,verification_lookup)
        item=build_digest_item(selected_record,summary_record,verification_result,rank=0)
        items.append(item)

    items=sort_digest_items(items)
    for index,item in enumerate(items,start=1):
        item["rank"]=index

    buckets={
        "verified_cn":[item for item in items if item["bucket"]=="verified_cn"],
        "ecosystem_signal":[item for item in items if item["bucket"]=="ecosystem_signal"],
        "insufficient_evidence":[item for item in items if item["bucket"]=="insufficient_evidence"],
    }

    verification_counts={}
    for item in items:
        verdict=item.get("claude_verdict") or "no_claude_verdict"
        verification_counts[verdict]=verification_counts.get(verdict,0)+1

    return {
        "generated_at":_utc_now_iso(),
        "date":date_tag,
        "input_paths":input_paths,
        "method_note":"Verified China-affiliated papers require institution/country/affiliation evidence. Chinese model usage alone is treated as an ecosystem signal, not proof of paper-level China affiliation.",
        "counts":{
            "selected_total":len(selected),
            "summaries_loaded":len(summaries),
            "verification_results_loaded":len(verification_results),
            "verified_cn":len(buckets["verified_cn"]),
            "ecosystem_signal":len(buckets["ecosystem_signal"]),
            "insufficient_evidence":len(buckets["insufficient_evidence"]),
            "verification_counts":verification_counts,
        },
        "buckets":buckets,
        "items":items,
    }


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

def _format_list(values:list[Any],fallback:str="not available")->str:
    cleaned=[str(value) for value in values if str(value).strip()]
    if not cleaned:
        return fallback
    return ", ".join(cleaned)


def _item_link(item:dict[str,Any])->str:
    title=str(item.get("title") or "(untitled)")
    url=str(item.get("url") or "")
    if url:
        return "["+title+"]("+url+")"
    return title


def write_item_markdown(lines:list[str],item:dict[str,Any],show_affiliation_detail:bool=True)->None:
    lines.append("### "+str(item.get("rank"))+". "+_item_link(item))
    lines.append("")
    lines.append("- **arXiv:** "+str(item.get("arxiv_id") or "not available"))
    lines.append("- **Topic:** "+str(item.get("summary",{}).get("topic") or "AI research"))
    lines.append("- **Scores:** significance="+str(item.get("significance_score"))+", AI relevance="+str(item.get("ai_relevance_score")))
    lines.append("- **Institutions:** "+_format_list(item.get("institutions") or []))
    lines.append("- **Countries:** "+_format_list(item.get("countries") or []))
    lines.append("- **Evidence bucket:** "+str(item.get("bucket_label"))+" — "+str(item.get("bucket_reason")))
    if item.get("claude_verdict"):
        lines.append("- **Claude verdict:** "+str(item.get("claude_verdict"))+" / confidence="+str(item.get("claude_confidence")))
    else:
        lines.append("- **Claude verdict:** not run or not matched")
    lines.append("")

    bullets=item.get("summary",{}).get("summary_bullets") or []
    if bullets:
        lines.append("**Summary**")
        lines.append("")
        for bullet in bullets[:4]:
            lines.append("- "+str(bullet))
        lines.append("")

    technical=item.get("summary",{}).get("technical_contribution")
    if technical:
        lines.append("**Technical contribution:** "+str(technical))
        lines.append("")

    why=item.get("summary",{}).get("why_it_matters")
    if why:
        lines.append("**Why it matters:** "+str(why))
        lines.append("")

    if show_affiliation_detail:
        evidence_for=item.get("evidence_for_cn") or []
        evidence_missing=item.get("evidence_against_or_missing") or []
        if evidence_for:
            lines.append("**Evidence for China affiliation:**")
            lines.append("")
            for evidence in evidence_for[:5]:
                lines.append("- "+str(evidence))
            lines.append("")
        if evidence_missing:
            lines.append("**Evidence limitations:**")
            lines.append("")
            for evidence in evidence_missing[:5]:
                lines.append("- "+str(evidence))
            lines.append("")


def write_markdown_digest(digest:dict[str,Any],path:Path)->None:
    counts=digest["counts"]
    buckets=digest["buckets"]
    date_tag=str(digest["date"])

    lines=[]
    lines.append("# CN AI Monitor Final Digest — "+date_tag)
    lines.append("")
    lines.append("Generated: "+str(digest["generated_at"]))
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append("- **Verified China-affiliated AI papers:** "+str(counts["verified_cn"]))
    lines.append("- **China ecosystem signals:** "+str(counts["ecosystem_signal"]))
    lines.append("- **Insufficient evidence / audit appendix:** "+str(counts["insufficient_evidence"]))
    lines.append("- **Selected papers before publishing filter:** "+str(counts["selected_total"]))
    lines.append("")
    lines.append("## Method note")
    lines.append("")
    lines.append(digest["method_note"])
    lines.append("")
    lines.append("This distinction is intentional: a paper that uses or benchmarks Qwen, DeepSeek, BGE, GLM, InternVL, Vidu, or another Chinese-origin model is strategically relevant, but it is not automatically a China-affiliated paper unless author/institution evidence supports that claim.")
    lines.append("")

    lines.append("## "+BUCKET_LABELS["verified_cn"])
    lines.append("")
    if buckets["verified_cn"]:
        for item in buckets["verified_cn"]:
            write_item_markdown(lines,item,show_affiliation_detail=True)
    else:
        lines.append("No papers met the verified China-affiliation threshold in this run.")
        lines.append("")

    lines.append("## "+BUCKET_LABELS["ecosystem_signal"])
    lines.append("")
    lines.append("These papers are useful for monitoring the China AI ecosystem, but they should not be described as verified China-affiliated papers.")
    lines.append("")
    if buckets["ecosystem_signal"]:
        for item in buckets["ecosystem_signal"]:
            write_item_markdown(lines,item,show_affiliation_detail=True)
    else:
        lines.append("No weak China ecosystem signals were identified.")
        lines.append("")

    lines.append("## "+BUCKET_LABELS["insufficient_evidence"])
    lines.append("")
    lines.append("These papers were selected or reviewed by the pipeline, but the available metadata does not provide enough evidence to present them as China-affiliated or even as strong ecosystem signals.")
    lines.append("")
    if buckets["insufficient_evidence"]:
        for item in buckets["insufficient_evidence"]:
            write_item_markdown(lines,item,show_affiliation_detail=True)
    else:
        lines.append("No papers fell into the insufficient-evidence bucket.")
        lines.append("")

    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(lines),encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv:Optional[list[str]]=None)->argparse.Namespace:
    parser=argparse.ArgumentParser(description="Assemble final evidence-aware CN AI monitor digest.")
    parser.add_argument("--selected",type=str,default=None,help="Path to selected_papers_YYYY-MM-DD.json. Default: latest.")
    parser.add_argument("--summaries",type=str,default=None,help="Path to summaries_YYYY-MM-DD.json. Default: latest if present.")
    parser.add_argument("--verification-results",type=str,default=None,help="Path to verification_results_YYYY-MM-DD.json. Default: latest if present.")
    parser.add_argument("--output-dir",type=str,default=None,help="Output directory. Default: <project>/data/processed.")
    parser.add_argument("--log-level",type=str,default="INFO",help="Logging level.")
    return parser.parse_args(argv)


def ensure_utf8_stdout()->None:
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower()!="utf-8":
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main(argv:Optional[list[str]]=None)->int:
    ensure_utf8_stdout()
    args=parse_args(argv)

    logging.basicConfig(
        level=getattr(logging,args.log_level.upper(),logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    project_root=find_project_root()

    if args.selected:
        selected_path=Path(args.selected).resolve()
    else:
        found=find_latest_file(project_root,"selected_papers_*.json")
        if found is None:
            LOGGER.error("No selected_papers_*.json found. Provide --selected.")
            return 2
        selected_path=found.resolve()

    if not selected_path.exists():
        LOGGER.error("Selected file not found: %s",selected_path)
        return 2

    if args.summaries:
        summaries_path=Path(args.summaries).resolve()
    else:
        found=find_latest_file(project_root,"summaries_*.json")
        summaries_path=found.resolve() if found else None

    if args.verification_results:
        verification_path=Path(args.verification_results).resolve()
    else:
        found=find_latest_file(project_root,"verification_results_*.json")
        verification_path=found.resolve() if found else None

    output_dir=Path(args.output_dir).resolve() if args.output_dir else project_root/"data"/"processed"
    date_tag=extract_date_tag(selected_path)

    LOGGER.info("Loading selected papers from %s",selected_path)
    selected=load_json_list(selected_path)

    summaries=[]
    if summaries_path and summaries_path.exists():
        LOGGER.info("Loading summaries from %s",summaries_path)
        summaries=load_json_list(summaries_path)
    else:
        LOGGER.warning("No summaries file found; final digest will use metadata-only summaries.")

    verification_results=[]
    if verification_path and verification_path.exists():
        LOGGER.info("Loading verification results from %s",verification_path)
        verification_results=load_json_list(verification_path)
    else:
        LOGGER.warning("No verification results found; review-level papers will remain insufficient evidence.")

    input_paths={
        "selected":str(selected_path),
        "summaries":str(summaries_path) if summaries_path else None,
        "verification_results":str(verification_path) if verification_path else None,
    }

    digest=assemble_digest(
        selected=selected,
        summaries=summaries,
        verification_results=verification_results,
        date_tag=date_tag,
        input_paths=input_paths,
    )

    final_json_path=output_dir/("final_digest_"+date_tag+".json")
    final_md_path=output_dir/("final_digest_"+date_tag+".md")

    write_json(digest,final_json_path)
    write_markdown_digest(digest,final_md_path)

    LOGGER.info("Wrote %s",final_json_path)
    LOGGER.info("Wrote %s",final_md_path)
    LOGGER.info("verified_cn: %s",digest["counts"]["verified_cn"])
    LOGGER.info("ecosystem_signal: %s",digest["counts"]["ecosystem_signal"])
    LOGGER.info("insufficient_evidence: %s",digest["counts"]["insufficient_evidence"])

    return 0


if __name__=="__main__":
    sys.exit(main())
