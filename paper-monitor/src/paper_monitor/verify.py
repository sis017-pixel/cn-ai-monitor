"""
src/paper_monitor/verify.py

Claude verification layer for cn-ai-monitor.

Purpose:
  Verify whether review-level selected papers have enough evidence to be treated
  as China-affiliated.

Inputs:
  - data/processed/selected_papers_YYYY-MM-DD.json
  - optional data/processed/summaries_YYYY-MM-DD.json

Outputs:
  - data/processed/verification_results_YYYY-MM-DD.json
  - data/processed/verified_papers_YYYY-MM-DD.json
  - data/processed/verification_report_YYYY-MM-DD.md

Important verification rule:
  Using/evaluating Chinese-origin models such as Qwen, DeepSeek, BGE, GLM,
  InternVL, InternLM, Vidu, Yi, Kimi, Baichuan, MiniCPM, ChatGLM, or ERNIE is
  not enough by itself to verify paper-level China affiliation. Model usage is
  only a weak China-ecosystem signal.

Verified China affiliation requires one of:
  - institution/country metadata showing a China-based author affiliation
  - explicit author/lab/company affiliation evidence in supplied metadata
  - strong, direct affiliation text in supplied paper metadata

This file uses requests instead of the Anthropic SDK to match the rest of the
project style.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any,Optional

import requests


LOGGER=logging.getLogger("verify")

ANTHROPIC_API_URL_DEFAULT="https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION_DEFAULT="2023-06-01"
ANTHROPIC_MODEL_DEFAULT="claude-sonnet-4-5-20250929"

VERDICTS=(
    "verified_cn",
    "likely_cn",
    "weak_cn_signal",
    "not_cn_affiliated",
    "insufficient_evidence",
)

CONFIDENCE_LEVELS=("high","medium","low")
BUCKETS=("verified_cn","review_candidate","exclude")

AFFILIATION_BASIS_VALUES=(
    "verified_institution_metadata",
    "verified_country_metadata",
    "explicit_institution_in_text",
    "explicit_chinese_company_or_lab",
    "author_affiliation_unclear",
    "chinese_model_used_only",
    "model_benchmark_only",
    "keyword_match_only",
    "no_evidence",
)

_DATE_RE=re.compile(r"(\d{4}-\d{2}-\d{2})")
_WS_RE=re.compile(r"\s+")


def _env_str(name:str,default:str)->str:
    value=os.getenv(name)
    if value is None or value.strip()=="":
        return default
    return value.strip()


def _env_int(name:str,default:int)->int:
    value=os.getenv(name)
    if value is None or value.strip()=="":
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning("Invalid int for %s=%s; using %s",name,value,default)
        return default


def _env_float(name:str,default:float)->float:
    value=os.getenv(name)
    if value is None or value.strip()=="":
        return default
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("Invalid float for %s=%s; using %s",name,value,default)
        return default


def _env_bool(name:str,default:bool)->bool:
    value=os.getenv(name)
    if value is None or value.strip()=="":
        return default
    return value.strip().lower() in {"1","true","yes","y","on"}


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


def _truncate(value:Any,max_chars:int=3500)->str:
    text=_clean_text(value)
    if len(text)<=max_chars:
        return text
    return text[:max_chars-1].rstrip()+"…"


def _normalize_title(title:Any)->str:
    text=_clean_text(title).lower()
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
    text=text.strip()
    return text


def _clean_arxiv_id_no_version(value:Any)->str:
    text=_clean_arxiv_id(value)
    return re.sub(r"v\d+$","",text)


def _get_original(record:dict[str,Any])->dict[str,Any]:
    original=record.get("original")
    if isinstance(original,dict):
        return original
    return {}


def _title(record:dict[str,Any])->str:
    title=_clean_text(record.get("title"))
    if title:
        return title
    original=_get_original(record)
    return _clean_text(original.get("title")) or "(untitled)"


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


def _summary_text(record:dict[str,Any])->str:
    for key in (
        "summary",
        "abstract",
        "tldr",
        "technical_contribution",
        "why_it_matters",
    ):
        value=_clean_text(record.get(key))
        if value:
            return value
    original=_get_original(record)
    for key in (
        "summary",
        "abstract",
        "tldr",
        "technical_contribution",
        "why_it_matters",
    ):
        value=_clean_text(original.get(key))
        if value:
            return value
    return ""


def _authors(record:dict[str,Any])->list[str]:
    values=_safe_list(record.get("authors"))
    if values:
        return [str(item) for item in values if str(item).strip()]
    original=_get_original(record)
    values=_safe_list(original.get("authors"))
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


def _score(record:dict[str,Any],key:str)->float:
    if key in record:
        value=record.get(key)
    else:
        value=_get_original(record).get(key)
    try:
        return float(value)
    except (TypeError,ValueError):
        return 0.0


def _get_field(record:dict[str,Any],key:str)->Any:
    if key in record:
        return record.get(key)
    return _get_original(record).get(key)


def _china_level(record:dict[str,Any])->str:
    return _clean_text(_get_field(record,"china_affiliation_level"))


def _selection_level(record:dict[str,Any])->str:
    return _clean_text(_get_field(record,"digest_selection_level"))


def _needs_review(record:dict[str,Any])->bool:
    return bool(_get_field(record,"needs_review"))


def _openalex_match_accepted(record:dict[str,Any])->Any:
    if "openalex_match_accepted" in record:
        return record.get("openalex_match_accepted")
    original=_get_original(record)
    if "openalex_match_accepted" in original:
        return original.get("openalex_match_accepted")
    return None


def _record_keys(record:dict[str,Any])->list[str]:
    keys=[]
    arxiv=_clean_arxiv_id(_arxiv_id(record))
    arxiv_no_version=_clean_arxiv_id_no_version(_arxiv_id(record))
    title=_normalize_title(_title(record))
    url=_clean_text(_url(record)).lower()

    if arxiv:
        keys.append("arxiv:"+arxiv)
    if arxiv_no_version:
        keys.append("arxiv_no_version:"+arxiv_no_version)
    if title:
        keys.append("title:"+title)
    if url:
        keys.append("url:"+url)

    original=_get_original(record)
    if original:
        original_arxiv=_clean_arxiv_id(original.get("arxiv_id"))
        original_arxiv_no_version=_clean_arxiv_id_no_version(original.get("arxiv_id"))
        original_title=_normalize_title(original.get("title"))
        original_url=_clean_text(original.get("url")).lower()
        if original_arxiv:
            keys.append("arxiv:"+original_arxiv)
        if original_arxiv_no_version:
            keys.append("arxiv_no_version:"+original_arxiv_no_version)
        if original_title:
            keys.append("title:"+original_title)
        if original_url:
            keys.append("url:"+original_url)

    deduped=[]
    seen=set()
    for key in keys:
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def _primary_record_key(record:dict[str,Any])->str:
    keys=_record_keys(record)
    if keys:
        return keys[0]
    return "title:"+_normalize_title(_title(record))


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


def extract_date_tag(path:Path)->str:
    match=_DATE_RE.search(path.name)
    if match:
        return match.group(1)
    return dt.date.today().isoformat()


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
    raise ValueError("Expected JSON list or object containing list at "+str(path))


def write_json(data:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2,default=str)


def build_summary_lookup(summaries:list[dict[str,Any]])->dict[str,dict[str,Any]]:
    lookup={}
    for item in summaries:
        for key in _record_keys(item):
            lookup[key]=item
    return lookup


def lookup_summary(record:dict[str,Any],summary_lookup:dict[str,dict[str,Any]])->Optional[dict[str,Any]]:
    for key in _record_keys(record):
        if key in summary_lookup:
            return summary_lookup[key]
    return None


def is_review_candidate(record:dict[str,Any])->bool:
    if not isinstance(record,dict):
        return False
    if _selection_level(record)=="review":
        return True
    if _china_level(record)=="review":
        return True
    if _needs_review(record):
        return True
    if _openalex_match_accepted(record) is False:
        return True
    return False


def is_strong_candidate(record:dict[str,Any])->bool:
    if not isinstance(record,dict):
        return False
    if _selection_level(record)=="strong":
        return True
    if _china_level(record) in {"strong","verified"}:
        return True
    if "CN" in _countries(record) and _institutions(record):
        return True
    return False


def select_candidates(records:list[dict[str,Any]],include_strong:bool)->list[dict[str,Any]]:
    selected=[]
    for record in records:
        if is_review_candidate(record):
            selected.append(record)
        elif include_strong and is_strong_candidate(record):
            selected.append(record)

    selected.sort(
        key=lambda item:(
            _score(item,"significance_score"),
            _score(item,"ai_relevance_score"),
        ),
        reverse=True,
    )
    return selected


def extract_openalex_authorships(record:dict[str,Any])->Any:
    openalex=record.get("openalex")
    if isinstance(openalex,dict) and openalex.get("authorships"):
        return openalex.get("authorships")

    original=_get_original(record)
    openalex=original.get("openalex")
    if isinstance(openalex,dict) and openalex.get("authorships"):
        return openalex.get("authorships")

    if record.get("authorships"):
        return record.get("authorships")
    if original.get("authorships"):
        return original.get("authorships")

    return None


def build_candidate_payload(record:dict[str,Any],summary_record:Optional[dict[str,Any]])->dict[str,Any]:
    original=_get_original(record)

    china_methods=_safe_list(record.get("china_match_methods")) or _safe_list(original.get("china_match_methods"))
    china_terms=_safe_list(record.get("china_match_terms")) or _safe_list(original.get("china_match_terms"))
    china_signals=_safe_list(record.get("china_signals")) or _safe_list(original.get("china_signals"))
    matched_keywords=_safe_list(record.get("matched_keywords")) or _safe_list(original.get("matched_keywords"))
    matched_model_hints=_safe_list(record.get("matched_model_hints")) or _safe_list(original.get("matched_model_hints"))
    risk_flags=_safe_list(record.get("risk_flags"))

    if summary_record:
        risk_flags=risk_flags or _safe_list(summary_record.get("risk_flags"))

    deepseek_summary=None
    if summary_record:
        deepseek_summary={
            "summary_mode":summary_record.get("summary_mode"),
            "topic":summary_record.get("topic"),
            "summary_bullets":summary_record.get("summary_bullets"),
            "technical_contribution":summary_record.get("technical_contribution"),
            "why_it_matters":summary_record.get("why_it_matters"),
            "china_affiliation_assessment":summary_record.get("china_affiliation_assessment"),
            "risk_flags":summary_record.get("risk_flags"),
            "llm_error":summary_record.get("llm_error"),
        }

    return {
        "title":_title(record),
        "authors":_authors(record),
        "abstract_or_summary":_truncate(_summary_text(record),3000),
        "arxiv_id":_arxiv_id(record),
        "url":_url(record),
        "published":record.get("published") or original.get("published"),
        "categories":record.get("categories") or original.get("categories"),
        "institutions":_institutions(record),
        "countries":_countries(record),
        "authorships":extract_openalex_authorships(record),
        "openalex_id":record.get("openalex_id") or original.get("openalex_id"),
        "openalex_title":record.get("openalex_title") or original.get("openalex_title"),
        "openalex_doi":record.get("openalex_doi") or original.get("openalex_doi"),
        "openalex_match_accepted":_openalex_match_accepted(record),
        "openalex_match_method":record.get("openalex_match_method") or original.get("openalex_match_method"),
        "openalex_match_score":record.get("openalex_match_score") or original.get("openalex_match_score"),
        "china_affiliation_level":_china_level(record),
        "digest_selection_level":_selection_level(record),
        "digest_selection_reason":record.get("digest_selection_reason") or original.get("digest_selection_reason"),
        "needs_review":_needs_review(record),
        "china_match_methods":china_methods,
        "china_match_terms":china_terms,
        "china_signals":china_signals,
        "matched_keywords":matched_keywords,
        "matched_model_hints":matched_model_hints,
        "significance_score":_score(record,"significance_score"),
        "ai_relevance_score":_score(record,"ai_relevance_score"),
        "core_ai_digest_signal":record.get("core_ai_digest_signal") or original.get("core_ai_digest_signal"),
        "deepseek_summary":deepseek_summary,
        "risk_flags":risk_flags,
    }


def build_output_schema()->dict[str,Any]:
    return {
        "type":"object",
        "additionalProperties":False,
        "properties":{
            "verdict":{
                "type":"string",
                "enum":list(VERDICTS),
            },
            "confidence":{
                "type":"string",
                "enum":list(CONFIDENCE_LEVELS),
            },
            "should_include_in_digest":{
                "type":"boolean",
            },
            "recommended_bucket":{
                "type":"string",
                "enum":list(BUCKETS),
            },
            "affiliation_basis":{
                "type":"array",
                "items":{
                    "type":"string",
                    "enum":list(AFFILIATION_BASIS_VALUES),
                },
            },
            "verified_institutions":{
                "type":"array",
                "items":{"type":"string"},
            },
            "evidence_for_cn":{
                "type":"array",
                "items":{"type":"string"},
            },
            "evidence_against_or_missing":{
                "type":"array",
                "items":{"type":"string"},
            },
            "rationale":{
                "type":"string",
            },
            "followup_checks":{
                "type":"array",
                "items":{"type":"string"},
            },
        },
        "required":[
            "verdict",
            "confidence",
            "should_include_in_digest",
            "recommended_bucket",
            "affiliation_basis",
            "verified_institutions",
            "evidence_for_cn",
            "evidence_against_or_missing",
            "rationale",
            "followup_checks",
        ],
    }


def normalize_verification(obj:Any)->dict[str,Any]:
    if not isinstance(obj,dict):
        obj={}

    affiliation_basis=obj.get("affiliation_basis")
    if not isinstance(affiliation_basis,list):
        affiliation_basis=[]

    cleaned_basis=[]
    for item in affiliation_basis:
        value=str(item)
        if value in AFFILIATION_BASIS_VALUES:
            cleaned_basis.append(value)

    out={
        "verdict":obj.get("verdict") or "insufficient_evidence",
        "confidence":obj.get("confidence") or "low",
        "should_include_in_digest":bool(obj.get("should_include_in_digest",False)),
        "recommended_bucket":obj.get("recommended_bucket") or "review_candidate",
        "affiliation_basis":cleaned_basis,
        "verified_institutions":[str(x) for x in _safe_list(obj.get("verified_institutions"))],
        "evidence_for_cn":[str(x) for x in _safe_list(obj.get("evidence_for_cn"))],
        "evidence_against_or_missing":[str(x) for x in _safe_list(obj.get("evidence_against_or_missing"))],
        "rationale":_clean_text(obj.get("rationale")),
        "followup_checks":[str(x) for x in _safe_list(obj.get("followup_checks"))],
    }

    if out["verdict"] not in VERDICTS:
        out["verdict"]="insufficient_evidence"
    if out["confidence"] not in CONFIDENCE_LEVELS:
        out["confidence"]="low"
    if out["recommended_bucket"] not in BUCKETS:
        out["recommended_bucket"]="review_candidate"

    if not out["rationale"]:
        out["rationale"]="No rationale returned."

    return out


def error_verification(error_msg:str)->dict[str,Any]:
    return {
        "verdict":"insufficient_evidence",
        "confidence":"low",
        "should_include_in_digest":False,
        "recommended_bucket":"review_candidate",
        "affiliation_basis":["no_evidence"],
        "verified_institutions":[],
        "evidence_for_cn":[],
        "evidence_against_or_missing":["Claude verification failed: "+error_msg],
        "rationale":"Verification could not be completed, so this paper remains in manual review.",
        "followup_checks":[
            "Manually inspect the arXiv PDF or paper website for author affiliations.",
            "Search OpenAlex/Semantic Scholar again later because metadata for new papers may lag.",
        ],
    }


def anthropic_headers(api_key:str,version:str)->dict[str,str]:
    return {
        "x-api-key":api_key,
        "anthropic-version":version,
        "content-type":"application/json",
    }


def build_prompt_payload(candidate_payload:dict[str,Any])->dict[str,Any]:
    return {
        "task":"Verify whether this AI paper has enough evidence to be treated as China-affiliated.",
        "strict_definition":"China-affiliated means at least one author/lab/company/institution is explicitly affiliated with mainland China, Hong Kong, Macau, or Taiwan, based on the provided metadata.",
        "decision_rules":[
            "verified_cn: provided institution/country metadata or explicit affiliation evidence shows China-based authors, labs, institutions, or companies.",
            "likely_cn: evidence strongly suggests China affiliation but is not fully verified.",
            "weak_cn_signal: only model usage, benchmarked model names, company/model keywords, or other weak ecosystem signals are present.",
            "not_cn_affiliated: evidence indicates non-China affiliation.",
            "insufficient_evidence: affiliation cannot be determined from the provided metadata.",
            "Do not treat use, fine-tuning, benchmarking, or comparison against Qwen, DeepSeek, BGE, GLM, InternVL, InternLM, Vidu, Yi, Kimi, Baichuan, MiniCPM, ChatGLM, or ERNIE as sufficient proof of author affiliation.",
            "Do not invent institutions, countries, labs, authors, or affiliations.",
        ],
        "paper":candidate_payload,
    }


def build_messages(candidate_payload:dict[str,Any])->list[dict[str,str]]:
    user_payload=build_prompt_payload(candidate_payload)
    return [
        {
            "role":"user",
            "content":json.dumps(user_payload,ensure_ascii=False),
        }
    ]


SYSTEM_PROMPT=(
    "You are a conservative verification analyst for a China AI research monitor. "
    "Your job is not to summarize a paper. Your job is to verify affiliation evidence. "
    "Use only the supplied metadata and summary. Do not use outside knowledge to invent or assert author affiliations. "
    "Chinese model usage alone is not sufficient evidence of China affiliation. "
    "Return only a JSON object matching the requested schema."
)


def build_anthropic_payload(candidate_payload:dict[str,Any],cfg:dict[str,Any],structured:bool=True)->dict[str,Any]:
    payload={
        "model":cfg["model"],
        "max_tokens":cfg["max_tokens"],
        "temperature":cfg["temperature"],
        "system":SYSTEM_PROMPT,
        "messages":build_messages(candidate_payload),
    }

    if structured:
        payload["output_config"]={
            "format":{
                "type":"json_schema",
                "schema":build_output_schema(),
            }
        }

    return payload


def is_retryable_status(status:int)->bool:
    return status==429 or 500<=status<=599


def extract_text_from_anthropic_response(data:dict[str,Any])->str:
    content=data.get("content")
    if not isinstance(content,list):
        raise ValueError("Claude response missing content list.")

    texts=[]
    for block in content:
        if not isinstance(block,dict):
            continue
        if block.get("type")=="text" and isinstance(block.get("text"),str):
            texts.append(block["text"])
        elif block.get("type") in {"json","tool_use"} and isinstance(block.get("input"),dict):
            return json.dumps(block["input"],ensure_ascii=False)

    text="\n".join(texts).strip()
    if not text:
        raise ValueError("Claude response contained no parseable text block.")
    return text


def parse_json_object(text:str)->dict[str,Any]:
    try:
        data=json.loads(text)
        if isinstance(data,dict):
            return data
    except json.JSONDecodeError:
        pass

    stripped=text.strip()
    if stripped.startswith("```"):
        stripped=re.sub(r"^```[a-zA-Z]*\s*","",stripped)
        stripped=re.sub(r"\s*```$","",stripped)
        try:
            data=json.loads(stripped)
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

    raise ValueError("Claude response did not contain a valid JSON object.")


def call_claude(candidate_payload:dict[str,Any],cfg:dict[str,Any])->dict[str,Any]:
    last_error=None
    structured=cfg["use_structured_outputs"]

    for attempt in range(1,cfg["retries"]+1):
        try:
            payload=build_anthropic_payload(candidate_payload,cfg,structured=structured)
            response=requests.post(
                cfg["api_url"],
                headers=anthropic_headers(cfg["api_key"],cfg["version"]),
                json=payload,
                timeout=(10,cfg["timeout"]),
            )

            if response.status_code==400 and structured and cfg["fallback_to_prompt_json"]:
                LOGGER.warning(
                    "Claude returned 400 with structured outputs for %s. Retrying once without output_config.",
                    candidate_payload.get("arxiv_id") or candidate_payload.get("title"),
                )
                structured=False
                continue

            if is_retryable_status(response.status_code):
                retry_after=response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_seconds=int(retry_after)
                else:
                    wait_seconds=cfg["retry_base"]*attempt
                if attempt<cfg["retries"]:
                    LOGGER.warning(
                        "Claude returned HTTP %s for %s; retrying in %ss.",
                        response.status_code,
                        candidate_payload.get("arxiv_id") or candidate_payload.get("title"),
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

            response.raise_for_status()
            data=response.json()
            text=extract_text_from_anthropic_response(data)
            parsed=parse_json_object(text)
            return normalize_verification(parsed)

        except Exception as exc:
            last_error=exc
            if attempt<cfg["retries"]:
                wait_seconds=cfg["retry_base"]*attempt
                LOGGER.warning(
                    "Claude verification failed for %s attempt %s/%s: %s",
                    candidate_payload.get("arxiv_id") or candidate_payload.get("title"),
                    attempt,
                    cfg["retries"],
                    exc,
                )
                time.sleep(wait_seconds)

    return error_verification(str(last_error))


def should_keep_verified(verification:dict[str,Any],include_likely:bool)->bool:
    verdict=verification.get("verdict")
    if verdict=="verified_cn":
        return True
    if include_likely and verdict=="likely_cn":
        return True
    return False


def make_result(
    record:dict[str,Any],
    summary_record:Optional[dict[str,Any]],
    verification:dict[str,Any],
    rank:int,
    include_likely:bool,
)->dict[str,Any]:
    payload=build_candidate_payload(record,summary_record)

    return {
        "rank":rank,
        "title":payload["title"],
        "arxiv_id":payload["arxiv_id"],
        "url":payload["url"],
        "authors":payload["authors"],
        "institutions":payload["institutions"],
        "countries":payload["countries"],
        "significance_score":payload["significance_score"],
        "ai_relevance_score":payload["ai_relevance_score"],
        "original_china_affiliation_level":payload["china_affiliation_level"],
        "original_digest_selection_level":payload["digest_selection_level"],
        "original_needs_review":payload["needs_review"],
        "openalex_match_accepted":payload["openalex_match_accepted"],
        "openalex_match_method":payload["openalex_match_method"],
        "openalex_match_score":payload["openalex_match_score"],
        "verification":verification,
        "verified_keep":should_keep_verified(verification,include_likely),
        "candidate_payload":payload,
        "original":record,
    }


def verify_candidates(
    candidates:list[dict[str,Any]],
    summary_lookup:dict[str,dict[str,Any]],
    cfg:dict[str,Any],
)->list[dict[str,Any]]:
    results=[]

    for index,record in enumerate(candidates,start=1):
        summary_record=lookup_summary(record,summary_lookup)
        candidate_payload=build_candidate_payload(record,summary_record)

        LOGGER.info(
            "[%s/%s] verifying: %s - %s",
            index,
            len(candidates),
            candidate_payload.get("arxiv_id") or "",
            str(candidate_payload.get("title"))[:100],
        )

        verification=call_claude(candidate_payload,cfg)
        result=make_result(record,summary_record,verification,index,cfg["include_likely"])
        results.append(result)

        if cfg["sleep_seconds"]>0 and index<len(candidates):
            time.sleep(cfg["sleep_seconds"])

    return results


def build_verified_papers(results:list[dict[str,Any]])->list[dict[str,Any]]:
    verified=[]
    for item in results:
        if not item.get("verified_keep"):
            continue
        original=dict(item.get("original") or {})
        original["claude_verification"]=item.get("verification")
        original["claude_verified_keep"]=True
        original["claude_verification_rank"]=item.get("rank")
        original["claude_recommended_bucket"]=item.get("verification",{}).get("recommended_bucket")
        verified.append(original)
    return verified


def verdict_counts(results:list[dict[str,Any]])->dict[str,int]:
    counts={verdict:0 for verdict in VERDICTS}
    for item in results:
        verdict=str(item.get("verification",{}).get("verdict") or "insufficient_evidence")
        if verdict not in counts:
            counts[verdict]=0
        counts[verdict]+=1
    return counts


def build_markdown_report(
    results:list[dict[str,Any]],
    verified_papers:list[dict[str,Any]],
    date_tag:str,
    include_likely:bool,
)->str:
    counts=verdict_counts(results)

    lines=[]
    lines.append("# CN AI Monitor Claude Verification Report — "+date_tag)
    lines.append("")
    lines.append("Generated: "+_utc_now_iso())
    lines.append("")
    lines.append("This report checks whether selected/review-level papers have enough evidence to be treated as China-affiliated.")
    lines.append("")
    lines.append("## Key rule")
    lines.append("")
    lines.append("Use of a Chinese-origin model such as Qwen, DeepSeek, BGE, GLM, InternVL, InternLM, Vidu, Yi, Kimi, Baichuan, MiniCPM, ChatGLM, or ERNIE is **not sufficient by itself** to verify paper-level China affiliation.")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("- **Candidates sent to Claude:** "+str(len(results)))
    lines.append("- **Kept as verified/likely:** "+str(len(verified_papers)))
    lines.append("- **Include likely_cn in verified set:** "+str(include_likely))
    lines.append("")
    lines.append("| Verdict | Count |")
    lines.append("|---|---:|")
    for verdict in VERDICTS:
        lines.append("| "+verdict+" | "+str(counts.get(verdict,0))+" |")
    lines.append("")

    if verified_papers:
        lines.append("## Papers kept as verified/likely")
        lines.append("")
        for index,item in enumerate([r for r in results if r.get("verified_keep")],start=1):
            verification=item.get("verification") or {}
            lines.append("### "+str(index)+". "+str(item.get("title") or "(untitled)"))
            lines.append("")
            lines.append("- **arXiv:** "+str(item.get("arxiv_id") or "not available"))
            if item.get("url"):
                lines.append("- **URL:** "+str(item.get("url")))
            lines.append("- **Verdict:** "+str(verification.get("verdict")))
            lines.append("- **Confidence:** "+str(verification.get("confidence")))
            lines.append("- **Recommended bucket:** "+str(verification.get("recommended_bucket")))
            institutions=verification.get("verified_institutions") or []
            if institutions:
                lines.append("- **Verified institutions:** "+", ".join(str(x) for x in institutions))
            lines.append("")
            lines.append("**Rationale:** "+str(verification.get("rationale") or ""))
            lines.append("")
    else:
        lines.append("## Papers kept as verified/likely")
        lines.append("")
        lines.append("No review-level papers met the verified/likely threshold.")
        lines.append("")

    lines.append("## All verification results")
    lines.append("")

    if not results:
        lines.append("No papers were sent to Claude verification.")
        lines.append("")
        return "\n".join(lines)

    for item in results:
        verification=item.get("verification") or {}
        lines.append("### "+str(item.get("rank"))+". "+str(item.get("title") or "(untitled)"))
        lines.append("")
        lines.append("- **arXiv:** "+str(item.get("arxiv_id") or "not available"))
        if item.get("url"):
            lines.append("- **URL:** "+str(item.get("url")))
        lines.append("- **Original China level:** "+str(item.get("original_china_affiliation_level")))
        lines.append("- **OpenAlex accepted:** "+str(item.get("openalex_match_accepted")))
        lines.append("- **Verdict:** "+str(verification.get("verdict")))
        lines.append("- **Confidence:** "+str(verification.get("confidence")))
        lines.append("- **Recommended bucket:** "+str(verification.get("recommended_bucket")))
        lines.append("- **Keep as verified/likely:** "+str(item.get("verified_keep")))
        lines.append("")
        lines.append("**Rationale:** "+str(verification.get("rationale") or ""))
        lines.append("")

        basis=verification.get("affiliation_basis") or []
        if basis:
            lines.append("**Affiliation basis:** "+", ".join(str(x) for x in basis))
            lines.append("")

        evidence_for=verification.get("evidence_for_cn") or []
        if evidence_for:
            lines.append("**Evidence for CN:**")
            lines.append("")
            for evidence in evidence_for:
                lines.append("- "+str(evidence))
            lines.append("")

        evidence_missing=verification.get("evidence_against_or_missing") or []
        if evidence_missing:
            lines.append("**Evidence against / missing:**")
            lines.append("")
            for evidence in evidence_missing:
                lines.append("- "+str(evidence))
            lines.append("")

        followups=verification.get("followup_checks") or []
        if followups:
            lines.append("**Follow-up checks:**")
            lines.append("")
            for check in followups:
                lines.append("- "+str(check))
            lines.append("")

    return "\n".join(lines)


def write_outputs(
    results:list[dict[str,Any]],
    verified_papers:list[dict[str,Any]],
    output_dir:Path,
    date_tag:str,
    include_likely:bool,
    input_path:Path,
    summaries_path:Optional[Path],
)->tuple[Path,Path,Path]:
    output_dir.mkdir(parents=True,exist_ok=True)

    verification_results_path=output_dir/("verification_results_"+date_tag+".json")
    verified_papers_path=output_dir/("verified_papers_"+date_tag+".json")
    verification_report_path=output_dir/("verification_report_"+date_tag+".md")

    result_object={
        "generated_at":_utc_now_iso(),
        "date":date_tag,
        "input":str(input_path),
        "summaries":str(summaries_path) if summaries_path else None,
        "include_likely":include_likely,
        "count":len(results),
        "verdict_counts":verdict_counts(results),
        "results":results,
    }

    write_json(result_object,verification_results_path)
    write_json(verified_papers,verified_papers_path)
    verification_report_path.write_text(
        build_markdown_report(results,verified_papers,date_tag,include_likely),
        encoding="utf-8",
    )

    return verification_results_path,verified_papers_path,verification_report_path


def ensure_utf8_stdout()->None:
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower()!="utf-8":
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def parse_args(argv:Optional[list[str]]=None)->argparse.Namespace:
    parser=argparse.ArgumentParser(
        description="Use Claude to verify China affiliation for selected/review-level papers."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to selected_papers_YYYY-MM-DD.json. Default: latest data/processed/selected_papers_*.json.",
    )
    parser.add_argument(
        "--summaries",
        type=str,
        default=None,
        help="Path to summaries_YYYY-MM-DD.json. Default: latest data/processed/summaries_*.json if present.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <project>/data/processed.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Verify top N candidates. 0 means all. Default: CLAUDE_VERIFY_TOP env or 10.",
    )
    parser.add_argument(
        "--include-strong",
        action="store_true",
        help="Also verify strong/verified papers. Default verifies only review-level candidates.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args(argv)


def build_config(args:argparse.Namespace)->dict[str,Any]:
    api_key=os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    top=args.top if args.top is not None else _env_int("CLAUDE_VERIFY_TOP",10)

    return {
        "api_key":api_key,
        "api_url":_env_str("ANTHROPIC_API_URL",ANTHROPIC_API_URL_DEFAULT),
        "version":_env_str("ANTHROPIC_VERSION",ANTHROPIC_VERSION_DEFAULT),
        "model":_env_str("ANTHROPIC_MODEL",ANTHROPIC_MODEL_DEFAULT),
        "max_tokens":_env_int("ANTHROPIC_MAX_TOKENS",1200),
        "temperature":_env_float("ANTHROPIC_TEMPERATURE",0.0),
        "retries":max(1,_env_int("ANTHROPIC_RETRIES",3)),
        "retry_base":_env_float("ANTHROPIC_RETRY_BASE_SECONDS",10.0),
        "timeout":_env_int("ANTHROPIC_TIMEOUT_SECONDS",120),
        "sleep_seconds":_env_float("ANTHROPIC_SLEEP_SECONDS",0.5),
        "include_likely":_env_bool("CLAUDE_INCLUDE_LIKELY",True),
        "top":top,
        "use_structured_outputs":_env_bool("ANTHROPIC_USE_STRUCTURED_OUTPUTS",True),
        "fallback_to_prompt_json":_env_bool("ANTHROPIC_FALLBACK_TO_PROMPT_JSON",True),
    }


def main(argv:Optional[list[str]]=None)->int:
    ensure_utf8_stdout()
    args=parse_args(argv)

    logging.basicConfig(
        level=getattr(logging,args.log_level.upper(),logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    project_root=find_project_root()

    try:
        cfg=build_config(args)
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        return 3

    if args.input:
        input_path=Path(args.input).resolve()
    else:
        found=find_latest_file(project_root,"selected_papers_*.json")
        if found is None:
            LOGGER.error("No selected_papers_*.json found. Provide --input.")
            return 2
        input_path=found.resolve()

    if not input_path.exists():
        LOGGER.error("Input file not found: %s",input_path)
        return 2

    if args.summaries:
        summaries_path=Path(args.summaries).resolve()
    else:
        summaries_path=find_latest_file(project_root,"summaries_*.json")
        summaries_path=summaries_path.resolve() if summaries_path else None

    if args.output_dir:
        output_dir=Path(args.output_dir).resolve()
    else:
        output_dir=project_root/"data"/"processed"

    date_tag=extract_date_tag(input_path)

    LOGGER.info("Loading selected papers from %s",input_path)
    selected=load_json_list(input_path)

    summaries=[]
    if summaries_path and summaries_path.exists():
        LOGGER.info("Loading summaries from %s",summaries_path)
        summaries=load_json_list(summaries_path)
    else:
        LOGGER.info("No summaries file found; continuing without summaries.")

    summary_lookup=build_summary_lookup(summaries)

    candidates=select_candidates(selected,args.include_strong)

    if cfg["top"] and cfg["top"]>0:
        candidates=candidates[:cfg["top"]]

    LOGGER.info("Claude verification mode")
    LOGGER.info("model: %s",cfg["model"])
    LOGGER.info("selected loaded: %s",len(selected))
    LOGGER.info("summaries indexed: %s",len(summary_lookup))
    LOGGER.info("candidates chosen: %s",len(candidates))
    LOGGER.info("top: %s",cfg["top"])
    LOGGER.info("include strong: %s",args.include_strong)
    LOGGER.info("include likely: %s",cfg["include_likely"])
    LOGGER.info("structured outputs: %s",cfg["use_structured_outputs"])

    results=verify_candidates(candidates,summary_lookup,cfg)
    verified_papers=build_verified_papers(results)

    verification_results_path,verified_papers_path,verification_report_path=write_outputs(
        results=results,
        verified_papers=verified_papers,
        output_dir=output_dir,
        date_tag=date_tag,
        include_likely=cfg["include_likely"],
        input_path=input_path,
        summaries_path=summaries_path,
    )

    LOGGER.info("Wrote %s",verification_results_path)
    LOGGER.info("Wrote %s",verified_papers_path)
    LOGGER.info("Wrote %s",verification_report_path)
    LOGGER.info("Verified/likely kept: %s / %s",len(verified_papers),len(results))

    counts=verdict_counts(results)
    for verdict,count in counts.items():
        LOGGER.info("%s: %s",verdict,count)

    return 0


if __name__=="__main__":
    sys.exit(main())
