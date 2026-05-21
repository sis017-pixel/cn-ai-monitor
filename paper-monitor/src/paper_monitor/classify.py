"""Classification layer for fetched AI papers.

Reads raw papers produced by fetch.py and writes four artifacts to data/processed/:

- classified_papers_YYYY-MM-DD.json
- selected_papers_YYYY-MM-DD.json
- review_candidates_YYYY-MM-DD.json
- classification_summary_YYYY-MM-DD.json

This module performs no network access.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass,field
from datetime import datetime
from pathlib import Path
from typing import Any,Iterable,Optional

try:
    import yaml  # type: ignore
    HAS_YAML=True
except ImportError:
    HAS_YAML=False


AI_CATEGORIES=frozenset(
    {
        "cs.AI",
        "cs.CL",
        "cs.CV",
        "cs.LG",
        "stat.ML",
        "eess.AS",
    }
)

AI_KEYWORDS_PHRASES:tuple[str,...]=(
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "large language model",
    "foundation model",
    "reasoning model",
    "vision-language",
    "vision language",
    "ai agent",
    "reinforcement learning",
    "text-to-video",
    "text to video",
    "video generation",
    "image generation",
    "mixture of experts",
    "instruction tuning",
    "tool use",
    "mathematical reasoning",
)

AI_KEYWORDS_TOKENS:tuple[str,...]=(
    "AI",
    "LLM",
    "VLM",
    "multimodal",
    "agentic",
    "diffusion",
    "benchmark",
    "dataset",
    "retrieval",
    "RAG",
    "embedding",
    "transformer",
    "MoE",
    "pretraining",
    "alignment",
    "coding",
)

FRONTIER_PHRASES:tuple[str,...]=(
    "technical report",
    "state-of-the-art",
    "state of the art",
    "foundation model",
    "large language model",
    "mixture of experts",
    "vision-language",
    "vision language",
    "text-to-video",
    "text to video",
    "video generation",
    "open-source",
    "open source",
    "open weight",
    "open weights",
)

FRONTIER_TOKENS:tuple[str,...]=(
    "frontier",
    "SOTA",
    "LLM",
    "MoE",
    "reasoning",
    "agentic",
    "multimodal",
    "benchmark",
    "dataset",
    "scaling",
)

CORE_AI_DIGEST_PHRASES:tuple[str,...]=(
    "large language model",
    "foundation model",
    "reasoning model",
    "vision-language",
    "vision language",
    "text-to-video",
    "text to video",
    "video generation",
    "image generation",
    "mixture of experts",
    "reinforcement learning",
    "instruction tuning",
    "agentic",
    "ai agent",
    "benchmark",
    "dataset",
    "retrieval augmented generation",
    "mathematical reasoning",
    "code generation",
    "tool use",
)

CORE_AI_DIGEST_TOKENS:tuple[str,...]=(
    "LLM",
    "VLM",
    "MoE",
    "RAG",
    "transformer",
    "diffusion",
    "multimodal",
    "pretraining",
    "alignment",
    "coding",
)

LOW_PRIORITY_TERMS:tuple[str,...]=(
    "survey",
    "position paper",
    "editorial",
    "commentary",
    "workshop",
)

CN_INSTITUTION_PHRASES:tuple[str,...]=(
    "Tsinghua",
    "Tsinghua University",
    "Peking University",
    "Zhejiang University",
    "Shanghai Jiao Tong",
    "Shanghai Jiao Tong University",
    "Fudan",
    "Fudan University",
    "University of Science and Technology of China",
    "Chinese Academy of Sciences",
    "Institute of Automation",
    "Shanghai AI Laboratory",
    "Shanghai AI Lab",
    "Beijing Academy of Artificial Intelligence",
    "Harbin Institute of Technology",
    "Nanjing University",
    "Renmin University",
    "Beihang",
    "Beihang University",
    "Xi'an Jiaotong",
    "Xi’an Jiaotong",
    "Xi'an Jiaotong University",
    "Sun Yat-sen University",
    "Wuhan University",
    "Huazhong University",
    "National University of Defense Technology",
    "University of Electronic Science and Technology of China",
    "Southern University of Science and Technology",
    "Sichuan University",
    "Xiamen University",
    "Tongji University",
    "Beijing Institute of Technology",
    "Dalian University of Technology",
    "South China University of Technology",
    "Central South University",
    "Hunan University",
    "East China Normal University",
    "East China University of Science and Technology",
    "ShanghaiTech University",
    "China University of Mining and Technology",
    "Nankai University",
    "Shandong University",
    "Jilin University",
    "Southeast University",
    "Northwestern Polytechnical University",
    "University of Hong Kong",
    "Hong Kong University of Science and Technology",
    "Chinese University of Hong Kong",
    "City University of Hong Kong",
    "Hong Kong Polytechnic University",
)

CN_INSTITUTION_TOKENS:tuple[str,...]=(
    "PKU",
    "SJTU",
    "USTC",
    "CAS",
    "BAAI",
    "HIT",
    "NUDT",
    "UESTC",
    "SUSTech",
    "CUHK",
    "HKUST",
    "HKU",
)

CN_COMPANY_PHRASES:tuple[str,...]=(
    "DeepSeek",
    "DeepSeek-AI",
    "Alibaba",
    "Alibaba Cloud",
    "Qwen",
    "Tencent",
    "Tencent AI Lab",
    "Hunyuan",
    "Baidu",
    "ERNIE",
    "ByteDance",
    "ByteDance Seed",
    "Doubao",
    "Huawei",
    "Pangu",
    "PanGu",
    "Moonshot AI",
    "Kimi",
    "MiniMax",
    "Hailuo",
    "Zhipu",
    "Zhipu AI",
    "Z.ai",
    "ChatGLM",
    "01.AI",
    "Baichuan",
    "Baichuan AI",
    "SenseTime",
    "SenseNova",
    "iFLYTEK",
    "SparkDesk",
    "Kuaishou",
    "Kling",
    "StepFun",
    "Shengshu",
    "Vidu",
    "ModelScope",
    "OpenBMB",
    "InternLM",
    "InternVL",
    "MiniCPM",
    "Aquila",
    "BGE",
    "Skywork",
    "TeleChat",
    "MOSS",
    "CogVideo",
    "CogVideoX",
)

AMBIGUOUS_MODEL_ALIASES:tuple[str,...]=(
    "Yi",
    "Seed",
    "GLM",
)

_TITLE_NORM_RE=re.compile(r"[^a-z0-9 ]+")
_WS_RE=re.compile(r"\s+")


@dataclass
class ChinaAffiliation:
    is_china_affiliated:bool=False
    level:str="none"
    methods:list[str]=field(default_factory=list)
    terms:list[str]=field(default_factory=list)


@dataclass
class AIRelevance:
    score:int=0
    reasons:list[str]=field(default_factory=list)


@dataclass
class Significance:
    score:int=0
    reasons:list[str]=field(default_factory=list)


def _word_match(haystack:str,needle:str)->bool:
    if not haystack or not needle:
        return False
    pattern=r"\b"+re.escape(needle)+r"\b"
    return re.search(pattern,haystack,re.IGNORECASE) is not None


def _phrase_match(haystack:str,needle:str)->bool:
    if not haystack or not needle:
        return False
    return needle.lower() in haystack.lower()


def _any_phrase(haystack:str,needles:Iterable[str])->bool:
    return any(_phrase_match(haystack,n) for n in needles)


def _any_word(haystack:str,needles:Iterable[str])->bool:
    return any(_word_match(haystack,n) for n in needles)


def _safe_list(value:Any)->list[Any]:
    if isinstance(value,list):
        return value
    return []


def _safe_str(value:Any)->str:
    if value is None:
        return ""
    if isinstance(value,list):
        return " ".join(str(x) for x in value)
    return str(value)


def _norm_title(title:str)->str:
    t=(title or "").strip().lower()
    t=_TITLE_NORM_RE.sub(" ",t)
    t=_WS_RE.sub(" ",t).strip()
    return t


def _clean_arxiv_id(arxiv_id:str)->str:
    return re.sub(r"v\d+$","",arxiv_id.strip().lower())


def _dedup_key(record:dict[str,Any])->str:
    arxiv_id=record.get("arxiv_id")
    if isinstance(arxiv_id,str) and arxiv_id.strip():
        return "arxiv:"+_clean_arxiv_id(arxiv_id)

    doi=record.get("doi")
    if isinstance(doi,str) and doi.strip():
        return "doi:"+doi.strip().lower()

    openalex=record.get("openalex_id")
    if isinstance(openalex,str) and openalex.strip():
        return "openalex:"+openalex.strip().lower()

    return "title:"+_norm_title(record.get("title") or "")


def _sort_key(record:dict[str,Any])->tuple[int,int,int,int]:
    cited=record.get("cited_by_count")
    if not isinstance(cited,(int,float)) or isinstance(cited,bool):
        cited=0

    return (
        0 if record.get("include_in_digest") else 1,
        -int(record.get("significance_score") or 0),
        -int(record.get("ai_relevance_score") or 0),
        -int(cited),
    )


def has_core_ai_digest_signal(paper:dict[str,Any])->bool:
    title=_safe_str(paper.get("title"))
    summary=_safe_str(paper.get("summary"))
    combined=title+" "+summary

    for phrase in CORE_AI_DIGEST_PHRASES:
        if _phrase_match(combined,phrase):
            return True

    for token in CORE_AI_DIGEST_TOKENS:
        if _word_match(combined,token):
            return True

    return False

def classify_ai_relevance(paper:dict[str,Any])->AIRelevance:
    result=AIRelevance()

    categories=_safe_list(paper.get("categories"))
    title=_safe_str(paper.get("title"))
    summary=_safe_str(paper.get("summary"))
    combined=title+" "+summary

    has_ai_category=any(
        isinstance(c,str) and c in AI_CATEGORIES
        for c in categories
    )

    if has_ai_category:
        result.score+=3
        result.reasons.append("ai_category")

    keyword_hits=0
    title_hit=False

    for phrase in AI_KEYWORDS_PHRASES:
        if _phrase_match(combined,phrase):
            keyword_hits+=1
            if _phrase_match(title,phrase):
                title_hit=True

    for token in AI_KEYWORDS_TOKENS:
        if _word_match(combined,token):
            keyword_hits+=1
            if _word_match(title,token):
                title_hit=True

    if keyword_hits>0:
        result.score+=min(keyword_hits,4)
        result.reasons.append("ai_keywords="+str(keyword_hits))

    if title_hit:
        result.score+=2
        result.reasons.append("ai_keyword_in_title")

    result.score=max(0,min(result.score,10))
    return result


def classify_china_affiliation(
    paper:dict[str,Any],
    institution_aliases:set[str],
    model_aliases:set[str],
)->ChinaAffiliation:
    result=ChinaAffiliation()

    institutions=_safe_list(paper.get("institutions"))
    institutions_str=[i for i in institutions if isinstance(i,str)]

    countries=_safe_list(paper.get("countries"))
    countries_upper={c.upper() for c in countries if isinstance(c,str)}

    title=_safe_str(paper.get("title"))
    summary=_safe_str(paper.get("summary"))
    combined_text=title+" "+summary

    metadata_text=" ".join(institutions_str)

    openalex_match_accepted=paper.get("openalex_match_accepted")
    has_openalex_fields=bool(paper.get("openalex_id") or paper.get("openalex_title"))

    # If fetch.py includes OpenAlex verification fields, only trust OpenAlex-derived
    # country/institution metadata as strong when the match was accepted.
    metadata_allowed=True
    if has_openalex_fields and openalex_match_accepted is False:
        metadata_allowed=False

    def _mark(level:str,method:str,term:str)->None:
        result.is_china_affiliated=True

        if level=="strong":
            result.level="strong"
        elif level=="review" and result.level!="strong":
            result.level="review"

        if method not in result.methods:
            result.methods.append(method)

        if term and term not in result.terms:
            result.terms.append(term)

    if metadata_allowed and "CN" in countries_upper:
        _mark("strong","country_code_cn","CN")

    if metadata_allowed and "HK" in countries_upper:
        _mark("review","country_code_hk_mo","HK")

    if metadata_allowed and "MO" in countries_upper:
        _mark("review","country_code_hk_mo","MO")

    if metadata_allowed:
        for inst in institutions_str:
            for phrase in CN_INSTITUTION_PHRASES:
                if _phrase_match(inst,phrase):
                    _mark("strong","institution_metadata",phrase)

            for token in CN_INSTITUTION_TOKENS:
                if _word_match(inst,token):
                    _mark("strong","institution_metadata",token)

            for phrase in CN_COMPANY_PHRASES:
                if _phrase_match(inst,phrase):
                    _mark("strong","institution_metadata",phrase)

        for alias in institution_aliases:
            if len(alias)<2:
                continue

            for inst in institutions_str:
                if _phrase_match(inst,alias):
                    _mark("strong","validation_institution_alias",alias)
                    break

    has_metadata_signal=(
        metadata_allowed
        and (
            "CN" in countries_upper
            or _any_phrase(metadata_text,CN_INSTITUTION_PHRASES)
            or _any_word(metadata_text,CN_INSTITUTION_TOKENS)
            or _any_phrase(metadata_text,CN_COMPANY_PHRASES)
            or any(
                len(alias)>=2 and _phrase_match(metadata_text,alias)
                for alias in institution_aliases
            )
        )
    )

    for phrase in CN_COMPANY_PHRASES:
        if _phrase_match(combined_text,phrase):
            if has_metadata_signal:
                if "text_company_alias" not in result.methods:
                    result.methods.append("text_company_alias")
                if phrase not in result.terms:
                    result.terms.append(phrase)
            else:
                _mark("review","text_company_alias",phrase)

    for alias in model_aliases:
        if len(alias)<3:
            continue

        if _phrase_match(combined_text,alias):
            if has_metadata_signal:
                if "validation_model_alias" not in result.methods:
                    result.methods.append("validation_model_alias")
                if alias not in result.terms:
                    result.terms.append(alias)
            else:
                _mark("review","validation_model_alias",alias)

    for alias in AMBIGUOUS_MODEL_ALIASES:
        if _word_match(combined_text,alias):
            if has_metadata_signal:
                if "ambiguous_model_alias_with_metadata" not in result.methods:
                    result.methods.append("ambiguous_model_alias_with_metadata")
                if alias not in result.terms:
                    result.terms.append(alias)
            else:
                _mark("review","ambiguous_model_alias_needs_review",alias)

    return result


def classify_significance(
    paper:dict[str,Any],
    ai:AIRelevance,
    china:ChinaAffiliation,
)->Significance:
    result=Significance()
    score=0

    score+=ai.score

    if ai.score>0:
        result.reasons.append("ai_relevance="+str(ai.score))

    if china.level=="strong":
        score+=4
        result.reasons.append("china_strong")
    elif china.level=="review":
        score+=2
        result.reasons.append("china_review")

    countries=_safe_list(paper.get("countries"))
    countries_upper={c.upper() for c in countries if isinstance(c,str)}

    if "CN" in countries_upper and paper.get("openalex_match_accepted") is not False:
        score+=3
        result.reasons.append("country_cn")

    if paper.get("arxiv_id"):
        score+=1
        result.reasons.append("has_arxiv_id")

    if paper.get("openalex_id") and paper.get("openalex_match_accepted") is not False:
        score+=1
        result.reasons.append("has_openalex_id")

    cited=paper.get("cited_by_count")
    if isinstance(cited,(int,float)) and not isinstance(cited,bool):
        cited_int=int(cited)

        if cited_int>=100:
            score+=5
            result.reasons.append("citations>=100")
        elif cited_int>=25:
            score+=3
            result.reasons.append("citations>=25")
        elif cited_int>=5:
            score+=1
            result.reasons.append("citations>=5")

    title=_safe_str(paper.get("title"))
    summary=_safe_str(paper.get("summary"))
    combined=title+" "+summary

    frontier_hits=0
    title_frontier=False

    for phrase in FRONTIER_PHRASES:
        if _phrase_match(combined,phrase):
            frontier_hits+=1
            if _phrase_match(title,phrase):
                title_frontier=True

    for token in FRONTIER_TOKENS:
        if _word_match(combined,token):
            frontier_hits+=1
            if _word_match(title,token):
                title_frontier=True

    if frontier_hits>0:
        score+=min(frontier_hits,5)
        result.reasons.append("frontier_terms="+str(frontier_hits))

    if title_frontier:
        score+=2
        result.reasons.append("frontier_in_title")

    for term in LOW_PRIORITY_TERMS:
        if _phrase_match(combined,term):
            score-=2
            result.reasons.append("low_priority:"+term)

    result.score=max(0,min(score,20))
    return result


def classify_paper(
    paper:dict[str,Any],
    institution_aliases:set[str],
    model_aliases:set[str],
    min_ai_score:int,
    min_significance_score:int,
)->dict[str,Any]:
    ai=classify_ai_relevance(paper)
    china=classify_china_affiliation(paper,institution_aliases,model_aliases)
    sig=classify_significance(paper,ai,china)
    core_ai_signal=has_core_ai_digest_signal(paper)

    needs_review=False

    if china.is_china_affiliated and china.level=="review":
        needs_review=True

    weak_methods={
        "text_company_alias",
        "validation_model_alias",
        "ambiguous_model_alias_needs_review",
        "country_code_hk_mo",
    }

    if any(method in weak_methods for method in china.methods):
        if (
            "institution_metadata" not in china.methods
            and "country_code_cn" not in china.methods
            and "validation_institution_alias" not in china.methods
        ):
            needs_review=True

    include=(
        china.is_china_affiliated
        and china.level=="strong"
        and ai.score>=min_ai_score
        and sig.score>=min_significance_score
        and core_ai_signal
    )

    return {
        "title":paper.get("title"),
        "source":paper.get("source"),
        "arxiv_id":paper.get("arxiv_id"),
        "openalex_id":paper.get("openalex_id"),
        "openalex_title":paper.get("openalex_title"),
        "openalex_match_score":paper.get("openalex_match_score"),
        "openalex_match_method":paper.get("openalex_match_method"),
        "openalex_match_accepted":paper.get("openalex_match_accepted"),
        "doi":paper.get("doi"),
        "url":paper.get("url"),
        "published":paper.get("published") or paper.get("publication_date") or paper.get("date"),
        "institutions":paper.get("institutions") or [],
        "countries":paper.get("countries") or [],
        "is_china_affiliated":china.is_china_affiliated,
        "china_affiliation_level":china.level,
        "china_match_methods":list(china.methods),
        "china_match_terms":list(china.terms),
        "ai_relevance_score":ai.score,
        "ai_relevance_reasons":list(ai.reasons),
        "core_ai_digest_signal":core_ai_signal,
        "significance_score":sig.score,
        "significance_reasons":list(sig.reasons),
        "cited_by_count":paper.get("cited_by_count"),
        "needs_review":needs_review,
        "include_in_digest":include,
        "original":paper,
    }


def load_validation_aliases(path:Optional[Path])->tuple[set[str],set[str]]:
    institution_aliases:set[str]=set()
    model_aliases:set[str]=set()

    if path is None or not path.exists():
        return institution_aliases,model_aliases

    if not HAS_YAML:
        print("Warning: PyYAML not installed; skipping validation aliases.",file=sys.stderr)
        return institution_aliases,model_aliases

    try:
        with path.open("r",encoding="utf-8") as f:
            data=yaml.safe_load(f)
    except Exception as exc:
        print("Warning: failed to parse "+str(path)+": "+str(exc),file=sys.stderr)
        return institution_aliases,model_aliases

    if data is None:
        return institution_aliases,model_aliases

    if isinstance(data,list):
        entries=data
    elif isinstance(data,dict):
        if isinstance(data.get("entries"),list):
            entries=data["entries"]
        elif isinstance(data.get("papers"),list):
            entries=data["papers"]
        elif isinstance(data.get("models"),list):
            entries=data["models"]
        else:
            entries=[v for v in data.values() if isinstance(v,dict)]
    else:
        entries=[]

    def _add(value:Any,target:set[str])->None:
        if isinstance(value,str):
            v=value.strip()
            if len(v)>=2:
                target.add(v)

    for entry in entries:
        if not isinstance(entry,dict):
            continue

        _add(entry.get("title"),model_aliases)

        aliases=entry.get("aliases") or []
        if isinstance(aliases,list):
            for alias in aliases:
                _add(alias,model_aliases)

        inst=entry.get("expected_institution")
        if isinstance(inst,str):
            _add(inst,institution_aliases)
        elif isinstance(inst,list):
            for item in inst:
                _add(item,institution_aliases)

    return institution_aliases,model_aliases


def find_project_root(start:Optional[Path]=None)->Path:
    candidates:list[Path]=[]

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


def find_latest_raw_file(project_root:Path)->Optional[Path]:
    raw_dir=project_root/"data"/"raw"

    if not raw_dir.exists():
        return None

    candidates=sorted(raw_dir.glob("papers_*.json"))

    if not candidates:
        return None

    return candidates[-1]


def extract_date_tag(path:Path)->str:
    match=re.search(r"(\d{4}-\d{2}-\d{2})",path.name)

    if match:
        return match.group(1)

    return datetime.now().strftime("%Y-%m-%d")


def _ensure_utf8_stdout()->None:
    try:
        if sys.stdout.encoding and sys.stdout.encoding.lower()!="utf-8":
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def main(argv:Optional[list[str]]=None)->int:
    _ensure_utf8_stdout()

    parser=argparse.ArgumentParser(
        description="Classify fetched AI papers for China affiliation and significance."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw papers JSON. Defaults to latest data/raw/papers_*.json.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for processed output. Defaults to <project>/data/processed.",
    )

    parser.add_argument(
        "--validation-set",
        type=str,
        default=None,
        help="Path to validation_set.yaml. Defaults to tests/validation_set.yaml.",
    )

    parser.add_argument("--min-ai-score",type=int,default=3)
    parser.add_argument("--min-significance-score",type=int,default=8)
    parser.add_argument("--top",type=int,default=10)

    args=parser.parse_args(argv)

    project_root=find_project_root()

    if args.input:
        input_path=Path(args.input).resolve()
    else:
        found=find_latest_raw_file(project_root)
        if found is None:
            print(
                "Error: no raw papers_*.json found in "+str(project_root/"data"/"raw"),
                file=sys.stderr,
            )
            return 1
        input_path=found

    if not input_path.exists():
        print("Error: input file not found: "+str(input_path),file=sys.stderr)
        return 1

    if args.output_dir:
        output_dir=Path(args.output_dir).resolve()
    else:
        output_dir=project_root/"data"/"processed"

    output_dir.mkdir(parents=True,exist_ok=True)

    if args.validation_set:
        validation_path=Path(args.validation_set).resolve()
    else:
        validation_path=project_root/"tests"/"validation_set.yaml"

    institution_aliases,model_aliases=load_validation_aliases(
        validation_path if validation_path.exists() else None
    )

    try:
        with input_path.open("r",encoding="utf-8") as f:
            raw=json.load(f)
    except Exception as exc:
        print("Error: failed to read "+str(input_path)+": "+str(exc),file=sys.stderr)
        return 1

    if not isinstance(raw,list):
        print("Error: expected a JSON list at "+str(input_path),file=sys.stderr)
        return 1

    classified:list[dict[str,Any]]=[]

    for paper in raw:
        if not isinstance(paper,dict):
            continue

        record=classify_paper(
            paper,
            institution_aliases,
            model_aliases,
            args.min_ai_score,
            args.min_significance_score,
        )
        classified.append(record)

    by_key:dict[str,dict[str,Any]]={}

    for record in classified:
        key=_dedup_key(record)
        existing=by_key.get(key)

        if existing is None or _sort_key(record)<_sort_key(existing):
            by_key[key]=record

    deduped=list(by_key.values())
    deduped.sort(key=_sort_key)

    selected=[
        record for record in deduped
        if record.get("include_in_digest")
    ]

    review_candidates=[
        record for record in deduped
        if record.get("needs_review") and record.get("is_china_affiliated")
    ]

    date_tag=extract_date_tag(input_path)

    classified_path=output_dir/("classified_papers_"+date_tag+".json")
    selected_path=output_dir/("selected_papers_"+date_tag+".json")
    review_path=output_dir/("review_candidates_"+date_tag+".json")
    summary_path=output_dir/("classification_summary_"+date_tag+".json")

    with classified_path.open("w",encoding="utf-8") as f:
        json.dump(deduped,f,ensure_ascii=False,indent=2)

    with selected_path.open("w",encoding="utf-8") as f:
        json.dump(selected,f,ensure_ascii=False,indent=2)

    with review_path.open("w",encoding="utf-8") as f:
        json.dump(review_candidates,f,ensure_ascii=False,indent=2)

    total=len(deduped)
    china_affiliated=sum(1 for r in deduped if r["is_china_affiliated"])
    china_strong=sum(1 for r in deduped if r["china_affiliation_level"]=="strong")
    china_review=sum(1 for r in deduped if r["china_affiliation_level"]=="review")
    needs_review_count=sum(1 for r in deduped if r["needs_review"])
    core_ai_count=sum(1 for r in deduped if r.get("core_ai_digest_signal"))

    summary={
        "run_date":datetime.now().isoformat(timespec="seconds"),
        "input_path":str(input_path),
        "classified_path":str(classified_path),
        "selected_path":str(selected_path),
        "review_candidates_path":str(review_path),
        "summary_path":str(summary_path),
        "thresholds":{
            "min_ai_score":args.min_ai_score,
            "min_significance_score":args.min_significance_score,
            "requires_core_ai_digest_signal":True,
        },
        "counts":{
            "raw_records":len(raw),
            "classified_records":total,
            "core_ai_digest_signal":core_ai_count,
            "china_affiliated":china_affiliated,
            "china_strong":china_strong,
            "china_review":china_review,
            "selected":len(selected),
            "review_candidates":len(review_candidates),
            "needs_review":needs_review_count,
        },
        "validation_aliases_loaded":{
            "yaml_available":HAS_YAML,
            "validation_path":str(validation_path) if validation_path.exists() else None,
            "institution_aliases":len(institution_aliases),
            "model_aliases":len(model_aliases),
        },
    }

    with summary_path.open("w",encoding="utf-8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)

    print("input:           "+str(input_path))
    print("classified out:  "+str(classified_path))
    print("selected out:    "+str(selected_path))
    print("review out:      "+str(review_path))
    print("summary out:     "+str(summary_path))
    print("total records:           "+str(total))
    print("core AI signals:         "+str(core_ai_count))
    print("China-affiliated:        "+str(china_affiliated))
    print("  strong:                "+str(china_strong))
    print("  review:                "+str(china_review))
    print("selected for digest:     "+str(len(selected)))
    print("review candidates:       "+str(len(review_candidates)))
    print("needs review:            "+str(needs_review_count))
    print("")
    print("Top selected papers:")

    top_n=selected[:max(0,args.top)]

    if not top_n:
        print("  (none)")

    for i,record in enumerate(top_n,start=1):
        title=record.get("title") or "(untitled)"
        sig=record.get("significance_score")
        ai_score=record.get("ai_relevance_score")

        insts=record.get("institutions") or []
        if not isinstance(insts,list):
            insts=[]

        countries=record.get("countries") or []
        if not isinstance(countries,list):
            countries=[]

        arxiv_id=record.get("arxiv_id") or "-"
        openalex_id=record.get("openalex_id") or "-"
        match_method=record.get("openalex_match_method") or "-"
        match_score=record.get("openalex_match_score")

        print(
            "  "+str(i)+". [sig="+str(sig)
            +" ai="+str(ai_score)+"] "+str(title)
        )
        print("     institutions: "+", ".join(str(x) for x in insts[:5]))
        print("     countries:    "+", ".join(str(x) for x in countries))
        print("     arxiv_id:     "+str(arxiv_id))
        print("     openalex_id:  "+str(openalex_id))
        print("     oa_match:     "+str(match_method)+" "+str(match_score))

    return 0


if __name__=="__main__":
    sys.exit(main())
