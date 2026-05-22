from __future__ import annotations

import difflib
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict,dataclass
from datetime import datetime,timedelta,timezone
from pathlib import Path
from typing import Any

import requests
from dateutil.parser import isoparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ARXIV_API_URL="https://export.arxiv.org/api/query"
OPENALEX_WORKS_URL="https://api.openalex.org/works"

CONTACT_EMAIL=os.getenv("CONTACT_EMAIL","sis017@ucsd.edu")
OPENALEX_API_KEY=os.getenv("OPENALEX_API_KEY")
USER_AGENT=f"UCSDPaperMonitorBot/1.0 (mailto:{CONTACT_EMAIL})"

ARXIV_DAYS=int(os.getenv("ARXIV_DAYS","7"))
ARXIV_BATCH_SIZE=int(os.getenv("ARXIV_BATCH_SIZE","50"))
ARXIV_MAX_PAGES=int(os.getenv("ARXIV_MAX_PAGES","10"))
ARXIV_SLEEP_SECONDS=float(os.getenv("ARXIV_SLEEP_SECONDS","6"))
ARXIV_RETRIES=int(os.getenv("ARXIV_RETRIES","4"))
ARXIV_RETRY_BASE_SECONDS=int(os.getenv("ARXIV_RETRY_BASE_SECONDS","60"))
ARXIV_PREFLIGHT_CHECK=os.getenv("ARXIV_PREFLIGHT_CHECK","true").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

OPENALEX_WORKERS=int(os.getenv("OPENALEX_WORKERS","2"))
OPENALEX_SLEEP_SECONDS=float(os.getenv("OPENALEX_SLEEP_SECONDS","0.35"))
OPENALEX_MAX_PAPERS=int(os.getenv("OPENALEX_MAX_PAPERS","1500"))
OPENALEX_MIN_TITLE_SCORE=float(os.getenv("OPENALEX_MIN_TITLE_SCORE","0.85"))
OPENALEX_PER_PAGE=max(1,min(int(os.getenv("OPENALEX_PER_PAGE","5")),100))

AI_CATEGORIES=[
    "cs.CL",
    "cs.LG",
    "cs.CV",
    "cs.AI",
]

ATOM_NS="{http://www.w3.org/2005/Atom}"


@dataclass
class Paper:
    source:str
    title:str
    authors:list[str]
    summary:str
    published:str
    updated:str
    arxiv_id:str|None
    url:str
    categories:list[str]
    doi:str|None=None
    openalex_id:str|None=None
    openalex_title:str|None=None
    openalex_doi:str|None=None
    openalex_match_score:float|None=None
    openalex_match_method:str|None=None
    openalex_match_accepted:bool=False
    openalex_candidates_returned:int|None=None
    openalex_candidates_checked:int|None=None
    institutions:list[str]|None=None
    countries:list[str]|None=None
    cited_by_count:int|None=None


def clean_text(text:str|None)->str:
    if not text:
        return ""

    return " ".join(text.split())


def normalize_title(title:str|None)->str:
    text=clean_text(title).lower()
    text=re.sub(r"[^a-z0-9]+"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text


def clean_arxiv_id(arxiv_id:str|None)->str|None:
    if not arxiv_id:
        return None

    cleaned=arxiv_id.strip().lower()
    cleaned=re.sub(r"^arxiv:","",cleaned)
    cleaned=re.sub(r"v\d+$","",cleaned)
    return cleaned


def clean_doi(doi:str|None)->str|None:
    if not doi:
        return None

    cleaned=doi.strip().lower()
    cleaned=cleaned.replace("https://doi.org/","")
    cleaned=cleaned.replace("http://doi.org/","")
    cleaned=cleaned.replace("doi:","")
    return cleaned


def title_similarity(left:str|None,right:str|None)->float:
    norm_left=normalize_title(left)
    norm_right=normalize_title(right)

    if not norm_left or not norm_right:
        return 0.0

    return difflib.SequenceMatcher(None,norm_left,norm_right).ratio()


def get_work_title(work:dict[str,Any])->str|None:
    title=work.get("title")
    if isinstance(title,str) and title.strip():
        return title

    display_name=work.get("display_name")
    if isinstance(display_name,str) and display_name.strip():
        return display_name

    return None


def get_work_arxiv_id(work:dict[str,Any])->str|None:
    ids=work.get("ids") or {}

    if not isinstance(ids,dict):
        return None

    raw=ids.get("arxiv")
    if not isinstance(raw,str):
        return None

    if "/abs/" in raw:
        raw=raw.split("/abs/")[-1]

    return clean_arxiv_id(raw)


def get_work_doi(work:dict[str,Any])->str|None:
    doi=work.get("doi")

    if isinstance(doi,str) and doi.strip():
        return clean_doi(doi)

    ids=work.get("ids") or {}
    if isinstance(ids,dict):
        raw=ids.get("doi")
        if isinstance(raw,str) and raw.strip():
            return clean_doi(raw)

    return None


def validate_openalex_match(paper:Paper,work:dict[str,Any])->tuple[bool,str,float]:
    paper_arxiv=clean_arxiv_id(paper.arxiv_id)
    work_arxiv=get_work_arxiv_id(work)

    if paper_arxiv and work_arxiv and paper_arxiv==work_arxiv:
        return True,"arxiv_id",1.0

    paper_doi=clean_doi(paper.doi)
    work_doi=get_work_doi(work)

    if paper_doi and work_doi and paper_doi==work_doi:
        return True,"doi",1.0

    work_title=get_work_title(work)
    score=title_similarity(paper.title,work_title)

    if score>=OPENALEX_MIN_TITLE_SCORE:
        return True,"title_similarity",round(score,4)

    return False,"rejected_title_mismatch",round(score,4)


def match_priority(method:str)->int:
    if method=="arxiv_id":
        return 3

    if method=="doi":
        return 2

    if method=="title_similarity":
        return 1

    return 0


def choose_best_openalex_work(
    paper:Paper,
    results:list[dict[str,Any]],
)->tuple[dict[str,Any]|None,bool,str,float]:
    best_work=None
    best_accepted=False
    best_method="no_match"
    best_score=0.0
    best_priority=0

    for candidate in results:
        if not isinstance(candidate,dict):
            continue

        accepted,method,score=validate_openalex_match(paper,candidate)
        priority=match_priority(method)

        should_replace=False

        if accepted and not best_accepted:
            should_replace=True
        elif accepted and best_accepted:
            if priority>best_priority:
                should_replace=True
            elif priority==best_priority and score>best_score:
                should_replace=True
        elif not accepted and not best_accepted and score>best_score:
            should_replace=True

        if should_replace:
            best_work=candidate
            best_accepted=accepted
            best_method=method
            best_score=score
            best_priority=priority

        if accepted and method in {"arxiv_id","doi"}:
            break

    return best_work,best_accepted,best_method,best_score


def check_arxiv_ip_status()->bool:
    print("Performing pre-flight check on arXiv API...")

    headers={"User-Agent":USER_AGENT}
    params={
        "search_query":"cat:cs.AI",
        "start":0,
        "max_results":1,
    }

    try:
        response=requests.get(
            ARXIV_API_URL,
            params=params,
            headers=headers,
            timeout=15,
        )

        if response.status_code==429:
            return False

        if response.status_code in (500,502,503,504):
            print(f"arXiv pre-flight returned transient status {response.status_code}. Continuing to real fetch with retries.")
            return True

        response.raise_for_status()
        return True

    except requests.RequestException as exc:
        print(f"arXiv pre-flight check failed: {exc}")
        return False


def get_retry_session(retries:int=5,backoff_factor:float=2.0)->requests.Session:
    session=requests.Session()
    session.headers.update({"User-Agent":USER_AGENT})

    retry=Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429,500,502,503,504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter=HTTPAdapter(max_retries=retry,pool_connections=20,pool_maxsize=20)
    session.mount("http://",adapter)
    session.mount("https://",adapter)

    return session


def fetch_arxiv_recent(
    session:requests.Session,
    days:int=ARXIV_DAYS,
    batch_size:int=ARXIV_BATCH_SIZE,
    max_pages:int=ARXIV_MAX_PAGES,
)->list[Paper]:
    cutoff=datetime.now(timezone.utc)-timedelta(days=days)
    query=" OR ".join(f"cat:{cat}" for cat in AI_CATEGORIES)

    papers=[]
    start=0
    page=1
    reached_cutoff=False

    while not reached_cutoff:
        if max_pages>0 and page>max_pages:
            print(f"Reached ARXIV_MAX_PAGES={max_pages}. Stopping arXiv fetch.")
            break

        params={
            "search_query":query,
            "start":start,
            "max_results":batch_size,
            "sortBy":"submittedDate",
            "sortOrder":"descending",
        }

        print(f"Fetching arXiv page {page}: start={start}, batch_size={batch_size}...")

        response=None

        for attempt in range(1,ARXIV_RETRIES+1):
            try:
                response=session.get(
                    ARXIV_API_URL,
                    params=params,
                    timeout=(10,180),
                )

                if response.status_code==200:
                    break

                if response.status_code in (429,500,502,503,504):
                    retry_after=response.headers.get("Retry-After")

                    if retry_after and retry_after.isdigit():
                        wait_seconds=int(retry_after)
                    else:
                        wait_seconds=ARXIV_RETRY_BASE_SECONDS*attempt

                    print(
                        f"\nWARNING: arXiv returned {response.status_code} on page {page}, "
                        f"attempt {attempt}/{ARXIV_RETRIES}."
                    )
                    print(f"Waiting {wait_seconds} seconds before retry...")
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()
                break

            except requests.RequestException as exc:
                wait_seconds=ARXIV_RETRY_BASE_SECONDS*attempt
                print(
                    f"\nWARNING: arXiv request failed on page {page}, "
                    f"attempt {attempt}/{ARXIV_RETRIES}: {exc}"
                )
                print(f"Waiting {wait_seconds} seconds before retry...")
                time.sleep(wait_seconds)

        if response is None or response.status_code!=200:
            print("\nWARNING: arXiv did not return a valid page after retries.")
            print("Stopping arXiv fetch early and saving gathered papers.")
            break

        root=ET.fromstring(response.text)
        entries=root.findall(f"{ATOM_NS}entry")

        if not entries:
            break

        for entry in entries:
            published_text=entry.findtext(f"{ATOM_NS}published")
            if not published_text:
                continue

            published_dt=isoparse(published_text)
            if published_dt.tzinfo is None:
                published_dt=published_dt.replace(tzinfo=timezone.utc)

            if published_dt<cutoff:
                reached_cutoff=True
                continue

            title=clean_text(entry.findtext(f"{ATOM_NS}title"))
            summary=clean_text(entry.findtext(f"{ATOM_NS}summary"))
            updated_text=entry.findtext(f"{ATOM_NS}updated")

            entry_id=entry.findtext(f"{ATOM_NS}id") or ""
            arxiv_id=entry_id.split("/abs/")[-1] if "/abs/" in entry_id else None

            authors=[]
            for author in entry.findall(f"{ATOM_NS}author"):
                name=author.findtext(f"{ATOM_NS}name")
                if name:
                    authors.append(clean_text(name))

            categories=[]
            for category in entry.findall(f"{ATOM_NS}category"):
                term=category.attrib.get("term")
                if term:
                    categories.append(term)

            doi=None
            for link in entry.findall(f"{ATOM_NS}link"):
                if link.attrib.get("title")=="doi":
                    doi=link.attrib.get("href")

            papers.append(
                Paper(
                    source="arXiv",
                    title=title,
                    authors=authors,
                    summary=summary,
                    published=published_text,
                    updated=updated_text or published_text,
                    arxiv_id=arxiv_id,
                    url=entry_id,
                    categories=categories,
                    doi=doi,
                )
            )

        start+=batch_size
        page+=1

        if not reached_cutoff:
            time.sleep(ARXIV_SLEEP_SECONDS)

    return papers


def enrich_single_paper(session:requests.Session,paper:Paper)->Paper:
    params={
        "search":paper.title,
        "filter":"institutions.country_code:CN",
        "per_page":OPENALEX_PER_PAGE,
        "select":"id,title,display_name,authorships,cited_by_count,publication_date,doi,ids",
        "mailto":CONTACT_EMAIL,
    }

    if OPENALEX_API_KEY:
        params["api_key"]=OPENALEX_API_KEY

    try:
        response=session.get(OPENALEX_WORKS_URL,params=params,timeout=30)

        if response.status_code==401:
            print("OpenAlex returned 401 Unauthorized. Check OPENALEX_API_KEY.")
            return paper

        if response.status_code==429:
            print("OpenAlex returned 429 rate limit. Leaving this paper unenriched.")
            return paper

        response.raise_for_status()

        data=response.json()
        results=data.get("results",[])

        if not isinstance(results,list):
            return paper

        paper.openalex_candidates_returned=len(results)
        paper.openalex_candidates_checked=len(results)

        if not results:
            return paper

        work,accepted,method,score=choose_best_openalex_work(paper,results)

        if work is None:
            return paper

        paper.openalex_title=get_work_title(work)
        paper.openalex_doi=get_work_doi(work)
        paper.openalex_match_score=score
        paper.openalex_match_method=method
        paper.openalex_match_accepted=accepted

        if not accepted:
            return paper

        institutions=[]
        countries=[]

        for authorship in work.get("authorships",[]):
            if not isinstance(authorship,dict):
                continue

            for inst in authorship.get("institutions",[]):
                if not isinstance(inst,dict):
                    continue

                name=inst.get("display_name")
                country=inst.get("country_code")

                if name:
                    institutions.append(name)
                if country:
                    countries.append(str(country).upper())

        paper.openalex_id=work.get("id")
        paper.institutions=sorted(set(institutions))
        paper.countries=sorted(set(countries))
        paper.cited_by_count=work.get("cited_by_count")

    except requests.RequestException as exc:
        print(f"OpenAlex lookup failed for {paper.arxiv_id}: {exc}")

    finally:
        if OPENALEX_SLEEP_SECONDS>0:
            time.sleep(OPENALEX_SLEEP_SECONDS)

    return paper


def save_papers(papers:list[Paper],out_path:Path)->None:
    with out_path.open("w",encoding="utf-8") as f:
        json.dump([asdict(p) for p in papers],f,ensure_ascii=False,indent=2)


def main()->None:
    if ARXIV_PREFLIGHT_CHECK:
        if not check_arxiv_ip_status():
            print("\nERROR: Your IP is currently rate-limited by arXiv.")
            print("Wait 15-30 minutes before running this script again.")
            print("Exiting safely to avoid extending the rate-limit window.\n")
            sys.exit(1)

        print("API check passed.")
        print("Sleeping 4 seconds before the real arXiv query to respect arXiv pacing.\n")
        time.sleep(4)
    else:
        print("Skipping arXiv pre-flight check because ARXIV_PREFLIGHT_CHECK is disabled.")

    if OPENALEX_API_KEY:
        print(f"OpenAlex API key detected. workers={OPENALEX_WORKERS}, sleep={OPENALEX_SLEEP_SECONDS}s.")
    else:
        print("Warning: No OPENALEX_API_KEY detected. Running unauthenticated OpenAlex requests.")

    print("\nConfiguration:")
    print(f"CONTACT_EMAIL={CONTACT_EMAIL}")
    print(f"OPENALEX_API_KEY detected={bool(OPENALEX_API_KEY)}")
    print(f"ARXIV_DAYS={ARXIV_DAYS}")
    print(f"ARXIV_BATCH_SIZE={ARXIV_BATCH_SIZE}")
    print(f"ARXIV_MAX_PAGES={ARXIV_MAX_PAGES}")
    print(f"ARXIV_SLEEP_SECONDS={ARXIV_SLEEP_SECONDS}")
    print(f"ARXIV_RETRIES={ARXIV_RETRIES}")
    print(f"ARXIV_RETRY_BASE_SECONDS={ARXIV_RETRY_BASE_SECONDS}")
    print(f"ARXIV_PREFLIGHT_CHECK={ARXIV_PREFLIGHT_CHECK}")
    print(f"OPENALEX_MAX_PAPERS={OPENALEX_MAX_PAPERS}")
    print(f"OPENALEX_WORKERS={OPENALEX_WORKERS}")
    print(f"OPENALEX_SLEEP_SECONDS={OPENALEX_SLEEP_SECONDS}")
    print(f"OPENALEX_MIN_TITLE_SCORE={OPENALEX_MIN_TITLE_SCORE}")
    print(f"OPENALEX_PER_PAGE={OPENALEX_PER_PAGE}")

    session=get_retry_session()

    print("\nFetching recent arXiv AI papers...")
    papers=fetch_arxiv_recent(session=session)
    print(f"Fetched {len(papers)} recent arXiv papers.")

    if not papers:
        print("No papers fetched. Exiting.")
        sys.exit(0)

    if OPENALEX_MAX_PAPERS>0:
        papers=papers[:OPENALEX_MAX_PAPERS]
        print(f"OPENALEX_MAX_PAPERS set. Only enriching first {len(papers)} papers.")

    out_dir=Path("data/raw")
    out_dir.mkdir(parents=True,exist_ok=True)

    today=datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path=out_dir/f"papers_{today}.json"

    print(
        f"\nEnriching with OpenAlex using verified matching "
        f"(workers={OPENALEX_WORKERS}, min_title_score={OPENALEX_MIN_TITLE_SCORE}, per_page={OPENALEX_PER_PAGE})..."
    )

    enriched_papers=[]
    chunk_size=100

    with ThreadPoolExecutor(max_workers=OPENALEX_WORKERS) as executor:
        for i in range(0,len(papers),chunk_size):
            chunk=papers[i:i+chunk_size]
            futures=[executor.submit(enrich_single_paper,session,p) for p in chunk]

            for future in as_completed(futures):
                enriched_papers.append(future.result())

            save_papers(enriched_papers,out_path)
            print(f"Processed and safely saved {len(enriched_papers)} / {len(papers)} papers...")

    accepted_matches=[
        paper for paper in enriched_papers
        if paper.openalex_match_accepted
    ]

    cn_matches=[
        paper for paper in accepted_matches
        if paper.countries and "CN" in paper.countries
    ]

    rejected_matches=[
        paper for paper in enriched_papers
        if paper.openalex_match_method=="rejected_title_mismatch"
    ]

    total_candidates=sum(
        paper.openalex_candidates_returned or 0
        for paper in enriched_papers
    )

    print(f"\nDONE.")
    print(f"OpenAlex accepted matches: {len(accepted_matches)} / {len(enriched_papers)}")
    print(f"Rejected OpenAlex title mismatches: {len(rejected_matches)}")
    print(f"OpenAlex candidates returned: {total_candidates}")
    print(f"Matched {len(cn_matches)} papers with verified CN institution signals.")
    print(f"Saved raw data to {out_path}")

    for paper in cn_matches[:5]:
        print("-"*80)
        print(paper.title)
        print("Institutions:",paper.institutions)
        print("Countries:",paper.countries)
        print("arXiv:",paper.arxiv_id)
        print("OpenAlex:",paper.openalex_id)
        print("Match:",paper.openalex_match_method,paper.openalex_match_score)
        print("Candidates returned:",paper.openalex_candidates_returned)


if __name__=="__main__":
    main()
