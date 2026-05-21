from __future__ import annotations

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dateutil.parser import isoparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ARXIV_API_URL = "https://export.arxiv.org/api/query"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"

CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "sis017@ucsd.edu")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
USER_AGENT = f"UCSDPaperMonitorBot/1.0 (mailto:{CONTACT_EMAIL})"

AI_CATEGORIES = [
    "cs.CL",
    "cs.LG",
    "cs.CV",
    "cs.AI",
]

ATOM_NS = "{http://www.w3.org/2005/Atom}"

@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    summary: str
    published: str
    updated: str
    arxiv_id: str | None
    url: str
    categories: list[str]
    doi: str | None = None
    openalex_id: str | None = None
    institutions: list[str] | None = None
    countries: list[str] | None = None
    cited_by_count: int | None = None

def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())

def check_arxiv_ip_status() -> bool:
    print("Performing pre-flight check on arXiv API...")

    headers = {"User-Agent": USER_AGENT}
    params = {
        "search_query": "cat:cs.AI",
        "start": 0,
        "max_results": 1,
    }

    try:
        response = requests.get(
            ARXIV_API_URL,
            params=params,
            headers=headers,
            timeout=15,
        )

        if response.status_code == 429:
            return False

        response.raise_for_status()
        return True

    except requests.RequestException as exc:
        print(f"arXiv pre-flight check failed: {exc}")
        return False

def get_retry_session(retries: int = 5, backoff_factor: float = 2.0) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def fetch_arxiv_recent(session: requests.Session, days: int = 7, batch_size: int = 100) -> list[Paper]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    query = " OR ".join(f"cat:{cat}" for cat in AI_CATEGORIES)

    papers = []
    start = 0
    page = 1
    reached_cutoff = False

    while not reached_cutoff:
        params = {
            "search_query": query,
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        print(f"Fetching arXiv page {page}: start={start}, batch_size={batch_size}...")

        response = session.get(ARXIV_API_URL, params=params, timeout=60)

        # 🛡️ THE SAFETY NET: Exit gracefully on 429, keep the papers!
        if response.status_code == 429:
            print("\n⚠️ WARNING: arXiv rate limit hit. Stopping fetch early, but saving the papers we successfully gathered!")
            break

        response.raise_for_status()

        root = ET.fromstring(response.text)
        entries = root.findall(f"{ATOM_NS}entry")

        if not entries:
            break

        for entry in entries:
            published_text = entry.findtext(f"{ATOM_NS}published")
            if not published_text:
                continue

            published_dt = isoparse(published_text)
            if published_dt.tzinfo is None:
                published_dt = published_dt.replace(tzinfo=timezone.utc)

            if published_dt < cutoff:
                reached_cutoff = True
                continue

            title = clean_text(entry.findtext(f"{ATOM_NS}title"))
            summary = clean_text(entry.findtext(f"{ATOM_NS}summary"))
            updated_text = entry.findtext(f"{ATOM_NS}updated")

            entry_id = entry.findtext(f"{ATOM_NS}id") or ""
            arxiv_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else None

            authors = []
            for author in entry.findall(f"{ATOM_NS}author"):
                name = author.findtext(f"{ATOM_NS}name")
                if name:
                    authors.append(clean_text(name))

            categories = []
            for category in entry.findall(f"{ATOM_NS}category"):
                term = category.attrib.get("term")
                if term:
                    categories.append(term)

            doi = None
            for link in entry.findall(f"{ATOM_NS}link"):
                if link.attrib.get("title") == "doi":
                    doi = link.attrib.get("href")

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

        start += batch_size
        page += 1

        if not reached_cutoff:
            time.sleep(4) # Respect arXiv's rules so we don't hit the 429

    return papers

def enrich_single_paper(session: requests.Session, paper: Paper) -> Paper:
    # Uses the expensive Title Search parameter
    params = {
        "search": paper.title,
        "filter": "institutions.country_code:CN",
        "per_page": 1,
        "select": "id,title,authorships,cited_by_count,publication_date,doi,ids",
        "mailto": CONTACT_EMAIL,
    }
    
    # Injects your API key to hit the Premium tier
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY

    try:
        # Zero sleep delay. Full speed ahead.
        response = session.get(OPENALEX_WORKS_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if results:
            work = results[0]
            institutions = []
            countries = []

            for authorship in work.get("authorships", []):
                for inst in authorship.get("institutions", []):
                    name = inst.get("display_name")
                    country = inst.get("country_code")
                    if name:
                        institutions.append(name)
                    if country:
                        countries.append(country)

            paper.openalex_id = work.get("id")
            paper.institutions = sorted(set(institutions))
            paper.countries = sorted(set(countries))
            paper.cited_by_count = work.get("cited_by_count")

    except requests.RequestException:
        pass
        
    return paper

def main() -> None:
    if not check_arxiv_ip_status():
        print("\n❌ ERROR: Your IP is currently rate-limited by arXiv.")
        print("Wait 15-30 minutes before running this script again.")
        print("Exiting safely to avoid extending the rate-limit window.\n")
        sys.exit(1)

    print("✅ API check passed.\n")

    if OPENALEX_API_KEY:
        print("🔑 Premium OpenAlex API key detected. Running in HIGH-SPEED mode.")
    else:
        print("⚠️ Warning: No OPENALEX_API_KEY detected. Premium mode requires an API key.")

    session = get_retry_session()

    print("\nFetching recent arXiv AI papers...")
    papers = fetch_arxiv_recent(session=session, days=7, batch_size=100)
    print(f"✅ Fetched {len(papers)} recent arXiv papers.")

    if not papers:
        print("No papers fetched. Exiting.")
        sys.exit(0)

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = out_dir / f"papers_{today}.json"

    print(f"\n🚀 Enriching with Premium OpenAlex (10 workers, NO DELAY)...")
    
    enriched_papers = []
    chunk_size = 100

    with ThreadPoolExecutor(max_workers=10) as executor:
        for i in range(0, len(papers), chunk_size):
            chunk = papers[i : i + chunk_size]
            futures = [executor.submit(enrich_single_paper, session, p) for p in chunk]
            
            for future in as_completed(futures):
                enriched_papers.append(future.result())
                
            with out_path.open("w", encoding="utf-8") as f:
                json.dump([asdict(p) for p in enriched_papers], f, ensure_ascii=False, indent=2)
                
            print(f"⏳ Processed and safely saved {len(enriched_papers)} / {len(papers)} papers...")

    cn_matches = [
        paper for paper in enriched_papers
        if paper.countries and "CN" in paper.countries
    ]

    print(f"\n🎯 DONE! Matched {len(cn_matches)} papers with CN institution signals.")

    for paper in cn_matches[:5]:
        print("-" * 80)
        print(paper.title)
        print("Institutions:", paper.institutions)
        print("Countries:", paper.countries)
        print("arXiv:", paper.arxiv_id)
        print("OpenAlex:", paper.openalex_id)

if __name__ == "__main__":
    main()