"""Fetch workplace discussions from Hacker News through the official Algolia HN Search API.

Usage:
    python -m workplace_rag.ingest_hn --pages 2

Writes data/raw/hn.jsonl (one post per line). Re-running merges and de-duplicates by id.
Author names are not stored: we only need what was said, not who said it.
"""

import argparse
import html
import json
import re
import time

import requests

from .config import RAW_DIR

API = "https://hn.algolia.com/api/v1/search"
OUT = RAW_DIR / "hn.jsonl"

# Workplace topics crossed with the India-focused companies and cities we care about.
QUERIES = [
    "work life balance india",
    "working at TCS",
    "working at Infosys",
    "Wipro work culture",
    "HCL Cognizant salary",
    "Bangalore startup work culture",
    "Hyderabad software job",
    "Indian IT services company",
    "Flipkart engineering culture",
    "Amazon India manager",
    "Google Bangalore",
    "Microsoft Hyderabad",
    "notice period india",
    "layoffs india startup",
    "toxic manager",
    "work life balance big tech",
    "on-call burnout",
    "return to office mandate",
    "remote work salary cut",
    "promotion process engineers",
]

MIN_CHARS = 200  # skip one-liners, they rarely carry an opinion

TAG_RE = re.compile(r"<[^>]+>")
# Monthly job-ad threads are ads, not experiences.
SKIP_THREADS = re.compile(r"who is hiring|who wants to be hired|freelancer\? seeking", re.I)


def clean(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("<p>", "\n\n")
    text = TAG_RE.sub("", text)
    return html.unescape(text).strip()


def fetch(query: str, page: int) -> list[dict]:
    resp = requests.get(
        API,
        params={"query": query, "tags": "(comment,ask_hn)", "hitsPerPage": 100, "page": page},
        headers={"User-Agent": "workplace-rag-research/0.1"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("hits", [])


def to_post(hit: dict, query: str) -> dict | None:
    text = clean(hit.get("comment_text") or hit.get("story_text"))
    title = hit.get("story_title") or hit.get("title") or ""
    if len(text) < MIN_CHARS or SKIP_THREADS.search(title):
        return None
    return {
        "id": f"hn_{hit['objectID']}",
        "source": "hackernews",
        "url": f"https://news.ycombinator.com/item?id={hit['objectID']}",
        "title": title,
        "text": text,
        "created_at": hit.get("created_at"),
        "query": query,
    }


def load_existing() -> dict[str, dict]:
    if not OUT.exists():
        return {}
    with OUT.open(encoding="utf-8") as f:
        return {p["id"]: p for p in map(json.loads, f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=2, help="pages of 100 hits per query")
    args = ap.parse_args()

    posts = {k: p for k, p in load_existing().items() if not SKIP_THREADS.search(p["title"])}
    before = len(posts)
    for q in QUERIES:
        for page in range(args.pages):
            hits = fetch(q, page)
            for hit in hits:
                post = to_post(hit, q)
                if post and post["id"] not in posts:
                    posts[post["id"]] = post
            time.sleep(0.5)  # be polite to the API
            if len(hits) < 100:
                break
        print(f"{q!r}: total {len(posts)}")

    with OUT.open("w", encoding="utf-8") as f:
        for p in posts.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(posts)} posts ({len(posts) - before} new) to {OUT}")


if __name__ == "__main__":
    main()
