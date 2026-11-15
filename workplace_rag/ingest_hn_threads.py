"""Fetch whole Ask HN threads about workplaces (question + replies) via the official Algolia HN API.

    python -m workplace_rag.ingest_hn_threads

Why: keyword search over all comments (ingest_hn.py) mostly finds side remarks inside unrelated
news threads. Replies to "Ask HN: Is working at Amazon worth it?" are first-hand answers to
exactly the kind of question job seekers ask, so they make far better evidence.

Writes data/raw/hn_threads.jsonl. Author names are not stored.
"""

import json
import time

import requests

from .config import RAW_DIR
from .ingest_hn import MIN_CHARS, SKIP_THREADS, clean

SEARCH = "https://hn.algolia.com/api/v1/search"
ITEM = "https://hn.algolia.com/api/v1/items/{}"
OUT = RAW_DIR / "hn_threads.jsonl"
HEADERS = {"User-Agent": "workplace-rag-research/0.1"}

QUERIES = [
    "what is it like working at", "experience working at", "worth working at", "work at Amazon", "work at Google", "work at Microsoft", "work at Meta Facebook",
    "TCS Infosys Wipro", "Indian IT company", "software engineer India", "Bangalore", "Hyderabad",
    "work life balance", "burnout", "toxic workplace", "bad manager", "on-call",
    "layoffs", "PIP performance improvement plan", "promotion", "salary negotiation", "big tech vs startup",
    "return to office", "first job", "fresher", "career advice",
]
MIN_COMMENTS = 15        # skip threads nobody answered
MAX_PER_THREAD = 60      # keep one huge thread from dominating the dataset
MAX_DEPTH = 2            # direct answers and replies to them; deeper is mostly back-and-forth


def find_threads() -> dict[str, dict]:
    threads: dict[str, dict] = {}
    for q in QUERIES:
        resp = requests.get(SEARCH, params={"query": q, "tags": "ask_hn", "hitsPerPage": 50,
                                            "numericFilters": f"num_comments>={MIN_COMMENTS}"},
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()
        for hit in resp.json().get("hits", []):
            if not SKIP_THREADS.search(hit.get("title") or ""):
                threads[hit["objectID"]] = {"title": hit["title"], "num_comments": hit.get("num_comments", 0)}
        time.sleep(0.5)
    return threads


def walk(node: dict, depth: int, title: str, out: list[dict]) -> None:
    for child in node.get("children") or []:
        if len(out) >= MAX_PER_THREAD:
            return
        text = clean(child.get("text"))
        if len(text) >= MIN_CHARS:
            out.append({
                "id": f"hn_{child['id']}",
                "source": "hackernews",
                "url": f"https://news.ycombinator.com/item?id={child['id']}",
                "title": title,
                "text": text,
                "created_at": child.get("created_at"),
                "query": "ask_hn_thread",
                "thread_kind": "ask_hn",
            })
        if depth < MAX_DEPTH:
            walk(child, depth + 1, title, out)


def main() -> None:
    threads = find_threads()
    print(f"Found {len(threads)} Ask HN threads with {MIN_COMMENTS}+ comments")
    posts: dict[str, dict] = {}
    for i, (tid, meta) in enumerate(threads.items(), 1):
        resp = requests.get(ITEM.format(tid), headers=HEADERS, timeout=60)
        if resp.status_code != 200:
            continue
        item = resp.json()
        out: list[dict] = []
        # The question itself is often a first-hand account too.
        if len(clean(item.get("text"))) >= MIN_CHARS:
            out.append({"id": f"hn_{tid}", "source": "hackernews",
                        "url": f"https://news.ycombinator.com/item?id={tid}", "title": meta["title"],
                        "text": clean(item.get("text")), "created_at": item.get("created_at"),
                        "query": "ask_hn_thread", "thread_kind": "ask_hn"})
        walk(item, 1, meta["title"], out)
        posts.update({p["id"]: p for p in out})
        if i % 20 == 0:
            print(f"  {i}/{len(threads)} threads, {len(posts)} comments")
        time.sleep(0.3)

    with OUT.open("w", encoding="utf-8") as f:
        for p in posts.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(posts)} posts from {len(threads)} threads to {OUT}")


if __name__ == "__main__":
    main()
