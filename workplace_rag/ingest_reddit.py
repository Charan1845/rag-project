"""Fetch workplace posts and top comments from Reddit through the official Data API (OAuth).

    python -m workplace_rag.ingest_reddit

Needs a free "script" app from https://www.reddit.com/prefs/apps, with these in .env:
    REDDIT_CLIENT_ID=...
    REDDIT_CLIENT_SECRET=...
    REDDIT_USER_AGENT=workplace-rag/0.1 (by u/<your_reddit_username>)

Uses application-only auth (read-only, no password). Author names are not stored, raw data stays in
data/ (gitignored) and is never redistributed, per the Reddit Data API terms for non-commercial use.
Writes data/raw/reddit.jsonl.
"""

import json
import os
import time

import requests

from .config import RAW_DIR

OUT = RAW_DIR / "reddit.jsonl"
SUBREDDITS = ["developersIndia", "IndianWorkplace", "cscareerquestions"]
QUERIES = [
    "work life balance", "TCS", "Infosys", "Wipro", "Cognizant", "Accenture", "HCL",
    "Amazon", "Google", "Microsoft", "Flipkart", "manager", "toxic", "layoffs", "hike", "appraisal",
    "notice period", "Hyderabad", "Bangalore", "Pune", "fresher", "PIP", "on-call", "promotion",
]
MIN_CHARS = 200
COMMENTS_PER_POST = 15
MIN_SCORE = 3  # skip downvoted / ignored comments


def token(session: requests.Session) -> str:
    cid, secret = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET")
    if not cid or not secret:
        raise SystemExit("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env (see this file's docstring).")
    resp = session.post("https://www.reddit.com/api/v1/access_token", auth=(cid, secret),
                        data={"grant_type": "client_credentials"}, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def get(session: requests.Session, path: str, **params) -> dict | list:
    for _ in range(3):
        resp = session.get(f"https://oauth.reddit.com{path}", params={**params, "raw_json": 1}, timeout=30)
        if resp.status_code == 429:  # rate limited: wait as told
            time.sleep(float(resp.headers.get("x-ratelimit-reset", 10)))
            continue
        resp.raise_for_status()
        # Stay well under the 100 requests/minute free-tier limit.
        if float(resp.headers.get("x-ratelimit-remaining", 100)) < 5:
            time.sleep(float(resp.headers.get("x-ratelimit-reset", 10)))
        time.sleep(0.7)
        return resp.json()
    raise RuntimeError(f"Rate limited too often on {path}")


def to_post(kind: str, d: dict, sub: str, title: str) -> dict | None:
    text = (d.get("selftext") if kind == "post" else d.get("body")) or ""
    if kind == "post" and d.get("title"):
        text = text.strip()
    if len(text) < MIN_CHARS or text in ("[deleted]", "[removed]"):
        return None
    return {
        "id": f"rd_{d['id']}",
        "source": "reddit",
        "url": f"https://www.reddit.com{d['permalink']}",
        "title": title,
        "text": text.strip(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(d.get("created_utc", 0))),
        "query": f"r/{sub}",
    }


def main() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = os.getenv("REDDIT_USER_AGENT", "workplace-rag/0.1")
    session.headers["Authorization"] = f"bearer {token(session)}"

    posts: dict[str, dict] = {}
    for sub in SUBREDDITS:
        for q in QUERIES:
            listing = get(session, f"/r/{sub}/search", q=q, restrict_sr=1, sort="relevance", t="all", limit=25)
            assert isinstance(listing, dict)
            for child in listing["data"]["children"]:
                d = child["data"]
                if f"rd_{d['id']}" in posts or d.get("num_comments", 0) < 3:
                    continue
                if p := to_post("post", d, sub, d["title"]):
                    posts[p["id"]] = p
                thread = get(session, f"/r/{sub}/comments/{d['id']}", sort="top", limit=COMMENTS_PER_POST, depth=1)
                assert isinstance(thread, list)
                for c in thread[1]["data"]["children"]:
                    cd = c.get("data", {})
                    if c.get("kind") == "t1" and cd.get("score", 0) >= MIN_SCORE:
                        if p := to_post("comment", cd, sub, d["title"]):
                            posts[p["id"]] = p
            print(f"r/{sub} {q!r}: total {len(posts)}")

    with OUT.open("w", encoding="utf-8") as f:
        for p in posts.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(posts)} posts to {OUT}")


if __name__ == "__main__":
    main()
