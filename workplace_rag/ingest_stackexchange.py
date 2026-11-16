"""Fetch questions and answers from The Workplace Stack Exchange via the official API.

    python -m workplace_rag.ingest_stackexchange

No key needed (anonymous quota: 300 requests/day; a full run uses ~60). Content is licensed
CC BY-SA, so every post keeps a link back to its source. Author names are not stored.

Writes data/raw/stackexchange.jsonl.
"""

import json
import time

import requests

from .config import RAW_DIR
from .ingest_hn import MIN_CHARS, clean

API = "https://api.stackexchange.com/2.3"
SITE = "workplace"
OUT = RAW_DIR / "stackexchange.jsonl"
HEADERS = {"User-Agent": "workplace-rag-research/0.1"}

# India first (the gap HN leaves), then employers and the topics job seekers ask about.
QUERIES = [
    "India", "Indian company", "Bangalore", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Noida", "Gurgaon",
    "TCS", "Infosys", "Wipro", "Cognizant", "Accenture", "HCL", "Tech Mahindra", "service based company",
    "notice period", "appraisal", "hike", "fresher", "bond", "onsite", "relieving letter",
    "Amazon", "Microsoft", "Facebook", "FAANG", "big tech",
]
TAGS = ["india"]
PAGES = {"India": 3, "india": 5}   # extra pages where there is lots of relevant content
MIN_ANSWER_SCORE = 2
MAX_ANSWERS = 8                   # per question, highest voted first


def get(path: str, **params) -> dict:
    params = {"site": SITE, "pagesize": 100, "filter": "withbody", **params}
    for attempt in range(3):
        resp = requests.get(f"{API}/{path}", params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("backoff"):          # the API asks clients to pause
                time.sleep(data["backoff"])
            return data
        time.sleep(5 * (attempt + 1))
    resp.raise_for_status()
    return {}


def find_questions() -> dict[int, dict]:
    questions: dict[int, dict] = {}
    searches = [{"q": q, "_pages": PAGES.get(q, 1)} for q in QUERIES]
    searches += [{"tagged": t, "_pages": PAGES.get(t, 1)} for t in TAGS]
    for s in searches:
        pages = s.pop("_pages")
        for page in range(1, pages + 1):
            data = get("search/advanced", sort="votes", order="desc", page=page, **s)
            for q in data.get("items", []):
                questions[q["question_id"]] = q
            if not data.get("has_more"):
                break
            time.sleep(0.3)
        print(f"  {s}: {len(questions)} questions so far (quota left {data.get('quota_remaining')})")
    return questions


def fetch_answers(qids: list[int]) -> list[dict]:
    answers: list[dict] = []
    for i in range(0, len(qids), 100):
        batch = ";".join(map(str, qids[i:i + 100]))
        page = 1
        while True:
            data = get(f"questions/{batch}/answers", sort="votes", order="desc", page=page)
            answers += data.get("items", [])
            if not data.get("has_more") or page >= 10:
                break
            page += 1
            time.sleep(0.3)
    return answers


def post(id_: str, url: str, title: str, body: str | None, created: int, query: str) -> dict:
    return {"id": id_, "source": "stackexchange", "url": url, "title": clean(title), "text": clean(body),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created)), "query": query}


def main() -> None:
    questions = find_questions()
    print(f"Found {len(questions)} questions")
    posts: dict[str, dict] = {}
    for qid, q in questions.items():
        p = post(f"se_q{qid}", q["link"], q["title"], q.get("body"), q["creation_date"], "workplace_se")
        if len(p["text"]) >= MIN_CHARS:
            posts[p["id"]] = p

    per_question: dict[int, int] = {}
    for a in fetch_answers(list(questions)):
        qid = a["question_id"]
        if a.get("score", 0) < MIN_ANSWER_SCORE or per_question.get(qid, 0) >= MAX_ANSWERS:
            continue
        p = post(f"se_a{a['answer_id']}", f"https://workplace.stackexchange.com/a/{a['answer_id']}",
                 questions[qid]["title"], a.get("body"), a["creation_date"], "workplace_se")
        if len(p["text"]) >= MIN_CHARS:
            posts[p["id"]] = p
            per_question[qid] = per_question.get(qid, 0) + 1

    with OUT.open("w", encoding="utf-8") as f:
        for p in posts.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_q = sum(p["id"].startswith("se_q") for p in posts.values())
    print(f"Saved {len(posts)} posts ({n_q} questions, {len(posts) - n_q} answers) to {OUT}")


if __name__ == "__main__":
    main()
