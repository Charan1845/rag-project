"""Turn messy free-text posts into structured records with a local LLM (Ollama).

Usage:
    python -m workplace_rag.extract --limit 50

Reads data/raw/*.jsonl, writes data/processed/posts.jsonl. Resumable: posts already
processed are skipped, so you can stop and restart at any time.
"""

import argparse
import json
import re
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from . import normalize
from .config import LLM_MODEL, OLLAMA_BASE_URL, PROCESSED_DIR, RAW_DIR
from .schema import Extraction

OUT = PROCESSED_DIR / "posts.jsonl"
MAX_CHARS = 4000

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract structured facts from online discussion posts about workplaces.\n"
            "Rules:\n"
            "- Only use what the post says. Never guess a company, city or role that is not stated or clearly implied.\n"
            "- is_workplace_experience is true only if the poster shares an experience or opinion about working "
            "at a company or in the industry (pay, hours, managers, culture, layoffs, growth, hiring...). "
            "News summaries, product talk and pure technical discussion are false.\n"
            "- Use short canonical company names (TCS, Infosys, Google, Amazon...).\n"
            "- experience_level: intern, junior (0-2 yrs), mid (3-6 yrs), senior (7+ yrs or staff/lead), "
            "manager, or unknown.\n"
            "- If is_workplace_experience is false, return empty opinions.\n"
            "- role: the poster's own job. 'I am a senior engineer' -> 'software engineer'. "
            "city/country: where the job is, e.g. 'Indian startup' -> country India.\n"
            "- One opinion per distinct (topic, company) claim, at most 5. claim is a short neutral paraphrase, "
            "not a quote, and never includes names of people.",
        ),
        ("human", "Thread title: Ask HN: Is it worth joining a service company?\n\nPost:\n"
                  "I spent 3 years at TCS in Hyderabad as a developer. Salary hikes were 4-6% a year and my manager "
                  "expected weekend calls during releases. The training program for freshers was actually good though."),
        ("ai", json.dumps({
            "is_workplace_experience": True, "companies": ["TCS"], "role": "software engineer",
            "city": "Hyderabad", "country": "India", "experience_level": "mid",
            "opinions": [
                {"topic": "compensation", "sentiment": "negative", "company": "TCS",
                 "claim": "Annual raises were low, around 4-6%."},
                {"topic": "work_life_balance", "sentiment": "negative", "company": "TCS",
                 "claim": "Managers expected weekend calls during releases."},
                {"topic": "learning_tech", "sentiment": "positive", "company": "TCS",
                 "claim": "The fresher training program was good."},
            ],
        }).replace("{", "{{").replace("}", "}}")),
        ("human", "Thread title: Show HN: A faster JSON parser in Rust\n\nPost:\n"
                  "Nice work. I benchmarked it against serde_json on my laptop and it was about 30% faster on large files."),
        ("ai", json.dumps({
            "is_workplace_experience": False, "companies": [], "role": None, "city": None,
            "country": None, "experience_level": "unknown", "opinions": [],
        }).replace("{", "{{").replace("}", "}}")),
        ("human", "Thread title: {title}\n\nPost:\n{text}"),
    ]
)


def build_chain():
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=4096,
                     # Cap output: some posts send the model into an endless loop that blocks Ollama.
                     num_predict=1024)
    return PROMPT | llm.with_structured_output(Extraction, method="json_schema")


def load_raw() -> list[dict]:
    """All raw posts from every source file, de-duplicated by id (the same HN comment can come
    from both keyword search and thread crawling)."""
    posts: dict[str, dict] = {}
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                posts.setdefault(p["id"], p)
    return list(posts.values())


FIRST_PERSON = re.compile(r"\b(I|I've|I'm|my)\b.{0,40}\b(work|worked|working|joined|manager|team|job)", re.I)
INDIA = re.compile(r"\b(india|indian|bangalore|bengaluru|hyderabad|pune|chennai|mumbai|gurgaon|noida)\b", re.I)


def worth_extracting(post: dict) -> bool:
    """Cheap pre-filter for big crawls: names an employer (or India) and speaks from experience.
    Extraction costs ~3 s/post on a laptop GPU, so we spend it where filters can use the result."""
    text = f"{post['title']} {post['text']}"
    named = any(normalize.mentioned(c, text) for c in set(normalize.COMPANY_ALIASES.values())) or INDIA.search(text)
    return bool(named and FIRST_PERSON.search(post["text"]))


def done_ids() -> set[str]:
    if not OUT.exists():
        return set()
    with OUT.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f}


def to_record(post: dict, ex: Extraction) -> dict:
    return normalize.record({
        **post,
        "is_workplace_experience": ex.is_workplace_experience,
        "companies": ex.companies,
        "role": ex.role,
        "city": ex.city,
        "country": ex.country,
        "experience_level": ex.experience_level,
        "opinions": [op.model_dump() for op in ex.opinions] if ex.is_workplace_experience else [],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process at most N new posts")
    ap.add_argument("--prefilter", action="store_true",
                    help="only posts that name an employer/India and are written from experience")
    args = ap.parse_args()

    chain = build_chain()
    skip = done_ids()
    todo = [p for p in load_raw() if p["id"] not in skip and (not args.prefilter or worth_extracting(p))]
    todo = todo[: args.limit]
    print(f"{len(skip)} already done, processing {len(todo)}")

    start = time.time()
    with OUT.open("a", encoding="utf-8") as f:
        for i, post in enumerate(todo, 1):
            ex = None
            for attempt in range(3):
                try:
                    ex = chain.invoke({"title": post["title"], "text": post["text"][:MAX_CHARS]})
                    break
                except Exception as e:  # dropped connection, malformed JSON...; skip after 3 tries
                    print(f"  ! {post['id']} try {attempt + 1}: {e.__class__.__name__}: {e}"[:200])
                    time.sleep(2)
            if ex is None:
                continue
            f.write(json.dumps(to_record(post, ex), ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - start) / i
                print(f"  {i}/{len(todo)}  {rate:.1f}s/post  ~{rate * (len(todo) - i) / 60:.0f} min left")


if __name__ == "__main__":
    main()
