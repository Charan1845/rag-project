"""Retrieval relevance evaluation: are the cited sources actually about the question?

    python -m workplace_rag.evaluate run --mode post        # retrieve sources for eval/questions.json
    python -m workplace_rag.evaluate run --mode opinion
    python -m workplace_rag.evaluate label                  # print sources that still need a label
    python -m workplace_rag.evaluate score                  # precision@k per mode

Questions carry their intended filters, so this measures retrieval, not question parsing.
Labels live in eval/labels.jsonl as {"qid", "post_id", "relevant"}: no post text, so they are safe to commit.
A source is relevant if the post says something about the asked company (or, for a city question,
about working in that city) on the asked topic.
"""

import argparse
import json
import re

from . import normalize
from .config import DATA_DIR, ROOT
from .query import resolve, retrieve
from .store import load_records

QUESTIONS = ROOT / "eval" / "questions.json"
LABELS = ROOT / "eval" / "labels.jsonl"
RUNS = DATA_DIR / "eval"
RUNS.mkdir(parents=True, exist_ok=True)


def run(mode: str, k: int) -> None:
    out = RUNS / f"run_{mode}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for q in json.loads(QUESTIONS.read_text(encoding="utf-8")):
            filters, dropped, matches = resolve(q["question"], dict(q["filters"]))
            sources = retrieve(q["question"], filters, matches, k=k, mode=mode)
            f.write(json.dumps({"qid": q["id"], "n_matches": len(matches), "dropped": dropped,
                                "sources": [s["id"] for s in sources]}) + "\n")
            print(f"{q['id']}: {len(matches)} matching posts, {len(sources)} sources")
    print(f"Saved {out}")


def load_labels() -> dict[tuple[str, str], bool]:
    if not LABELS.exists():
        return {}
    with LABELS.open(encoding="utf-8") as f:
        return {(d["qid"], d["post_id"]): d["relevant"] for d in map(json.loads, f) if d}


def neutral_excerpt(text: str, q: dict) -> str:
    """Sentences that mention the asked company or city: the same view whichever mode found the post."""
    f = q["filters"]
    terms = []
    if f.get("company"):
        terms = [a for a, c in normalize.COMPANY_ALIASES.items() if c == f["company"]] + [f["company"].lower()]
    elif f.get("city"):
        terms = [a for a, c in normalize.CITY_ALIASES.items() if c == f["city"]]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    hits = [i for i, s in enumerate(sents) if any(re.search(rf"\b{re.escape(t)}\b", s.lower()) for t in terms)]
    keep = sorted({j for i in hits[:3] for j in (i, i + 1) if j < len(sents)})
    return " | ".join(sents[j][:300] for j in keep) or text[:400]


def label() -> None:
    questions = {q["id"]: q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))}
    records = {r["id"]: r for r in load_records()}
    labels = load_labels()
    todo = []
    for path in sorted(RUNS.glob("run_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for row in map(json.loads, f):
                todo += [(row["qid"], pid) for pid in row["sources"] if (row["qid"], pid) not in labels]
    seen = set()
    for qid, pid in todo:
        if (qid, pid) in seen:
            continue
        seen.add((qid, pid))
        q, r = questions[qid], records[pid]
        print(f"### {qid} | {pid} | Q: {q['question']}\nTITLE: {r['title']}\n{neutral_excerpt(r['text'], q)}\n")
    print(f"{len(seen)} unlabelled")


def score() -> None:
    labels = load_labels()
    print(f"{'mode':<8} {'P@k':>6} {'P@3':>6} {'top-1':>6} {'labelled':>9}")
    for path in sorted(RUNS.glob("run_*.jsonl")):
        rel = tot = rel3 = tot3 = top1 = n_q = missing = 0
        with path.open(encoding="utf-8") as f:
            for row in map(json.loads, f):
                ls = [labels.get((row["qid"], p)) for p in row["sources"]]
                missing += sum(x is None for x in ls)
                known = [x for x in ls if x is not None]
                rel += sum(known)
                tot += len(known)
                rel3 += sum(x for x in ls[:3] if x is not None)
                tot3 += sum(x is not None for x in ls[:3])
                if ls and ls[0] is not None:
                    n_q += 1
                    top1 += ls[0]
        mode = path.stem.removeprefix("run_")
        pct = lambda a, b: f"{a / b:.0%}" if b else "n/a"  # noqa: E731
        print(f"{mode:<8} {pct(rel, tot):>6} {pct(rel3, tot3):>6} {pct(top1, n_q):>6} "
              f"{tot:>4}/{tot + missing:<4}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "label", "score"])
    ap.add_argument("--mode", choices=["post", "opinion"], default="opinion")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    {"run": lambda: run(args.mode, args.k), "label": label, "score": score}[args.cmd]()


if __name__ == "__main__":
    main()
