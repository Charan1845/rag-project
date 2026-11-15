"""Ask a question from the terminal.

Usage:
    python -m workplace_rag.cli "How is work life balance at TCS?"
    python -m workplace_rag.cli --stats
"""

import argparse
from collections import Counter

from .query import ask
from .store import load_records


def print_stats() -> None:
    recs = load_records(workplace_only=False)
    work = [r for r in recs if r["is_workplace_experience"]]
    print(f"{len(recs)} posts extracted, {len(work)} are workplace experiences\n")
    for label, counter in [
        ("Companies", Counter(c for r in work for c in r["companies"])),
        ("Cities", Counter(r["city"] for r in work if r["city"])),
        ("Roles", Counter(r["role"] for r in work if r["role"])),
        ("Topics", Counter(o["topic"] for r in work for o in r["opinions"])),
    ]:
        print(label + ": " + ", ".join(f"{k} ({v})" for k, v in counter.most_common(12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--stats", action="store_true", help="show what the dataset covers")
    args = ap.parse_args()
    if args.stats or not args.question:
        print_stats()
        return

    a = ask(args.question)
    print(f"Filters: {a.filters}" + (f"  (relaxed: dropped {a.dropped_filters})" if a.dropped_filters else ""))
    print(f"Matching posts: {a.n_posts}   Sentiment: {a.sentiment}")
    if a.thin_evidence:
        print(f"WARNING: based on only {a.n_posts} post{'s' if a.n_posts != 1 else ''}; "
              "treat as anecdotes, not a trend.")
    print("\n" + a.evidence + "\n" + a.answer + "\n")
    for s in a.sources:
        print(f"[{s['n']}] {s['title'][:70]} - {s['url']}")


if __name__ == "__main__":
    main()
