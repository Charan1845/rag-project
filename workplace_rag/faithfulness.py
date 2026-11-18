"""Answer faithfulness: is each claim in a generated answer supported by the post it cites?

    python -m workplace_rag.faithfulness generate  # answer eval/questions.json (slow: runs the full RAG)
    python -m workplace_rag.faithfulness judge     # judge every (claim, cited post) pair
    python -m workplace_rag.faithfulness score     # summary
    python -m workplace_rag.faithfulness show --verdict unsupported   # read pairs to check the judge

Each answer is split into sentences, and each sentence into the claim each citation backs:
"One poster says A [2], while others say B [3][5]." gives (A, post 2), (B, post 3), (B, post 5).
The judge (qwen2.5:7b, temperature 0) reads the cited post's original text, not our extracted claims,
so errors introduced during extraction are caught too.

Outputs go to data/eval/answers.jsonl and faithfulness.jsonl (they contain post text, so they stay
out of git).
"""

import argparse
import json
import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .config import DATA_DIR, LLM_MODEL, OLLAMA_BASE_URL, ROOT
from .query import ask
from .store import load_records

QUESTIONS = ROOT / "eval" / "questions.json"
ANSWERS = DATA_DIR / "eval" / "answers.jsonl"
OUT = DATA_DIR / "eval" / "faithfulness.jsonl"
POST_CHARS = 4000


class Verdict(BaseModel):
    quote: str = Field(description="The words in the post most relevant to the claim, copied exactly, "
                                   "or '' if nothing is relevant.")
    stance: Literal["affirms", "partly_affirms", "denies", "not_mentioned"] = Field(
        description="What the post's author says about the claim.")
    reason: str = Field(description="One short sentence explaining the stance.")


# The judge only reports the post's stance; code maps it to a verdict. Asked for a verdict directly,
# the 7B model called a post that rejects a claim ("no one gives kickbacks") support for it.
STANCE_VERDICT = {"affirms": "supported", "partly_affirms": "partial",
                  "denies": "unsupported", "not_mentioned": "unsupported"}

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check a claim from a summary against one source post. Report the post author's stance:\n"
     "- affirms: the author says this (paraphrase is fine).\n"
     "- partly_affirms: the author says something related, but the claim adds details, generalizes, "
     "changes the strength, or attributes it to a company the post does not discuss.\n"
     "- denies: the author says the opposite, or mentions the claim only to reject it "
     "('people say X, but that is nonsense' DENIES X).\n"
     "- not_mentioned: the post does not address the claim.\n"
     "Judge only the substance of the claim. Ignore framing about how many people hold a view or the "
     "overall lean ('one commenter', 'others', 'many', 'overall, posters lean negative'): those come "
     "from elsewhere. The full sentence is given for context only."),
    ("human", "POST (title: {title}):\n{post}\n\nFULL SENTENCE (context): {sentence}\n\n"
              "CLAIM TO CHECK: {claim}"),
])

CITE_RE = re.compile(r"\[(\d+)\]")
GROUP_RE = re.compile(r"(?:\s*\[\d+\])+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text.strip()) if s.strip()]


def claims(sentence: str) -> list[tuple[str, list[int]]]:
    """Pair each citation group with the text it follows: 'A [2], while B [3][5].' ->
    [('A', [2]), ('while B', [3, 5])]. A group with almost no text before it backs the whole sentence."""
    out, start = [], 0
    whole = CITE_RE.sub("", sentence).strip()
    for m in GROUP_RE.finditer(sentence):
        text = sentence[start:m.start()].strip(" ,;:.")
        nums = [int(n) for n in CITE_RE.findall(m.group())]
        out.append((text if len(text.split()) >= 4 else whole, nums))
        start = m.end()
    return out


def judge_chain():
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192, num_predict=300)
    return JUDGE_PROMPT | llm.with_structured_output(Verdict, method="json_schema")


def generate() -> None:
    with ANSWERS.open("w", encoding="utf-8") as f:
        for q in json.loads(QUESTIONS.read_text(encoding="utf-8")):
            a = ask(q["question"], dict(q["filters"]))
            sources = {str(s["n"]): s["id"] for s in a.sources}
            row = {"qid": q["id"], "question": q["question"], "answer": a.answer, "sources": sources}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{q['id']}: {len(sentences(a.answer))} sentences, {len(a.sources)} sources")
    print(f"Saved {ANSWERS}")


def judge() -> None:
    records = {r["id"]: r for r in load_records(workplace_only=False)}
    chain = judge_chain()
    with ANSWERS.open(encoding="utf-8") as fa, OUT.open("w", encoding="utf-8") as f:
        for a in map(json.loads, fa):
            for i, sent in enumerate(sentences(a["answer"])):
                row = {"qid": a["qid"], "question": a["question"], "i": i, "sentence": sent, "judgements": []}
                for claim, nums in claims(sent):
                    for n in nums:
                        pid = a["sources"].get(str(n))
                        if pid is None:  # cites a number that was never given
                            row["judgements"].append({"n": n, "post_id": None, "claim": claim,
                                                      "verdict": "unsupported", "reason": "no such source"})
                            continue
                        rec = records[pid]
                        v = chain.invoke({"title": rec["title"], "post": rec["text"][:POST_CHARS],
                                          "sentence": CITE_RE.sub("", sent), "claim": claim})
                        row["judgements"].append({"n": n, "post_id": pid, "claim": claim,
                                                  "verdict": STANCE_VERDICT[v.stance], "stance": v.stance,
                                                  "quote": v.quote, "reason": v.reason})
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{a['qid']} judged")
    print(f"Saved {OUT}")
    score()


def load() -> list[dict]:
    with OUT.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


LABELS = ROOT / "eval" / "faithfulness_labels.jsonl"


def load_labels() -> dict[tuple[str, int, int], str]:
    """Hand labels for one answers.jsonl run (ids only). Regenerating answers invalidates them."""
    if not LABELS.exists():
        return {}
    with LABELS.open(encoding="utf-8") as f:
        return {(d["qid"], d["i"], d["n"]): d["label"] for d in map(json.loads, f) if d}


def summarize(name: str, rows: list[dict], verdict) -> None:
    """verdict(row, judgement) -> 'supported' | 'partial' | 'unsupported' | None (unlabelled)."""
    pct = lambda a, b: f"{a / b:.0%}" if b else "n/a"  # noqa: E731
    sents = [[verdict(r, j) for j in r["judgements"]] for r in rows if r["judgements"]]
    sents = [vs for vs in sents if None not in vs]
    pairs = [v for vs in sents for v in vs]
    # A sentence is fully supported if every citation fully supports its claim, and at least
    # partly supported if none of its citations is unsupported.
    full = sum(all(v == "supported" for v in vs) for vs in sents)
    some = sum("unsupported" not in vs for vs in sents)
    print(f"{name:<12} pairs {len(pairs):>3}: supported {pct(pairs.count('supported'), len(pairs)):>4}, "
          f"partial {pct(pairs.count('partial'), len(pairs)):>4}, "
          f"unsupported {pct(pairs.count('unsupported'), len(pairs)):>4} | sentences {len(sents)}: "
          f"fully supported {pct(full, len(sents))}, at least partly {pct(some, len(sents))}")


def score() -> None:
    rows = load()
    cited = sum(bool(r["judgements"]) for r in rows)
    print(f"Answer sentences: {len(rows)}, with a citation: {cited} ({cited / len(rows):.0%})")
    labels = load_labels()
    summarize("LLM judge", rows, lambda r, j: j["verdict"])
    if not labels:
        return
    summarize("hand labels", rows, lambda r, j: labels.get((r["qid"], r["i"], j["n"])))
    both = [(labels[(r["qid"], r["i"], j["n"])], j["verdict"]) for r in rows for j in r["judgements"]
            if (r["qid"], r["i"], j["n"]) in labels]
    agree = sum(h == m for h, m in both)
    missed = sum(h == "unsupported" and m != "unsupported" for h, m in both)
    n_unsup = sum(h == "unsupported" for h, _ in both)
    print(f"Judge agrees with hand labels on {agree}/{len(both)} pairs ({agree / len(both):.0%}); "
          f"it caught {n_unsup - missed} of {n_unsup} hand-labelled unsupported claims.")


def show(verdict: str | None) -> None:
    records = {r["id"]: r for r in load_records(workplace_only=False)}
    for r in load():
        for j in r["judgements"]:
            if verdict and j["verdict"] != verdict:
                continue
            post = records[j["post_id"]]["text"][:700].replace("\n", " ") if j["post_id"] else ""
            print(f"### {r['qid']} s{r['i']} [{j['n']}] {j['post_id']} -> {j['verdict'].upper()}\n"
                  f"CLAIM: {j.get('claim', r['sentence'])}\nQUOTE: {j.get('quote', '')}\n"
                  f"JUDGE: {j.get('stance', '')}: {j['reason']}\nPOST: {post}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "judge", "score", "show"])
    ap.add_argument("--verdict", choices=["supported", "partial", "unsupported"])
    args = ap.parse_args()
    {"generate": generate, "judge": judge, "score": score, "show": lambda: show(args.verdict)}[args.cmd]()


if __name__ == "__main__":
    main()
