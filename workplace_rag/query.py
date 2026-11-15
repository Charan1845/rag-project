"""Answer workplace questions with structured filters, counts and citations.

Flow: question -> filters (LLM) -> matching posts (exact, over extracted metadata)
      -> sentiment counts over ALL matches -> semantic top-k among matches -> cited answer.
"""

from collections import Counter
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from . import normalize
from .config import LLM_MODEL, MIN_EVIDENCE, OLLAMA_BASE_URL
from .schema import Level, Topic
from .store import load_records, vectorstore


class Filters(BaseModel):
    company: str | None = Field(None, description="Employer asked about, if any")
    city: str | None = None
    country: str | None = None
    role: str | None = Field(None, description="Job role asked about, e.g. 'software engineer'")
    experience_level: Level | None = None
    topic: Topic | None = Field(None, description="Main workplace topic of the question, if any")


FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extract search filters from a question about workplaces. Leave a field null unless the "
               "question names it. Topic hints: 'WLB'/'hours'/'overtime' -> work_life_balance; "
               "'pay'/'salary'/'hikes'/'bonus' -> compensation; 'manager'/'managers'/'boss'/'leadership' -> "
               "management; 'promotion'/'growth' -> career_growth; 'layoffs'/'job security' -> "
               "layoffs_job_security; 'interview'/'hiring' -> interview_hiring; 'WFH'/'RTO' -> remote_work. "
               "'SDE'/'developer' mean software engineer; 'fresher' means junior."),
    ("human", "{question}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You summarize what people in online discussions say about workplaces, for job seekers.\n"
     "Rules:\n"
     "- Use ONLY the numbered posts below. Cite them like [1] or [2][3] after each claim.\n"
     "- Attribute everything: 'posters say', 'one commenter reports'. Never state claims about a "
     "company as fact.\n"
     "- Mention both positive and negative views if they exist. The counts are of OPINIONS, not posts "
     "(one post can hold several opinions): say e.g. '7 of 13 opinions were negative', never 'N posts'.\n"
     "- Only attribute a view to a company if that post's 'company' field lists it.\n"
     "- If the posts do not answer the question, say so plainly.\n"
     "- 4-8 sentences. No names of people."),
    ("human", "Question: {question}\n\n{n} posts match. Opinion counts across them ({total} opinions): "
              "{stats}\n\n"
              "Posts:\n{context}"),
])

# Least important first: when nothing matches we drop these one at a time.
RELAX_ORDER = ["role", "experience_level", "city", "country", "topic"]


@dataclass
class Answer:
    question: str
    filters: dict
    dropped_filters: list[str]
    n_posts: int
    sentiment: dict
    answer: str
    sources: list[dict] = field(default_factory=list)
    thin_evidence: bool = False


def _llm():
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192)


def parse_filters(question: str) -> dict:
    f = (FILTER_PROMPT | _llm().with_structured_output(Filters, method="json_schema")).invoke(
        {"question": question})
    # Exact matches against the alias lists beat the 7B model, which sometimes skips a field.
    return {
        "company": normalize.find_in_text(question, normalize.COMPANY_ALIASES) or normalize.company(f.company),
        "city": normalize.find_in_text(question, normalize.CITY_ALIASES) or normalize.city(f.city),
        "country": normalize.country(f.country),
        "role": normalize.role(f.role),
        "experience_level": f.experience_level if f.experience_level != "unknown" else None,
        "topic": f.topic if f.topic != "other" else None,
    }


def _opinions(rec: dict, f: dict) -> list[dict]:
    return [o for o in rec["opinions"]
            if (not f.get("topic") or o["topic"] == f["topic"])
            and (not f.get("company") or o["company"] == f["company"])]


def match(records: list[dict], f: dict) -> list[dict]:
    out = []
    for r in records:
        if f.get("company") and f["company"] not in r["companies"]:
            continue
        if f.get("city") and r["city"] != f["city"]:
            continue
        if f.get("country") and r["country"] != f["country"]:
            continue
        if f.get("role") and r["role"] != f["role"]:
            continue
        if f.get("experience_level") and r["experience_level"] != f["experience_level"]:
            continue
        if f.get("topic") and not _opinions(r, f):
            continue
        out.append(r)
    return out


def ask(question: str, filters: dict | None = None, k: int = 8) -> Answer:
    f = filters if filters is not None else parse_filters(question)
    f = {key: v for key, v in f.items() if v}
    records = load_records()

    dropped: list[str] = []
    matches = match(records, f)
    for key in RELAX_ORDER:
        if matches or key not in f:
            continue
        f.pop(key)
        dropped.append(key)
        matches = match(records, f)

    sentiment = Counter(o["sentiment"] for r in matches for o in _opinions(r, f))
    if not matches:
        return Answer(question, f, dropped, 0, {}, "No posts in the dataset match this question yet.")

    by_id = {r["id"]: r for r in matches}
    where = {"post_id": {"$in": list(by_id)}} if f else None
    hits = vectorstore().similarity_search(question, k=k * 2, filter=where)

    sources, seen = [], set()
    for doc in hits:  # one source per post, best chunk first
        pid = doc.metadata["post_id"]
        if pid in seen or pid not in by_id:
            continue
        seen.add(pid)
        r = by_id[pid]
        sources.append({"n": len(sources) + 1, "id": pid, "url": r["url"], "title": r["title"],
                        "snippet": doc.page_content, "opinions": _opinions(r, f) or r["opinions"],
                        "company": ", ".join(r["companies"]), "city": r["city"]})
        if len(sources) == k:
            break

    context = "\n\n".join(
        f"[{s['n']}] (company: {s['company'] or 'unspecified'}; city: {s['city'] or 'unspecified'})\n"
        f"{s['snippet'][:1200]}" for s in sources)
    stats = ", ".join(f"{v} {s}" for s, v in sentiment.most_common()) or "not available"
    text = (ANSWER_PROMPT | _llm()).invoke(
        {"question": question, "n": len(matches), "total": sum(sentiment.values()), "stats": stats,
         "context": context}).content

    thin = len(matches) < MIN_EVIDENCE
    return Answer(question, f, dropped, len(matches), dict(sentiment), text, sources, thin)
