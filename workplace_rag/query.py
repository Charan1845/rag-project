"""Answer workplace questions with structured filters, counts and citations.

Flow: question -> filters (LLM) -> matching posts (exact, over extracted metadata)
      -> sentiment counts over ALL matches -> semantic top-k among matches -> cited answer.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from . import normalize
from .config import LLM_MODEL, MIN_EVIDENCE, OLLAMA_BASE_URL
from .schema import Level, Topic
from .store import load_records, opinion_store, vectorstore


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
     "- Mention both positive and negative views if they exist. The overall lean given below was "
     "computed from all matching opinions: state it in words and do not contradict it.\n"
     "- Do NOT state any counts, fractions or percentages of posts or opinions: exact counts are shown "
     "to the reader separately.\n"
     "- Only attribute a view to a company if that post's 'company' field lists it.\n"
     "- If the posts do not answer the question, say so plainly.\n"
     "- 4-8 sentences. No names of people."),
    ("human", "Question: {question}\n\nOverall lean of all matching opinions: {lean}\n\nPosts:\n{context}"),
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
    evidence: str = ""


def _llm():
    return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=8192)


EMPLOYER_RE = re.compile(r"\b(?:at|for|about|joining|join|with)\s+([A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*)*)")


def named_employer(question: str) -> str | None:
    """Last-resort company detection for names we have no alias for ("WLB at Zoho").

    Missing a company is worse than an empty result: without it we would answer from every
    post and present that as being about the company the user asked about.
    """
    for m in EMPLOYER_RE.finditer(question):
        name = m.group(1).strip(" ?.")
        if normalize.city(name) in normalize.INDIAN_CITIES or normalize.find_in_text(name, normalize.CITY_ALIASES):
            continue
        if name.lower() in normalize.COUNTRIES or name.lower() in {"i", "a", "the"}:
            continue
        return normalize.company(name)
    return None


def parse_filters(question: str) -> dict:
    f = (FILTER_PROMPT | _llm().with_structured_output(Filters, method="json_schema")).invoke(
        {"question": question})
    # Exact matches against keyword lists beat the 7B model, which sometimes skips or invents a field.
    # Topics come from keywords only. The model invents one for general questions ("Is it worth
    # joining Flipkart?" -> compensation), which wrongly narrows them; no keyword = general question.
    topic = keyword_match(question, TOPIC_KEYWORDS)
    level = keyword_match(question, LEVEL_KEYWORDS)
    return {
        "company": (normalize.find_in_text(question, normalize.COMPANY_ALIASES)
                    or normalize.company(f.company) or named_employer(question)),
        "city": normalize.find_in_text(question, normalize.CITY_ALIASES) or normalize.city(f.city),
        "country": normalize.country(f.country),
        "role": keyword_match(question, ROLE_KEYWORDS) or normalize.role(f.role),
        "experience_level": level or (f.experience_level if f.experience_level != "unknown" else None),
        "topic": topic,
    }


TOPIC_KEYWORDS = {
    "work_life_balance": r"wlb|work[- ]?life|hours|overtime|weekends?|burn ?out|on-?call|stress",
    "compensation": r"pay|paid|salar(?:y|ies)|hikes?|appraisal|bonus|comp|compensation|ctc|stock|rsus?",
    "management": r"managers?|boss|leadership|manag(?:ement|ing)",
    "career_growth": r"promotions?|promoted|career|growth|grow",
    "layoffs_job_security": r"layoffs?|laid off|fired|job security|pips?|hire[- ]to[- ]fire",
    "interview_hiring": r"interviews?|hiring|recruit(?:er|ing|ment)?",
    "remote_work": r"wfh|remote|rto|return to office|work from home|hybrid",
    "culture": r"culture|politics|toxic|environment",
    "learning_tech": r"learn(?:ing)?|tech stack|training",
}
LEVEL_KEYWORDS = {
    "junior": r"freshers?|new grads?|graduates?|juniors?|entry[- ]level|campus|freshers'",
    "intern": r"interns?|internships?",
    "senior": r"seniors?|staff engineers?|principal engineers?",
    "manager": r"engineering managers?",
}
ROLE_KEYWORDS = {
    "software engineer": r"sdes?|swes?|software (?:engineers?|developers?)|developers?|devs?|programmers?",
    "data scientist": r"data scientists?",
    "product manager": r"product managers?|pms?",
    "devops engineer": r"devops|sres?",
    "qa engineer": r"qa|testers?",
}


def keyword_match(question: str, table: dict[str, str]) -> str | None:
    q = question.lower()
    for value, pattern in table.items():
        if re.search(rf"\b(?:{pattern})\b", q):
            return value
    return None


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
        # Role and level are soft: most posts never state them, so only exclude posts that state a
        # *different* one (a manager's post when asking about freshers), not posts that are silent.
        if f.get("role") and r["role"] and r["role"] != f["role"]:
            continue
        if f.get("experience_level") and r["experience_level"] not in (f["experience_level"], "unknown"):
            continue
        if f.get("topic") and not _opinions(r, f):
            continue
        out.append(r)
    return out


def _source(n: int, r: dict, snippet: str, opinions: list[dict]) -> dict:
    return {"n": n, "id": r["id"], "url": r["url"], "title": r["title"], "snippet": snippet.replace("�", ""),
            "opinions": opinions, "company": ", ".join(r["companies"]), "city": r["city"],
            "source": r.get("source", "hackernews")}


def retrieve_posts(question: str, f: dict, matches: list[dict], k: int) -> list[dict]:
    """v1: rank whole-post chunks by similarity to the question."""
    by_id = {r["id"]: r for r in matches}
    where = {"post_id": {"$in": list(by_id)}} if f else None
    # langchain-chroma annotates filter as dict[str, str], but Chroma accepts operators like $in.
    hits = vectorstore().similarity_search(question, k=k * 2, filter=where)  # ty: ignore[invalid-argument-type]
    sources, seen = [], set()
    for doc in hits:  # one source per post, best chunk first
        pid = doc.metadata["post_id"]
        if pid in seen or pid not in by_id:
            continue
        seen.add(pid)
        r = by_id[pid]
        sources.append(_source(len(sources) + 1, r, doc.page_content, _opinions(r, f) or r["opinions"]))
        if len(sources) == k:
            break
    return sources


STOP = set("""about after again also been being could does doing from have having here into just like more most
other over really same some such than that their them then there these they this those through very what when where
which while with would your posters poster says said company people""".split())
WORD_RE = re.compile(r"[a-z][a-z0-9'-]{3,}")
SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def excerpt(text: str, claims: list[str], company: str | None, max_sentences: int = 3) -> str:
    """The few sentences of a post that best support its matched claims (lexical overlap)."""
    want = {w for c in claims for w in WORD_RE.findall(c.lower())} - STOP
    if company:
        want |= {company.lower()} | {a for a, canon in normalize.COMPANY_ALIASES.items() if canon == company}
    sents = [s.strip() for s in SENT_RE.split(text) if len(s.strip()) > 20]
    if not sents:
        return text[:600]
    scored = sorted(range(len(sents)), key=lambda i: -len(want & set(WORD_RE.findall(sents[i].lower()))))
    keep = sorted(scored[:max_sentences])
    return " ... ".join(sents[i][:400] for i in keep)


def retrieve_opinions(question: str, f: dict, matches: list[dict], k: int) -> list[dict]:
    """v2: rank the short extracted opinions instead of whole posts, then group them by post.

    Each opinion is one focused claim ("TCS: managers expect weekend calls"), so a long comment that
    mentions the company once can no longer outrank a post that is actually about the question.
    """
    by_id = {r["id"]: r for r in matches}
    clauses: list[dict] = [{"post_id": {"$in": list(by_id)}}]
    if f.get("company"):
        clauses.append({"company": f["company"]})
    if f.get("topic"):
        clauses.append({"topic": f["topic"]})
    where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
    hits = opinion_store().similarity_search("search_query: " + question, k=k * 5, filter=where)
    if not hits and len(clauses) > 1:
        # No opinion is explicitly about this company/topic (e.g. multi-company posts where the
        # extractor left the company empty): rank all opinions within the matching posts instead.
        hits = opinion_store().similarity_search("search_query: " + question, k=k * 5, filter=clauses[0])

    grouped: dict[str, list[dict]] = {}
    for doc in hits:  # keep rank order: first post seen = best claim
        pid = doc.metadata["post_id"]
        if pid in by_id:
            grouped.setdefault(pid, []).append(
                {"topic": doc.metadata["topic"], "sentiment": doc.metadata["sentiment"],
                 "company": doc.metadata["company"] or None, "claim": doc.metadata["claim"]})
    sources = []
    for pid, ops in list(grouped.items())[:k]:
        r = by_id[pid]
        claims = [o["claim"] for o in ops[:3]]
        sources.append(_source(len(sources) + 1, r, excerpt(r["text"], claims, f.get("company")), ops[:3]))
    return sources


def resolve(question: str, filters: dict | None) -> tuple[dict, list[str], list[dict]]:
    """Filters (parsed or given), which ones had to be dropped, and the posts that match."""
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
    return f, dropped, matches


def retrieve(question: str, f: dict, matches: list[dict], k: int = 8, mode: str = "opinion") -> list[dict]:
    if not matches:
        return []
    return (retrieve_opinions if mode == "opinion" else retrieve_posts)(question, f, matches, k)


def ask(question: str, filters: dict | None = None, k: int = 8, mode: str = "opinion") -> Answer:
    f, dropped, matches = resolve(question, filters)
    sentiment = Counter(o["sentiment"] for r in matches for o in _opinions(r, f))
    if not matches:
        return Answer(question, f, dropped, 0, {}, "No posts in the dataset match this question yet.")
    sources = retrieve(question, f, matches, k, mode)

    def block(s: dict) -> str:
        claims = "\n".join(f"- {o['claim']}" for o in s["opinions"][:3])
        return (f"[{s['n']}] (company: {s['company'] or 'unspecified'}; city: {s['city'] or 'unspecified'})\n"
                f"Extracted claims:\n{claims}\nExcerpt: {s['snippet'][:1200]}")

    context = "\n\n".join(block(s) for s in sources)
    text = strip_counts((ANSWER_PROMPT | _llm()).invoke(
        {"question": question, "context": context, "lean": lean(sentiment)}).content)
    thin = len(matches) < MIN_EVIDENCE
    return Answer(question, f, dropped, len(matches), dict(sentiment), text, sources, thin,
                  evidence_line(len(matches), sentiment, f.get("topic")))


# "7 out of 10", "7 of 13 opinions", "68%": the 7B model miscounts, so numbers come from code only.
_NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)"
COUNT_RE = re.compile(rf"\b{_NUM}\s+(?:out\s+)?of\s+(?:the\s+)?{_NUM}\b|\b\d+(?:\.\d+)?\s?%|\bper\s?cent\b", re.I)


def strip_counts(text: str) -> str:
    """Drop sentences where the model still states a count despite the prompt."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sents if not COUNT_RE.search(s)]
    return " ".join(kept) if kept else text


def lean(sentiment: Counter | dict) -> str:
    """Overall direction in words, from the real counts, for the model to state without numbers."""
    total = sum(sentiment.values())
    if total < 3:
        return "too few opinions to say"
    neg, pos = sentiment.get("negative", 0) / total, sentiment.get("positive", 0) / total
    if neg >= 0.6:
        return "mostly negative"
    if pos >= 0.6:
        return "mostly positive"
    if neg > pos * 1.5:
        return "mixed, leaning negative"
    if pos > neg * 1.5:
        return "mixed, leaning positive"
    return "mixed"


def evidence_line(n_posts: int, sentiment: Counter | dict, topic: str | None) -> str:
    """The one sentence with numbers in every answer, written by code so it is always right."""
    total = sum(sentiment.values())
    about = f" on {topic.replace('_', ' ')}" if topic else ""
    if not total:
        return f"Based on {n_posts} matching post{'s' if n_posts != 1 else ''}."
    # No positive/negative split here: forum posts skew negative, so the counts read like a company
    # rating they are not. The lean still steers the answer's wording.
    return (f"Based on {n_posts} matching post{'s' if n_posts != 1 else ''} with {total} "
            f"opinion{'s' if total != 1 else ''}{about}.")
