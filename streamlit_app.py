"""Streamlit UI for the workplace insights RAG.

Run:  streamlit run streamlit_app.py
Needs Ollama running locally with qwen2.5:7b and nomic-embed-text, and a built index.
"""

from collections import Counter
from dataclasses import asdict
from typing import get_args

import pandas as pd
import streamlit as st

from workplace_rag.config import MIN_EVIDENCE
from workplace_rag.query import ask, parse_filters
from workplace_rag.schema import Topic
from workplace_rag.store import load_records

st.set_page_config(page_title="Workplace insights", page_icon=":material/work:")

AUTO = "Auto-detect"
EXAMPLES = [
    "How is work life balance at TCS?",
    "How are managers at Amazon?",
    "Is Infosys good for freshers?",
    "What do people in Bangalore say about pay?",
]
TOPICS = [t for t in get_args(Topic) if t != "other"]
SOURCE_NAMES = {"hackernews": "Hacker News", "stackexchange": "Workplace Stack Exchange", "reddit": "Reddit"}


# ---------- cached data and model calls ----------

@st.cache_data(ttl="1h")
def dataset() -> dict:
    """Option lists for the filter dropdowns plus headline stats."""
    all_recs = load_records(workplace_only=False)
    recs = [r for r in all_recs if r["is_workplace_experience"]]
    companies = Counter(c for r in recs for c in r["companies"])
    cities = Counter(r["city"] for r in recs if r["city"])
    return {
        "n_all": len(all_recs),
        "n_work": len(recs),
        "n_opinions": sum(len(r["opinions"]) for r in recs),
        # Only offer values with enough posts to be worth filtering on.
        "companies": [c for c, n in companies.most_common() if n >= 3],
        "cities": [c for c, n in cities.most_common() if n >= 2],
        "top_companies": companies.most_common(8),
    }


@st.cache_data(max_entries=200, show_spinner=False)
def cached_filters(question: str) -> dict:
    return parse_filters(question)


@st.cache_data(max_entries=200, show_spinner=False)
def cached_answer(question: str, filters: tuple) -> dict:
    return asdict(ask(question, dict(filters)))


def label(topic: str) -> str:
    return topic.replace("_", " ").capitalize()


def use_example() -> None:
    st.session_state.question = st.session_state.example
    st.session_state.example = None
    st.session_state.run = True


# ---------- sidebar ----------

data = dataset()
with st.sidebar:
    st.header("About the data")
    st.metric("Workplace posts", f"{data['n_work']:,}", help=f"Out of {data['n_all']:,} posts collected")
    st.metric("Extracted opinions", f"{data['n_opinions']:,}")
    st.caption("Most discussed employers")
    st.table(pd.DataFrame(data["top_companies"], columns=["Company", "Posts"]).set_index("Company"),
             alt="Companies with the most workplace posts")
    st.caption(
        "Sources: Hacker News and The Workplace Stack Exchange (CC BY-SA), via their official APIs. "
        "Author names are not stored. "
        "Answers summarize what posters say; they are anecdotes, not verified facts about any company."
    )

# ---------- question ----------

st.title("Workplace insights", anchor=False)
st.caption("Ask what people actually say about working somewhere. Every answer shows how much evidence it rests on.")

st.session_state.setdefault("run", False)
st.pills("Try an example", EXAMPLES, key="example", on_change=use_example)

# A form so pressing Enter in the question box submits it.
with st.form("ask_form", border=False):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        question = st.text_input("Question", key="question", placeholder="e.g. How is work life balance at TCS?")
        if st.form_submit_button("Ask", type="primary", icon=":material/search:", key="ask"):
            st.session_state.run = True

    with st.expander("Filters", icon=":material/tune:"):
        st.caption("Leave on auto-detect to read filters from the question. Pick a value to override it.")
        with st.container(horizontal=True):
            company = st.selectbox("Company", [AUTO, *data["companies"]], key="f_company")
            city = st.selectbox("City", [AUTO, *data["cities"]], key="f_city")
            topic = st.selectbox("Topic", [AUTO, *TOPICS], format_func=lambda t: t if t == AUTO else label(t),
                                 key="f_topic")

if st.session_state.run and question.strip():
    st.session_state.run = False
    with st.spinner("Reading posts and writing an answer (runs locally, ~10-30 s)..."):
        filters = cached_filters(question.strip())
        for key, value in (("company", company), ("city", city), ("topic", topic)):
            if value != AUTO:
                filters[key] = value
        st.session_state.result = cached_answer(question.strip(), tuple(sorted(filters.items())))

# ---------- answer ----------

result = st.session_state.get("result")
if result:
    f = result["filters"]
    with st.container(horizontal=True):
        if not f:
            st.badge("No filters: searching all posts", color="gray")
        for key, value in f.items():
            st.badge(f"{key.replace('_', ' ')}: {label(value) if key == 'topic' else value}",
                     icon=":material/filter_alt:")
    if result["dropped_filters"]:
        st.info(f"Nothing matched every filter, so these were dropped: {', '.join(result['dropped_filters'])}.",
                icon=":material/filter_alt_off:")

    if result["n_posts"] == 0:
        st.warning(result["answer"], title="No matching posts", icon=":material/search_off:")
        st.stop()

    n_opinions = sum(result["sentiment"].values())
    with st.container(horizontal=True):
        st.metric("Matching posts", result["n_posts"], border=True)
        st.metric("Opinions", n_opinions, border=True)

    if result["thin_evidence"]:
        n = result["n_posts"]
        st.warning(f"Based on only {n} post{'s' if n != 1 else ''} (under {MIN_EVIDENCE}). Treat this as anecdotes, "
                   "not a trend.", title="Thin evidence", icon=":material/warning:")

    with st.container(border=True):
        st.subheader("What posters say", anchor=False)
        st.caption(result.get("evidence", ""))
        st.markdown(result["answer"])

    st.subheader(f"Sources ({len(result['sources'])})", anchor=False)
    for s in result["sources"]:
        meta = " · ".join(x for x in (s["company"], s["city"]) if x)
        with st.expander(f"[{s['n']}] {s['title'] or 'Untitled thread'}" + (f"  ({meta})" if meta else "")):
            for op in s["opinions"][:5]:
                icon = {"negative": ":red[:material/thumb_down:]", "positive": ":green[:material/thumb_up:]"}.get(
                    op["sentiment"], ":gray[:material/remove:]")
                st.markdown(f"{icon} **{label(op['topic'])}**: {op['claim']}")
            st.caption("Excerpt")
            st.text(s["snippet"][:800] + ("..." if len(s["snippet"]) > 800 else ""))
            st.link_button(f"Open on {SOURCE_NAMES.get(s.get('source', ''), 'source site')}", s["url"],
                           icon=":material/open_in_new:")
