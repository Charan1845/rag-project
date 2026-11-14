# Workplace insights RAG

Ask what people actually say about working somewhere, e.g. *"How is work-life balance for SDEs at TCS in Hyderabad?"*,
and get an answer built from community discussions, with **counts and citations** instead of a vague summary.

> Status: 🚧 MVP. Hacker News source working; Reddit source next.

## Why not just ask Glassdoor or ChatGPT?

- Posts are messy free text. Before indexing, a local LLM **extracts structure** from each one: company, role, city,
  country, experience level, and per-topic opinions with sentiment.
- Questions are turned into **exact filters** over that structure, so "TCS in Hyderabad" really means posts about TCS
  in Hyderabad, not whatever is semantically nearby.
- Sentiment counts are computed over **all** matching posts, not just the top-k retrieved ones.
- When evidence is thin, the answer says so: *"based on only N posts"*. If nothing matches, filters are relaxed one
  at a time and the answer reports which were dropped.
- Answers are phrased as *"posters say…"*, never as facts about a company.

## Pipeline

```
ingest_hn.py   Algolia HN Search API  ->  data/raw/hn.jsonl
extract.py     qwen2.5:7b (Ollama), JSON-schema output  ->  data/processed/posts.jsonl
index.py       nomic-embed-text  ->  Chroma (data/chroma)
query.py       question -> filters -> matching posts -> counts + semantic top-k -> cited answer
```

Everything runs locally and free (tested on an RTX 5060 Laptop GPU with 8 GB of VRAM).

## Run it

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements.txt
ollama pull qwen2.5:7b && ollama pull nomic-embed-text

python -m workplace_rag.ingest_hn --pages 1
python -m workplace_rag.extract            # resumable, ~3 s/post
python -m workplace_rag.index
python -m workplace_rag.cli --stats
python -m workplace_rag.cli "How is work life balance at TCS?"
```

## Data and ethics

- Sources: official APIs only (Hacker News via Algolia; Reddit Data API planned). No Glassdoor/AmbitionBox scraping.
- Author names are not stored. Raw data is never committed (`data/` is gitignored).
- Non-commercial project for learning.

## Roadmap

- [ ] Reddit source (r/developersIndia, r/IndianWorkplace, r/cscareerquestions)
- [ ] Streamlit UI
- [ ] Evaluation: ~200 hand-labelled posts for extraction accuracy, answer faithfulness checks,
      comparison with Stack Overflow Developer Survey benchmarks
