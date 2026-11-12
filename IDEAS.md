# Candidate project ideas (not decided yet)

Backlog of RAG project directions, grounded in real/public data sources, to pick from.

## Real-world, data available now
- **SEC filings analyst assistant** — real 10-K/10-Q filings + earnings call transcripts from SEC EDGAR (free, no auth). Answer analyst-style questions with citations to the exact filing section.
- **Government scheme/benefits navigator** — real scheme PDFs from india.gov.in / myscheme.gov.in. Answer eligibility questions with citations to the actual clause.
- **Tenant/consumer rights assistant** — real state rental law statutes / consumer protection acts. High stakes if wrong, so forces proper citation + confidence handling.
- **Open civic data assistant** — a city's own open data portal (council minutes, building codes, transit schedules). Locally verifiable, easy to demo.
- **Drug interaction / medication checker** — openFDA's free API (drug labels, adverse events). Strict "cite or refuse" policy.

## Production-system flavored
- **Multi-tenant RAG-as-a-service** — per-customer isolated namespaces, auth, usage metering.
- **RAG cost/latency benchmark** — compare chunking strategies, embedding models, caching, vector DBs on cost-per-query and p95 latency.
- **Fully local/offline RAG** — llama.cpp/Ollama + local embeddings + local vector store, no external API calls.
- **Feedback-driven re-ranking RAG** — thumbs up/down logged and used to re-weight retrieval over time.
- **Support-ticket triage RAG** — retrieve similar past tickets + resolution macros, output confidence for auto-close vs. escalate.

## Technically differentiated
- **GraphRAG** — knowledge graph + embeddings for multi-hop questions vector-only retrieval can't answer.
- **RAG fact-checker** — claim in, verdict + cited evidence out (support/refute/insufficient).
- **Freshness-aware self-updating RAG** — scheduled crawl, incremental re-embedding of only diffed content, time-decay scoring.
- **Text-to-SQL grounded by retrieval** — retrieve relevant schema/example queries before generating SQL, instead of dumping the whole schema in the prompt.
- **Debate-and-judge RAG** — two agents retrieve opposing evidence and argue; a judge model weighs citations for a final verdict.

## Decision needed
Pick one lane: resume/interview impact (SEC filings, support triage) vs. technical depth (GraphRAG, fact-checker) vs. personal/social-impact story (government scheme navigator).
