# Project ideas

Backlog of RAG project ideas brainstormed so far, grouped by theme.

## Chosen project

**Workplace insights for job seekers.** Answers questions like "how is work-life balance for software engineers at Company X in Hyderabad?" from real community discussions, filtered to people in the same role, city and situation, with counts and citations.

- **Data:** Reddit (r/developersIndia, r/IndianWorkplace, r/cscareerquestions) through the official Reddit Data API, and Hacker News through its official API. Stack Overflow Developer Survey (ODbL) for benchmarks. Optionally a small survey of seniors/alumni, collected with consent.
- **No scraping** of Glassdoor, AmbitionBox or Indeed, and no scraped Kaggle copies of them.
- **Core challenge:** posts are free text, so company, role, city, experience level and opinions must be extracted before indexing. Answers say "based on only N posts" when evidence is thin.
- **Evaluation:** hand-label ~200 posts for extraction accuracy, check that every claim traces to real posts, compare aggregates with the survey benchmarks.
- **Rules:** non-commercial, raw data never committed, answers phrased as "posters say...", never as facts about named companies.
- **Still open:** which LLM to use (free/local vs. API key).

## Review-based ideas (runners-up)

- **Critics vs. audience gap explainer.** Why did critics love a movie that audiences hated? Contrastive retrieval per group. Critics: Rotten Tomatoes excerpts or Wikipedia "Critical response"; audiences: Stanford IMDb reviews or TMDB API.
- **Car reliability assistant.** NHTSA complaints and recalls API (official). Failure patterns by model year and mileage.
- **Steam game review advisor.** Official Steam reviews API. Weight by playtime at review time; detect review bombing.
- **Hotel finder by trip type.** 515K Hotel Reviews Europe (Kaggle). Trip-type tags (family, couple, business) enable persona filtering and its evaluation.
- **App review → bug ticket converter.** Google Play review datasets. Deduplicated bug tickets with app version and user counts.
- **"What should I order?" assistant.** Yelp Open Dataset. Dish-level opinions.
- **Spoiler-free book review assistant.** UCSD Goodreads dataset with sentence-level spoiler labels.
- **Skincare advisor by skin type.** Sephora reviews (Kaggle, scraped: check license).
- **Airline route and cabin advisor.** Skytrax reviews (Kaggle, scraped: check license).
- **Product review shopper assistant.** Amazon Reviews 2023. Dropped: too close to Amazon's own features and data-terms concerns.

## Real-world public data (India-focused)

- **Government scheme navigator.** Scheme PDFs from india.gov.in / myscheme.gov.in; eligibility answers with clause citations.
- **Telugu/Hindi farmer advisory bot.** ICAR/KVK advisories; cross-lingual retrieval (Telugu question, English documents).
- **JoSAA college counselor.** Past cutoff ranks, NIRF data, college brochures; table lookups plus text retrieval.
- **Indian court precedent finder.** Supreme Court judgments (Indian Kanoon, open datasets).
- **Parliament debate search.** Lok Sabha / Rajya Sabha transcripts, cited by speaker and date.
- **Income Tax Act assistant.** Act text and CBDT circulars, section citations, not tax advice.
- **RTI application drafter.** RTI Act and CIC decisions.
- **Government tender finder.** CPPP / GeM tenders with eligibility checks for small businesses.
- **GATE/UPSC exam prep tutor.** Previous-year papers and NCERT textbooks.
- **NPTEL lecture search.** Answers link to the exact timestamp in the lecture video.
- **Election manifesto comparer.** Neutral side-by-side of party promises with page citations.
- **Food label safety checker.** Open Food Facts API plus FSSAI rules.
- **Thirukkural / Bhagavad Gita verse finder.** Public domain texts, cross-language retrieval by meaning.

## Finance, legal and health

- **SEC filings analyst assistant.** 10-K/10-Q filings and transcripts from SEC EDGAR.
- **Tenant/consumer rights assistant.** Rental and consumer protection statutes.
- **Insurance claim coverage checker.** Policy wording PDFs and IRDAI rules.
- **Regulatory diff assistant.** Compare versions of a law or spec ("what changed between v1 and v2?").
- **Drug interaction checker.** openFDA API, strict "cite or refuse".
- **Clinical trial matcher.** ClinicalTrials.gov API; reasoning over eligibility criteria.
- **Blood test report explainer.** MedlinePlus API, not medical advice.
- **Greenwashing detector.** Marketing claims checked against companies' own sustainability reports.

## Developer tools

- **GitHub issue duplicate finder.** Issues closed as "duplicate" give ground-truth labels.
- **Code review assistant in a repo's style.** Past PR review comments from the GitHub API.
- **Documentation gap finder.** RAG in reverse: user questions the docs can't answer.
- **Version-aware Stack Overflow helper.** SO data dump; flags answers outdated for your library version.
- **Library migration helper.** Changelogs and migration guides (pandas 2, Pydantic v2, React hooks).
- **Dependency vulnerability explainer.** NVD CVE API and GitHub Security Advisories.
- **Open-source license compatibility checker.** SPDX license texts.
- **Website accessibility (WCAG) checker.** Cites the exact success criterion each issue breaks.
- **Incident/on-call copilot.** Postmortems and runbooks; similar past incidents for a new error.
- **Codebase Q&A.** "Where is X implemented / why was Y changed?"
- **Text-to-SQL grounded by retrieval.** Retrieve relevant schema and example queries before generating SQL.

## Research, education and knowledge

- **Related-work writer with citation checks.** arXiv and Semantic Scholar; verifies every cited paper exists.
- **Kaggle winning-solution finder.** Top-team write-ups compared across competitions.
- **Dataset finder for ML.** Hugging Face, Kaggle and data.gov.in dataset cards.
- **Math lemma finder for Lean.** English description to mathlib lemma.
- **Lab protocol finder.** protocols.io API.
- **IPCC climate report Q&A.** Keeps the report's own confidence labels.
- **Historical newspaper search.** Chronicling America (Library of Congress), noisy OCR text.
- **NASA lessons-learned search.** NASA's public Lessons Learned database.
- **Aviation accident report analyzer.** NTSB and DGCA/AAIB reports, patterns across incidents.
- **Wikipedia "citation needed" fixer.** Finds sources for unsourced claims.
- **System design knowledge base.** Engineering blogs (Netflix, Uber, Swiggy, Zerodha) via RSS.
- **Patent prior-art search.** USPTO / Google Patents public data.
- **Electronics datasheet Q&A.** Answers from datasheet tables and pin diagrams.

## Everyday and fun

- **Cricket stats + commentary Q&A.** Cricsheet ball-by-ball data plus match reports.
- **Sports rulebook referee.** ICC/FIFA laws, cited clause for each ruling.
- **Chess position explainer.** Lichess database (CC0); retrieval over board positions.
- **Museum art search.** Met Museum Open Access API (CC0).
- **Location-aware trip planner.** Wikivoyage plus OpenStreetMap with distance limits.
- **Spoiler-free game wiki assistant.** Retrieval limited to the player's progress.
- **Car fault code diagnosis.** OBD-II codes plus repair forum threads.
- **Terms of service explainer.** ToS;DR ratings as evaluation labels.
- **Offline disaster-response assistant.** NDMA guidelines, no internet, SMS-length answers.
- **News framing comparer.** GDELT coverage of the same event across outlets.
- **Bug bounty report learner.** Publicly disclosed HackerOne reports.
- **Meme / cultural reference explainer.** Meme origins and slang sources.
- **Startup/hackathon idea originality checker.** Devpost, YC directory, Product Hunt.
- **Resume ↔ job description gap finder.** Public job posting datasets.
- **Personal "second brain".** Your own notes, highlights and saved articles.
- **Voice-native RAG.** Speech in, speech out, low latency.

## Production-system flavored

- **Multi-tenant RAG-as-a-service.** Per-customer isolation, auth, usage metering.
- **RAG cost/latency benchmark.** Chunking, embeddings, caching and vector DBs compared on cost and p95 latency.
- **Fully local/offline RAG.** llama.cpp/Ollama, local embeddings and vector store.
- **Feedback-driven re-ranking.** Thumbs up/down used to re-weight retrieval.
- **Support-ticket triage.** Similar past tickets, suggested fix, auto-close vs. escalate.
- **Freshness-aware self-updating RAG.** Scheduled crawl, incremental re-embedding, time-decay scoring.
- **Multimodal and hybrid/agentic variants.** Text+image retrieval, BM25 + vectors + re-ranking, multi-hop retrieval.

## Technically differentiated

- **GraphRAG.** Knowledge graph plus embeddings for multi-hop questions.
- **RAG fact-checker.** Claim in, verdict and cited evidence out.
- **Debate-and-judge RAG.** Two agents retrieve opposing evidence; a judge weighs it.
- **RAG with evaluation and guardrails.** Groundedness checks, citations, an eval harness.
