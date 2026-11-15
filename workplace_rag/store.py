"""Load extracted records and open the vector store."""

import json
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from . import normalize
from .config import CHROMA_DIR, COLLECTION, EMBED_MODEL, OLLAMA_BASE_URL, PROCESSED_DIR

POSTS = PROCESSED_DIR / "posts.jsonl"


def load_records(workplace_only: bool = True) -> list[dict]:
    """All extracted posts, re-normalized. Later lines win if a post appears twice."""
    if not POSTS.exists():
        return []
    recs: dict[str, dict] = {}
    with POSTS.open(encoding="utf-8") as f:
        for line in f:
            r = normalize.record(json.loads(line))
            recs[r["id"]] = r
    out = list(recs.values())
    return [r for r in out if r["is_workplace_experience"]] if workplace_only else out


@lru_cache(maxsize=1)
def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


@lru_cache(maxsize=1)
def vectorstore() -> Chroma:
    """Post chunks (v1 retrieval)."""
    return Chroma(collection_name=COLLECTION, embedding_function=_embeddings(), persist_directory=str(CHROMA_DIR))


@lru_cache(maxsize=1)
def opinion_store() -> Chroma:
    """One document per extracted opinion (v2 retrieval). nomic-embed-text expects
    'search_document: ' / 'search_query: ' prefixes, so documents and queries carry them."""
    return Chroma(collection_name="opinions", embedding_function=_embeddings(),
                  persist_directory=str(CHROMA_DIR))
