"""Embed workplace posts into Chroma.

Usage:
    python -m workplace_rag.index

Only posts the extractor marked as workplace experiences are indexed. Each chunk keeps
its post id, so retrieval can be restricted to posts that match structured filters.
Incremental: chunks already in the store are skipped.
"""

from langchain_core.documents import Document

from .store import load_records, vectorstore

CHUNK = 1500
OVERLAP = 200


def chunks(text: str) -> list[str]:
    if len(text) <= CHUNK:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + CHUNK, len(text))
        if end < len(text):  # break on a paragraph or sentence if we can
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + CHUNK // 2:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP
    return out


def main() -> None:
    vs = vectorstore()
    existing = set(vs.get(include=[])["ids"])
    docs, ids = [], []
    for r in load_records():
        for i, piece in enumerate(chunks(r["text"])):
            cid = f"{r['id']}#{i}"
            if cid in existing:
                continue
            ids.append(cid)
            docs.append(Document(
                page_content=f"{r['title']}\n\n{piece}" if r["title"] else piece,
                metadata={"post_id": r["id"], "url": r["url"], "source": r["source"],
                          "created_at": r.get("created_at") or ""},
            ))
    for i in range(0, len(docs), 64):
        vs.add_documents(docs[i:i + 64], ids=ids[i:i + 64])
        print(f"  embedded {min(i + 64, len(docs))}/{len(docs)}")
    print(f"Index has {len(existing) + len(docs)} chunks ({len(docs)} new)")


if __name__ == "__main__":
    main()
