# rag_chroma.py
from pathlib import Path
from typing import List, Tuple, Dict
import json

from dotenv import load_dotenv
from openai import OpenAI

import chromadb


load_dotenv()  # load OPENAI_API_KEY from .env

client = OpenAI()

# ---------- Embeddings helper ----------

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Call OpenAI embeddings and return a list of embedding vectors.
    """
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    usage = getattr(resp, "usage", None)
    if usage:
        print(
            f"[EMBED] items={len(texts)} "
            f"tokens={getattr(usage, 'total_tokens', 0)}"
        )

    return [d.embedding for d in resp.data]


# ---------- Chroma setup (local, persistent) ----------

# New architecture: use PersistentClient to store on disk
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="docs",
    embedding_function=None,  # we still provide embeddings ourselves
)


# ---------- Ingestion ----------

def ingest_folder(folder: str = "data") -> None:
    """
    Read all .txt files from a folder, embed them, and upsert into Chroma.
    """
    paths = list(Path(folder).glob("*.txt"))
    if not paths:
        print(f"[ingest] No .txt files found in {folder}")
        return

    ids: List[str] = []
    docs: List[str] = []

    for p in paths:
        content = p.read_text(encoding="utf-8")
        ids.append(p.stem)
        docs.append(content)

    embs = embed_texts(docs)

    collection.upsert(
        ids=ids,
        documents=docs,
        embeddings=embs,
    )

    print(f"[ingest] Ingested {len(ids)} docs into Chroma from {folder}")


# ---------- Retrieval ----------

def retrieve(query: str, k: int = 3) -> List[Tuple[str, str, float]]:
    """
    Embed the query and return top-k (id, document, similarity) triples.
    """
    q_emb = embed_texts([query])[0]

    result = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        # 'ids' is NOT allowed here; it's always returned automatically
        include=["documents", "distances"],
    )

    docs = result["documents"][0]
    ids = result["ids"][0]          # still available
    dists = result["distances"][0]  # smaller = closer
    #print(dists)
    sims = [1.0 - float(d) for d in dists]
    return list(zip(ids, docs, sims))


# ---------- RAG answer ----------

SYSTEM_RAG = """
You are a careful assistant.

You are given some context chunks from a private knowledge base.
Follow these rules:
- Use ONLY the provided context to answer the question.
- If the answer is not clearly supported by the context, say "I don't know based on the given documents."
- Do NOT fabricate or guess.
- When possible, mention which document(s) you used (by id).
"""


def answer_with_rag(query: str, k: int = 3) -> Tuple[str, List[str], Dict[str, int]]:
    """
    Retrieve top-k docs for the query, then call the LLM with the context.
    Returns (answer_text, list_of_source_ids, usage_dict).
    """
    hits = retrieve(query, k=k)

    if not hits:
        return "I couldn't find any relevant context.", [], {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    context_blocks = []
    source_ids: List[str] = []

    for doc_id, doc_text, sim in hits:
        context_blocks.append(f"[{doc_id}] {doc_text}")
        source_ids.append(doc_id)

    context = "\n\n---\n\n".join(context_blocks)

    resp = client.responses.create(
        model="gpt-5-mini",
        instructions=SYSTEM_RAG,
        input=(
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Remember to cite document ids where relevant."
        ),
    )

    answer_text = resp.output_text

    usage = getattr(resp, "usage", None)
    usage_dict = {
        "input_tokens": getattr(usage, "input_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "output_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage and hasattr(usage, "total_tokens") else (
            (getattr(usage, "input_tokens", 0) or 0)
            + (getattr(usage, "output_tokens", 0) or 0)
            if usage else 0
        ),
    }

    return answer_text, source_ids, usage_dict

def get_docs_by_ids(ids: List[str]) -> List[str]:
    """
    Fetch raw document texts from Chroma by their ids.
    """
    if not ids:
        return []

    res = collection.get(
        ids=ids,
        include=["documents"],
    )

    # In new Chroma, "documents" is a flat list of strings
    docs = res.get("documents") or []
    return docs

if __name__ == "__main__":
    # 1) One-time (or occasional) ingestion
    ingest_folder("data")

    # 2) Test query
    q = "What does an AI engineer typically do?"
    ans, src = answer_with_rag(q, k=3)
    print("\n=== QUESTION ===")
    print(q)
    print("\n=== ANSWER ===")
    print(ans)
    print("\n=== SOURCES ===")
    print(json.dumps(src, indent=2))
    sources = get_docs_by_ids(src)
    for i, source in enumerate(sources):
        print(f"\n--- Document {src[i]} ---\n{source}\n")
# ---------- Step evaluation ----------