"""
src/rag/qa.py

Query phase of RAG.
Loads the prebuilt FAISS index + chunk metadata, embeds an incoming question,
retrieves the top-K most relevant chunks, and asks Groq Llama 3.1 8B to
answer grounded in those chunks with citations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import faiss
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
load_dotenv()

INDEX_PATH = Path("data/rag/index/index.faiss")
CHUNKS_PATH = Path("data/rag/index/chunks.json")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_K = 4
GENERATION_TEMPERATURE = 0.1

_embedder: Optional[SentenceTransformer] = None
_index: Optional[faiss.Index] = None
_chunks: Optional[list] = None
_groq: Optional[Groq] = None


def _load_components():
    global _embedder, _index, _chunks, _groq
    if _embedder is None:
        logger.info(f"Loading embedder: {EMBED_MODEL_ID}")
        _embedder = SentenceTransformer(EMBED_MODEL_ID)
    if _index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. Run: python -m src.rag.ingest"
            )
        logger.info(f"Loading FAISS index: {INDEX_PATH}")
        _index = faiss.read_index(str(INDEX_PATH))
    if _chunks is None:
        with open(CHUNKS_PATH) as f:
            _chunks = json.load(f)
        logger.info(f"Loaded {len(_chunks)} chunks from {CHUNKS_PATH}")
    if _groq is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment (.env)")
        _groq = Groq(api_key=api_key)
    return _embedder, _index, _chunks, _groq


def retrieve(question: str, k: int = DEFAULT_K) -> list[dict]:
    embedder, index, chunks, _ = _load_components()
    qvec = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype(
        "float32"
    )
    scores, idxs = index.search(qvec, k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        if idx == -1:
            continue
        c = chunks[idx]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "source": c["source"],
                "chunk_id": c["id"],
                "text": c["text"],
            }
        )
    return results


SYSTEM_PROMPT = (
    "You are a compliance assistant for a bank. "
    "Answer the user's question STRICTLY based on the numbered context passages provided. "
    "Cite sources inline as [1], [2], etc. matching the numbered passages. "
    "If the context does not contain enough information to answer, reply exactly: "
    '"I don\'t have enough information in the provided policy documents to answer this." '
    "Do not use any outside knowledge. Be concise and professional."
)


def _build_context(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[{c['rank']}] (source: {c['source']})\n{c['text']}\n")
    return "\n".join(lines)


def answer_question(question: str, k: int = DEFAULT_K) -> dict:
    *_, groq = _load_components()
    retrieved = retrieve(question, k=k)
    if not retrieved:
        return {"answer": "No relevant policy passages found.", "sources": [], "model": GROQ_MODEL}

    context = _build_context(retrieved)
    user_msg = f"Context passages:\n\n{context}\nQuestion: {question}\n\nAnswer:"

    completion = groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=GENERATION_TEMPERATURE,
        max_tokens=512,
    )
    answer = completion.choices[0].message.content.strip()

    source_meta = [
        {"rank": c["rank"], "source": c["source"], "chunk_id": c["chunk_id"], "score": c["score"]}
        for c in retrieved
    ]
    return {"answer": answer, "sources": source_meta, "model": GROQ_MODEL}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is the three lines of defence in AML risk management?"
    print(f"\nQuestion: {q}\n")
    result = answer_question(q)
    print(f"Answer:\n{result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  [{s['rank']}] {s['source']} (score={s['score']:.3f})")
