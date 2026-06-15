"""
src/rag/ingest.py

Build phase of RAG.
Extracts text from PDFs in data/rag/source_pdfs/, chunks them,
embeds the chunks with sentence-transformers, builds a FAISS index,
and saves both the index and the chunk metadata to data/rag/index/.

Run once after adding new PDFs:
    python -m src.rag.ingest
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PDF_DIR        = Path("data/rag/source_pdfs")
INDEX_DIR      = Path("data/rag/index")
INDEX_PATH     = INDEX_DIR / "index.faiss"
CHUNKS_PATH    = INDEX_DIR / "chunks.json"

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE     = 800     # characters
CHUNK_OVERLAP  = 100     # characters
BATCH_SIZE     = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_text(pdf_path: Path) -> str:
    """Pull all text from a PDF, with light cleanup."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text)
    full_text = "\n".join(pages)
    # collapse repeated whitespace
    return " ".join(full_text.split())


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-windowed chunks."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")
    print(f"Found {len(pdf_files)} PDF(s) in {PDF_DIR}")

    # 1. Extract + chunk
    all_chunks: list[dict] = []
    for pdf_path in pdf_files:
        print(f"  Extracting: {pdf_path.name}")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{pdf_path.stem}__chunk{i:04d}",
                "source": pdf_path.name,
                "text": chunk,
            })
        print(f"    -> {len(chunks)} chunks from {len(text):,} chars")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # 2. Embed
    print(f"\nLoading embedding model: {EMBED_MODEL_ID}")
    model = SentenceTransformer(EMBED_MODEL_ID)
    texts = [c["text"] for c in all_chunks]

    print("Embedding chunks...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # so inner product == cosine similarity
    ).astype("float32")
    print(f"Embeddings shape: {vectors.shape}")

    # 3. Build FAISS index (inner product on normalized vectors == cosine sim)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"FAISS index size: {index.ntotal} vectors of dim {dim}")

    # 4. Save
    faiss.write_index(index, str(INDEX_PATH))
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved index to {INDEX_PATH}")
    print(f"✅ Saved chunks  to {CHUNKS_PATH}")


if __name__ == "__main__":
    build_index()