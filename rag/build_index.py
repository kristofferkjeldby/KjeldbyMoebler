"""Build a FAISS index over the product catalog for RAG retrieval.

Usage:
    python -m rag.build_index

Reads catalog/data/catalog.json, embeds each product with a local
sentence-transformers model, and writes:
  - rag/data/catalog.index   (FAISS index, inner-product over normalized vectors)
  - rag/data/catalog_meta.json (parallel list of product dicts, same order as the index)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from catalog.product_format import product_to_embedding_text  # noqa: E402
from config import CATALOG_PATH, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, RAG_METADATA_PATH  # noqa: E402


def main() -> None:
    if not CATALOG_PATH.exists():
        raise SystemExit(f"No catalog found at {CATALOG_PATH}. Run catalog/generate_catalog.py first.")

    products = json.loads(CATALOG_PATH.read_text())
    print(f"Loaded {len(products)} products.")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # multilingual-e5 expects a "passage: " prefix on embedded documents
    # (asymmetric with the "query: " prefix used at retrieval time).
    texts = ["passage: " + product_to_embedding_text(p) for p in products]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    RAG_METADATA_PATH.write_text(json.dumps(products, indent=2))

    print(f"Wrote FAISS index ({index.ntotal} vectors, dim={dim}) to {FAISS_INDEX_PATH}")
    print(f"Wrote metadata to {RAG_METADATA_PATH}")


if __name__ == "__main__":
    main()
