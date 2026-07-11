"""FAISS vector index for semantic retrieval."""
import json
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

from config import ARTIFACTS_DIR, EMBEDDING_DIM


class FaissVectorStore:
    """FAISS-based vector search for tokens, news, and bundles."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_map: List[str] = []
        self.normalized = True

    def build(self, embeddings: Dict[str, np.ndarray]):
        if not embeddings:
            raise ValueError("No embeddings provided")

        keys = list(embeddings.keys())
        vectors = np.vstack([embeddings[k] for k in keys]).astype(np.float32)

        # L2 normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.id_map = keys

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> List[Tuple[str, float]]:
        if self.index is None:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, len(self.id_map)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.id_map[idx], float(score)))
        return results

    def save(self, path=None):
        path = path or ARTIFACTS_DIR / "faiss_index"
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "id_map.json", "w") as f:
            json.dump(self.id_map, f)

    @classmethod
    def load(cls, path=None):
        path = path or ARTIFACTS_DIR / "faiss_index"
        store = cls()
        store.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "id_map.json") as f:
            store.id_map = json.load(f)
        return store
