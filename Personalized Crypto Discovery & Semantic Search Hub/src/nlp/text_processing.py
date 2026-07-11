"""Text cleaning and TF-IDF sparse retrieval for hybrid search."""
import re
import string
from typing import List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ARTIFACTS_DIR


def clean_text(text: str) -> str:
    """Normalize unstructured crypto text for sparse retrieval."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TfidfSparseRetriever:
    """TF-IDF based sparse keyword matching component."""

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_ids: List[str] = []
        self.matrix = None

    def fit(self, documents: List[dict], text_field: str = "text", id_field: str = "id"):
        texts = [clean_text(doc[text_field]) for doc in documents]
        self.doc_ids = [doc[id_field] for doc in documents]
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 20) -> List[tuple]:
        if self.vectorizer is None or self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([clean_text(query)])
        scores = (self.matrix @ query_vec.T).toarray().flatten()
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def save(self, path=None):
        path = path or ARTIFACTS_DIR / "tfidf_retriever.joblib"
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=None):
        path = path or ARTIFACTS_DIR / "tfidf_retriever.joblib"
        return joblib.load(path)
