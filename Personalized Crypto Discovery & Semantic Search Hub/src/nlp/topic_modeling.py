"""LDA topic modeling for crypto news categorization."""
import json
from typing import List, Optional

import joblib
import numpy as np
from gensim import corpora
from gensim.models import LdaModel

from config import ARTIFACTS_DIR, LDA_NUM_TOPICS, LDA_TOPIC_LABELS
from src.nlp.text_processing import clean_text


class NewsTopicModeler:
    """LDA-based topic modeling for DeFi, NFTs, Regulation, etc."""

    def __init__(self, num_topics: int = LDA_NUM_TOPICS):
        self.num_topics = num_topics
        self.lda_model: Optional[LdaModel] = None
        self.dictionary = None
        self.topic_labels = LDA_TOPIC_LABELS

    def fit(self, news_articles: List[dict]):
        texts = [
            clean_text(f"{a['title']} {a['content']}").split()
            for a in news_articles
        ]
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=1, no_above=0.9)
        corpus = [self.dictionary.doc2bow(t) for t in texts]

        self.lda_model = LdaModel(
            corpus=corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=42,
            passes=15,
            alpha="auto",
        )
        self._news_ids = [a["id"] for a in news_articles]
        self._corpus = corpus
        self._cached_topics = self.get_article_topics()
        return self

    def predict_topic(self, text: str) -> dict:
        if self.lda_model is None:
            return {"topic_id": -1, "topic_label": "Unknown", "confidence": 0.0}

        bow = self.dictionary.doc2bow(clean_text(text).split())
        topics = self.lda_model.get_document_topics(bow)
        if not topics:
            return {"topic_id": -1, "topic_label": "Unknown", "confidence": 0.0}

        best = max(topics, key=lambda x: x[1])
        topic_id = best[0]
        label = self.topic_labels[topic_id] if topic_id < len(self.topic_labels) else f"Topic_{topic_id}"
        return {"topic_id": int(topic_id), "topic_label": label, "confidence": round(float(best[1]), 4)}

    def get_article_topics(self) -> List[dict]:
        if hasattr(self, "_cached_topics") and self._cached_topics:
            return self._cached_topics
        if not hasattr(self, "_corpus"):
            # Rebuild from news data when loaded from disk without corpus
            from config import DATA_DIR
            with open(DATA_DIR / "news.json") as f:
                news = json.load(f)
            results = []
            for article in news:
                text = clean_text(f"{article['title']} {article['content']}")
                topic = self.predict_topic(text)
                results.append({
                    "news_id": article["id"],
                    "topic_id": topic["topic_id"],
                    "topic_label": topic["topic_label"],
                    "confidence": topic["confidence"],
                })
            self._cached_topics = results
            return results
        results = []
        for news_id, bow in zip(self._news_ids, self._corpus):
            topics = self.lda_model.get_document_topics(bow)
            if topics:
                best = max(topics, key=lambda x: x[1])
                topic_id = best[0]
                label = self.topic_labels[topic_id] if topic_id < len(self.topic_labels) else f"Topic_{topic_id}"
                results.append({
                    "news_id": news_id,
                    "topic_id": int(topic_id),
                    "topic_label": label,
                    "confidence": round(float(best[1]), 4),
                })
        return results

    def save(self, path=None):
        path = path or ARTIFACTS_DIR / "lda_model"
        path.mkdir(parents=True, exist_ok=True)
        self.lda_model.save(str(path / "lda.gensim"))
        self.dictionary.save(str(path / "dictionary.gensim"))
        with open(path / "news_ids.json", "w") as f:
            json.dump(self._news_ids, f)
        with open(path / "article_topics.json", "w") as f:
            json.dump(self._cached_topics, f)

    @classmethod
    def load(cls, path=None):
        path = path or ARTIFACTS_DIR / "lda_model"
        model = cls()
        model.lda_model = LdaModel.load(str(path / "lda.gensim"))
        model.dictionary = corpora.Dictionary.load(str(path / "dictionary.gensim"))
        with open(path / "news_ids.json") as f:
            model._news_ids = json.load(f)
        topics_path = path / "article_topics.json"
        if topics_path.exists():
            with open(topics_path) as f:
                model._cached_topics = json.load(f)
        return model
