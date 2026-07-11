"""Decision Tree classifier for search query intent routing."""
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from config import ARTIFACTS_DIR, DATA_DIR, INTENT_CLASSES
from src.nlp.text_processing import clean_text


class IntentClassifier:
    """Classify queries as informational, transactional, or exploratory."""

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[DecisionTreeClassifier] = None
        self.classes = INTENT_CLASSES

    def _augment_training_data(self) -> tuple:
        """Build training set from labeled queries + keyword rules."""
        with open(DATA_DIR / "search_queries.json") as f:
            labeled = json.load(f)
        with open(DATA_DIR / "intent_keywords.json") as f:
            keywords = json.load(f)

        texts, labels = [], []
        for entry in labeled:
            texts.append(entry["query"])
            labels.append(entry["intent"])

        # Augment with keyword-based synthetic examples
        templates = {
            "informational": [
                "what is {kw} in crypto",
                "how does {kw} work",
                "explain {kw} blockchain news",
                "latest {kw} regulation update",
            ],
            "transactional": [
                "buy {kw} tokens now",
                "purchase best {kw} coins",
                "invest in low fee {kw}",
                "trade {kw} assets today",
            ],
            "exploratory": [
                "trending {kw} coins",
                "discover new {kw} projects",
                "explore {kw} metaverse tokens",
                "top {kw} gaming cryptocurrencies",
            ],
        }
        kw_list = ["defi", "nft", "layer1", "metaverse", "bitcoin", "solana"]
        for intent, patterns in templates.items():
            for pattern in patterns:
                for kw in kw_list:
                    texts.append(pattern.format(kw=kw))
                    labels.append(intent)

        return texts, labels

    def fit(self):
        texts, labels = self._augment_training_data()
        cleaned = [clean_text(t) for t in texts]
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(cleaned)

        self.classifier = DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
        )
        self.classifier.fit(X, labels)
        return self

    def predict(self, query: str) -> dict:
        if self.classifier is None:
            return {"intent": "exploratory", "confidence": 0.0, "routing": "hybrid_search"}

        cleaned = clean_text(query)
        X = self.vectorizer.transform([cleaned])
        intent = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]
        confidence = float(max(proba))

        routing = {
            "informational": "news_and_education",
            "transactional": "token_trade_pipeline",
            "exploratory": "hybrid_search",
        }[intent]

        return {
            "intent": intent,
            "confidence": round(float(confidence), 4),
            "routing": routing,
            "probabilities": {
                cls: round(float(p), 4)
                for cls, p in zip(self.classifier.classes_, proba)
            },
        }

    def save(self, path=None):
        path = path or ARTIFACTS_DIR / "intent_classifier.joblib"
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=None):
        path = path or ARTIFACTS_DIR / "intent_classifier.joblib"
        return joblib.load(path)
