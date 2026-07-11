"""Hybrid semantic + sparse search orchestrator."""
import json
from typing import Dict, List, Optional

from config import (
    DATA_DIR,
    DEFAULT_TOP_K,
    HYBRID_DENSE_WEIGHT,
    HYBRID_SPARSE_WEIGHT,
)
from src.nlp.intent_classifier import IntentClassifier
from src.nlp.ner import CryptoNER
from src.nlp.text_processing import TfidfSparseRetriever
from src.nlp.topic_modeling import NewsTopicModeler
from src.search.embeddings import EmbeddingEngine
from src.search.ltr_ranker import LambdaMARTRanker
from src.search.vector_store import FaissVectorStore


class HybridSearchEngine:
    """
    End-to-end search pipeline:
    1. Intent classification (Decision Tree)
    2. NER entity extraction (spaCy + crypto dictionary)
    3. Dense vector search (FAISS + BERT embeddings)
    4. Sparse TF-IDF search
    5. Hybrid fusion
    6. LambdaMART re-ranking
    """

    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = FaissVectorStore()
        self.tfidf_retriever = TfidfSparseRetriever()
        self.intent_classifier = IntentClassifier()
        self.ner: Optional[CryptoNER] = None
        self.ltr_ranker = LambdaMARTRanker()
        self.topic_modeler = NewsTopicModeler()
        self.tokens = []
        self.news = []
        self.bundles = []
        self._corpus_docs = []
        self._doc_lookup = {}

    def load_data(self):
        with open(DATA_DIR / "tokens.json") as f:
            self.tokens = json.load(f)
        with open(DATA_DIR / "news.json") as f:
            self.news = json.load(f)
        with open(DATA_DIR / "bundles.json") as f:
            self.bundles = json.load(f)

        self.ner = CryptoNER(self.tokens)

        # Build unified corpus for TF-IDF
        self._corpus_docs = []
        self._doc_lookup = {}

        for token in self.tokens:
            doc = {
                "id": f"token:{token['id']}",
                "text": f"{token['name']} {token['symbol']} {token['description']} {' '.join(token['tags'])}",
                "type": "token",
                "data": token,
            }
            self._corpus_docs.append(doc)
            self._doc_lookup[doc["id"]] = doc

        for article in self.news:
            doc = {
                "id": f"news:{article['id']}",
                "text": f"{article['title']} {article['content']}",
                "type": "news",
                "data": article,
            }
            self._corpus_docs.append(doc)
            self._doc_lookup[doc["id"]] = doc

        for bundle in self.bundles:
            doc = {
                "id": f"bundle:{bundle['id']}",
                "text": f"{bundle['name']} {bundle['description']}",
                "type": "bundle",
                "data": bundle,
            }
            self._corpus_docs.append(doc)
            self._doc_lookup[doc["id"]] = doc

    def build_indexes(self, fine_tune: bool = True):
        """Build all search indexes and train classifiers."""
        self.load_data()

        # Build corpus texts for vocabulary
        corpus_texts = [doc["text"] for doc in self._corpus_docs]

        # Generate embeddings
        all_embeddings = {}
        all_embeddings.update(self.embedding_engine.encode_items(self.tokens, "token", corpus_texts))
        all_embeddings.update(self.embedding_engine.encode_items(self.news, "news", corpus_texts))
        all_embeddings.update(self.embedding_engine.encode_items(self.bundles, "bundle", corpus_texts))

        # Optional BERT fine-tuning on query-doc pairs
        if fine_tune:
            with open(DATA_DIR / "search_queries.json") as f:
                queries = json.load(f)
            token_map = {t["id"]: t for t in self.tokens}
            pairs = []
            for q in queries:
                for tid in q.get("relevant_tokens", []):
                    if tid in token_map:
                        t = token_map[tid]
                        doc_text = f"{t['name']} {t['description']}"
                        pairs.append((q["query"], doc_text))
            if pairs:
                self.embedding_engine.fine_tune(pairs, epochs=3)
                # Re-encode after fine-tuning
                all_embeddings = {}
                all_embeddings.update(self.embedding_engine.encode_items(self.tokens, "token", corpus_texts))
                all_embeddings.update(self.embedding_engine.encode_items(self.news, "news", corpus_texts))
                all_embeddings.update(self.embedding_engine.encode_items(self.bundles, "bundle", corpus_texts))

        self.embedding_engine.save()
        self.vector_store.build(all_embeddings)
        self.vector_store.save()

        # TF-IDF sparse index
        self.tfidf_retriever.fit(self._corpus_docs)
        self.tfidf_retriever.save()

        # Intent classifier
        self.intent_classifier.fit()
        self.intent_classifier.save()

        # LDA topic model
        self.topic_modeler.fit(self.news)
        self.topic_modeler.save()

        # LTR ranker
        self.ltr_ranker.fit()
        self.ltr_ranker.save()

    def load_indexes(self):
        """Load pre-built indexes from disk."""
        self.load_data()
        self.embedding_engine.load_embeddings()
        self.vector_store = FaissVectorStore.load()
        self.tfidf_retriever = TfidfSparseRetriever.load()
        self.intent_classifier = IntentClassifier.load()
        self.topic_modeler = NewsTopicModeler.load()
        self.ltr_ranker = LambdaMARTRanker.load()

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, item_types: Optional[List[str]] = None) -> dict:
        """Execute full hybrid search pipeline."""
        # Step 1: Intent classification
        intent_result = self.intent_classifier.predict(query)

        # Step 2: NER entity extraction
        entities = self.ner.extract_entities(query)

        # Step 3: Dense vector search
        query_emb = self.embedding_engine.encode_query(query)
        dense_results = self.vector_store.search(query_emb, top_k=top_k * 3)

        # Step 4: Sparse TF-IDF search
        sparse_results = self.tfidf_retriever.search(query, top_k=top_k * 3)

        # Step 5: Hybrid fusion
        fused_scores: Dict[str, dict] = {}

        for doc_id, score in dense_results:
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"dense_score": 0, "sparse_score": 0}
            fused_scores[doc_id]["dense_score"] = score

        max_sparse = max((s for _, s in sparse_results), default=1.0) or 1.0
        for doc_id, score in sparse_results:
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"dense_score": 0, "sparse_score": 0}
            fused_scores[doc_id]["sparse_score"] = score / max_sparse

        candidates = []
        for doc_id, scores in fused_scores.items():
            doc = self._doc_lookup.get(doc_id)
            if not doc:
                continue
            if item_types and doc["type"] not in item_types:
                continue

            hybrid = (
                HYBRID_DENSE_WEIGHT * scores["dense_score"]
                + HYBRID_SPARSE_WEIGHT * scores["sparse_score"]
            )

            item_data = doc["data"]
            candidate = {
                "id": item_data["id"],
                "type": doc["type"],
                "dense_score": round(scores["dense_score"], 4),
                "sparse_score": round(scores["sparse_score"], 4),
                "hybrid_score": round(hybrid, 4),
                "data": item_data,
            }

            if doc["type"] == "token":
                candidate["market_cap_rank"] = item_data.get("market_cap_rank", 100)
                fee_map = {"low": 3, "medium": 2, "high": 1, "variable": 2, "n/a": 1}
                candidate["fee_score"] = fee_map.get(item_data.get("fee_tier", "n/a"), 1)
                candidate["category_match"] = (
                    1 if item_data.get("category", "").lower() in query.lower() else 0
                )
            else:
                candidate["market_cap_rank"] = 50
                candidate["fee_score"] = 2
                candidate["category_match"] = 0

            candidates.append(candidate)

        # Route by intent
        if intent_result["routing"] == "news_and_education":
            candidates = [c for c in candidates if c["type"] in ("news", "token")]
        elif intent_result["routing"] == "token_trade_pipeline":
            candidates = [c for c in candidates if c["type"] in ("token", "bundle")]

        # Step 6: LambdaMART re-ranking
        ranked = self.ltr_ranker.rank(candidates, intent_result["intent"], entities)

        # Topic for news results
        query_topic = self.topic_modeler.predict_topic(query)

        results = []
        for item in ranked[:top_k]:
            result = {
                "id": item["id"],
                "type": item["type"],
                "scores": {
                    "dense": float(item["dense_score"]),
                    "sparse": float(item["sparse_score"]),
                    "hybrid": float(item["hybrid_score"]),
                    "ltr": float(item.get("ltr_score", item["hybrid_score"])),
                },
                "data": item["data"],
            }
            if item["type"] == "news":
                topic = self.topic_modeler.predict_topic(
                    f"{item['data']['title']} {item['data']['content']}"
                )
                result["topic"] = {
                    "topic_id": int(topic["topic_id"]),
                    "topic_label": topic["topic_label"],
                    "confidence": float(topic["confidence"]),
                }
            results.append(result)

        return {
            "query": query,
            "intent": intent_result,
            "entities": entities,
            "query_topic": query_topic,
            "results": results,
            "total_candidates": len(candidates),
        }
