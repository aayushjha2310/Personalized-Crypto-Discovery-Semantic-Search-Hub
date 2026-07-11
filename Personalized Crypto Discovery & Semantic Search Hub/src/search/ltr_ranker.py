"""LambdaMART-style Learning to Rank using LightGBM."""
import json
from typing import Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, DATA_DIR


class LambdaMARTRanker:
    """
    Re-rank retrieved search results based on likelihood of user trade.
    Uses LightGBM with LambdaRank objective (LambdaMART).
    """

    FEATURE_COLS = [
        "dense_score", "sparse_score", "hybrid_score",
        "market_cap_rank", "fee_score", "category_match",
        "entity_match", "intent_transactional", "intent_informational",
    ]

    def __init__(self):
        self.model: Optional[lgb.Booster] = None

    def _build_training_frame(self) -> pd.DataFrame:
        with open(DATA_DIR / "ltr_training.json") as f:
            ltr_data = json.load(f)
        with open(DATA_DIR / "tokens.json") as f:
            tokens = {t["id"]: t for t in json.load(f)}

        rows = []
        for record in ltr_data:
            token = tokens[record["item_id"]]
            rows.append({
                "query_id": record["query_id"],
                "item_id": record["item_id"],
                "label": record["label"],
                "dense_score": np.random.uniform(0.3, 0.9) if record["label"] > 0 else np.random.uniform(0.0, 0.4),
                "sparse_score": np.random.uniform(0.2, 0.8) if record["label"] > 0 else np.random.uniform(0.0, 0.3),
                "hybrid_score": 0.0,
                "market_cap_rank": record["market_cap_rank"],
                "fee_score": record["fee_score"],
                "category_match": record["category_match"],
                "entity_match": 1 if record["label"] >= 2 else 0,
                "intent_transactional": 1 if "buy" in record["query"].lower() or "fee" in record["query"].lower() else 0,
                "intent_informational": 1 if "what" in record["query"].lower() or "how" in record["query"].lower() else 0,
            })
            rows[-1]["hybrid_score"] = 0.65 * rows[-1]["dense_score"] + 0.35 * rows[-1]["sparse_score"]

        return pd.DataFrame(rows)

    def fit(self):
        df = self._build_training_frame()
        df = df.sort_values("query_id")

        X = df[self.FEATURE_COLS].values
        y = df["label"].values
        groups = df.groupby("query_id").size().values

        train_set = lgb.Dataset(X, label=y, group=groups)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 5,
            "verbose": -1,
            "seed": 42,
        }
        self.model = lgb.train(params, train_set, num_boost_round=50)
        return self

    def rank(
        self,
        candidates: List[dict],
        intent: str,
        entities: dict,
    ) -> List[dict]:
        """Re-rank candidate items using LTR model."""
        if self.model is None or not candidates:
            return sorted(candidates, key=lambda x: x.get("hybrid_score", 0), reverse=True)

        intent_map = {"transactional": 0, "informational": 1, "exploratory": 2}
        entity_tokens = set(entities.get("tokens", []))

        features = []
        for c in candidates:
            item_id = c.get("id", c.get("item_id", ""))
            features.append([
                c.get("dense_score", 0.0),
                c.get("sparse_score", 0.0),
                c.get("hybrid_score", 0.0),
                c.get("market_cap_rank", 100),
                c.get("fee_score", 1),
                c.get("category_match", 0),
                1 if item_id in entity_tokens else 0,
                1 if intent == "transactional" else 0,
                1 if intent == "informational" else 0,
            ])

        scores = self.model.predict(np.array(features))
        for c, score in zip(candidates, scores):
            c["ltr_score"] = float(score)

        return sorted(candidates, key=lambda x: x["ltr_score"], reverse=True)

    def save(self, path=None):
        path = path or ARTIFACTS_DIR / "ltr_model.lgb"
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path=None):
        path = path or ARTIFACTS_DIR / "ltr_model.lgb"
        ranker = cls()
        ranker.model = lgb.Booster(model_file=str(path))
        return ranker
