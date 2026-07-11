"""Two-tower recommendation architecture (PyTorch) for personalized crypto discovery."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    HIDDEN_DIM,
    ITEM_EMBED_DIM,
    TWO_TOWER_BATCH_SIZE,
    TWO_TOWER_EPOCHS,
    TWO_TOWER_LR,
    USER_EMBED_DIM,
)


class UserTower(nn.Module):
    """Encode user features: portfolio, risk appetite, trading frequency."""

    def __init__(self, num_users: int, num_categories: int, embed_dim: int = USER_EMBED_DIM):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.risk_embed = nn.Embedding(3, 16)
        self.freq_embed = nn.Embedding(20, 16)
        self.category_embed = nn.Embedding(num_categories, 32)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + 16 + 16 + 32, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, embed_dim),
        )

    def forward(self, user_idx, risk_idx, freq_idx, category_idx):
        u = self.user_embed(user_idx)
        r = self.risk_embed(risk_idx)
        f = self.freq_embed(freq_idx)
        c = self.category_embed(category_idx)
        x = torch.cat([u, r, f, c], dim=-1)
        return F.normalize(self.fc(x), dim=-1)


class ItemTower(nn.Module):
    """Encode item features: type, category/theme, market cap."""

    def __init__(self, num_items: int, num_types: int, num_themes: int, embed_dim: int = ITEM_EMBED_DIM):
        super().__init__()
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.type_embed = nn.Embedding(num_types, 16)
        self.theme_embed = nn.Embedding(num_themes, 32)
        self.market_cap_fc = nn.Linear(1, 16)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + 16 + 32 + 16, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_DIM, embed_dim),
        )

    def forward(self, item_idx, type_idx, theme_idx, market_cap):
        i = self.item_embed(item_idx)
        t = self.type_embed(type_idx)
        th = self.theme_embed(theme_idx)
        mc = self.market_cap_fc(market_cap.unsqueeze(-1))
        x = torch.cat([i, t, th, mc], dim=-1)
        return F.normalize(self.fc(x), dim=-1)


class TwoTowerModel(nn.Module):
    """Two-tower architecture for token, news, and bundle recommendations."""

    def __init__(self, num_users, num_items, num_categories, num_types, num_themes):
        super().__init__()
        self.user_tower = UserTower(num_users, num_categories)
        self.item_tower = ItemTower(num_items, num_types, num_themes)

    def forward(self, user_features, item_features):
        user_vec = self.user_tower(*user_features)
        item_vec = self.item_tower(*item_features)
        return (user_vec * item_vec).sum(dim=-1)

    def recommend(self, user_features, all_item_features, top_k=10):
        self.eval()
        with torch.no_grad():
            user_vec = self.user_tower(*user_features)
            item_vecs = self.item_tower(*all_item_features)
            scores = torch.matmul(user_vec, item_vecs.T)
            top_scores, top_indices = torch.topk(scores, min(top_k, scores.shape[-1]))
        return top_scores[0].tolist(), top_indices[0].tolist()


class InteractionDataset(Dataset):
    def __init__(self, interactions, encoders):
        self.data = interactions
        self.enc = encoders

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        user_id = self.enc["user"][row["user_id"]]
        item_id = self.enc["item"][row["item_id"]]
        risk = row["risk_appetite"]
        freq = min(row["trading_frequency"], 19)
        item_type = self.enc["type"][row["item_type"]]
        theme = self.enc["theme"].get(row.get("theme", "unknown"), 0)
        market_cap = row.get("market_cap_rank", 50) / 100.0
        label = float(row.get("traded", row.get("clicked", 0)))
        category = self.enc["user_category"].get(row["user_id"], 0)
        return user_id, item_id, risk, freq, item_type, theme, market_cap, category, label


class TwoTowerRecommender:
    """Production-style two-tower recommender for Company X users."""

    def __init__(self):
        self.model: Optional[TwoTowerModel] = None
        self.encoders: Dict = {}
        self.item_catalog: List[dict] = []
        self.users: List[dict] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_catalog(self):
        with open(DATA_DIR / "tokens.json") as f:
            tokens = json.load(f)
        with open(DATA_DIR / "news.json") as f:
            news = json.load(f)
        with open(DATA_DIR / "bundles.json") as f:
            bundles = json.load(f)
        with open(DATA_DIR / "users.json") as f:
            self.users = json.load(f)

        self.item_catalog = []
        for t in tokens:
            self.item_catalog.append({
                "id": t["id"], "type": "token", "theme": t["category"],
                "market_cap_rank": t["market_cap_rank"], "data": t,
            })
        for n in news:
            self.item_catalog.append({
                "id": n["id"], "type": "news", "theme": n.get("topic_hint", "Trading"),
                "market_cap_rank": 50, "data": n,
            })
        for b in bundles:
            self.item_catalog.append({
                "id": b["id"], "type": "bundle", "theme": b["theme"],
                "market_cap_rank": 30, "data": b,
            })

    def _build_encoders(self, interactions: List[dict]):
        user_ids = sorted({u["id"] for u in self.users})
        item_ids = sorted({item["id"] for item in self.item_catalog})
        types = sorted({item["type"] for item in self.item_catalog})
        themes = sorted({item["theme"] for item in self.item_catalog})
        categories = sorted({c for u in self.users for c in u.get("preferred_categories", [])})

        category_map = {c: i for i, c in enumerate(categories)} if categories else {"default": 0}
        default_cat = list(category_map.keys())[0]

        self.encoders = {
            "user": {uid: i for i, uid in enumerate(user_ids)},
            "item": {iid: i for i, iid in enumerate(item_ids)},
            "type": {t: i for i, t in enumerate(types)},
            "theme": {t: i for i, t in enumerate(themes)},
            "category": category_map,
            "user_category": {
                u["id"]: category_map.get(
                    u["preferred_categories"][0] if u.get("preferred_categories") else default_cat,
                    0,
                )
                for u in self.users
            },
        }

        # Enrich interactions with theme info
        item_map = {item["id"]: item for item in self.item_catalog}
        for row in interactions:
            item = item_map.get(row["item_id"], {})
            row["theme"] = item.get("theme", "Trading")
            row["market_cap_rank"] = item.get("market_cap_rank", 50)

    def train(self):
        self._build_catalog()
        with open(DATA_DIR / "interactions.json") as f:
            interactions = json.load(f)
        self._build_encoders(interactions)

        num_users = len(self.encoders["user"])
        num_items = len(self.encoders["item"])
        num_categories = max(len(self.encoders["category"]), 1)
        num_types = len(self.encoders["type"])
        num_themes = len(self.encoders["theme"])

        self.model = TwoTowerModel(num_users, num_items, num_categories, num_types, num_themes)
        self.model.to(self.device)

        dataset = InteractionDataset(interactions, self.encoders)
        loader = DataLoader(dataset, batch_size=TWO_TOWER_BATCH_SIZE, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=TWO_TOWER_LR)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for epoch in range(TWO_TOWER_EPOCHS):
            total_loss = 0
            for batch in loader:
                u, i, r, f, it, th, mc, cat, label = [b.to(self.device) if torch.is_tensor(b) else torch.tensor(b).to(self.device) for b in batch]
                optimizer.zero_grad()
                logits = self.model((u, r, f, cat), (i, it, th, mc.float()))
                loss = criterion(logits, label.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self._save_artifacts()
        return {"epochs": TWO_TOWER_EPOCHS, "final_loss": round(total_loss / len(loader), 4)}

    def recommend(self, user_id: str, top_k: int = 10) -> List[dict]:
        if self.model is None:
            self.load()

        user = next((u for u in self.users if u["id"] == user_id), None)
        if not user:
            return []

        self.model.eval()
        u_idx = torch.tensor([self.encoders["user"][user_id]]).to(self.device)
        risk_map = {"low": 0, "medium": 1, "high": 2}
        freq_map = {"low": 3, "medium": 8, "high": 15}
        r_idx = torch.tensor([risk_map[user["risk_appetite"]]]).to(self.device)
        f_idx = torch.tensor([min(freq_map[user["trading_frequency"]], 19)]).to(self.device)
        c_idx = torch.tensor([self.encoders["user_category"][user_id]]).to(self.device)

        item_indices = []
        type_indices = []
        theme_indices = []
        market_caps = []
        for item in self.item_catalog:
            item_indices.append(self.encoders["item"][item["id"]])
            type_indices.append(self.encoders["type"][item["type"]])
            theme_indices.append(self.encoders["theme"][item["theme"]])
            market_caps.append(item["market_cap_rank"] / 100.0)

        i_idx = torch.tensor(item_indices).to(self.device)
        it_idx = torch.tensor(type_indices).to(self.device)
        th_idx = torch.tensor(theme_indices).to(self.device)
        mc = torch.tensor(market_caps, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            user_vec = self.model.user_tower(u_idx, r_idx, f_idx, c_idx)
            item_vecs = self.model.item_tower(i_idx, it_idx, th_idx, mc)
            scores = torch.matmul(user_vec, item_vecs.T)[0]
            top_scores, top_indices = torch.topk(scores, min(top_k, len(self.item_catalog)))

        recommendations = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            item = self.item_catalog[idx]
            recommendations.append({
                "id": item["id"],
                "type": item["type"],
                "score": round(score, 4),
                "data": item["data"],
            })
        return recommendations

    def _save_artifacts(self):
        path = ARTIFACTS_DIR / "two_tower"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "model.pt")
        with open(path / "encoders.json", "w") as f:
            json.dump(self.encoders, f)
        with open(path / "item_catalog.json", "w") as f:
            json.dump(self.item_catalog, f)
        with open(path / "users.json", "w") as f:
            json.dump(self.users, f)

    def load(self):
        self._build_catalog()
        path = ARTIFACTS_DIR / "two_tower"
        with open(path / "encoders.json") as f:
            self.encoders = json.load(f)
        with open(path / "item_catalog.json") as f:
            self.item_catalog = json.load(f)
        with open(path / "users.json") as f:
            self.users = json.load(f)

        num_users = len(self.encoders["user"])
        num_items = len(self.encoders["item"])
        num_categories = max(len(self.encoders.get("category", {})), 1)
        num_types = len(self.encoders["type"])
        num_themes = len(self.encoders["theme"])

        self.model = TwoTowerModel(num_users, num_items, num_categories, num_types, num_themes)
        self.model.load_state_dict(torch.load(path / "model.pt", map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
