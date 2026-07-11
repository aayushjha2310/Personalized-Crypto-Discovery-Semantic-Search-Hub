"""BERT-style embedding generation using PyTorch (fully offline-capable)."""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from config import ARTIFACTS_DIR, EMBEDDING_DIM

# Optional online fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    HAS_SENTENCE_TRANSFORMERS = False


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Vocabulary:
    def __init__(self, max_vocab: int = 8000):
        self.token2id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.max_vocab = max_vocab

    def build(self, texts: List[str]):
        freq: Dict[str, int] = {}
        for text in texts:
            for tok in _tokenize(text):
                freq[tok] = freq.get(tok, 0) + 1
        for tok, _ in sorted(freq.items(), key=lambda x: -x[1])[: self.max_vocab - 2]:
            self.token2id[tok] = len(self.token2id)

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        ids = [self.token2id.get(t, 1) for t in _tokenize(text)][:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.token2id, f)

    @classmethod
    def load(cls, path: Path):
        vocab = cls()
        with open(path) as f:
            vocab.token2id = json.load(f)
        return vocab


class CryptoBERTEncoder(nn.Module):
    """
    Lightweight BERT-style transformer encoder for crypto semantic search.
    Built with PyTorch — works fully offline without HuggingFace downloads.
    """

    def __init__(self, vocab_size: int, embed_dim: int = EMBEDDING_DIM, max_len: int = 64):
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        mask = (input_ids == 0)
        x = self.transformer(x, src_key_padding_mask=mask)
        # Mean pooling over non-pad tokens
        mask_exp = (~mask).unsqueeze(-1).float()
        pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return F.normalize(self.projection(pooled), dim=-1)


class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], vocab: Vocabulary):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, d = self.pairs[idx]
        return (
            torch.tensor(self.vocab.encode(q)),
            torch.tensor(self.vocab.encode(d)),
        )


class EmbeddingEngine:
    """
    Generate dense embeddings for tokens, news, and bundles.
    Uses local PyTorch BERT encoder (offline) with optional sentence-transformers fallback.
    """

    def __init__(self, model_name: str = "local-crypto-bert"):
        self.model_name = model_name
        self.model: Optional[nn.Module] = None
        self.vocab: Optional[Vocabulary] = None
        self.use_local = True
        self.item_embeddings: Dict[str, np.ndarray] = {}
        self.item_metadata: Dict[str, dict] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_item_text(self, item: dict, item_type: str) -> str:
        if item_type == "token":
            tags = " ".join(item.get("tags", []))
            return f"{item['name']} {item['symbol']} {item['category']} {item['description']} {tags}"
        if item_type == "news":
            return f"{item['title']} {item['content']} {item.get('source', '')}"
        if item_type == "bundle":
            return f"{item['name']} {item['description']} {item.get('theme', '')}"
        return str(item)

    def _init_local_model(self, corpus_texts: List[str]):
        self.vocab = Vocabulary()
        self.vocab.build(corpus_texts)
        self.model = CryptoBERTEncoder(len(self.vocab.token2id), EMBEDDING_DIM)
        self.model.to(self.device)
        self.model.eval()
        self.use_local = True

    def _try_load_sentence_transformer(self) -> bool:
        if not HAS_SENTENCE_TRANSFORMERS:
            return False
        try:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.use_local = False
            return True
        except Exception:
            return False

    def load_model(self, corpus_texts: Optional[List[str]] = None):
        if self.model is not None:
            return self.model

        model_path = ARTIFACTS_DIR / "embeddings" / "crypto_bert.pt"
        vocab_path = ARTIFACTS_DIR / "embeddings" / "vocab.json"

        if model_path.exists() and vocab_path.exists():
            self.vocab = Vocabulary.load(vocab_path)
            self.model = CryptoBERTEncoder(len(self.vocab.token2id), EMBEDDING_DIM)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
            self.use_local = True
            return self.model

        if corpus_texts:
            self._init_local_model(corpus_texts)
            return self.model

        if not self._try_load_sentence_transformer():
            raise RuntimeError("No embedding model available. Run build_indexes first.")
        return self.model

    def _encode_local(self, texts: List[str]) -> np.ndarray:
        self.model.eval()
        batch = torch.tensor([self.vocab.encode(t) for t in texts]).to(self.device)
        with torch.no_grad():
            emb = self.model(batch).cpu().numpy()
        return emb.astype(np.float32)

    def encode_items(self, items: List[dict], item_type: str, corpus_texts: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        if self.model is None:
            texts_for_vocab = corpus_texts or [self._build_item_text(it, item_type) for it in items]
            self.load_model(corpus_texts=texts_for_vocab)

        texts = [self._build_item_text(item, item_type) for item in items]

        if self.use_local:
            embeddings = self._encode_local(texts)
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        result = {}
        for item, emb in zip(items, embeddings):
            key = f"{item_type}:{item['id']}"
            result[key] = emb.astype(np.float32)
            self.item_metadata[key] = {"type": item_type, "id": item["id"], "data": item}
        self.item_embeddings.update(result)
        return result

    def encode_query(self, query: str) -> np.ndarray:
        if self.model is None:
            self.load_model(corpus_texts=[query])
        if self.use_local:
            return self._encode_local([query])[0]
        return self.model.encode([query], convert_to_numpy=True)[0].astype(np.float32)

    def fine_tune(self, training_pairs: List[tuple], epochs: int = 3):
        """Fine-tune local BERT encoder on crypto query-document pairs (contrastive loss)."""
        if not self.use_local or self.model is None:
            return

        self.model.train()
        dataset = PairDataset(training_pairs, self.vocab)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        for _ in range(epochs):
            for q_ids, d_ids in loader:
                q_ids, d_ids = q_ids.to(self.device), d_ids.to(self.device)
                optimizer.zero_grad()
                q_emb = self.model(q_ids)
                d_emb = self.model(d_ids)
                # InfoNCE-style contrastive loss
                logits = torch.matmul(q_emb, d_emb.T) / 0.05
                labels = torch.arange(logits.size(0), device=self.device)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()

        self.model.eval()
        self._save_local_model()

    def _save_local_model(self):
        path = ARTIFACTS_DIR / "embeddings"
        path.mkdir(parents=True, exist_ok=True)
        if self.use_local and self.model is not None:
            torch.save(self.model.state_dict(), path / "crypto_bert.pt")
            self.vocab.save(path / "vocab.json")

    def save(self, path=None):
        path = path or ARTIFACTS_DIR / "embeddings"
        path.mkdir(parents=True, exist_ok=True)
        self._save_local_model()
        np.savez_compressed(
            path / "item_embeddings.npz",
            **{k.replace(":", "_"): v for k, v in self.item_embeddings.items()},
        )
        with open(path / "item_metadata.json", "w") as f:
            json.dump(self.item_metadata, f)

    def load_embeddings(self, path=None):
        path = path or ARTIFACTS_DIR / "embeddings"
        self.load_model()
        data = np.load(path / "item_embeddings.npz")
        self.item_embeddings = {
            k.replace("_", ":", 1): data[k] for k in data.files
        }
        with open(path / "item_metadata.json") as f:
            self.item_metadata = json.load(f)
