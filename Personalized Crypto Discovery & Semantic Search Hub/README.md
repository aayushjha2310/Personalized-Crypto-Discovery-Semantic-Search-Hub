# Company X: Personalized Crypto Discovery & Semantic Search Hub

A complete end-to-end prototype implementing hybrid semantic search, Learning to Rank, and two-tower personalized recommendations for crypto tokens, news articles, and investment bundles (Crypto Kuber Packs).

## Architecture

```
User Query + User Profile
    │
    ├─► Decision Tree Intent Classifier (informational / transactional / exploratory)
    ├─► spaCy NER + Crypto Entity Dictionary
    │
    ├─► Dense Search: PyTorch BERT Encoder → FAISS Vector Index
    ├─► Sparse Search: Text Cleaning → TF-IDF
    │
    ├─► Hybrid Fusion (65% dense + 35% sparse)
    ├─► LambdaMART Re-ranking (LightGBM)
    │
    ├─► Ranked Search Results (tokens, news, bundles)
    │
    └─► Two-Tower Model (PyTorch) → Personalized Recommendations
```

## Components Implemented

| Skill | Component | Technology |
|-------|-----------|------------|
| Skill 6 | Two-Tower Recommendation | PyTorch user/item towers |
| Skill 7 | Semantic Search + LTR | BERT embeddings, FAISS, LambdaMART (LightGBM) |
| Skill 8 | Fine-tuned Embeddings | PyTorch CryptoBERT encoder (offline-capable) |
| Skill 1 | NER | spaCy + crypto entity dictionary |
| Skill 3 | Hybrid Search | TF-IDF sparse + dense vector fusion |
| Skill 2 | Topic Modeling + Intent | LDA (gensim), Decision Tree (sklearn) |

## Quick Start (Windows)

### Option A: One-command setup
```bash
python setup.py
python run.py
```

### Option B: Manual setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/train_pipeline.py
python run.py
```

Open **http://localhost:8081** in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/pipeline/status` | ML component status |
| GET | `/api/discover?q=...&user_id=u1` | Combined search + recommendations |
| GET | `/api/search?q=...` | Hybrid semantic search |
| POST | `/api/search` | Search with JSON body |
| GET | `/api/recommend/{user_id}` | Personalized recommendations |
| GET | `/api/news/topics` | LDA topic assignments |
| GET | `/api/users` | List demo users |
| GET | `/api/tokens` | List all tokens |
| GET | `/api/news` | List all news articles |
| GET | `/api/bundles` | List investment bundles |

## Example Queries

- `"low fee Layer 1 tokens for smart contracts"` → SOL, NEAR, AVAX, Layer 1 Pack
- `"trending metaverse coins"` → SAND, MANA, AXS, Metaverse Pack
- `"what is Ethereum DeFi TVL"` → DeFi news + ETH, UNI, AAVE
- `"buy DeFi lending tokens"` → AAVE, UNI, lending news
- `"crypto regulation news today"` → SEC, MiCA, regulation articles

## Project Structure

```
Project-2/
├── config.py                  # Configuration
├── run.py                     # Start API server
├── setup.py                     # One-command setup (Gmail-safe)
├── requirements.txt
├── data/                      # Generated datasets
├── artifacts/                 # Trained models & indexes (generated)
├── scripts/
│   ├── generate_data.py       # Synthetic data generator
│   ├── train_pipeline.py      # End-to-end training
│   └── test_api.py            # API smoke tests
├── src/
│   ├── nlp/                   # NER, TF-IDF, LDA, Intent
│   ├── search/                # Embeddings, FAISS, Hybrid, LTR
│   ├── recommendation/        # Two-tower PyTorch model
│   └── api/                   # FastAPI backend
└── frontend/
    └── index.html             # Web UI (Discovery Hub)
```

## Demo Users

| User | Risk | Frequency | Portfolio |
|------|------|-----------|-----------|
| Alice (u1) | Low | Low | BTC, ETH, XRP |
| Bob (u2) | High | High | SOL, SAND, MANA, AXS |
| Carol (u3) | Medium | Medium | ETH, UNI, AAVE, LINK |
| David (u4) | Medium | High | ARB, OP, MATIC, SOL |
| Eve (u5) | High | Medium | DOGE, SHIB, AXS |

## Verify Installation

```bash
python scripts/test_api.py
```

All search, recommendation, discover, and LDA topic endpoints should pass.

## Sharing via Gmail

Gmail blocks ZIPs containing `.ps1`, `.html` (with JS), `.pt`, `.faiss`, `.lgb`, `.joblib`, and other binary/script files.

**Create a Gmail-safe ZIP:**
```bash
python scripts/package_for_email.py
```

This produces `Project-2-GmailSafe.zip` (~1 MB) with source code, data, and reports only.
The recipient runs `setup.py` to regenerate trained models locally.

See `EMAIL_INSTRUCTIONS.txt` for full details.
