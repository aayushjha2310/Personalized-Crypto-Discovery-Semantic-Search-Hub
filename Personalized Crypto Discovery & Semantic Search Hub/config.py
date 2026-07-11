"""Project configuration and paths."""
import os
from pathlib import Path

# Avoid TensorFlow import issues on Windows (use PyTorch only)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

for directory in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Embedding model (BERT-based sentence transformer)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Search parameters
HYBRID_DENSE_WEIGHT = 0.65
HYBRID_SPARSE_WEIGHT = 0.35
DEFAULT_TOP_K = 10

# Two-tower recommendation
USER_EMBED_DIM = 64
ITEM_EMBED_DIM = 64
HIDDEN_DIM = 128
TWO_TOWER_EPOCHS = 15
TWO_TOWER_BATCH_SIZE = 64
TWO_TOWER_LR = 1e-3

# Server
API_PORT = 8081
API_HOST = "0.0.0.0"

# LDA topics
LDA_NUM_TOPICS = 6
LDA_TOPIC_LABELS = ["DeFi", "NFTs", "Regulation", "Layer1", "Metaverse", "Trading"]

# Intent classes
INTENT_CLASSES = ["informational", "transactional", "exploratory"]
