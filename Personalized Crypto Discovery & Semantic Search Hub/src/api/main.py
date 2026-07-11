"""FastAPI backend for Crypto Discovery Hub."""
import json
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from config import DATA_DIR  # noqa: E402
from src.recommendation.two_tower import TwoTowerRecommender  # noqa: E402
from src.search.hybrid_search import HybridSearchEngine  # noqa: E402

app = FastAPI(
    title="Company X Crypto Discovery Hub",
    description="Personalized crypto discovery with hybrid semantic search and two-tower recommendations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engines (loaded at startup)
search_engine: Optional[HybridSearchEngine] = None
recommender: Optional[TwoTowerRecommender] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    item_types: Optional[list] = None


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 10


class DiscoverRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = 8


@app.on_event("startup")
async def startup():
    global search_engine, recommender
    search_engine = HybridSearchEngine()
    recommender = TwoTowerRecommender()

    artifacts_exist = (ROOT / "artifacts" / "faiss_index" / "index.faiss").exists()
    if artifacts_exist:
        search_engine.load_indexes()
        recommender.load()
    else:
        search_engine.build_indexes()
        recommender.train()


@app.get("/")
async def root():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "search_ready": search_engine is not None,
        "recommender_ready": recommender is not None,
    }


@app.get("/api/users")
async def list_users():
    with open(DATA_DIR / "users.json") as f:
        return json.load(f)


@app.get("/api/tokens")
async def list_tokens():
    with open(DATA_DIR / "tokens.json") as f:
        return json.load(f)


@app.get("/api/news")
async def list_news():
    with open(DATA_DIR / "news.json") as f:
        return json.load(f)


@app.get("/api/bundles")
async def list_bundles():
    with open(DATA_DIR / "bundles.json") as f:
        return json.load(f)


@app.post("/api/search")
async def search(request: SearchRequest):
    if not search_engine:
        raise HTTPException(503, "Search engine not initialized")
    return search_engine.search(
        query=request.query,
        top_k=request.top_k,
        item_types=request.item_types,
    )


@app.get("/api/search")
async def search_get(
    q: str = Query(..., description="Natural language search query"),
    top_k: int = Query(10, ge=1, le=50),
):
    if not search_engine:
        raise HTTPException(503, "Search engine not initialized")
    return search_engine.search(query=q, top_k=top_k)


@app.post("/api/recommend")
async def recommend(request: RecommendRequest):
    if not recommender:
        raise HTTPException(503, "Recommender not initialized")
    results = recommender.recommend(request.user_id, top_k=request.top_k)
    if not results:
        raise HTTPException(404, f"User {request.user_id} not found")
    return {"user_id": request.user_id, "recommendations": results}


@app.get("/api/recommend/{user_id}")
async def recommend_get(user_id: str, top_k: int = Query(10, ge=1, le=50)):
    if not recommender:
        raise HTTPException(503, "Recommender not initialized")
    results = recommender.recommend(user_id, top_k=top_k)
    if not results:
        raise HTTPException(404, f"User {user_id} not found")
    return {"user_id": user_id, "recommendations": results}


@app.get("/api/news/topics")
async def news_topics():
    if not search_engine or not search_engine.topic_modeler:
        raise HTTPException(503, "Topic model not initialized")
    return search_engine.topic_modeler.get_article_topics()


@app.get("/api/discover")
async def discover(
    q: str = Query(..., description="Natural language search query"),
    user_id: str = Query(..., description="User ID for personalized recommendations"),
    top_k: int = Query(8, ge=1, le=20),
):
    """Combined discovery: semantic search + personalized recommendations."""
    if not search_engine or not recommender:
        raise HTTPException(503, "Engines not initialized")

    with open(DATA_DIR / "users.json") as f:
        users = {u["id"]: u for u in json.load(f)}
    if user_id not in users:
        raise HTTPException(404, f"User {user_id} not found")

    search_results = search_engine.search(query=q, top_k=top_k)
    recommendations = recommender.recommend(user_id, top_k=top_k)

    return {
        "query": q,
        "user": users[user_id],
        "search": search_results,
        "recommendations": recommendations,
    }


@app.post("/api/discover")
async def discover_post(request: DiscoverRequest):
    if not search_engine or not recommender:
        raise HTTPException(503, "Engines not initialized")

    with open(DATA_DIR / "users.json") as f:
        users = {u["id"]: u for u in json.load(f)}
    if request.user_id not in users:
        raise HTTPException(404, f"User {request.user_id} not found")

    search_results = search_engine.search(query=request.query, top_k=request.top_k)
    recommendations = recommender.recommend(request.user_id, top_k=request.top_k)

    return {
        "query": request.query,
        "user": users[request.user_id],
        "search": search_results,
        "recommendations": recommendations,
    }


@app.get("/api/pipeline/status")
async def pipeline_status():
    """Show which ML components are loaded and ready."""
    artifact_checks = {
        "faiss_index": (ROOT / "artifacts" / "faiss_index" / "index.faiss").exists(),
        "bert_embeddings": (ROOT / "artifacts" / "embeddings" / "crypto_bert.pt").exists(),
        "tfidf_retriever": (ROOT / "artifacts" / "tfidf_retriever.joblib").exists(),
        "intent_classifier": (ROOT / "artifacts" / "intent_classifier.joblib").exists(),
        "lda_model": (ROOT / "artifacts" / "lda_model" / "lda.gensim").exists(),
        "ltr_ranker": (ROOT / "artifacts" / "ltr_model.lgb").exists(),
        "two_tower_model": (ROOT / "artifacts" / "two_tower" / "model.pt").exists(),
    }
    return {
        "status": "ready" if all(artifact_checks.values()) else "partial",
        "components": {
            "semantic_search": "BERT (PyTorch) + FAISS",
            "sparse_search": "TF-IDF",
            "hybrid_fusion": "65% dense + 35% sparse",
            "learning_to_rank": "LambdaMART (LightGBM)",
            "intent_classification": "Decision Tree",
            "entity_extraction": "spaCy NER + Crypto Dictionary",
            "topic_modeling": "LDA (gensim)",
            "recommendations": "Two-Tower (PyTorch)",
        },
        "artifacts": artifact_checks,
    }


# Serve frontend static assets
frontend_dir = ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
