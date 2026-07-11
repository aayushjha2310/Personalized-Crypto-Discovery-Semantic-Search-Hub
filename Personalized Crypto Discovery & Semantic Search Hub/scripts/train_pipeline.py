"""End-to-end pipeline: generate data, train all models, build indexes."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    print("=" * 60)
    print("Company X Crypto Discovery Hub — Training Pipeline")
    print("=" * 60)

    # Step 1: Generate synthetic data
    print("\n[1/4] Generating synthetic dataset...")
    from scripts.generate_data import main as gen_data
    gen_data()

    # Step 2: Build search indexes (BERT embeddings, FAISS, TF-IDF, LDA, Intent, LTR)
    print("\n[2/4] Building hybrid search indexes...")
    from src.search.hybrid_search import HybridSearchEngine
    search = HybridSearchEngine()
    search.build_indexes(fine_tune=True)
    print("  - BERT embeddings generated and fine-tuned")
    print("  - FAISS vector index built")
    print("  - TF-IDF sparse retriever trained")
    print("  - LDA topic model trained")
    print("  - Decision Tree intent classifier trained")
    print("  - LambdaMART (LightGBM) ranker trained")

    # Step 3: Train two-tower recommender
    print("\n[3/4] Training two-tower recommendation model...")
    from src.recommendation.two_tower import TwoTowerRecommender
    rec = TwoTowerRecommender()
    metrics = rec.train()
    print(f"  - Training complete: {metrics}")

    # Step 4: Smoke test
    print("\n[4/4] Running smoke tests...")
    search.load_indexes()
    rec.load()

    test_queries = [
        "low fee Layer 1 tokens for smart contracts",
        "trending metaverse coins",
        "crypto regulation news today",
    ]
    for q in test_queries:
        result = search.search(q, top_k=3)
        top = result["results"][0] if result["results"] else None
        top_name = top["data"].get("name") or top["data"].get("title", "N/A") if top else "N/A"
        print(f"  Query: '{q}'")
        print(f"    Intent: {result['intent']['intent']} | Top: {top_name}")

    for uid in ["u1", "u2", "u3"]:
        recs = rec.recommend(uid, top_k=3)
        names = [r["data"].get("name") or r["data"].get("title", r["id"]) for r in recs]
        print(f"  User {uid} recommendations: {', '.join(names)}")

    print("\n" + "=" * 60)
    print("Pipeline complete! Start the server with:")
    print("  python run.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
