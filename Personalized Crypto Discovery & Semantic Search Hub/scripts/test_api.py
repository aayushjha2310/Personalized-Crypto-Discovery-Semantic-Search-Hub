"""Full API smoke tests for the Crypto Discovery Hub."""
import requests

BASE = "http://localhost:8081"

def test_health():
    r = requests.get(f"{BASE}/api/health", timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    print("  health: OK")

def test_pipeline_status():
    r = requests.get(f"{BASE}/api/pipeline/status", timeout=30)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("ready", "partial")
    print(f"  pipeline: {data['status']} ({sum(data['artifacts'].values())}/{len(data['artifacts'])} artifacts)")

def test_search():
    queries = [
        "low fee Layer 1 tokens for smart contracts",
        "trending metaverse coins",
        "crypto regulation news today",
    ]
    for q in queries:
        r = requests.get(f"{BASE}/api/search", params={"q": q, "top_k": 3}, timeout=60)
        assert r.status_code == 200, f"Search failed for: {q}"
        data = r.json()
        assert len(data["results"]) > 0
        top = data["results"][0]["data"].get("name") or data["results"][0]["data"].get("title")
        print(f"  search '{q[:40]}...' -> {top}")

def test_recommend():
    for uid in ["u1", "u2", "u3"]:
        r = requests.get(f"{BASE}/api/recommend/{uid}", params={"top_k": 5}, timeout=30)
        assert r.status_code == 200
        recs = r.json()["recommendations"]
        assert len(recs) > 0
        print(f"  recommend {uid}: {len(recs)} items")

def test_discover():
    r = requests.get(
        f"{BASE}/api/discover",
        params={"q": "trending metaverse coins", "user_id": "u2", "top_k": 5},
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert "search" in data and "recommendations" in data
    assert len(data["search"]["results"]) > 0
    assert len(data["recommendations"]) > 0
    print(f"  discover: search={len(data['search']['results'])}, recs={len(data['recommendations'])}")

def test_topics():
    r = requests.get(f"{BASE}/api/pipeline/status", timeout=30)
    r = requests.get(f"{BASE}/api/news/topics", timeout=30)
    assert r.status_code == 200
    topics = r.json()
    assert len(topics) > 0
    labels = set(t["topic_label"] for t in topics)
    print(f"  topics: {len(topics)} articles, {len(labels)} clusters")

def main():
    print("=== API SMOKE TESTS ===")
    test_health()
    test_pipeline_status()
    test_search()
    test_recommend()
    test_discover()
    test_topics()
    print("\nAll tests passed!")

if __name__ == "__main__":
    main()
