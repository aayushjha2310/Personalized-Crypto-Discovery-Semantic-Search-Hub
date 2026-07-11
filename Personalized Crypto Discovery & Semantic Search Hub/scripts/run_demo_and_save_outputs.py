"""
Run full Crypto Discovery Hub demo and save textual + visual outputs.
Outputs saved to: outputs/
"""
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE = "http://localhost:8081"

SEARCH_QUERIES = [
    "low fee Layer 1 tokens for smart contracts",
    "trending metaverse coins",
    "what is Ethereum DeFi TVL",
    "buy DeFi lending tokens",
    "crypto regulation news today",
]

USERS = ["u1", "u2", "u3", "u4", "u5"]


def fetch(endpoint, params=None):
    r = requests.get(f"{BASE}{endpoint}", params=params or {}, timeout=120)
    r.raise_for_status()
    return r.json()


def save_json(name, data):
    path = OUTPUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def item_title(item):
    d = item.get("data", item)
    return d.get("name") or d.get("title") or d.get("symbol") or item.get("id", "?")


def run_demo():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "=" * 70,
        "Company X — Crypto Discovery Hub — Demo Run Report",
        f"Generated: {timestamp}",
        "=" * 70,
        "",
    ]
    all_data = {"timestamp": timestamp, "health": None, "pipeline": None, "searches": {}, "recommendations": {}, "discover": {}, "topics": None}

    # Health & pipeline
    print("Fetching health & pipeline status...")
    all_data["health"] = fetch("/api/health")
    all_data["pipeline"] = fetch("/api/pipeline/status")
    report_lines += [
        "## System Status",
        f"Health: {all_data['health']['status']}",
        f"Pipeline: {all_data['pipeline']['status']}",
        f"Artifacts ready: {sum(all_data['pipeline']['artifacts'].values())}/{len(all_data['pipeline']['artifacts'])}",
        "",
        "### ML Components",
    ]
    for comp, tech in all_data["pipeline"]["components"].items():
        report_lines.append(f"  - {comp}: {tech}")
    report_lines.append("")

    # Search queries
    print("Running semantic searches...")
    report_lines += ["## Semantic Search Results", ""]
    search_chart_data = []

    for q in SEARCH_QUERIES:
        result = fetch("/api/search", {"q": q, "top_k": 5})
        all_data["searches"][q] = result
        report_lines.append(f"### Query: \"{q}\"")
        report_lines.append(f"  Intent: {result['intent']['intent']} ({result['intent']['confidence']:.0%})")
        report_lines.append(f"  Routing: {result['intent']['routing']}")
        report_lines.append(f"  Entities: tokens={result['entities'].get('tokens', [])}, categories={result['entities'].get('categories', [])}")
        report_lines.append("  Top Results:")
        for i, r in enumerate(result["results"][:5], 1):
            title = item_title(r)
            ltr = r["scores"]["ltr"]
            report_lines.append(f"    {i}. [{r['type']}] {title} (LTR: {ltr:.3f})")
            search_chart_data.append({"query": q[:30], "item": title[:25], "ltr": ltr, "intent": result["intent"]["intent"]})
        report_lines.append("")

    # Recommendations
    print("Running recommendations for all users...")
    report_lines += ["## Personalized Recommendations (Two-Tower)", ""]
    rec_chart_data = []

    for uid in USERS:
        result = fetch(f"/api/recommend/{uid}", {"top_k": 8})
        all_data["recommendations"][uid] = result
        report_lines.append(f"### User {uid}")
        for i, r in enumerate(result["recommendations"][:8], 1):
            title = item_title(r)
            report_lines.append(f"  {i}. [{r['type']}] {title} (score: {r['score']:.3f})")
            rec_chart_data.append({"user": uid, "item": title[:25], "score": r["score"], "type": r["type"]})
        report_lines.append("")

    # Discover (combined)
    print("Running combined discover flow...")
    discover_q = "trending metaverse coins"
    discover_user = "u2"
    all_data["discover"] = fetch("/api/discover", {"q": discover_q, "user_id": discover_user, "top_k": 6})
    d = all_data["discover"]
    report_lines += [
        "## Combined Discovery Hub",
        f"Query: \"{discover_q}\" | User: {d['user']['name']} ({discover_user})",
        f"Risk: {d['user']['risk_appetite']} | Portfolio: {', '.join(d['user']['portfolio'])}",
        "",
        "Search Results:",
    ]
    for i, r in enumerate(d["search"]["results"][:6], 1):
        report_lines.append(f"  {i}. {item_title(r)} (LTR: {r['scores']['ltr']:.3f})")
    report_lines += ["", "Personalized Recommendations:"]
    for i, r in enumerate(d["recommendations"][:6], 1):
        report_lines.append(f"  {i}. {item_title(r)} (score: {r['score']:.3f})")
    report_lines.append("")

    # LDA topics
    print("Fetching LDA topic assignments...")
    all_data["topics"] = fetch("/api/news/topics")
    topic_counts = Counter(t["topic_label"] for t in all_data["topics"])
    report_lines += ["## LDA News Topic Clusters", ""]
    for label, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {label}: {count} articles")
    report_lines.append("")

    # Save textual outputs
    save_json("demo_results.json", all_data)
    report_path = OUTPUT_DIR / "demo_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved: {report_path}")

    # Generate visualizations
    print("Generating visualizations...")
    generate_visualizations(all_data, search_chart_data, rec_chart_data, topic_counts)
    generate_html_report(all_data, report_lines, topic_counts)
    generate_markdown_report(all_data, report_lines, topic_counts)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    return OUTPUT_DIR


def generate_visualizations(all_data, search_chart_data, rec_chart_data, topic_counts):
    plt.style.use("dark_background")
    fig_dir = OUTPUT_DIR

    # 1. Search LTR scores
    fig, ax = plt.subplots(figsize=(12, 6))
    queries_short = [d["query"] for d in search_chart_data[:15]]
    scores = [d["ltr"] for d in search_chart_data[:15]]
    colors = plt.cm.viridis([s / max(scores) if scores else 0 for s in scores])
    ax.barh(range(len(queries_short)), scores, color=colors)
    ax.set_yticks(range(len(queries_short)))
    ax.set_yticklabels([f"{d['query']} → {d['item']}" for d in search_chart_data[:15]], fontsize=8)
    ax.set_xlabel("LambdaMART (LTR) Score")
    ax.set_title("Semantic Search — Top Result LTR Scores by Query")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(fig_dir / "chart_search_ltr_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Intent distribution
    intents = Counter(data["intent"]["intent"] for data in all_data["searches"].values())
    fig, ax = plt.subplots(figsize=(8, 6))
    labels, counts = zip(*intents.items()) if intents else ([], [])
    ax.pie(counts, labels=labels, autopct="%1.0f%%", colors=["#00d4aa", "#6c5ce7", "#ffd93d"])
    ax.set_title("Query Intent Classification (Decision Tree)")
    plt.tight_layout()
    fig.savefig(fig_dir / "chart_intent_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Recommendation scores by user
    fig, ax = plt.subplots(figsize=(12, 6))
    users = USERS
    avg_scores = []
    for uid in users:
        recs = all_data["recommendations"][uid]["recommendations"]
        avg_scores.append(sum(r["score"] for r in recs) / len(recs) if recs else 0)
    bars = ax.bar(users, avg_scores, color="#6c5ce7", edgecolor="#00d4aa")
    ax.set_xlabel("User ID")
    ax.set_ylabel("Avg Recommendation Score")
    ax.set_title("Two-Tower Recommendations — Avg Score by User")
    for bar, score in zip(bars, avg_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{score:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "chart_recommendation_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. LDA topic distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(topic_counts.keys())
    counts = list(topic_counts.values())
    ax.bar(labels, counts, color="#ffd93d", edgecolor="#00d4aa")
    ax.set_xlabel("Topic Cluster")
    ax.set_ylabel("Article Count")
    ax.set_title("LDA News Topic Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / "chart_lda_topics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Pipeline architecture diagram
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor("#0b0f1a")
    fig.patch.set_facecolor("#0b0f1a")

    boxes = [
        (5, 9, "User Query", "#00d4aa"),
        (2, 7, "Decision Tree\nIntent Classifier", "#6c5ce7"),
        (5, 7, "spaCy NER +\nCrypto Dictionary", "#6c5ce7"),
        (8, 7, "LDA Topic\nModel", "#6c5ce7"),
        (2, 5, "BERT Embeddings\n(PyTorch)", "#ffd93d"),
        (5, 5, "TF-IDF\nSparse Search", "#ffd93d"),
        (8, 5, "FAISS\nVector Index", "#ffd93d"),
        (5, 3, "Hybrid Fusion\n(65% + 35%)", "#00d4aa"),
        (5, 1.5, "LambdaMART\nRe-ranking", "#00d4aa"),
        (8, 1.5, "Two-Tower\nRecommendations", "#6c5ce7"),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(plt.Rectangle((x - 0.9, y - 0.4), 1.8, 0.8, fill=True, facecolor=color, alpha=0.3, edgecolor=color))
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color="white")

    arrows = [(5, 8.6, 5, 7.4), (5, 8.6, 2, 7.4), (5, 8.6, 8, 7.4),
              (2, 6.6, 2, 5.4), (5, 6.6, 5, 5.4), (8, 6.6, 8, 5.4),
              (2, 4.6, 5, 3.4), (5, 4.6, 5, 3.4), (8, 4.6, 5, 3.4),
              (5, 2.6, 5, 1.9), (5, 8.6, 8, 1.9)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#8b95a8", lw=1))

    ax.set_title("Company X Crypto Discovery Hub — ML Pipeline Architecture", color="white", fontsize=14, pad=20)
    plt.tight_layout()
    fig.savefig(fig_dir / "chart_pipeline_architecture.png", dpi=150, bbox_inches="tight", facecolor="#0b0f1a")
    plt.close()

    print("Saved 5 chart images to outputs/")


def generate_html_report(all_data, report_lines, topic_counts):
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Crypto Discovery Hub — Demo Report</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #0b0f1a; color: #e8ecf4; padding: 2rem; max-width: 1200px; margin: 0 auto; }}
    h1 {{ color: #00d4aa; }}
    h2 {{ color: #6c5ce7; margin-top: 2rem; border-bottom: 1px solid #2a3555; padding-bottom: 0.5rem; }}
    pre {{ background: #141b2d; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.85rem; white-space: pre-wrap; }}
    img {{ max-width: 100%; border-radius: 8px; margin: 1rem 0; border: 1px solid #2a3555; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
    .badge {{ display: inline-block; padding: 0.2rem 0.6rem; background: #1c2540; border-radius: 4px; margin: 0.2rem; font-size: 0.8rem; }}
  </style>
</head>
<body>
  <h1>Company X — Crypto Discovery Hub</h1>
  <p>Demo run generated: {all_data['timestamp']}</p>
  <p>
    <span class="badge">Status: {all_data['health']['status']}</span>
    <span class="badge">Pipeline: {all_data['pipeline']['status']}</span>
    <span class="badge">Artifacts: {sum(all_data['pipeline']['artifacts'].values())}/7</span>
  </p>

  <h2>Pipeline Architecture</h2>
  <img src="chart_pipeline_architecture.png" alt="Pipeline Architecture">

  <h2>Visual Analytics</h2>
  <div class="grid">
    <div><h3>Search LTR Scores</h3><img src="chart_search_ltr_scores.png" alt="Search LTR"></div>
    <div><h3>Intent Distribution</h3><img src="chart_intent_distribution.png" alt="Intent"></div>
    <div><h3>Recommendation Scores</h3><img src="chart_recommendation_scores.png" alt="Recommendations"></div>
    <div><h3>LDA Topic Clusters</h3><img src="chart_lda_topics.png" alt="LDA Topics"></div>
  </div>

  <h2>Full Text Report</h2>
  <pre>{chr(10).join(report_lines)}</pre>
</body>
</html>"""
    path = OUTPUT_DIR / "demo_report.html"
    path.write_text(html, encoding="utf-8")
    print(f"Saved: {path}")


def generate_markdown_report(all_data, report_lines, topic_counts):
    """Save demo report as Markdown (converted from demo_report.html content)."""
    md_lines = [
        "# Company X — Crypto Discovery Hub",
        "",
        f"**Demo run generated:** {all_data['timestamp']}",
        "",
        "| Status | Pipeline | Artifacts |",
        "|--------|----------|-----------|",
        f"| {all_data['health']['status']} | {all_data['pipeline']['status']} | {sum(all_data['pipeline']['artifacts'].values())}/7 |",
        "",
        "---",
        "",
        "## Pipeline Architecture",
        "",
        "![Pipeline Architecture](chart_pipeline_architecture.png)",
        "",
        "---",
        "",
        "## Visual Analytics",
        "",
        "### Search LTR Scores",
        "",
        "![Search LTR Scores](chart_search_ltr_scores.png)",
        "",
        "### Intent Distribution",
        "",
        "![Intent Distribution](chart_intent_distribution.png)",
        "",
        "### Recommendation Scores",
        "",
        "![Recommendation Scores](chart_recommendation_scores.png)",
        "",
        "### LDA Topic Clusters",
        "",
        "![LDA Topic Clusters](chart_lda_topics.png)",
        "",
        "---",
        "",
    ]

    # Append formatted report body (skip the ASCII banner lines)
    body_started = False
    for line in report_lines:
        if line.startswith("## "):
            body_started = True
        if not body_started:
            continue
        if line.startswith("### Query:"):
            md_lines.extend(["", line.replace("### Query:", "### Query:"), ""])
        elif line.startswith("  Intent:") or line.startswith("  Routing:") or line.startswith("  Entities:"):
            key, val = line.strip().split(":", 1)
            md_lines.append(f"- **{key.strip()}:**{val}")
        elif line.startswith("  Top Results:") or line.startswith("  Search Results:") or line.startswith("  Personalized Recommendations:"):
            md_lines.extend(["", f"**{line.strip()}**", ""])
        elif line.startswith("  ") and ". [" in line:
            md_lines.append(line.strip())
        elif line.startswith("### User"):
            md_lines.extend(["", line, ""])
        elif line.startswith("## Combined"):
            md_lines.extend(["", line, ""])
        elif line.startswith("Query:") or line.startswith("Risk:"):
            md_lines.append(f"**{line}**")
        elif line.startswith("  ") and ":" in line and "articles" in line:
            topic, count = line.strip().rsplit(":", 1)
            md_lines.append(f"- {topic.strip()}: {count.strip()}")
        elif line.startswith("## LDA"):
            md_lines.extend(["", line, "", "| Topic | Articles |", "|-------|----------|"])
        elif line.startswith("  ") and "articles" in line:
            topic, count = line.strip().rsplit(":", 1)
            md_lines.append(f"| {topic.strip()} | {count.replace('articles', '').strip()} |")
        elif line.startswith("## ") and "LDA" not in line:
            md_lines.extend(["", line, ""])
        elif line.startswith("### ML"):
            md_lines.extend(["", line, ""])
        elif line.startswith("  - "):
            md_lines.append(line.replace("  - ", "- "))
        elif line.startswith("Health:") or line.startswith("Pipeline:") or line.startswith("Artifacts"):
            key, val = line.split(":", 1)
            md_lines.append(f"- **{key.strip()}:** {val.strip()}")

    path = OUTPUT_DIR / "demo_report.md"
    path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {path}")


if __name__ == "__main__":
    try:
        out = run_demo()
        print(f"\nDone! Open {out / 'demo_report.html'} in your browser.")
    except requests.exceptions.ConnectionError:
        print("ERROR: Server not running. Start it with: python run.py")
        sys.exit(1)
