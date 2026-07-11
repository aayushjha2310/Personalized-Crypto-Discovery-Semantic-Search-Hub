"""Generate synthetic crypto discovery dataset for prototype training."""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

TOKENS = [
    {"id": "btc", "symbol": "BTC", "name": "Bitcoin", "category": "Layer1",
     "description": "Bitcoin is the original decentralized digital currency with proof-of-work consensus and store-of-value use case.",
     "tags": ["layer1", "store-of-value", "proof-of-work"], "fee_tier": "medium", "market_cap_rank": 1},
    {"id": "eth", "symbol": "ETH", "name": "Ethereum", "category": "Layer1",
     "description": "Ethereum is a smart contract platform enabling DeFi, NFTs, and decentralized applications with EVM compatibility.",
     "tags": ["layer1", "smart-contracts", "defi", "evm"], "fee_tier": "medium", "market_cap_rank": 2},
    {"id": "sol", "symbol": "SOL", "name": "Solana", "category": "Layer1",
     "description": "Solana is a high-throughput Layer 1 blockchain optimized for low-fee smart contracts and fast finality.",
     "tags": ["layer1", "smart-contracts", "low-fee", "high-throughput"], "fee_tier": "low", "market_cap_rank": 5},
    {"id": "avax", "symbol": "AVAX", "name": "Avalanche", "category": "Layer1",
     "description": "Avalanche offers sub-second finality and customizable subnets for enterprise and DeFi smart contract deployments.",
     "tags": ["layer1", "smart-contracts", "subnets", "defi"], "fee_tier": "low", "market_cap_rank": 12},
    {"id": "matic", "symbol": "MATIC", "name": "Polygon", "category": "Layer2",
     "description": "Polygon scales Ethereum with low-fee Layer 2 solutions for gaming, DeFi, and NFT marketplaces.",
     "tags": ["layer2", "ethereum", "low-fee", "gaming"], "fee_tier": "low", "market_cap_rank": 15},
    {"id": "link", "symbol": "LINK", "name": "Chainlink", "category": "Oracle",
     "description": "Chainlink provides decentralized oracle infrastructure connecting smart contracts to real-world data feeds.",
     "tags": ["oracle", "defi", "infrastructure"], "fee_tier": "n/a", "market_cap_rank": 14},
    {"id": "uni", "symbol": "UNI", "name": "Uniswap", "category": "DeFi",
     "description": "Uniswap is a leading decentralized exchange protocol for automated market making on Ethereum.",
     "tags": ["defi", "dex", "ethereum", "amm"], "fee_tier": "variable", "market_cap_rank": 20},
    {"id": "aave", "symbol": "AAVE", "name": "Aave", "category": "DeFi",
     "description": "Aave is a decentralized lending protocol allowing users to borrow and lend crypto assets with flash loans.",
     "tags": ["defi", "lending", "borrowing"], "fee_tier": "variable", "market_cap_rank": 45},
    {"id": "sand", "symbol": "SAND", "name": "The Sandbox", "category": "Metaverse",
     "description": "The Sandbox is a virtual world metaverse platform where users buy land NFTs and create gaming experiences.",
     "tags": ["metaverse", "gaming", "nft", "virtual-land"], "fee_tier": "medium", "market_cap_rank": 80},
    {"id": "mana", "symbol": "MANA", "name": "Decentraland", "category": "Metaverse",
     "description": "Decentraland is a trending metaverse coin powering a 3D virtual world with user-owned digital real estate.",
     "tags": ["metaverse", "virtual-reality", "nft", "gaming"], "fee_tier": "medium", "market_cap_rank": 95},
    {"id": "axs", "symbol": "AXS", "name": "Axie Infinity", "category": "Gaming",
     "description": "Axie Infinity is a play-to-earn gaming token in the metaverse and NFT gaming ecosystem.",
     "tags": ["gaming", "metaverse", "play-to-earn", "nft"], "fee_tier": "low", "market_cap_rank": 110},
    {"id": "ada", "symbol": "ADA", "name": "Cardano", "category": "Layer1",
     "description": "Cardano is a research-driven proof-of-stake Layer 1 blockchain for smart contracts with academic rigor.",
     "tags": ["layer1", "proof-of-stake", "smart-contracts"], "fee_tier": "low", "market_cap_rank": 8},
    {"id": "dot", "symbol": "DOT", "name": "Polkadot", "category": "Layer0",
     "description": "Polkadot enables cross-chain interoperability connecting multiple blockchains through parachains.",
     "tags": ["interoperability", "parachain", "layer0"], "fee_tier": "low", "market_cap_rank": 11},
    {"id": "atom", "symbol": "ATOM", "name": "Cosmos", "category": "Layer0",
     "description": "Cosmos provides the Inter-Blockchain Communication protocol for sovereign connected blockchains.",
     "tags": ["interoperability", "ibc", "layer0"], "fee_tier": "low", "market_cap_rank": 25},
    {"id": "xrp", "symbol": "XRP", "name": "Ripple", "category": "Payments",
     "description": "XRP facilitates fast cross-border payments and remittances for financial institutions.",
     "tags": ["payments", "remittance", "enterprise"], "fee_tier": "low", "market_cap_rank": 6},
    {"id": "doge", "symbol": "DOGE", "name": "Dogecoin", "category": "Meme",
     "description": "Dogecoin is a community-driven meme coin used for micro-tipping and social payments.",
     "tags": ["meme", "community", "payments"], "fee_tier": "low", "market_cap_rank": 9},
    {"id": "shib", "symbol": "SHIB", "name": "Shiba Inu", "category": "Meme",
     "description": "Shiba Inu is a trending meme token with an ecosystem including DEX and metaverse initiatives.",
     "tags": ["meme", "community", "metaverse"], "fee_tier": "low", "market_cap_rank": 16},
    {"id": "arb", "symbol": "ARB", "name": "Arbitrum", "category": "Layer2",
     "description": "Arbitrum is an optimistic rollup Layer 2 scaling Ethereum with low fees for DeFi and smart contracts.",
     "tags": ["layer2", "ethereum", "rollup", "low-fee", "defi"], "fee_tier": "low", "market_cap_rank": 30},
    {"id": "op", "symbol": "OP", "name": "Optimism", "category": "Layer2",
     "description": "Optimism scales Ethereum using optimistic rollups for cheaper smart contract execution.",
     "tags": ["layer2", "ethereum", "rollup", "defi"], "fee_tier": "low", "market_cap_rank": 35},
    {"id": "near", "symbol": "NEAR", "name": "NEAR Protocol", "category": "Layer1",
     "description": "NEAR Protocol is a sharded proof-of-stake blockchain designed for developer-friendly smart contracts.",
     "tags": ["layer1", "sharding", "smart-contracts", "low-fee"], "fee_tier": "low", "market_cap_rank": 22},
]

NEWS = [
    {"id": "n1", "title": "Ethereum DeFi TVL Surges Past $50B Milestone",
     "content": "Total value locked in Ethereum DeFi protocols reached a new high as lending and DEX volumes spike amid institutional interest.",
     "source": "CryptoDaily", "topic_hint": "DeFi"},
    {"id": "n2", "title": "SEC Proposes New Crypto Regulation Framework",
     "content": "US regulators unveiled draft rules for digital asset custody and exchange compliance affecting token listings nationwide.",
     "source": "RegWatch", "topic_hint": "Regulation"},
    {"id": "n3", "title": "Metaverse Land Sales Hit Record in The Sandbox",
     "content": "Virtual real estate NFTs in metaverse platforms saw record trading volume as brands enter the virtual world economy.",
     "source": "MetaVerse Times", "topic_hint": "Metaverse"},
    {"id": "n4", "title": "Solana Network Upgrade Cuts Transaction Fees",
     "content": "Solana validators approved a protocol upgrade reducing fees for smart contract deployments on the high-throughput Layer 1.",
     "source": "ChainNews", "topic_hint": "Layer1"},
    {"id": "n5", "title": "NFT Market Shows Signs of Recovery in Q4",
     "content": "Blue-chip NFT collections rebound as trading platforms introduce royalty enforcement and creator tools.",
     "source": "NFT Insider", "topic_hint": "NFTs"},
    {"id": "n6", "title": "Bitcoin ETF Inflows Drive Institutional Trading Volume",
     "content": "Spot Bitcoin ETF products attracted billions in inflows reshaping crypto trading patterns among hedge funds.",
     "source": "TradeWire", "topic_hint": "Trading"},
    {"id": "n7", "title": "Aave Launches Cross-Chain Lending on Arbitrum",
     "content": "Leading DeFi lending protocol expands to Layer 2 with lower gas fees attracting retail borrowers.",
     "source": "DeFi Pulse", "topic_hint": "DeFi"},
    {"id": "n8", "title": "EU MiCA Regulation Takes Effect for Stablecoins",
     "content": "European crypto regulation MiCA now governs stablecoin issuance with strict reserve and disclosure requirements.",
     "source": "RegWatch", "topic_hint": "Regulation"},
    {"id": "n9", "title": "Axie Infinity Announces New Metaverse Game Mode",
     "content": "Play-to-earn gaming studio unveils metaverse expansion with interoperable avatar NFTs across virtual worlds.",
     "source": "GameChain", "topic_hint": "Metaverse"},
    {"id": "n10", "title": "Avalanche Subnets Power Enterprise Blockchain Pilots",
     "content": "Fortune 500 companies pilot private subnets on Avalanche for supply chain smart contract automation.",
     "source": "Enterprise Ledger", "topic_hint": "Layer1"},
    {"id": "n11", "title": "Uniswap v4 Hooks Spark DeFi Innovation Wave",
     "content": "Customizable AMM hooks enable novel DeFi primitives including dynamic fees and on-chain limit orders.",
     "source": "DeFi Pulse", "topic_hint": "DeFi"},
    {"id": "n12", "title": "Polygon Partners with Gaming Studios for NFT Launch",
     "content": "Low-fee Layer 2 network signs deals with major gaming publishers for in-game NFT marketplaces.",
     "source": "NFT Insider", "topic_hint": "NFTs"},
    {"id": "n13", "title": "Cosmos IBC Volume Breaks Monthly Record",
     "content": "Inter-blockchain communication transfers between Cosmos zones surge as cross-chain DeFi gains traction.",
     "source": "ChainNews", "topic_hint": "Layer1"},
    {"id": "n14", "title": "Crypto Day Traders Shift to Layer 2 Tokens",
     "content": "High-frequency traders migrate to Arbitrum and Optimism tokens amid lower fees and faster settlement.",
     "source": "TradeWire", "topic_hint": "Trading"},
    {"id": "n15", "title": "Decentraland Hosts Virtual Fashion Week Event",
     "content": "Trending metaverse coin MANA rallied as luxury brands showcased digital wearables in a virtual world event.",
     "source": "MetaVerse Times", "topic_hint": "Metaverse"},
    {"id": "n16", "title": "Chainlink Oracles Integrate Real-World Asset Feeds",
     "content": "DeFi protocols adopt Chainlink price feeds for tokenized treasury bills and commodity-backed assets.",
     "source": "DeFi Pulse", "topic_hint": "DeFi"},
    {"id": "n17", "title": "India Drafts Crypto Tax Compliance Guidelines",
     "content": "New regulation requires exchanges to report transaction data with enhanced KYC for retail traders.",
     "source": "RegWatch", "topic_hint": "Regulation"},
    {"id": "n18", "title": "NEAR Protocol Hackathon Yields 200 Smart Contract dApps",
     "content": "Developer-friendly Layer 1 attracts builders focusing on low-fee smart contract deployments.",
     "source": "ChainNews", "topic_hint": "Layer1"},
    {"id": "n19", "title": "OpenSea Expands NFT Curation for Digital Art",
     "content": "NFT marketplace introduces verified creator collections driving renewed collector interest.",
     "source": "NFT Insider", "topic_hint": "NFTs"},
    {"id": "n20", "title": "Altcoin Rotation Strategy Gains Among Swing Traders",
     "content": "Trading desks rotate from Bitcoin into high-beta Layer 1 and metaverse tokens during risk-on sessions.",
     "source": "TradeWire", "topic_hint": "Trading"},
]

BUNDLES = [
    {"id": "b1", "name": "Crypto Kuber DeFi Pack",
     "description": "Curated bundle of top DeFi tokens including UNI, AAVE, and LINK for yield and DEX exposure.",
     "token_ids": ["uni", "aave", "link"], "risk_level": "medium", "theme": "DeFi"},
    {"id": "b2", "name": "Crypto Kuber Metaverse Pack",
     "description": "Trending metaverse coins bundle with SAND, MANA, and AXS for virtual world and gaming exposure.",
     "token_ids": ["sand", "mana", "axs"], "risk_level": "high", "theme": "Metaverse"},
    {"id": "b3", "name": "Crypto Kuber Layer 1 Pack",
     "description": "Low-fee Layer 1 tokens for smart contracts: SOL, NEAR, AVAX, and ADA.",
     "token_ids": ["sol", "near", "avax", "ada"], "risk_level": "medium", "theme": "Layer1"},
    {"id": "b4", "name": "Crypto Kuber Blue Chip Pack",
     "description": "Conservative bundle of BTC, ETH, and XRP for long-term portfolio stability.",
     "token_ids": ["btc", "eth", "xrp"], "risk_level": "low", "theme": "BlueChip"},
    {"id": "b5", "name": "Crypto Kuber Layer 2 Pack",
     "description": "Ethereum scaling tokens ARB, OP, and MATIC for low-fee DeFi activity.",
     "token_ids": ["arb", "op", "matic"], "risk_level": "medium", "theme": "Layer2"},
    {"id": "b6", "name": "Crypto Kuber Meme Pack",
     "description": "High-risk meme coin bundle with DOGE and SHIB for speculative trading.",
     "token_ids": ["doge", "shib"], "risk_level": "high", "theme": "Meme"},
]

USERS = [
    {"id": "u1", "name": "Alice", "risk_appetite": "low", "trading_frequency": "low",
     "portfolio": ["btc", "eth", "xrp"], "preferred_categories": ["Layer1", "Payments"]},
    {"id": "u2", "name": "Bob", "risk_appetite": "high", "trading_frequency": "high",
     "portfolio": ["sol", "sand", "mana", "axs"], "preferred_categories": ["Metaverse", "Gaming"]},
    {"id": "u3", "name": "Carol", "risk_appetite": "medium", "trading_frequency": "medium",
     "portfolio": ["eth", "uni", "aave", "link"], "preferred_categories": ["DeFi", "Oracle"]},
    {"id": "u4", "name": "David", "risk_appetite": "medium", "trading_frequency": "high",
     "portfolio": ["arb", "op", "matic", "sol"], "preferred_categories": ["Layer2", "Layer1"]},
    {"id": "u5", "name": "Eve", "risk_appetite": "high", "trading_frequency": "medium",
     "portfolio": ["doge", "shib", "axs"], "preferred_categories": ["Meme", "Gaming"]},
]

SEARCH_QUERIES = [
    {"query": "low fee Layer 1 tokens for smart contracts", "intent": "transactional",
     "relevant_tokens": ["sol", "near", "avax", "ada"], "relevant_news": ["n4", "n18"]},
    {"query": "trending metaverse coins", "intent": "exploratory",
     "relevant_tokens": ["sand", "mana", "axs", "shib"], "relevant_news": ["n3", "n9", "n15"]},
    {"query": "what is Ethereum DeFi TVL", "intent": "informational",
     "relevant_tokens": ["eth", "uni", "aave"], "relevant_news": ["n1", "n7", "n11"]},
    {"query": "buy DeFi lending tokens", "intent": "transactional",
     "relevant_tokens": ["aave", "uni", "link"], "relevant_news": ["n7", "n16"]},
    {"query": "crypto regulation news today", "intent": "informational",
     "relevant_tokens": [], "relevant_news": ["n2", "n8", "n17"]},
    {"query": "best NFT marketplace tokens", "intent": "exploratory",
     "relevant_tokens": ["sand", "mana"], "relevant_news": ["n5", "n12", "n19"]},
    {"query": "Layer 2 rollup tokens with low gas", "intent": "transactional",
     "relevant_tokens": ["arb", "op", "matic"], "relevant_news": ["n14"]},
    {"query": "how does Chainlink oracle work", "intent": "informational",
     "relevant_tokens": ["link"], "relevant_news": ["n16"]},
    {"query": "explore gaming play to earn coins", "intent": "exploratory",
     "relevant_tokens": ["axs", "sand", "mana"], "relevant_news": ["n9"]},
    {"query": "purchase Bitcoin ETF related assets", "intent": "transactional",
     "relevant_tokens": ["btc", "eth"], "relevant_news": ["n6"]},
    {"query": "cross chain interoperability protocols", "intent": "informational",
     "relevant_tokens": ["dot", "atom"], "relevant_news": ["n13"]},
    {"query": "meme coins for speculation", "intent": "transactional",
     "relevant_tokens": ["doge", "shib"], "relevant_news": []},
    {"query": "Avalanche subnet enterprise blockchain", "intent": "informational",
     "relevant_tokens": ["avax"], "relevant_news": ["n10"]},
    {"query": "discover new DeFi DEX tokens", "intent": "exploratory",
     "relevant_tokens": ["uni", "link"], "relevant_news": ["n11"]},
    {"query": "Polygon gaming NFT low fee", "intent": "transactional",
     "relevant_tokens": ["matic", "sand"], "relevant_news": ["n12"]},
]

INTENT_KEYWORDS = {
    "informational": ["what", "how", "why", "explain", "news", "today", "work", "guide"],
    "transactional": ["buy", "purchase", "invest", "trade", "low fee", "best", "tokens for"],
    "exploratory": ["trending", "explore", "discover", "new", "metaverse", "gaming"],
}


def generate_interactions():
    """Generate user-item interaction logs for two-tower training and LTR."""
    interactions = []
    item_pool = (
        [(t["id"], "token", t["category"]) for t in TOKENS]
        + [(n["id"], "news", n["topic_hint"]) for n in NEWS]
        + [(b["id"], "bundle", b["theme"]) for b in BUNDLES]
    )

    for user in USERS:
        risk_map = {"low": 0, "medium": 1, "high": 2}
        freq_map = {"low": 3, "medium": 8, "high": 15}
        num_events = freq_map[user["trading_frequency"]]

        for _ in range(num_events):
            # Bias toward portfolio and preferred categories
            candidates = []
            for item_id, item_type, theme in item_pool:
                score = random.random()
                if item_id in user["portfolio"]:
                    score += 0.5
                if theme in user["preferred_categories"]:
                    score += 0.3
                if item_type == "bundle" and user["risk_appetite"] == "high":
                    score += 0.2
                candidates.append((score, item_id, item_type))

            candidates.sort(reverse=True)
            top = candidates[:5]
            chosen = random.choice(top)
            interactions.append({
                "user_id": user["id"],
                "item_id": chosen[1],
                "item_type": chosen[2],
                "clicked": 1,
                "traded": 1 if random.random() < 0.4 else 0,
                "risk_appetite": risk_map[user["risk_appetite"]],
                "trading_frequency": freq_map[user["trading_frequency"]],
            })

        # Negative samples
        for _ in range(num_events // 2):
            item_id, item_type, _ = random.choice(item_pool)
            interactions.append({
                "user_id": user["id"],
                "item_id": item_id,
                "item_type": item_type,
                "clicked": 0,
                "traded": 0,
                "risk_appetite": risk_map[user["risk_appetite"]],
                "trading_frequency": freq_map[user["trading_frequency"]],
            })

    return interactions


def generate_ltr_training_data():
    """Generate query-token relevance labels for LambdaMART."""
    ltr_data = []
    token_map = {t["id"]: t for t in TOKENS}

    for entry in SEARCH_QUERIES:
        query = entry["query"]
        relevant = set(entry["relevant_tokens"])
        for token in TOKENS:
            label = 3 if token["id"] in relevant else 0
            # Partial relevance from tag overlap
            if label == 0:
                query_words = set(query.lower().split())
                tag_overlap = len(query_words & set(" ".join(token["tags"]).split()))
                if tag_overlap > 0:
                    label = 1
                if token["category"].lower() in query.lower():
                    label = max(label, 2)

            ltr_data.append({
                "query": query,
                "query_id": hash(query) % 100000,
                "item_id": token["id"],
                "label": label,
                "market_cap_rank": token["market_cap_rank"],
                "fee_score": {"low": 3, "medium": 2, "high": 1, "variable": 2, "n/a": 1}.get(token["fee_tier"], 1),
                "category_match": 1 if token["category"].lower() in query.lower() else 0,
            })
    return ltr_data


def main():
    datasets = {
        "tokens.json": TOKENS,
        "news.json": NEWS,
        "bundles.json": BUNDLES,
        "users.json": USERS,
        "search_queries.json": SEARCH_QUERIES,
        "interactions.json": generate_interactions(),
        "ltr_training.json": generate_ltr_training_data(),
        "intent_keywords.json": INTENT_KEYWORDS,
    }

    for filename, data in datasets.items():
        path = DATA_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {path} ({len(data) if isinstance(data, list) else 'dict'} records)")


if __name__ == "__main__":
    main()
