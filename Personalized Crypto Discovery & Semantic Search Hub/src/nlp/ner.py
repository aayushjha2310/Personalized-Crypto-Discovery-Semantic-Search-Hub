"""spaCy NER + crypto entity dictionary for query understanding."""
import re
from typing import Dict, List, Set

import spacy


class CryptoNER:
    """Extract token names, symbols, and blockchain protocols from queries."""

    PROTOCOL_PATTERNS = {
        "ethereum": ["ethereum", "eth", "evm"],
        "solana": ["solana", "sol"],
        "avalanche": ["avalanche", "avax"],
        "polygon": ["polygon", "matic"],
        "cosmos": ["cosmos", "atom", "ibc"],
        "polkadot": ["polkadot", "dot", "parachain"],
        "bitcoin": ["bitcoin", "btc"],
        "arbitrum": ["arbitrum", "arb"],
        "optimism": ["optimism", "op"],
        "near": ["near protocol", "near"],
    }

    CATEGORY_PATTERNS = {
        "layer1": ["layer 1", "layer1", "l1"],
        "layer2": ["layer 2", "layer2", "l2", "rollup"],
        "defi": ["defi", "dex", "lending", "yield", "amm"],
        "metaverse": ["metaverse", "virtual world", "virtual land"],
        "nft": ["nft", "nfts", "digital art"],
        "gaming": ["gaming", "play to earn", "game"],
        "meme": ["meme", "meme coin"],
    }

    def __init__(self, tokens: List[dict]):
        self.tokens = tokens
        self.symbol_map = {t["symbol"].lower(): t for t in tokens}
        self.name_map = {t["name"].lower(): t for t in tokens}
        self.id_map = {t["id"]: t for t in tokens}
        self._nlp = None

    def _load_spacy(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def extract_entities(self, query: str) -> Dict[str, List]:
        """Hybrid NER: spaCy entities + crypto dictionary matching."""
        query_lower = query.lower()
        nlp = self._load_spacy()
        doc = nlp(query)

        found_tokens: Set[str] = set()
        found_protocols: Set[str] = set()
        found_categories: Set[str] = set()
        spacy_entities = []

        # spaCy NER entities (ORG, PRODUCT, etc.)
        for ent in doc.ents:
            spacy_entities.append({"text": ent.text, "label": ent.label_})
            ent_lower = ent.text.lower()
            if ent_lower in self.name_map:
                found_tokens.add(self.name_map[ent_lower]["id"])
            if ent_lower in self.symbol_map:
                found_tokens.add(self.symbol_map[ent_lower]["id"])

        # Dictionary-based token matching
        for token in self.tokens:
            patterns = [
                token["name"].lower(),
                token["symbol"].lower(),
                token["id"].lower(),
            ] + token.get("tags", [])
            for pattern in patterns:
                if re.search(rf"\b{re.escape(pattern)}\b", query_lower):
                    found_tokens.add(token["id"])

        # Protocol extraction
        for protocol, keywords in self.PROTOCOL_PATTERNS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", query_lower):
                    found_protocols.add(protocol)
                    if kw in self.symbol_map:
                        found_tokens.add(self.symbol_map[kw]["id"])
                    if kw in self.name_map:
                        found_tokens.add(self.name_map[kw]["id"])

        # Category / theme extraction
        for category, keywords in self.CATEGORY_PATTERNS.items():
            for kw in keywords:
                if kw in query_lower:
                    found_categories.add(category)

        return {
            "tokens": list(found_tokens),
            "protocols": list(found_protocols),
            "categories": list(found_categories),
            "spacy_entities": spacy_entities,
        }
