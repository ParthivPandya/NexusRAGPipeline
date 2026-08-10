"""
NEXUS RAG — Pillar 3: Query DNA Classifier
===========================================

Classifies every incoming query across 10 semantic dimensions
simultaneously, then computes exact retrieval strategy weights.

Dimensions:
    1. factual       — "What is X?"
    2. analytical    — "Why did X happen?"
    3. temporal      — "What was X in 2020?"
    4. procedural    — "How do I do X?"
    5. comparative   — "X vs Y?"
    6. causal        — "What caused X?"
    7. adversarial   — "Argue for X"
    8. creative      — "Imagine if X…"
    9. counterfactual — "What if X hadn't occurred?"
    10. multilingual — Cross-language retrieval required
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QueryDNA:
    """
    Complete query profile with 10 semantic dimensions
    and computed retrieval strategy weights.
    """
    raw_query:         str
    normalized_query:  str
    detected_language: str = "en"         # ISO 639-1

    # 10 semantic dimensions — each 0.0 to 1.0
    factual:           float = 0.0
    analytical:        float = 0.0
    temporal:          float = 0.0
    procedural:        float = 0.0
    comparative:       float = 0.0
    causal:            float = 0.0
    adversarial:       float = 0.0
    creative:          float = 0.0
    counterfactual:    float = 0.0
    multilingual:      float = 0.0

    # Computed retrieval weights — sum to 1.0
    retrieval_strategy: dict = field(default_factory=dict)

    @property
    def dominant_dimension(self) -> str:
        """Return the highest-scoring dimension."""
        dims = {
            "factual": self.factual, "analytical": self.analytical,
            "temporal": self.temporal, "procedural": self.procedural,
            "comparative": self.comparative, "causal": self.causal,
            "adversarial": self.adversarial, "creative": self.creative,
            "counterfactual": self.counterfactual, "multilingual": self.multilingual,
        }
        return max(dims, key=dims.get)

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "raw_query": self.raw_query,
            "normalized_query": self.normalized_query,
            "detected_language": self.detected_language,
            "dimensions": {
                "factual": self.factual, "analytical": self.analytical,
                "temporal": self.temporal, "procedural": self.procedural,
                "comparative": self.comparative, "causal": self.causal,
                "adversarial": self.adversarial, "creative": self.creative,
                "counterfactual": self.counterfactual, "multilingual": self.multilingual,
            },
            "dominant_dimension": self.dominant_dimension,
            "retrieval_strategy": self.retrieval_strategy,
        }


class QueryDNAClassifier:
    """
    Classify queries across 10 semantic dimensions and compute
    optimal retrieval strategy weights per query.
    """

    # ── Signal Word Lists ─────────────────────────────────────

    TEMPORAL_SIGNALS = [
        "in 2020", "in 2021", "in 2022", "in 2023", "in 2024", "in 2025", "in 2026",
        "last year", "back in", "at the time", "historically", "previously",
        "used to", "when did", "since when", "years ago", "decades ago",
        "in the past", "before", "after", "during",
    ]

    CAUSAL_SIGNALS = [
        "caused", "why did", "led to", "resulted in", "reason for",
        "triggered by", "consequence of", "what caused", "because of",
        "root cause", "contributing factor", "origin of",
    ]

    COUNTERFACTUAL_SIGNALS = [
        "what if", "if hadn't", "had not", "suppose", "imagine",
        "would have", "hypothetically", "in an alternate", "assuming",
        "if instead", "what would happen",
    ]

    PROCEDURAL_SIGNALS = [
        "how to", "how do i", "steps to", "guide for", "tutorial",
        "walk me through", "show me how", "instructions for",
        "step by step", "procedure for", "method to", "way to",
    ]

    COMPARATIVE_SIGNALS = [
        " vs ", " versus ", "compare", "difference between",
        "which is better", "pros and cons", "advantages of",
        "disadvantages of", "similarities between", "contrast",
        "compared to", " or ", "which one",
    ]

    ADVERSARIAL_SIGNALS = [
        "argue for", "argue against", "defend", "case for",
        "case against", "debate", "justify", "convince",
        "persuade", "advocate",
    ]

    CREATIVE_SIGNALS = [
        "imagine", "creative", "story", "invent", "design",
        "brainstorm", "novel approach", "innovative", "unique way",
        "what could", "envision",
    ]

    ANALYTICAL_SIGNALS = [
        "why", "how", "explain", "analyze", "analyse", "interpret",
        "evaluate", "assess", "examine", "investigate", "understand",
        "reason behind", "mechanism of", "implications of",
    ]

    def __init__(self, encoder=None):
        self._encoder = encoder

    def classify(self, query: str) -> QueryDNA:
        """
        Classify a query into 10 semantic dimensions and
        compute retrieval strategy weights.

        Args:
            query: The raw query string

        Returns:
            QueryDNA with all dimensions and strategy weights
        """
        lang = self._detect_language(query)
        norm = query.lower().strip()

        # Score all 10 dimensions
        dims = self._score_dimensions(norm)

        # Compute retrieval strategy weights from dimensions
        strategy = self._compute_strategy(dims)

        return QueryDNA(
            raw_query=query,
            normalized_query=norm,
            detected_language=lang,
            retrieval_strategy=strategy,
            **dims,
        )

    def _detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return "en"

    def _score_dimensions(self, query: str) -> dict:
        """Score all 10 semantic dimensions for the query."""
        q = query.lower()

        temporal = min(sum(1 for s in self.TEMPORAL_SIGNALS if s in q) * 0.4, 1.0)
        causal = min(sum(1 for s in self.CAUSAL_SIGNALS if s in q) * 0.5, 1.0)
        counterfactual = min(sum(1 for s in self.COUNTERFACTUAL_SIGNALS if s in q) * 0.6, 1.0)
        procedural = min(sum(1 for s in self.PROCEDURAL_SIGNALS if s in q) * 0.5, 1.0)
        comparative = min(sum(1 for s in self.COMPARATIVE_SIGNALS if s in q) * 0.5, 1.0)
        adversarial = 0.8 if any(w in q for w in self.ADVERSARIAL_SIGNALS) else 0.0
        creative = 0.8 if any(w in q for w in self.CREATIVE_SIGNALS) else 0.0

        # Factual: question-word queries
        factual = 0.8 if q.startswith(("what ", "who ", "when ", "where ")) else 0.3

        # Analytical: explanation-seeking
        analytical = min(sum(1 for s in self.ANALYTICAL_SIGNALS if s in q) * 0.35, 1.0)
        if analytical < 0.3 and any(w in q for w in ["why", "how", "explain"]):
            analytical = 0.7

        # Multilingual: non-English query
        multilingual = 0.9 if self._detect_language(query) != "en" else 0.0

        return {
            "factual": round(factual, 3),
            "analytical": round(analytical, 3),
            "temporal": round(temporal, 3),
            "procedural": round(procedural, 3),
            "comparative": round(comparative, 3),
            "causal": round(causal, 3),
            "adversarial": round(adversarial, 3),
            "creative": round(creative, 3),
            "counterfactual": round(counterfactual, 3),
            "multilingual": round(multilingual, 3),
        }

    def _compute_strategy(self, dims: dict) -> dict:
        """
        Combine dimension scores into retriever activation weights.
        Weights sum to 1.0 — each retriever gets a proportional share.
        """
        raw = {
            "dense_hnsw":     0.40 * dims["factual"] + 0.20 * dims["analytical"],
            "bm25_sparse":    0.40 * dims["factual"] + 0.20 * dims["procedural"],
            "causal_graph":   0.80 * dims["causal"] + 0.70 * dims["counterfactual"],
            "temporal_index": 0.90 * dims["temporal"],
            "multi_branch":   0.80 * dims["comparative"],
            "xrag_bridge":    0.90 * dims["multilingual"],
            "fuzzy_creative": 0.80 * dims["creative"],
        }

        total = sum(raw.values()) + 1e-9
        return {k: round(v / total, 4) for k, v in raw.items()}
