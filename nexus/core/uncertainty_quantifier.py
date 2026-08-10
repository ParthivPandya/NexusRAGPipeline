"""
NEXUS RAG — Pillar 6: Calibrated Uncertainty Quantifier
=======================================================

Every factual claim in the generated answer gets an individual
confidence score computed from:
    - Source count (25%)
    - Recency (20%)
    - Agreement ratio (25%)
    - Domain authority (15%)
    - Semantic support (15%)

Labels: High (≥0.80) | Medium (0.60–0.80) | Low (0.40–0.60) | Uncertain (<0.40)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from nexus.core.unified_encoder import Modality, UnifiedChunk

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceClaim:
    """A single claim with its computed confidence score."""
    claim_text:      str
    confidence:      float       # 0.0 – 1.0
    label:           str         # "High" | "Medium" | "Low" | "Uncertain"
    source_count:    int = 0
    recency_days:    float = 0.0
    agreement_ratio: float = 0.0
    authority_score: float = 0.0
    chunk_ids:       list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "text": self.claim_text,
            "confidence": self.confidence,
            "label": self.label,
            "source_count": self.source_count,
            "recency_days": self.recency_days,
            "agreement_ratio": self.agreement_ratio,
            "authority_score": self.authority_score,
        }


class UncertaintyQuantifier:
    """
    Per-claim confidence scoring based on multi-factor analysis.
    Each factual claim gets an individual score so users see
    exactly how reliable each statement is.
    """

    THRESHOLDS = {
        "High":      (0.80, 1.01),
        "Medium":    (0.60, 0.80),
        "Low":       (0.40, 0.60),
        "Uncertain": (0.00, 0.40),
    }

    def __init__(self, embedding_model: str = "intfloat/e5-large-v2"):
        self._model_name = embedding_model
        self._model = None

    def _lazy_init(self):
        """Lazy load the embedding model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("UncertaintyQuantifier loaded model: %s", self._model_name)
        except Exception as e:
            logger.warning("Embedding model unavailable: %s", e)

    def quantify(
        self, claim: str, chunks: list[UnifiedChunk]
    ) -> ConfidenceClaim:
        """
        Compute confidence for a single claim against retrieved chunks.

        Args:
            claim: The factual claim to score
            chunks: Retrieved chunks that may support the claim

        Returns:
            ConfidenceClaim with multi-factor confidence score
        """
        if not chunks:
            return ConfidenceClaim(
                claim_text=claim,
                confidence=0.0,
                label="Uncertain",
                source_count=0,
                recency_days=999.0,
                agreement_ratio=0.0,
                authority_score=0.0,
                chunk_ids=[],
            )

        source_n = len(chunks)
        recency = self._avg_recency_days(chunks)
        agreement = self._agreement_ratio(claim, chunks)
        authority = float(np.mean([c.credibility_score for c in chunks]))
        sem_support = self._avg_semantic_support(claim, chunks)

        # Weighted combination
        confidence = (
            0.25 * min(source_n / 5.0, 1.0)           # 5+ sources = max
            + 0.20 * max(1.0 - recency / 365.0, 0.0)  # Fresher = better
            + 0.25 * agreement
            + 0.15 * authority
            + 0.15 * sem_support
        )
        confidence = round(min(max(confidence, 0.0), 1.0), 3)

        # Determine label
        label = "Uncertain"
        for lbl, (lo, hi) in self.THRESHOLDS.items():
            if lo <= confidence < hi:
                label = lbl
                break

        return ConfidenceClaim(
            claim_text=claim,
            confidence=confidence,
            label=label,
            source_count=source_n,
            recency_days=round(recency, 1),
            agreement_ratio=round(agreement, 3),
            authority_score=round(authority, 3),
            chunk_ids=[c.id for c in chunks],
        )

    def quantify_claims(
        self, claims: list[str], chunks: list[UnifiedChunk]
    ) -> list[ConfidenceClaim]:
        """Quantify confidence for multiple claims at once."""
        return [self.quantify(claim, chunks) for claim in claims]

    def _avg_recency_days(self, chunks: list[UnifiedChunk]) -> float:
        """Average age of chunks in days."""
        ages = []
        for c in chunks:
            try:
                dt = datetime.fromisoformat(c.temporal_valid_from)
                ages.append((datetime.utcnow() - dt).days)
            except (ValueError, TypeError):
                ages.append(180)  # Default: 6 months
        return float(np.mean(ages)) if ages else 180.0

    def _agreement_ratio(self, claim: str, chunks: list[UnifiedChunk]) -> float:
        """Fraction of text chunks that semantically support the claim."""
        self._lazy_init()
        if self._model is None:
            return 0.5  # Neutral fallback

        text_chunks = [
            c for c in chunks
            if c.modality == Modality.TEXT and isinstance(c.content, str)
        ]
        if not text_chunks:
            return 0.0

        try:
            c_emb = self._model.encode(claim)
            supporting = sum(
                1 for c in text_chunks
                if float(np.dot(c_emb, self._model.encode(c.content[:256]))) > 0.65
            )
            return supporting / max(len(text_chunks), 1)
        except Exception as e:
            logger.error("Agreement ratio calculation failed: %s", e)
            return 0.5

    def _avg_semantic_support(self, claim: str, chunks: list[UnifiedChunk]) -> float:
        """Average semantic similarity between claim and chunk contents."""
        self._lazy_init()
        if self._model is None:
            return 0.5

        text_chunks = [
            c for c in chunks
            if c.modality == Modality.TEXT and isinstance(c.content, str)
        ]
        if not text_chunks:
            return 0.0

        try:
            c_emb = self._model.encode(claim)
            scores = [
                float(np.dot(c_emb, self._model.encode(c.content[:256])))
                for c in text_chunks
            ]
            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logger.error("Semantic support calculation failed: %s", e)
            return 0.5
