"""
NEXUS RAG — Gap 7: Modality-Aware Reranker
==========================================

Research: CoRe-MMRAG (arXiv 2506.02544, 2025)
Boosts the modality best suited to each query type.
"""

from __future__ import annotations

import logging
from datetime import datetime

from nexus.core.unified_encoder import Modality, UnifiedChunk
from nexus.core.query_classifier import QueryDNA

logger = logging.getLogger(__name__)


class ModalityAwareReranker:
    """
    Cross-encoder reranker that accounts for content modality.
    Each query type has optimal modality weights.
    """

    QUERY_MODALITY_WEIGHTS = {
        "factual":      {"text": 0.70, "table": 0.80, "code": 0.40, "image": 0.30},
        "quantitative": {"table": 0.90, "text": 0.50, "image": 0.40, "code": 0.30},
        "procedural":   {"text": 0.80, "code": 0.90, "image": 0.60, "table": 0.30},
        "spatial":      {"image": 0.90, "text": 0.50, "table": 0.30, "code": 0.20},
        "temporal":     {"text": 0.80, "table": 0.70, "image": 0.40, "code": 0.20},
        "causal":       {"text": 0.90, "table": 0.60, "image": 0.30, "code": 0.20},
        "comparative":  {"table": 0.85, "text": 0.70, "image": 0.50, "code": 0.30},
        "creative":     {"image": 0.70, "text": 0.70, "code": 0.40, "table": 0.20},
    }

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._cross_enc = None

    def _lazy_init(self):
        if self._cross_enc is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._cross_enc = CrossEncoder(self._model_name)
        except Exception as e:
            logger.warning("Cross-encoder unavailable: %s", e)

    def rerank(
        self,
        query: str,
        dna: QueryDNA,
        chunks: list[UnifiedChunk],
        top_k: int = 5,
    ) -> list[UnifiedChunk]:
        """
        Rerank chunks using modality-aware scoring.

        Final score = 0.45 * semantic + 0.25 * modality_bonus
                    + 0.15 * freshness + 0.15 * credibility
        """
        self._lazy_init()

        q_type = self._dominant_type(dna)
        weights = self.QUERY_MODALITY_WEIGHTS.get(q_type, {})

        scored = []
        for c in chunks:
            # Cross-encoder semantic score
            text_repr = c.content if isinstance(c.content, str) else str(c.content)
            sem_score = self._semantic_score(query, text_repr)

            modality_bonus = weights.get(c.modality.value, 0.50)
            freshness = self._freshness(c)
            credibility = c.credibility_score

            final = (
                0.45 * sem_score
                + 0.25 * modality_bonus
                + 0.15 * freshness
                + 0.15 * credibility
            )
            scored.append((final, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [c for _, c in scored[:top_k]]

        logger.debug(
            "Reranked %d → %d chunks (query_type=%s)",
            len(chunks), len(result), q_type,
        )
        return result

    def _semantic_score(self, query: str, text: str) -> float:
        """Compute semantic relevance score using cross-encoder."""
        if self._cross_enc is None:
            # Fallback: simple keyword overlap
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            return len(q_words & t_words) / max(len(q_words), 1)

        try:
            score = float(self._cross_enc.predict([[query, text[:512]]])[0])
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    def _dominant_type(self, dna: QueryDNA) -> str:
        """Get the dominant query type from DNA dimensions."""
        candidates = {
            "factual": dna.factual,
            "procedural": dna.procedural,
            "causal": dna.causal,
            "temporal": dna.temporal,
            "comparative": dna.comparative,
            "creative": dna.creative,
        }
        return max(candidates, key=candidates.get)

    def _freshness(self, c: UnifiedChunk) -> float:
        """Compute freshness score for streaming chunks."""
        if c.metadata.get("is_streaming"):
            try:
                age_h = (
                    datetime.utcnow()
                    - datetime.fromisoformat(c.metadata["freshness_ts"])
                ).total_seconds() / 3600
                return max(1.0 - age_h / 24.0, 0.0)
            except (ValueError, KeyError):
                return 0.5
        return 0.5
