"""
NEXUS RAG — Gap 1: Epistemic Sufficiency Engine
================================================

Research: Ghafouri et al. 2025 "Epistemic Mismatch"
           Entropic Claim Resolution (arXiv 2603.28444, March 2026)

Computes Shannon entropy over competing answer hypotheses.
- Low entropy  → ANSWER confidently
- Mid entropy  → RETRIEVE_MORE
- High entropy → ABSTAIN and recommend better sources
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from nexus.core.unified_encoder import Modality, UnifiedChunk

logger = logging.getLogger(__name__)


class EpistemicDecision(Enum):
    ANSWER        = "answer"
    ABSTAIN       = "abstain"
    RETRIEVE_MORE = "retrieve_more"


@dataclass
class EpistemicReport:
    """Result of epistemic sufficiency evaluation."""
    decision:          EpistemicDecision
    entropy:           float
    epsilon:           float
    reason:            str
    suggested_sources: list[str] = field(default_factory=list)
    partial_evidence:  list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "entropy": self.entropy,
            "epsilon": self.epsilon,
            "reason": self.reason,
            "suggested_sources": self.suggested_sources,
            "partial_evidence": self.partial_evidence,
        }


class EpistemicSufficiencyEngine:
    """
    Shannon entropy-based epistemic sufficiency evaluation.
    Terminates dynamically when epistemically sufficient,
    not after fixed top-k.

    Research: Entropic Claim Resolution — arXiv 2603.28444, March 2026.
    """

    def __init__(
        self,
        epsilon: float = 0.15,
        max_entropy: float = 0.85,
        model_name: str = "intfloat/e5-large-v2",
    ):
        self.epsilon = epsilon          # ≤ this → answer
        self.max_entropy = max_entropy  # ≥ this → abstain
        self._model_name = model_name
        self._model = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except Exception as e:
            logger.warning("Embedding model unavailable: %s", e)

    def evaluate(
        self,
        query: str,
        evidence: list[UnifiedChunk],
        hypotheses: list[str],
    ) -> EpistemicReport:
        """
        Evaluate epistemic sufficiency of evidence for answering the query.

        Args:
            query: The user's question
            evidence: Retrieved context chunks
            hypotheses: Candidate answers sampled from LLM

        Returns:
            EpistemicReport with ANSWER/ABSTAIN/RETRIEVE_MORE decision
        """
        H = self._entropy(hypotheses, evidence)

        if H <= self.epsilon:
            return EpistemicReport(
                decision=EpistemicDecision.ANSWER,
                entropy=round(H, 4),
                epsilon=self.epsilon,
                reason=f"Entropy {H:.4f} ≤ {self.epsilon} — sufficient evidence",
                suggested_sources=[],
                partial_evidence=[],
            )

        if H >= self.max_entropy:
            return EpistemicReport(
                decision=EpistemicDecision.ABSTAIN,
                entropy=round(H, 4),
                epsilon=self.epsilon,
                reason=f"Entropy {H:.4f} ≥ {self.max_entropy} — no signal found",
                suggested_sources=self._suggest_sources(query),
                partial_evidence=self._best_partial(evidence, query),
            )

        return EpistemicReport(
            decision=EpistemicDecision.RETRIEVE_MORE,
            entropy=round(H, 4),
            epsilon=self.epsilon,
            reason=f"Entropy {H:.4f} — need more evidence",
            suggested_sources=[],
            partial_evidence=[],
        )

    def _entropy(
        self, hypotheses: list[str], evidence: list[UnifiedChunk]
    ) -> float:
        """
        Compute normalized Shannon entropy over hypothesis distribution.
        Each hypothesis is weighted by its average similarity to evidence.
        """
        if not hypotheses:
            return 1.0

        self._lazy_init()
        if self._model is None:
            # Fallback: uniform entropy (maximum uncertainty)
            return 0.5

        try:
            h_embs = self._model.encode(hypotheses)
            e_texts = [
                c.content[:256] for c in evidence
                if c.modality == Modality.TEXT and isinstance(c.content, str)
            ]

            if not e_texts:
                return 1.0

            e_embs = self._model.encode(e_texts)

            # Support score = avg cosine sim vs evidence
            support = np.array([
                float(np.mean([np.dot(h, e) for e in e_embs]))
                for h in h_embs
            ])
            support = np.clip(support, 1e-9, None)
            probs = support / support.sum()

            # Shannon entropy
            H = float(-np.sum(probs * np.log2(probs + 1e-12)))
            max_H = math.log2(len(hypotheses)) if len(hypotheses) > 1 else 1.0

            # Normalize to [0, 1]
            return H / max_H if max_H > 0 else 1.0

        except Exception as e:
            logger.error("Entropy calculation failed: %s", e)
            return 0.5

    def _suggest_sources(self, query: str) -> list[str]:
        """Suggest domain-specific external sources when abstaining."""
        domain_map = {
            ("drug", "medicine", "clinical", "dose", "treatment"):
                "https://pubmed.ncbi.nlm.nih.gov",
            ("law", "regulation", "statute", "court", "legal"):
                "https://www.westlaw.com",
            ("stock", "sec", "filing", "earnings", "revenue", "financial"):
                "https://www.sec.gov/edgar",
            ("research", "paper", "arxiv", "study", "academic"):
                "https://arxiv.org",
            ("code", "programming", "software", "api", "library"):
                "https://stackoverflow.com",
        }
        q = query.lower()
        return [url for kws, url in domain_map.items() if any(k in q for k in kws)]

    def _best_partial(
        self, evidence: list[UnifiedChunk], query: str
    ) -> list[str]:
        """Return the most relevant partial evidence even when abstaining."""
        if not evidence:
            return []

        self._lazy_init()
        if self._model is None:
            # Fallback: return first 3 chunks
            return [
                c.content[:200] for c in evidence[:3]
                if isinstance(c.content, str)
            ]

        try:
            q_emb = self._model.encode(query)
            scored = sorted(
                [
                    (float(np.dot(q_emb, self._model.encode(c.content[:256]))), c)
                    for c in evidence
                    if c.modality == Modality.TEXT and isinstance(c.content, str)
                ],
                reverse=True,
            )
            return [c.content[:200] for _, c in scored[:3]]
        except Exception:
            return [
                c.content[:200] for c in evidence[:3]
                if isinstance(c.content, str)
            ]
