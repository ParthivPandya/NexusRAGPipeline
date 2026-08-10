"""
NEXUS RAG — Pillar 5: Conflict Resolution Engine
=================================================

Detects when retrieved chunks contradict each other using NLI,
scores each chunk's credibility across four factors, chooses a
resolution strategy, and generates a transparent user-facing
message when sources disagree.

Strategies:
    - HIGHER_CREDIBILITY: Use the more credible source
    - MORE_RECENT: Use the more recent source
    - MORE_CORROBORATED: Use the source with more corroboration
    - SURFACE_CONFLICT: Present both views transparently to the user
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

from nexus.core.unified_encoder import Modality, UnifiedChunk

logger = logging.getLogger(__name__)


class Strategy(Enum):
    """Conflict resolution strategies."""
    HIGHER_CREDIBILITY = "use_higher_credibility"
    MORE_RECENT        = "use_more_recent"
    MORE_CORROBORATED  = "use_more_corroborated"
    SURFACE_CONFLICT   = "surface_conflict_to_user"


@dataclass
class ConflictReport:
    """Result of conflict detection and resolution."""
    detected:        bool
    conflicts:       list[dict] = field(default_factory=list)
    strategy:        Optional[Strategy] = None
    resolved_chunks: list[UnifiedChunk] = field(default_factory=list)
    user_message:    str = ""

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "detected": self.detected,
            "num_conflicts": len(self.conflicts),
            "strategy": self.strategy.value if self.strategy else None,
            "user_message": self.user_message,
        }


class ConflictResolver:
    """
    NLI-based conflict detection and multi-strategy resolution.
    Uses cross-encoder NLI model for contradiction detection
    and multi-factor credibility scoring for resolution.
    """

    CONTRADICTION_THRESHOLD = 0.70

    # Known high-authority domains
    HIGH_AUTHORITY = {
        ".gov", ".edu", "pubmed", "nature.com",
        "science.org", "arxiv.org", "sec.gov",
        "who.int", "cdc.gov", "nih.gov",
    }

    def __init__(self, nli_model: str = "cross-encoder/nli-deberta-v3-small"):
        self._nli_model_name = nli_model
        self._nli = None

    def _lazy_init_nli(self):
        """Lazy load the NLI model."""
        if self._nli is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._nli = CrossEncoder(self._nli_model_name)
            logger.info("Loaded NLI model: %s", self._nli_model_name)
        except Exception as e:
            logger.warning("NLI model unavailable: %s — using fallback", e)

    def resolve(self, chunks: list[UnifiedChunk]) -> ConflictReport:
        """
        Detect and resolve conflicts among retrieved chunks.

        Args:
            chunks: List of retrieved chunks to check for contradictions

        Returns:
            ConflictReport with resolution details
        """
        if len(chunks) < 2:
            return ConflictReport(
                detected=False,
                resolved_chunks=chunks,
            )

        # Detect contradictions
        conflicts = self._detect(chunks)

        if not conflicts:
            return ConflictReport(
                detected=False,
                resolved_chunks=chunks,
            )

        logger.info("Detected %d conflicts", len(conflicts))

        # Score credibility for all chunks
        for c in chunks:
            c.credibility_score = self._credibility(c)

        # Choose resolution strategy
        strategy = self._choose_strategy(conflicts, chunks)

        # Apply resolution
        resolved = self._apply(strategy, conflicts, chunks)

        return ConflictReport(
            detected=True,
            conflicts=conflicts,
            strategy=strategy,
            resolved_chunks=resolved,
            user_message=self._user_msg(conflicts, strategy),
        )

    def _detect(self, chunks: list[UnifiedChunk]) -> list[dict]:
        """
        Detect contradictions between text chunks using NLI.

        Returns list of conflict dicts with chunk pairs and confidence scores.
        """
        self._lazy_init_nli()

        text_chunks = [
            c for c in chunks
            if c.modality == Modality.TEXT and isinstance(c.content, str)
        ]

        if len(text_chunks) < 2:
            return []

        conflicts = []

        for i, a in enumerate(text_chunks):
            for b in text_chunks[i + 1:]:
                score = self._check_contradiction(a.content, b.content)
                if score >= self.CONTRADICTION_THRESHOLD:
                    conflicts.append({
                        "chunk_a": a,
                        "chunk_b": b,
                        "confidence": score,
                    })

        return conflicts

    def _check_contradiction(self, text_a: str, text_b: str) -> float:
        """
        Check if two texts contradict each other using NLI model.

        Returns:
            Contradiction score (0.0 to 1.0)
        """
        if self._nli is None:
            # Fallback: simple heuristic based on negation words
            return self._heuristic_contradiction(text_a, text_b)

        try:
            pairs = [[text_a[:512], text_b[:512]]]
            scores = self._nli.predict(pairs, apply_softmax=True)
            # scores shape: (1, 3) → [contradiction, neutral, entailment]
            return float(scores[0][0])
        except Exception as e:
            logger.error("NLI prediction failed: %s", e)
            return 0.0

    def _heuristic_contradiction(self, text_a: str, text_b: str) -> float:
        """
        Simple heuristic for detecting contradictions without NLI model.
        Checks for opposing numbers, negation patterns, etc.
        """
        import re

        # Extract numbers from both texts
        nums_a = set(re.findall(r"\b\d[\d,.]+\b", text_a))
        nums_b = set(re.findall(r"\b\d[\d,.]+\b", text_b))

        # If same entity but different numbers, likely contradiction
        if nums_a and nums_b and nums_a != nums_b:
            # Check if they share common non-numeric words (same topic)
            words_a = set(text_a.lower().split()) - nums_a
            words_b = set(text_b.lower().split()) - nums_b
            overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
            if overlap > 0.3:
                return 0.75

        # Check for negation patterns
        negation_words = {"not", "never", "no", "isn't", "wasn't", "aren't", "don't", "doesn't"}
        has_neg_a = bool(negation_words & set(text_a.lower().split()))
        has_neg_b = bool(negation_words & set(text_b.lower().split()))
        if has_neg_a != has_neg_b:
            return 0.60

        return 0.0

    def _credibility(self, c: UnifiedChunk) -> float:
        """
        Multi-factor credibility score.

        Factors:
            - Recency (30%): Newer sources score higher
            - Corroboration (25%): More sources confirming = higher
            - Domain authority (25%): .gov, .edu, etc. score higher
            - Consistency (20%): Fewer known contradictions = higher
        """
        recency = max(0.0, 1.0 - self._age_days(c) / 365.0)
        corroboration = min(c.metadata.get("source_count", 1) / 5.0, 1.0)
        domain_rank = self._domain_authority(c.metadata.get("source_url", ""))
        consistency = 1.0 - 0.1 * len(c.metadata.get("known_contradictions", []))
        consistency = max(0.0, consistency)

        return round(
            0.30 * recency
            + 0.25 * corroboration
            + 0.25 * domain_rank
            + 0.20 * consistency,
            3,
        )

    def _age_days(self, c: UnifiedChunk) -> float:
        """Calculate age of chunk in days from its temporal_valid_from."""
        try:
            dt = datetime.fromisoformat(c.temporal_valid_from)
            return (datetime.utcnow() - dt).days
        except (ValueError, TypeError):
            return 180.0  # Default: assume 6 months old

    def _domain_authority(self, url: str) -> float:
        """
        Heuristic domain authority score.
        Government, educational, and major research domains score highest.
        """
        if not url:
            return 0.40

        url_lower = url.lower()
        if any(d in url_lower for d in self.HIGH_AUTHORITY):
            return 0.90
        if url_lower.startswith("https://"):
            return 0.65
        return 0.40

    def _choose_strategy(self, conflicts: list[dict], chunks: list[UnifiedChunk]) -> Strategy:
        """
        Choose the best resolution strategy based on conflict characteristics.
        """
        if not conflicts:
            return Strategy.SURFACE_CONFLICT

        a = conflicts[0]["chunk_a"]
        b = conflicts[0]["chunk_b"]

        cred_gap = abs(a.credibility_score - b.credibility_score)
        age_gap = abs(self._age_days(a) - self._age_days(b))

        # If credibility gap is large, trust the more credible source
        if cred_gap > 0.25:
            return Strategy.HIGHER_CREDIBILITY

        # If age gap is large, trust the more recent source
        if age_gap > 365:
            return Strategy.MORE_RECENT

        # Otherwise, surface the conflict to the user transparently
        return Strategy.SURFACE_CONFLICT

    def _apply(
        self, strategy: Strategy, conflicts: list[dict], chunks: list[UnifiedChunk]
    ) -> list[UnifiedChunk]:
        """Apply the chosen resolution strategy."""
        if not conflicts:
            return chunks

        if strategy == Strategy.HIGHER_CREDIBILITY:
            a, b = conflicts[0]["chunk_a"], conflicts[0]["chunk_b"]
            loser = a if a.credibility_score < b.credibility_score else b
            return [c for c in chunks if c.id != loser.id]

        if strategy == Strategy.MORE_RECENT:
            a, b = conflicts[0]["chunk_a"], conflicts[0]["chunk_b"]
            older = a if self._age_days(a) > self._age_days(b) else b
            return [c for c in chunks if c.id != older.id]

        # SURFACE_CONFLICT: return all chunks, let user decide
        return chunks

    def _user_msg(self, conflicts: list[dict], strategy: Strategy) -> str:
        """Generate a user-facing message about the conflict."""
        if not conflicts:
            return ""

        a = conflicts[0]["chunk_a"]
        b = conflicts[0]["chunk_b"]

        a_text = a.content[:80] if isinstance(a.content, str) else str(a.content)[:80]
        b_text = b.content[:80] if isinstance(b.content, str) else str(b.content)[:80]

        if strategy == Strategy.SURFACE_CONFLICT:
            return (
                f"⚠️ Sources disagree: '{a_text}…' vs '{b_text}…'. "
                f"Both views are presented; verify independently."
            )

        if strategy == Strategy.HIGHER_CREDIBILITY:
            winner = a if a.credibility_score > b.credibility_score else b
            return (
                f"ℹ️ A conflict was detected and resolved by credibility score "
                f"({winner.credibility_score:.2f}). The more credible source was used."
            )

        if strategy == Strategy.MORE_RECENT:
            return "ℹ️ A conflict was detected and resolved by recency — the newer source was used."

        return ""
