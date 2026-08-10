"""
NEXUS RAG — Gap 8: Failure Forensics Engine
============================================

Research: RAGChecker (Ru et al., NeurIPS 2024), arXiv 2408.02854.

Diagnoses exactly WHY an answer was wrong:
    - RETRIEVAL:     Correct info not retrieved
    - UTILIZATION:   Retrieved but not used (lost-in-the-middle)
    - COMPREHENSION: Fact misread (wrong number, date, name)
    - REASONING:     Multi-hop logic broke
    - PARAMETRIC:    LLM's own weights overrode context
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from nexus.core.unified_encoder import Modality, UnifiedChunk

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    RETRIEVAL     = "retrieval_failure"
    UTILIZATION   = "context_utilization"
    COMPREHENSION = "comprehension_failure"
    REASONING     = "reasoning_failure"
    PARAMETRIC    = "parametric_knowledge"
    NONE          = "no_failure"


@dataclass
class Diagnosis:
    """Forensic diagnosis of a failed answer."""
    mode:             FailureMode
    description:      str
    confidence:       float
    recommended_fix:  str
    component_to_fix: str

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "description": self.description,
            "confidence": self.confidence,
            "recommended_fix": self.recommended_fix,
            "component_to_fix": self.component_to_fix,
        }


class FailureForensicsEngine:
    """
    Diagnoses exactly WHY an answer was wrong — not just THAT it was wrong.
    Uses a cascade of 5 tests to attribute the error to a specific component.
    """

    def __init__(self, nli_model: str = "cross-encoder/nli-deberta-v3-small",
                 emb_model: str = "intfloat/e5-large-v2"):
        self._nli_model_name = nli_model
        self._emb_model_name = emb_model
        self._nli = None
        self._emb = None

    def _lazy_init(self):
        if self._nli is None:
            try:
                from sentence_transformers import CrossEncoder
                self._nli = CrossEncoder(self._nli_model_name)
            except Exception as e:
                logger.warning("NLI model unavailable: %s", e)
        if self._emb is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._emb = SentenceTransformer(self._emb_model_name)
            except Exception as e:
                logger.warning("Embedding model unavailable: %s", e)

    def diagnose(
        self,
        query: str,
        ground_truth: str,
        generated_answer: str,
        retrieved_chunks: list[UnifiedChunk],
        corpus_sample: Optional[list[UnifiedChunk]] = None,
    ) -> Diagnosis:
        """
        Run cascade of 5 diagnostic tests.

        Args:
            query: Original user query
            ground_truth: Known correct answer (from user feedback)
            generated_answer: The (incorrect) generated answer
            retrieved_chunks: Chunks that were retrieved
            corpus_sample: Broader sample from corpus for retrieval check
        """
        self._lazy_init()
        corpus_sample = corpus_sample or []

        # Test 1: Is correct info in retrieved chunks?
        if not self._info_in_chunks(ground_truth, retrieved_chunks):
            if corpus_sample and self._info_in_chunks(ground_truth, corpus_sample):
                return Diagnosis(
                    mode=FailureMode.RETRIEVAL,
                    description="Correct information exists in corpus but was not in top-k results",
                    confidence=0.90,
                    recommended_fix="Improve embedding model or lower similarity threshold",
                    component_to_fix="AdaptiveRetrievalRouter",
                )

        # Test 2: Was retrieved content used in the answer?
        if not self._content_utilized(retrieved_chunks, generated_answer):
            return Diagnosis(
                mode=FailureMode.UTILIZATION,
                description="Correct chunk was retrieved but LLM ignored it (lost-in-the-middle)",
                confidence=0.85,
                recommended_fix="Reorder context: place high-value chunks at start or end",
                component_to_fix="ContextAssembler",
            )

        # Test 3: Was a specific fact misread?
        if not self._facts_read_correctly(retrieved_chunks, generated_answer):
            return Diagnosis(
                mode=FailureMode.COMPREHENSION,
                description="LLM misread a specific fact (wrong number, date, or name)",
                confidence=0.80,
                recommended_fix="Add structured fact-extraction step before final answer",
                component_to_fix="SelfHealingVerifier",
            )

        # Test 4: Did multi-hop reasoning fail?
        if not self._reasoning_correct(query, retrieved_chunks, generated_answer):
            return Diagnosis(
                mode=FailureMode.REASONING,
                description="Multi-hop reasoning chain broke — could not connect retrieved facts",
                confidence=0.75,
                recommended_fix="Add chain-of-thought reasoning verification before final output",
                component_to_fix="CausalCounterfactualLayer",
            )

        # Test 5: LLM parametric knowledge error
        return Diagnosis(
            mode=FailureMode.PARAMETRIC,
            description="LLM's own training knowledge overrode the retrieved context",
            confidence=0.70,
            recommended_fix="Increase system prompt weighting on context over parametric knowledge",
            component_to_fix="GenerationLayer",
        )

    def _info_in_chunks(
        self, truth: str, chunks: list[UnifiedChunk], threshold: float = 0.65
    ) -> bool:
        """Check if the truth is semantically present in any chunk."""
        if self._emb is None:
            # Fallback: keyword overlap
            truth_words = set(truth.lower().split())
            for c in chunks:
                if isinstance(c.content, str):
                    chunk_words = set(c.content.lower().split())
                    overlap = len(truth_words & chunk_words) / max(len(truth_words), 1)
                    if overlap > 0.3:
                        return True
            return False

        import numpy as np
        t_emb = self._emb.encode(truth)
        return any(
            float(np.dot(t_emb, self._emb.encode(c.content[:256]))) >= threshold
            for c in chunks
            if c.modality == Modality.TEXT and isinstance(c.content, str)
        )

    def _content_utilized(
        self, chunks: list[UnifiedChunk], answer: str, threshold: float = 0.40
    ) -> bool:
        """Check if retrieved content was actually used in the answer."""
        if self._emb is None:
            return True  # Can't determine, assume utilized

        import numpy as np
        a_emb = self._emb.encode(answer)
        return any(
            float(np.dot(a_emb, self._emb.encode(c.content[:256]))) >= threshold
            for c in chunks
            if c.modality == Modality.TEXT and isinstance(c.content, str)
        )

    def _facts_read_correctly(
        self, chunks: list[UnifiedChunk], answer: str
    ) -> bool:
        """Check for numeric/date hallucinations."""
        chunk_text = " ".join(
            c.content for c in chunks
            if c.modality == Modality.TEXT and isinstance(c.content, str)
        )
        chunk_nums = set(re.findall(r"\b\d[\d,.]+\b", chunk_text))
        answer_nums = set(re.findall(r"\b\d[\d,.]+\b", answer))
        invented = answer_nums - chunk_nums
        return len(invented) == 0

    def _reasoning_correct(
        self, query: str, chunks: list[UnifiedChunk], answer: str
    ) -> bool:
        """Check if answer can be logically derived from chunks."""
        if self._nli is None:
            return True  # Can't determine

        combined = " ".join(
            c.content[:256] for c in chunks[:5]
            if c.modality == Modality.TEXT and isinstance(c.content, str)
        )
        if not combined:
            return True

        try:
            scores = self._nli.predict([[combined, answer]], apply_softmax=True)
            return float(scores[0][2]) > 0.45  # entailment score
        except Exception:
            return True
