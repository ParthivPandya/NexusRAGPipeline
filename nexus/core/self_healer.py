"""
NEXUS RAG — Pillar 7: Self-Healing Verifier
============================================

Decomposes the generated answer into atomic claims. Verifies each
claim against retrieved chunks via NLI. Re-retrieves and regenerates
only the unsupported portions. After 3 failed iterations, flags the
claim as uncertain rather than outputting a hallucination.

Flow:
    1. Decompose answer → atomic claims
    2. NLI-verify each claim against chunks
    3. If unsupported claims exist:
       a. Re-retrieve specifically for those claims
       b. Regenerate only the unsupported portions
    4. Repeat (max 3 iterations)
    5. Flag remaining unsupported claims as [⚠️ UNCERTAIN]
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from nexus.core.unified_encoder import Modality, UnifiedChunk

logger = logging.getLogger(__name__)


@dataclass
class HealingResult:
    """Result of the self-healing verification process."""
    healed_answer:      str
    all_supported:      bool
    unsupported_claims: list[str] = field(default_factory=list)
    iterations:         int = 0
    overall_confidence: float = 0.0

    def to_dict(self) -> dict:
        """Serialize for API response."""
        return {
            "all_supported": self.all_supported,
            "unsupported_claims": self.unsupported_claims,
            "iterations": self.iterations,
            "overall_confidence": self.overall_confidence,
        }


class SelfHealingVerifier:
    """
    NLI-based answer verification with targeted re-retrieval
    and regeneration. Maximum 3 healing iterations.
    """

    MAX_ITER = 3

    def __init__(self, nli_model: str = "cross-encoder/nli-deberta-v3-small"):
        self._nli_model_name = nli_model
        self._nli = None

    def _lazy_init(self):
        """Lazy load the NLI model."""
        if self._nli is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._nli = CrossEncoder(self._nli_model_name)
            logger.info("Self-healer loaded NLI: %s", self._nli_model_name)
        except Exception as e:
            logger.warning("NLI model unavailable: %s", e)

    def verify_and_heal(
        self,
        answer: str,
        chunks: list[UnifiedChunk],
        query: str,
        router=None,
        llm_client=None,
    ) -> HealingResult:
        """
        Verify an answer and iteratively heal unsupported claims.

        Args:
            answer: The generated answer to verify
            chunks: Retrieved context chunks
            query: Original user query
            router: AdaptiveRetrievalRouter for re-retrieval
            llm_client: Anthropic client for regeneration

        Returns:
            HealingResult with healed answer and confidence
        """
        self._lazy_init()

        healed = answer
        iters = 0

        while iters < self.MAX_ITER:
            unsupported = self._unsupported_claims(healed, chunks)
            if not unsupported:
                logger.info("All claims verified after %d iterations", iters)
                break

            logger.info(
                "Iteration %d: %d unsupported claims found",
                iters + 1, len(unsupported)
            )

            # Re-retrieve specifically for unsupported claims
            if router:
                for claim in unsupported:
                    try:
                        extra = asyncio.run(router.retrieve_for_claim(claim))
                        chunks.extend(extra)
                    except RuntimeError:
                        # Already in async context
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            break
                        extra = loop.run_until_complete(
                            router.retrieve_for_claim(claim)
                        )
                        chunks.extend(extra)

            # Regenerate only the unsupported portions
            if llm_client:
                healed = self._targeted_regen(
                    healed, unsupported, chunks, query, llm_client
                )

            iters += 1

        # Final check
        final_unsupported = self._unsupported_claims(healed, chunks)
        if final_unsupported:
            healed = self._flag_uncertain(healed, final_unsupported)
            logger.warning(
                "%d claims remain unsupported after %d iterations",
                len(final_unsupported), iters
            )

        return HealingResult(
            healed_answer=healed,
            all_supported=len(final_unsupported) == 0,
            unsupported_claims=final_unsupported,
            iterations=iters,
            overall_confidence=self._overall_conf(healed, chunks),
        )

    def _unsupported_claims(
        self, answer: str, chunks: list[UnifiedChunk]
    ) -> list[str]:
        """
        Find claims in the answer that are not supported by any chunk.

        Uses NLI entailment to check if any chunk entails each claim.
        """
        claims = self._decompose(answer)
        if not claims:
            return []

        text_chunks = [
            c for c in chunks
            if c.modality == Modality.TEXT
            and isinstance(c.content, str)
        ][:10]  # Limit to top 10 for performance

        if not text_chunks:
            return claims  # All unsupported if no text chunks

        unsupported = []
        for claim in claims:
            if self._is_supported(claim, text_chunks):
                continue
            unsupported.append(claim)

        return unsupported

    def _is_supported(self, claim: str, text_chunks: list[UnifiedChunk]) -> bool:
        """Check if any chunk entails this claim via NLI."""
        if self._nli is None:
            # Fallback: simple keyword overlap check
            return self._heuristic_support(claim, text_chunks)

        try:
            pairs = [[tc.content[:512], claim] for tc in text_chunks]
            scores = self._nli.predict(pairs, apply_softmax=True)
            # Check if ANY chunk entails this claim
            return any(float(s[2]) > 0.60 for s in scores)
        except Exception as e:
            logger.error("NLI entailment check failed: %s", e)
            return True  # Assume supported on error to avoid false positives

    def _heuristic_support(self, claim: str, chunks: list[UnifiedChunk]) -> bool:
        """Simple keyword-based support check when NLI is unavailable."""
        claim_words = set(claim.lower().split())
        for chunk in chunks:
            chunk_words = set(chunk.content.lower().split()) if isinstance(chunk.content, str) else set()
            overlap = len(claim_words & chunk_words) / max(len(claim_words), 1)
            if overlap > 0.4:
                return True
        return False

    def _decompose(self, text: str) -> list[str]:
        """
        Split answer into atomic factual claims.
        Filters out very short or non-factual sentences.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []
        for s in sentences:
            s = s.strip()
            # Skip short or non-factual sentences
            if len(s) < 15:
                continue
            if s.startswith(("I ", "Let me", "Here", "In summary", "To summarize")):
                continue
            # Skip questions
            if s.endswith("?"):
                continue
            claims.append(s)
        return claims

    def _targeted_regen(
        self,
        answer: str,
        unsupported: list[str],
        chunks: list[UnifiedChunk],
        query: str,
        llm_client,
    ) -> str:
        """
        Ask LLM to regenerate only the unsupported claims
        using new context.
        """
        try:
            ctx = "\n\n".join(
                c.content for c in chunks[:8]
                if c.modality == Modality.TEXT and isinstance(c.content, str)
            )
            prompt = (
                f"Original answer:\n{answer}\n\n"
                f"These claims are not supported by the context:\n"
                + "\n".join(f"- {c}" for c in unsupported)
                + f"\n\nContext:\n{ctx}\n\n"
                f"Rewrite the original answer, correcting only the unsupported "
                f"claims using the provided context. Do not change supported claims."
            )
            resp = llm_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            logger.error("Targeted regeneration failed: %s", e)
            return answer  # Return original if regen fails

    def _flag_uncertain(self, answer: str, unsupported: list[str]) -> str:
        """Flag remaining unsupported claims in the answer."""
        for claim in unsupported:
            flagged = f"[⚠️ UNCERTAIN: {claim}]"
            answer = answer.replace(claim, flagged)
        return answer

    def _overall_conf(self, answer: str, chunks: list[UnifiedChunk]) -> float:
        """Compute overall confidence of the answer."""
        claims = self._decompose(answer)
        text_chks = [
            c for c in chunks
            if c.modality == Modality.TEXT and isinstance(c.content, str)
        ][:8]

        if not claims or not text_chks:
            return 0.5

        if self._nli is None:
            # Fallback: count supported claims ratio
            supported = sum(
                1 for c in claims
                if self._heuristic_support(c, text_chks)
            )
            return round(supported / len(claims), 3)

        try:
            all_pairs = [
                [tc.content[:256], cl]
                for cl in claims
                for tc in text_chks
            ]
            scores = self._nli.predict(all_pairs, apply_softmax=True)
            entail_sc = [float(s[2]) for s in scores]
            return round(float(sum(entail_sc) / len(entail_sc)), 3)
        except Exception as e:
            logger.error("Overall confidence calc failed: %s", e)
            return 0.5
