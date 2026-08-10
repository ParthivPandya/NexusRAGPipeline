"""
NEXUS RAG — Pillar 8: Self-Optimizing Feedback Loop
====================================================

Records every query, retrieval decision, and user signal.
Boosts chunk weights for retrievals that led to positive outcomes.
Penalizes unhelpful chunks. Every 1,000 queries, triggers a
fine-tuning job on the bi-encoder using successful query-chunk pairs.

Signals:
    - positive:  User explicitly approved the answer
    - negative:  User marked answer as wrong/unhelpful
    - ignored:   User didn't interact (implicit negative)
    - follow_up: User asked a clarifying question (weak negative)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from nexus.core.unified_encoder import Modality, UnifiedChunk

logger = logging.getLogger(__name__)


class FeedbackLearner:
    """
    Self-optimizing feedback loop that learns from every interaction.
    Uses PostgreSQL for persistence and Celery for async fine-tuning.
    """

    FINE_TUNE_EVERY = 1000

    def __init__(self, db_conn=None, config: Optional[dict] = None):
        self._conn = db_conn
        self._config = config or {}
        self._total = 0
        self._memory_log: list[dict] = []  # In-memory fallback
        self._memory_weights: dict[str, float] = {}

        if db_conn:
            self._total = self._load_total()

    def record(
        self,
        query: str,
        chunks: list[UnifiedChunk],
        answer: str,
        signal: str,
        latency_ms: float,
        failure: Optional[str] = None,
    ) -> dict:
        """
        Record a query interaction and apply feedback.

        Args:
            query: The user's query
            chunks: Retrieved chunks used for the answer
            answer: The generated answer
            signal: User feedback signal
            latency_ms: Response latency
            failure: Optional failure type from forensics

        Returns:
            Dict with feedback processing summary
        """
        result = {
            "recorded": True,
            "signal": signal,
            "weights_updated": False,
            "fine_tune_triggered": False,
        }

        # Store the interaction
        self._store_interaction(query, chunks, answer, signal, latency_ms, failure)

        # Apply feedback signals
        if signal == "positive":
            self._boost(chunks)
            self._store_positive_pair(query, chunks)
            result["weights_updated"] = True
        elif signal == "negative":
            useless = self._low_contribution_chunks(chunks, answer)
            self._penalize(useless)
            result["weights_updated"] = True

        # Check if fine-tuning should be triggered
        self._total += 1
        if self._total % self.FINE_TUNE_EVERY == 0:
            self._schedule_fine_tune()
            result["fine_tune_triggered"] = True

        return result

    def get_chunk_weight(self, chunk_id: str) -> float:
        """Get the current weight for a chunk."""
        if self._conn:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        "SELECT weight FROM chunk_weights WHERE chunk_id = %s LIMIT 1",
                        (chunk_id,),
                    )
                    row = cur.fetchone()
                    return row[0] if row else 1.0
            except Exception:
                pass
        return self._memory_weights.get(chunk_id, 1.0)

    def get_stats(self) -> dict:
        """Get feedback loop statistics."""
        return {
            "total_queries": self._total,
            "fine_tune_threshold": self.FINE_TUNE_EVERY,
            "queries_until_fine_tune": self.FINE_TUNE_EVERY - (self._total % self.FINE_TUNE_EVERY),
            "memory_log_size": len(self._memory_log),
        }

    # ── Private: Storage ──────────────────────────────────────

    def _store_interaction(
        self, query, chunks, answer, signal, latency_ms, failure
    ):
        """Store interaction in PostgreSQL or memory."""
        entry = {
            "query": query,
            "chunk_ids": [c.id for c in chunks],
            "answer": answer[:500],
            "signal": signal,
            "latency_ms": latency_ms,
            "failure": failure,
            "ts": datetime.utcnow().isoformat(),
        }

        if self._conn:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO feedback_log
                            (query, chunk_ids, answer, user_signal, latency_ms,
                             failure_type, ts)
                        VALUES (%s, %s, %s, %s, %s, %s, now())
                    """, (
                        query, [c.id for c in chunks], answer[:500],
                        signal, latency_ms, failure,
                    ))
                self._conn.commit()
                return
            except Exception as e:
                logger.error("Failed to store interaction in DB: %s", e)

        # Fallback to memory
        self._memory_log.append(entry)
        if len(self._memory_log) > 10000:
            self._memory_log = self._memory_log[-5000:]

    def _boost(self, chunks: list[UnifiedChunk]):
        """Boost weights for chunks that contributed to positive feedback."""
        ids = [c.id for c in chunks]
        if self._conn:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("""
                        UPDATE chunk_weights
                           SET weight         = LEAST(weight * 1.05, 2.0),
                               positive_count = positive_count + 1,
                               updated_at     = now()
                         WHERE chunk_id = ANY(%s)
                    """, (ids,))
                self._conn.commit()
                return
            except Exception as e:
                logger.error("Boost failed: %s", e)

        # Memory fallback
        for cid in ids:
            current = self._memory_weights.get(cid, 1.0)
            self._memory_weights[cid] = min(current * 1.05, 2.0)

    def _penalize(self, chunks: list[UnifiedChunk]):
        """Penalize weights for chunks that didn't contribute."""
        ids = [c.id for c in chunks]
        if self._conn:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("""
                        UPDATE chunk_weights
                           SET weight         = GREATEST(weight * 0.95, 0.1),
                               negative_count = negative_count + 1,
                               updated_at     = now()
                         WHERE chunk_id = ANY(%s)
                    """, (ids,))
                self._conn.commit()
                return
            except Exception as e:
                logger.error("Penalize failed: %s", e)

        # Memory fallback
        for cid in ids:
            current = self._memory_weights.get(cid, 1.0)
            self._memory_weights[cid] = max(current * 0.95, 0.1)

    def _store_positive_pair(self, query: str, chunks: list[UnifiedChunk]):
        """Store successful query-chunk pair for fine-tuning."""
        top_chunk = chunks[0] if chunks else None
        if not top_chunk:
            return

        if self._conn:
            try:
                with self._conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO fine_tune_pairs (query, positive_chunk_id, ts)
                        VALUES (%s, %s, now())
                    """, (query, top_chunk.id))
                self._conn.commit()
            except Exception as e:
                logger.error("Store positive pair failed: %s", e)

    def _low_contribution_chunks(
        self, chunks: list[UnifiedChunk], answer: str
    ) -> list[UnifiedChunk]:
        """
        Identify chunks that didn't contribute to the answer
        (low semantic similarity between chunk and answer).
        """
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer("intfloat/e5-large-v2")
            a_emb = m.encode(answer)
            return [
                c for c in chunks
                if c.modality == Modality.TEXT
                and isinstance(c.content, str)
                and float(np.dot(a_emb, m.encode(c.content[:256]))) < 0.40
            ]
        except Exception:
            return []  # Can't determine — return empty

    def _schedule_fine_tune(self):
        """Queue async fine-tuning job via Celery."""
        try:
            from nexus.tasks import run_fine_tune_job
            run_fine_tune_job.delay()
            logger.info("Fine-tuning job scheduled (total queries: %d)", self._total)
        except Exception as e:
            logger.warning("Could not schedule fine-tune job: %s", e)

    def _load_total(self) -> int:
        """Load total query count from database."""
        try:
            with self._conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM feedback_log")
                return cur.fetchone()[0]
        except Exception:
            return 0
