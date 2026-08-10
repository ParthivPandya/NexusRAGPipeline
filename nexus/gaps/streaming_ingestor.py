"""
NEXUS RAG — Gap 5: Streaming Real-Time Ingestor
================================================

Research: "From Static to Dynamic RAG" (arXiv 2508.05662, 2025)

Micro-batch streaming with 50ms windows.
Compatible with: Kafka, Kinesis, WebSocket, RSS, MQTT.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from nexus.core.unified_encoder import Modality, UnifiedChunk

logger = logging.getLogger(__name__)


class StreamingIngestor:
    """
    Micro-batch streaming ingestor with configurable batch windows.
    Supports async consumption from any stream source.
    """

    BATCH_MS = 50
    MAX_BUFFER = 10_000
    FRESHNESS_HOURS = 24

    def __init__(self, encoder=None, qdrant=None, bm25=None, kg=None):
        self.encoder = encoder
        self.qdrant = qdrant
        self.bm25 = bm25
        self.kg = kg
        self.buffer: deque = deque(maxlen=self.MAX_BUFFER)
        self._last_flush = datetime.utcnow()
        self._total_ingested = 0

    async def consume(self, source):
        """Consume an async iterable stream."""
        async for message in source:
            self.buffer.append(message)
            elapsed_ms = (datetime.utcnow() - self._last_flush).total_seconds() * 1000
            if len(self.buffer) >= 100 or elapsed_ms >= self.BATCH_MS:
                await self._flush()

    async def ingest_message(self, message: dict) -> Optional[str]:
        """Ingest a single message immediately."""
        self.buffer.append(message)
        await self._flush()
        return message.get("id", str(uuid4()))

    async def _flush(self):
        """Flush buffer: parse, encode, store."""
        if not self.buffer:
            return

        batch = list(self.buffer)
        self.buffer.clear()
        self._last_flush = datetime.utcnow()

        # 1. Parse messages → chunks
        parse_tasks = [self._parse(m) for m in batch]
        parsed = await asyncio.gather(*parse_tasks)
        chunks = [c for c in parsed if c is not None]

        if not chunks:
            return

        # 2. Encode
        if self.encoder:
            for chunk in chunks:
                try:
                    chunk.embedding = await asyncio.to_thread(
                        self.encoder.encode, chunk.content, chunk.modality
                    )
                except Exception as e:
                    logger.error("Encoding failed for chunk %s: %s", chunk.id, e)

        # 3. Evict stale streaming chunks
        await self._evict_stale()

        # 4. Upsert into Qdrant
        if self.qdrant:
            try:
                points = []
                for c in chunks:
                    if c.embedding is not None:
                        points.append({
                            "id": c.id,
                            "vector": c.embedding.tolist(),
                            "payload": {
                                **c.metadata,
                                "text": c.content if isinstance(c.content, str) else "",
                                "modality": c.modality.value,
                                "is_streaming": True,
                                "freshness_ts": datetime.utcnow().isoformat(),
                                "language": c.language,
                                "valid_from": c.temporal_valid_from,
                                "credibility": c.credibility_score,
                            },
                        })
                if points:
                    self.qdrant.upsert(
                        collection_name="nexus_knowledge",
                        points=points,
                    )
            except Exception as e:
                logger.error("Qdrant upsert failed: %s", e)

        # 5. Update BM25 and knowledge graph
        if self.bm25:
            text_contents = [
                c.content for c in chunks
                if isinstance(c.content, str)
            ]
            if text_contents and hasattr(self.bm25, 'add_documents'):
                self.bm25.add_documents(text_contents)

        if self.kg:
            self.kg.ingest_chunks(chunks)

        self._total_ingested += len(chunks)
        logger.debug("Flushed %d streaming chunks (total: %d)", len(chunks), self._total_ingested)

    async def _evict_stale(self):
        """Remove streaming chunks older than FRESHNESS_HOURS."""
        if not self.qdrant:
            return

        cutoff = (datetime.utcnow() - timedelta(hours=self.FRESHNESS_HOURS)).isoformat()
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
            self.qdrant.delete(
                collection_name="nexus_knowledge",
                points_selector=Filter(must=[
                    FieldCondition(key="is_streaming", match=MatchValue(value=True)),
                    FieldCondition(key="freshness_ts", range=Range(lt=cutoff)),
                ]),
            )
        except Exception as e:
            logger.debug("Stale eviction skipped: %s", e)

    async def _parse(self, message: dict) -> Optional[UnifiedChunk]:
        """Parse a raw message into a UnifiedChunk."""
        text = message.get("text") or message.get("content") or str(message)
        if not text or len(text.strip()) < 5:
            return None

        return UnifiedChunk(
            id=message.get("id", str(uuid4())),
            content=text,
            modality=Modality.TEXT,
            embedding=None,
            metadata={
                "source": message.get("source", "stream"),
                "is_streaming": True,
            },
            context_prefix="",
            causal_node_ids=[],
            temporal_valid_from=datetime.utcnow().isoformat(),
            temporal_valid_until="present",
            credibility_score=message.get("credibility", 0.5),
            language=message.get("language", "en"),
            chunk_boundary_type="streaming",
        )

    def freshness_score(self, chunk: UnifiedChunk) -> float:
        """Retrieval bonus for fresher streaming chunks."""
        if not chunk.metadata.get("is_streaming"):
            return 0.5
        try:
            age_h = (
                datetime.utcnow()
                - datetime.fromisoformat(chunk.metadata["freshness_ts"])
            ).total_seconds() / 3600
            return max(1.0 - age_h / self.FRESHNESS_HOURS, 0.0)
        except (ValueError, KeyError, TypeError):
            return 0.0

    @property
    def stats(self) -> dict:
        return {
            "buffer_size": len(self.buffer),
            "total_ingested": self._total_ingested,
            "last_flush": self._last_flush.isoformat(),
        }
