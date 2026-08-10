"""
NEXUS RAG — Storage: Qdrant Vector Store
=========================================

Manages the Qdrant vector database connection, collection
setup, and HNSW configuration.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QdrantStore:
    """Qdrant vector store wrapper with collection management."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "nexus_knowledge",
        dim: int = 1024,
    ):
        self.host = host
        self.port = port
        self.collection = collection
        self.dim = dim
        self._client = None

    @property
    def client(self):
        """Lazy-initialize Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(host=self.host, port=self.port)
                logger.info("Connected to Qdrant at %s:%d", self.host, self.port)
            except Exception as e:
                logger.error("Qdrant connection failed: %s", e)
                raise
        return self._client

    def setup(self):
        """Create collection with HNSW config and payload indexes."""
        try:
            from qdrant_client.models import (
                Distance, VectorParams, HnswConfigDiff,
                PayloadSchemaType, OptimizersConfigDiff,
            )

            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection in collections:
                logger.info("Collection '%s' already exists", self.collection)
                return

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000),
            )

            # Create payload indexes
            for field_name, field_type in [
                ("modality", PayloadSchemaType.KEYWORD),
                ("language", PayloadSchemaType.KEYWORD),
                ("is_streaming", PayloadSchemaType.BOOL),
                ("valid_from", PayloadSchemaType.FLOAT),
                ("credibility", PayloadSchemaType.FLOAT),
            ]:
                self.client.create_payload_index(
                    self.collection, field_name, field_type
                )

            logger.info("Created Qdrant collection: %s (dim=%d)", self.collection, self.dim)

        except Exception as e:
            logger.error("Qdrant setup failed: %s", e)
            raise

    def health(self) -> dict:
        """Check Qdrant health."""
        try:
            info = self.client.get_collection(self.collection)
            return {
                "status": "ok",
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
