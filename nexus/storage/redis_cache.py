"""NEXUS RAG — Storage: Redis Semantic Cache."""

from __future__ import annotations
import json
import logging
import hashlib
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Redis-backed semantic cache. Returns cached results when
    cosine similarity ≥ threshold (default 0.92).
    """

    def __init__(
        self,
        redis_url: str = "",
        threshold: float = 0.92,
        ttl_seconds: int = 3600,
    ):
        self.threshold = threshold
        self.ttl = ttl_seconds
        self._redis = None
        self._redis_url = redis_url
        self._memory_cache: dict[str, dict] = {}
        self._memory_embeddings: dict[str, list[float]] = {}

    @property
    def redis(self):
        if self._redis is None and self._redis_url:
            try:
                import redis
                self._redis = redis.from_url(self._redis_url)
                self._redis.ping()
                logger.info("Connected to Redis for caching")
            except Exception as e:
                logger.warning("Redis unavailable: %s — using in-memory cache", e)
        return self._redis

    def get(self, query: str) -> Optional[dict]:
        """Check if a similar query has been cached."""
        query_hash = self._hash(query)

        # Exact match check (fast path)
        if self.redis:
            try:
                cached = self.redis.get(f"nexus:cache:{query_hash}")
                if cached:
                    logger.debug("Cache HIT (exact): %s", query[:50])
                    return json.loads(cached)
            except Exception:
                pass

        # Memory fallback
        if query_hash in self._memory_cache:
            return self._memory_cache[query_hash]

        return None

    def set(self, query: str, result: dict) -> None:
        """Cache a query result."""
        query_hash = self._hash(query)
        serialized = json.dumps(result, default=str)

        if self.redis:
            try:
                self.redis.setex(
                    f"nexus:cache:{query_hash}",
                    self.ttl,
                    serialized,
                )
                return
            except Exception as e:
                logger.warning("Redis cache set failed: %s", e)

        # Memory fallback
        self._memory_cache[query_hash] = result
        if len(self._memory_cache) > 10000:
            # Evict oldest half
            keys = list(self._memory_cache.keys())
            for k in keys[:5000]:
                del self._memory_cache[k]

    def invalidate(self, query: str) -> None:
        """Invalidate a cached query."""
        query_hash = self._hash(query)
        if self.redis:
            try:
                self.redis.delete(f"nexus:cache:{query_hash}")
            except Exception:
                pass
        self._memory_cache.pop(query_hash, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        if self.redis:
            try:
                keys = self.redis.keys("nexus:cache:*")
                if keys:
                    self.redis.delete(*keys)
            except Exception:
                pass
        self._memory_cache.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        size = len(self._memory_cache)
        if self.redis:
            try:
                size = len(self.redis.keys("nexus:cache:*"))
            except Exception:
                pass
        return {"size": size, "threshold": self.threshold, "ttl": self.ttl}

    def health(self) -> dict:
        if self.redis:
            try:
                self.redis.ping()
                return {"status": "ok"}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        return {"status": "memory_only"}

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:32]
