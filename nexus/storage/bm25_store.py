"""NEXUS RAG — Storage: BM25 Sparse Retrieval Store."""

from __future__ import annotations
import logging
from typing import Optional

from nexus.core.unified_encoder import Modality, UnifiedChunk

logger = logging.getLogger(__name__)


class BM25Store:
    """BM25Okapi wrapper with document add/remove/search."""

    def __init__(self):
        self._corpus: list[str] = []
        self._chunks: list[UnifiedChunk] = []
        self._bm25 = None

    def add_documents(self, texts: list[str], chunks: Optional[list[UnifiedChunk]] = None):
        """Add documents to the BM25 index."""
        self._corpus.extend(texts)
        if chunks:
            self._chunks.extend(chunks)
        self._rebuild()

    def search(self, tokens: list[str], top_k: int = 10) -> list[UnifiedChunk]:
        """Search using BM25."""
        if not self._bm25 or not self._corpus:
            return []

        try:
            import numpy as np
            scores = self._bm25.get_scores(tokens)
            top_idx = np.argsort(scores)[::-1][:top_k]

            results = []
            for i in top_idx:
                if scores[i] <= 0:
                    break
                if i < len(self._chunks):
                    chunk = self._chunks[i]
                    chunk.retrieval_score = float(scores[i])
                    results.append(chunk)
                else:
                    from uuid import uuid4
                    results.append(UnifiedChunk(
                        id=str(uuid4()),
                        content=self._corpus[i],
                        modality=Modality.TEXT,
                        embedding=None,
                        metadata={},
                        context_prefix="",
                        retrieval_score=float(scores[i]),
                    ))
            return results
        except Exception as e:
            logger.error("BM25 search failed: %s", e)
            return []

    def remove_documents(self, indices: list[int]):
        """Remove documents by index."""
        for idx in sorted(indices, reverse=True):
            if idx < len(self._corpus):
                self._corpus.pop(idx)
            if idx < len(self._chunks):
                self._chunks.pop(idx)
        self._rebuild()

    def find_documents_containing(self, target: str) -> list[int]:
        """Find document indices containing target text."""
        return [
            i for i, doc in enumerate(self._corpus)
            if target.lower() in doc.lower()
        ]

    def _rebuild(self):
        """Rebuild the BM25 index."""
        if not self._corpus:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [doc.lower().split() for doc in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank_bm25 not available")

    @property
    def size(self) -> int:
        return len(self._corpus)
