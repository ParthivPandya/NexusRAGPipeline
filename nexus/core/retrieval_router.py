"""
NEXUS RAG — Pillar 4: Adaptive Retrieval Router
================================================

Runs all active retrievers in parallel (async), then fuses results
using weighted Reciprocal Rank Fusion. Each retriever's contribution
is controlled by the Query DNA strategy weights.

Retrievers:
    - Dense HNSW (Qdrant vector search)
    - BM25/SPLADE (sparse keyword search)
    - Causal Graph Traversal (Neo4j)
    - Temporal Search (filtered vector search)
    - Multi-Branch (comparative query decomposition)
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import numpy as np

from nexus.core.unified_encoder import Modality, UnifiedChunk, UnifiedEncoder
from nexus.core.query_classifier import QueryDNA

logger = logging.getLogger(__name__)


class AdaptiveRetrievalRouter:
    """
    Parallel multi-retriever with weighted Reciprocal Rank Fusion.
    Activates retrievers based on Query DNA strategy weights.
    """

    # Minimum weight threshold to activate a retriever
    ACTIVATION_THRESHOLD = 0.05

    def __init__(
        self,
        qdrant=None,
        bm25=None,
        kg=None,
        encoder: Optional[UnifiedEncoder] = None,
        config: Optional[dict] = None,
    ):
        self.qdrant = qdrant
        self.bm25 = bm25
        self.kg = kg
        self.encoder = encoder or UnifiedEncoder()
        self._config = config or {}
        self._collection = self._config.get("collection", "nexus_knowledge")

    async def retrieve(
        self, query_dna: QueryDNA, top_k: int = 20
    ) -> list[UnifiedChunk]:
        """
        Run all active retrievers in parallel and fuse results.

        Args:
            query_dna: Classified query with retrieval strategy weights
            top_k: Maximum number of results to return

        Returns:
            Fused and ranked list of UnifiedChunks
        """
        strategy = query_dna.retrieval_strategy
        tasks: list = []
        weights: list[float] = []
        retriever_names: list[str] = []

        # Activate retrievers based on strategy weights
        if strategy.get("dense_hnsw", 0) > self.ACTIVATION_THRESHOLD:
            tasks.append(self._dense(query_dna, top_k))
            weights.append(strategy["dense_hnsw"])
            retriever_names.append("dense_hnsw")

        if strategy.get("bm25_sparse", 0) > self.ACTIVATION_THRESHOLD:
            tasks.append(self._sparse(query_dna, top_k))
            weights.append(strategy["bm25_sparse"])
            retriever_names.append("bm25_sparse")

        if strategy.get("causal_graph", 0) > self.ACTIVATION_THRESHOLD:
            tasks.append(self._causal(query_dna, top_k))
            weights.append(strategy["causal_graph"])
            retriever_names.append("causal_graph")

        if strategy.get("temporal_index", 0) > self.ACTIVATION_THRESHOLD:
            tasks.append(self._temporal(query_dna, top_k))
            weights.append(strategy["temporal_index"])
            retriever_names.append("temporal_index")

        if strategy.get("multi_branch", 0) > self.ACTIVATION_THRESHOLD:
            tasks.append(self._multi_branch(query_dna, top_k))
            weights.append(strategy["multi_branch"])
            retriever_names.append("multi_branch")

        if not tasks:
            # Fallback: always run dense retrieval
            logger.warning("No retrievers activated — falling back to dense search")
            tasks.append(self._dense(query_dna, top_k))
            weights.append(1.0)
            retriever_names.append("dense_hnsw_fallback")

        # Run all retrievers in parallel
        logger.info(
            "Activating %d retrievers: %s",
            len(tasks), ", ".join(retriever_names)
        )
        result_lists = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        valid_weights = []
        for result, weight, name in zip(result_lists, weights, retriever_names):
            if isinstance(result, Exception):
                logger.error("Retriever %s failed: %s", name, result)
                continue
            valid_results.append(result)
            valid_weights.append(weight)

        if not valid_results:
            logger.error("All retrievers failed — returning empty results")
            return []

        # Fuse results using weighted RRF
        fused = self._rrf(valid_results, valid_weights, top_k)
        logger.info("Retrieved %d chunks after RRF fusion", len(fused))
        return fused

    # ── Individual Retrievers ─────────────────────────────────

    async def _dense(self, dna: QueryDNA, top_k: int) -> list[UnifiedChunk]:
        """Dense HNSW vector search via Qdrant."""
        if not self.qdrant:
            return []

        vec = self.encoder.encode(dna.normalized_query, Modality.TEXT)
        try:
            hits = self.qdrant.search(
                collection_name=self._collection,
                query_vector=vec.tolist(),
                limit=top_k,
            )
            return [self._hit_to_chunk(h) for h in hits]
        except Exception as e:
            logger.error("Dense search failed: %s", e)
            return []

    async def _sparse(self, dna: QueryDNA, top_k: int) -> list[UnifiedChunk]:
        """BM25 sparse keyword search."""
        if not self.bm25:
            return []

        try:
            tokens = dna.normalized_query.split()
            results = self.bm25.search(tokens, top_k=top_k)
            return results
        except Exception as e:
            logger.error("Sparse search failed: %s", e)
            return []

    async def _causal(self, dna: QueryDNA, top_k: int) -> list[UnifiedChunk]:
        """Causal graph traversal via Knowledge Graph."""
        if not self.kg:
            return []

        try:
            chains = self.kg.find_causal_chain(dna.normalized_query, depth=3)
            # Convert chain nodes to pseudo-chunks for fusion
            chunks = []
            for chain in chains[:top_k]:
                if isinstance(chain, dict):
                    text = " → ".join(chain.get("nodes", []))
                    chunks.append(UnifiedChunk(
                        id=f"causal_{hash(text)}",
                        content=text,
                        modality=Modality.TEXT,
                        embedding=None,
                        metadata={"source": "causal_graph", "chain": chain},
                        context_prefix="Causal chain: ",
                        credibility_score=chain.get("confidence", 0.5),
                        language=dna.detected_language,
                        chunk_boundary_type="causal",
                    ))
            return chunks
        except Exception as e:
            logger.error("Causal search failed: %s", e)
            return []

    async def _temporal(self, dna: QueryDNA, top_k: int) -> list[UnifiedChunk]:
        """Temporal-filtered vector search."""
        if not self.qdrant:
            return []

        year = self._extract_year(dna.normalized_query)
        if not year:
            return []

        vec = self.encoder.encode(dna.normalized_query, Modality.TEXT)
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            hits = self.qdrant.search(
                collection_name=self._collection,
                query_vector=vec.tolist(),
                query_filter=Filter(must=[
                    FieldCondition(key="year", match=MatchValue(value=year))
                ]),
                limit=top_k,
            )
            return [self._hit_to_chunk(h) for h in hits]
        except Exception as e:
            logger.error("Temporal search failed: %s", e)
            # Fallback: regular dense search
            return await self._dense(dna, top_k)

    async def _multi_branch(self, dna: QueryDNA, top_k: int) -> list[UnifiedChunk]:
        """For comparative queries: retrieve for each entity separately."""
        entities = self._extract_comparison_entities(dna.normalized_query)
        if len(entities) <= 1:
            return await self._dense(dna, top_k)

        # Create sub-queries for each entity
        branch_tasks = []
        per_branch_k = max(top_k // len(entities), 3)

        for entity in entities:
            sub_dna = QueryDNA(
                raw_query=entity,
                normalized_query=entity.lower(),
                detected_language=dna.detected_language,
                retrieval_strategy={"dense_hnsw": 1.0},
                factual=0.8,
            )
            branch_tasks.append(self._dense(sub_dna, per_branch_k))

        branch_results = await asyncio.gather(*branch_tasks, return_exceptions=True)
        combined = []
        for result in branch_results:
            if not isinstance(result, Exception):
                combined.extend(result)
        return combined

    # ── Reciprocal Rank Fusion ────────────────────────────────

    def _rrf(
        self,
        result_lists: list[list],
        weights: list[float],
        top_k: int,
        k: int = 60,
    ) -> list[UnifiedChunk]:
        """
        Weighted Reciprocal Rank Fusion.

        RRF formula: score(d) = Σ weight_i / (k + rank_i(d))

        Args:
            result_lists: Results from each retriever
            weights: Weight for each retriever
            top_k: Number of results to return
            k: RRF constant (default 60)
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, UnifiedChunk] = {}

        for result_list, weight in zip(result_lists, weights):
            for rank, chunk in enumerate(result_list):
                if not hasattr(chunk, 'id'):
                    continue
                cid = chunk.id
                scores[cid] = scores.get(cid, 0.0) + weight / (k + rank + 1)
                chunk_map[cid] = chunk

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Set retrieval score on chunks
        results = []
        for cid, score in ranked[:top_k]:
            chunk = chunk_map[cid]
            chunk.retrieval_score = score
            results.append(chunk)

        return results

    # ── Utility Methods ───────────────────────────────────────

    async def retrieve_for_claim(self, claim: str) -> list[UnifiedChunk]:
        """Targeted retrieval for a single claim during self-healing."""
        if not self.qdrant:
            return []

        vec = self.encoder.encode(claim, Modality.TEXT)
        try:
            hits = self.qdrant.search(
                collection_name=self._collection,
                query_vector=vec.tolist(),
                limit=5,
            )
            return [self._hit_to_chunk(h) for h in hits]
        except Exception as e:
            logger.error("Claim retrieval failed: %s", e)
            return []

    def _extract_year(self, text: str) -> Optional[int]:
        """Extract a 4-digit year from text."""
        match = re.search(r"\b(19|20)\d{2}\b", text)
        return int(match.group()) if match else None

    def _extract_comparison_entities(self, text: str) -> list[str]:
        """Extract entities being compared in a comparative query."""
        for sep in [" vs ", " versus ", " compared to ", " and "]:
            if sep in text.lower():
                parts = text.lower().split(sep)
                return [p.strip() for p in parts if p.strip()]
        return [text]

    def _hit_to_chunk(self, hit) -> UnifiedChunk:
        """Convert a Qdrant search hit to a UnifiedChunk."""
        p = hit.payload or {}
        return UnifiedChunk(
            id=str(hit.id),
            content=p.get("text", ""),
            modality=Modality(p.get("modality", "text")),
            embedding=np.array([]),
            metadata=p,
            context_prefix=p.get("context_prefix", ""),
            causal_node_ids=p.get("causal_node_ids", []),
            temporal_valid_from=p.get("valid_from", ""),
            temporal_valid_until=p.get("valid_until", "present"),
            credibility_score=p.get("credibility", 0.5),
            language=p.get("language", "en"),
            chunk_boundary_type=p.get("boundary_type", "semantic"),
            retrieval_score=hit.score,
        )
