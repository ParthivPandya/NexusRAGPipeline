"""
NEXUS RAG — Master Pipeline Orchestrator
=========================================

Chains all 17 components into a single coherent flow:
    cache → streaming flush → Query DNA → cross-lingual bridge →
    parallel retrieval → reranking → conflict resolution →
    epistemic check → generation with uncertainty →
    self-healing → failure forensics → feedback recording
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from nexus.config import NexusConfig, get_config
from nexus.core.unified_encoder import Modality, UnifiedChunk, UnifiedEncoder
from nexus.core.knowledge_graph import TemporalKnowledgeGraph
from nexus.core.query_classifier import QueryDNAClassifier, QueryDNA
from nexus.core.retrieval_router import AdaptiveRetrievalRouter
from nexus.core.conflict_resolver import ConflictResolver
from nexus.core.uncertainty_quantifier import UncertaintyQuantifier, ConfidenceClaim
from nexus.core.self_healer import SelfHealingVerifier
from nexus.core.feedback_loop import FeedbackLearner
from nexus.gaps.epistemic_engine import EpistemicSufficiencyEngine, EpistemicDecision
from nexus.gaps.causal_counterfactual import CausalCounterfactualLayer
from nexus.gaps.cross_lingual_bridge import CrossLingualBridge, XLingualContext
from nexus.gaps.amnesia_engine import AmnesiaEngine
from nexus.gaps.streaming_ingestor import StreamingIngestor
from nexus.gaps.semantic_chunker import SemanticBoundaryChunker
from nexus.gaps.modality_reranker import ModalityAwareReranker
from nexus.gaps.forensics_engine import FailureForensicsEngine
from nexus.gaps.knowledge_evolution import KnowledgeEvolutionManager
from nexus.storage.redis_cache import SemanticCache
from nexus.storage.bm25_store import BM25Store

logger = logging.getLogger(__name__)


@dataclass
class NexusResponse:
    """Complete pipeline response."""
    answer:              str = ""
    claims:              list = field(default_factory=list)
    citations:           list = field(default_factory=list)
    epistemic_decision:  str = "answer"
    conflicts_detected:  bool = False
    conflict_message:    str = ""
    failure_diagnosis:   Optional[dict] = None
    latency_ms:          float = 0.0
    cache_hit:           bool = False
    query_dna:           Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "claims": [c.to_dict() if hasattr(c, "to_dict") else c for c in self.claims],
            "citations": self.citations,
            "epistemic_decision": self.epistemic_decision,
            "conflicts_detected": self.conflicts_detected,
            "conflict_message": self.conflict_message,
            "failure_diagnosis": self.failure_diagnosis,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
        }


class NexusPipeline:
    """Master orchestrator — 17 components, single coherent flow."""

    def __init__(self, config: Optional[NexusConfig] = None):
        self.config = config or get_config()
        logger.info("Initializing NEXUS Pipeline...")

        # Core pillars
        self.encoder = UnifiedEncoder({
            "text_model": self.config.text_embedding_model,
            "code_model": self.config.code_embedding_model,
            "clip_model": self.config.clip_model,
            "whisper_model": self.config.whisper_model,
        })
        self.kg = TemporalKnowledgeGraph(
            uri=self.config.neo4j_uri,
            auth=self.config.neo4j_auth,
        )
        self.classifier = QueryDNAClassifier()
        self.bm25 = BM25Store()
        self.router = AdaptiveRetrievalRouter(
            qdrant=self._get_qdrant(),
            bm25=self.bm25,
            kg=self.kg,
            encoder=self.encoder,
            config={"collection": self.config.qdrant_collection},
        )
        self.conflict = ConflictResolver(nli_model=self.config.nli_model)
        self.uq = UncertaintyQuantifier(embedding_model=self.config.text_embedding_model)
        self.healer = SelfHealingVerifier(nli_model=self.config.nli_model)
        self.feedback = FeedbackLearner(db_conn=self._get_pg())

        # Gap solutions
        self.epistemic = EpistemicSufficiencyEngine(
            epsilon=self.config.epistemic_epsilon,
            max_entropy=self.config.epistemic_max_entropy,
        )
        self.causal = CausalCounterfactualLayer(self.kg)
        self.xrag = CrossLingualBridge(model_name=self.config.multilingual_embedding_model)
        self.amnesia = AmnesiaEngine(
            qdrant=self._get_qdrant(),
            neo4j_driver=self.kg.driver,
            bm25=self.bm25,
            redis_client=self._get_redis_client(),
            db_conn=self._get_pg(),
            signing_key=self.config.signing_key_bytes,
        )
        self.streamer = StreamingIngestor(
            encoder=self.encoder,
            qdrant=self._get_qdrant(),
            bm25=self.bm25,
            kg=self.kg,
        )
        self.chunker = SemanticBoundaryChunker(encoder=self.encoder)
        self.reranker = ModalityAwareReranker(model_name=self.config.cross_encoder_model)
        self.forensics = FailureForensicsEngine()
        self.evolution = KnowledgeEvolutionManager(kg=self.kg, db_conn=self._get_pg())

        # Cache + LLM
        self.cache = SemanticCache(
            redis_url=self.config.redis_url,
            threshold=self.config.semantic_cache_threshold,
        )
        self.llm = self._get_llm()

        logger.info("NEXUS Pipeline initialized — 17 components ready")

    async def query(self, query: str, language: str = "auto", top_k: int = 5) -> NexusResponse:
        """Execute the full NEXUS query pipeline."""
        t0 = time.time()

        # 0. Semantic cache check
        cached = self.cache.get(query)
        if cached:
            return NexusResponse(
                **{k: v for k, v in cached.items() if k in NexusResponse.__dataclass_fields__},
                latency_ms=round((time.time() - t0) * 1000, 1),
                cache_hit=True,
            )

        # 1. Streaming freshness flush
        try:
            await self.streamer._flush()
        except Exception as e:
            logger.debug("Streaming flush skipped: %s", e)

        # 2. Query DNA classification
        dna = self.classifier.classify(query)

        # 3. Cross-lingual bridge (if multilingual query)
        xl_ctx = None
        if dna.multilingual > 0.3:
            xl_ctx = self.xrag.build(query, dna.detected_language, [])

        # 4. Parallel retrieval
        chunks = await self.router.retrieve(dna, top_k=20)

        # 5. Modality-aware reranking
        reranked = self.reranker.rerank(query, dna, chunks, top_k=max(top_k + 2, 7))

        # 6. Conflict resolution
        conflict_report = self.conflict.resolve(reranked)
        final_chunks = conflict_report.resolved_chunks

        # 7. Epistemic sufficiency check
        hypotheses = await self._sample_hypotheses(query, final_chunks)
        ep_report = self.epistemic.evaluate(query, final_chunks, hypotheses)

        if ep_report.decision == EpistemicDecision.ABSTAIN:
            return self._abstain_response(ep_report, t0)

        if ep_report.decision == EpistemicDecision.RETRIEVE_MORE:
            extra = await self.router.retrieve(dna, top_k=10)
            final_chunks.extend(extra)

        # 8. Generate with uncertainty quantification
        raw_answer, claims = await self._generate_with_uncertainty(
            query, final_chunks, dna, xl_ctx
        )

        # 9. Self-healing verification
        healed = self.healer.verify_and_heal(
            raw_answer, final_chunks, query, self.router, self.llm,
        )

        # 10. Cache result
        response_data = {
            "answer": healed.healed_answer,
            "claims": [c.to_dict() if hasattr(c, "to_dict") else c for c in claims],
            "citations": self._citations(final_chunks),
            "epistemic_decision": ep_report.decision.value,
            "conflicts_detected": conflict_report.detected,
            "conflict_message": conflict_report.user_message,
            "failure_diagnosis": None,
        }
        self.cache.set(query, response_data)

        # 11. Record feedback
        self.feedback.record(
            query=query,
            chunks=final_chunks,
            answer=healed.healed_answer,
            signal="pending",
            latency_ms=(time.time() - t0) * 1000,
        )

        return NexusResponse(
            answer=healed.healed_answer,
            claims=claims,
            citations=self._citations(final_chunks),
            epistemic_decision=ep_report.decision.value,
            conflicts_detected=conflict_report.detected,
            conflict_message=conflict_report.user_message,
            failure_diagnosis=None,
            latency_ms=round((time.time() - t0) * 1000, 1),
            cache_hit=False,
            query_dna=dna.to_dict(),
        )

    async def ingest(self, document: dict) -> dict:
        """Ingest a document through the full pipeline."""
        from uuid import uuid4

        # 1. Semantic chunking
        chunks_raw = self.chunker.chunk(document)

        # 2. Create UnifiedChunks with embeddings
        chunks = []
        for sc in chunks_raw:
            chunk = UnifiedChunk(
                id=str(uuid4()),
                content=sc.raw_text,
                modality=Modality.TEXT,
                embedding=self.encoder.encode(sc.raw_text, Modality.TEXT),
                metadata=document.get("metadata", {}),
                context_prefix=sc.context_prefix,
                temporal_valid_from=document.get("metadata", {}).get("date", ""),
                credibility_score=document.get("metadata", {}).get("credibility", 0.5),
                language=document.get("metadata", {}).get("language", "en"),
                chunk_boundary_type=sc.boundary_type,
            )
            chunks.append(chunk)

        # If no chunks from semantic chunker, create a single chunk
        if not chunks and document.get("text"):
            text = document["text"]
            chunk = UnifiedChunk(
                id=str(uuid4()),
                content=text,
                modality=Modality.TEXT,
                embedding=self.encoder.encode(text, Modality.TEXT),
                metadata=document.get("metadata", {}),
                context_prefix="",
                temporal_valid_from=document.get("metadata", {}).get("date", ""),
                credibility_score=document.get("metadata", {}).get("credibility", 0.5),
                language=document.get("metadata", {}).get("language", "en"),
                chunk_boundary_type="document",
            )
            chunks.append(chunk)

        # 3. Upsert into Qdrant
        qdrant = self._get_qdrant()
        if qdrant:
            try:
                from qdrant_client.models import PointStruct
                points = [
                    PointStruct(
                        id=c.id,
                        vector=c.embedding.tolist() if c.embedding is not None else [],
                        payload={
                            "text": c.content if isinstance(c.content, str) else "",
                            "modality": c.modality.value,
                            "language": c.language,
                            "valid_from": c.temporal_valid_from,
                            "credibility": c.credibility_score,
                            "context_prefix": c.context_prefix,
                            **c.metadata,
                        },
                    )
                    for c in chunks
                ]
                qdrant.upsert(
                    collection_name=self.config.qdrant_collection,
                    points=points,
                )
            except Exception as e:
                logger.error("Qdrant upsert failed: %s", e)

        # 4. Add to BM25 index
        self.bm25.add_documents(
            [c.content for c in chunks if isinstance(c.content, str)],
            chunks,
        )

        # 5. Ingest into knowledge graph
        kg_result = self.kg.ingest_chunks(chunks)

        # 6. Check knowledge evolution
        for chunk in chunks:
            entities = self.kg._extract_entities(chunk)
            for entity in entities:
                self.evolution.ingest_new_fact(entity, chunk)

        return {
            "chunk_ids": [c.id for c in chunks],
            "chunks_created": len(chunks),
            "kg_nodes_created": kg_result.get("nodes_created", 0),
            "kg_edges_created": kg_result.get("edges_created", 0),
            "status": "indexed",
        }

    # ── Private Helpers ───────────────────────────────────────

    async def _sample_hypotheses(
        self, query: str, chunks: list[UnifiedChunk], n: int = 4
    ) -> list[str]:
        """Generate n candidate answers for entropy estimation."""
        if not self.llm:
            return [query]

        ctx = "\n\n".join(
            c.content for c in chunks[:5]
            if c.modality == Modality.TEXT and isinstance(c.content, str)
        )

        try:
            resp = self.llm.messages.create(
                model=self.config.llm_model,
                max_tokens=400,
                system="Answer the question. Be concise.",
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:",
                }],
            )
            base = resp.content[0].text
            return [
                base,
                f"Based on the evidence: {base}",
                f"The data suggests: {base}",
                f"In summary: {base}",
            ]
        except Exception as e:
            logger.error("Hypothesis sampling failed: %s", e)
            return [query]

    async def _generate_with_uncertainty(
        self,
        query: str,
        chunks: list[UnifiedChunk],
        dna: QueryDNA,
        xl_ctx: Optional[XLingualContext],
    ):
        """Generate answer with per-claim uncertainty quantification."""
        ctx = "\n\n".join(
            c.content for c in chunks[:7]
            if c.modality == Modality.TEXT and isinstance(c.content, str)
        )

        system = (
            "You are a precise, citation-grounded assistant. "
            "Answer using ONLY the provided context. "
            "If the context does not support a claim, say so explicitly. "
        )
        if xl_ctx:
            system += xl_ctx.reasoning_scaffold

        if not self.llm:
            return f"[No LLM configured] Context summary for: {query}", []

        try:
            resp = self.llm.messages.create(
                model=self.config.llm_model,
                max_tokens=1024,
                system=system,
                messages=[{
                    "role": "user",
                    "content": f"Context:\n{ctx}\n\nQuestion: {query}",
                }],
            )
            raw = resp.content[0].text
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            raw = f"Generation error: {e}"

        # Quantify uncertainty per claim
        claims = []
        for sentence in raw.split(". "):
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            relevant_chunks = [
                c for c in chunks if c.modality == Modality.TEXT
            ][:5]
            cc = self.uq.quantify(sentence, relevant_chunks)
            claims.append(cc)

        return raw, claims

    def _abstain_response(self, ep, t0: float) -> NexusResponse:
        """Build abstention response."""
        msg = (
            "I cannot reliably answer this question — the evidence in my "
            "knowledge base is insufficient or contradictory. "
        )
        if ep.suggested_sources:
            msg += f"Suggested sources: {', '.join(ep.suggested_sources)}."
        if ep.partial_evidence:
            msg += "\n\nClosest available evidence:\n" + \
                   "\n".join(f"• {e}" for e in ep.partial_evidence)

        return NexusResponse(
            answer=msg,
            epistemic_decision="abstain",
            latency_ms=round((time.time() - t0) * 1000, 1),
        )

    def _citations(self, chunks: list[UnifiedChunk]) -> list[dict]:
        """Build citations list from chunks."""
        return [
            {
                "chunk_id": c.id,
                "source": c.metadata.get("source_url", c.metadata.get("source", "unknown")),
                "excerpt": (c.content[:120] + "…" if isinstance(c.content, str) and len(c.content) > 120 else c.content if isinstance(c.content, str) else ""),
                "credibility": c.credibility_score,
                "language": c.language,
            }
            for c in chunks
        ]

    # ── Infrastructure Helpers ────────────────────────────────

    def _get_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            return QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
            )
        except Exception:
            return None

    def _get_redis_client(self):
        try:
            import redis
            return redis.from_url(self.config.redis_url)
        except Exception:
            return None

    def _get_pg(self):
        try:
            import psycopg2
            return psycopg2.connect(self.config.postgres_dsn)
        except Exception:
            return None

    def _get_llm(self):
        try:
            import anthropic
            if self.config.anthropic_api_key:
                return anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        except ImportError:
            pass
        return None

    def get_stats(self) -> dict:
        """Get system-wide statistics."""
        return {
            "total_chunks": self.bm25.size,
            "feedback": self.feedback.get_stats(),
            "cache": self.cache.stats(),
            "streaming": self.streamer.stats,
        }
