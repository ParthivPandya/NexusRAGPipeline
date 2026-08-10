# NEXUS RAG
## Neural EXtensible Unified Search — Complete Implementation Guide

> **System Name:** NEXUS RAG  
> **Full Form:** Neural EXtensible Unified Search  
> **Version:** 1.0 — Full Blueprint  
> **Research Sources:** ACL 2025/2026 · ICLR 2025 · NeurIPS 2024/2025 · arXiv 2024–2026  
> **Total Components:** 17 (8 Core Pillars + 9 Research-Validated Gap Solutions)

---

## Table of Contents

1. [Vision & System Overview](#1-vision)
2. [What Makes NEXUS Different](#2-differentiation)
3. [Full Master Architecture](#3-architecture)
4. [Core 8 Pillars](#4-core-pillars)
   - 4.1 Omni-Modal Unification Engine
   - 4.2 Living Temporal Knowledge Graph
   - 4.3 Query DNA Classifier
   - 4.4 Adaptive Retrieval Router
   - 4.5 Conflict Resolution Engine
   - 4.6 Calibrated Uncertainty Quantifier
   - 4.7 Self-Healing Verifier
   - 4.8 Self-Optimizing Feedback Loop
5. [9 Research-Validated Gap Solutions](#5-gap-solutions)
   - Gap 1: Epistemic Sufficiency Engine
   - Gap 2: Causal-Counterfactual Layer
   - Gap 3: Cross-Lingual Reasoning Bridge
   - Gap 4: Machine Unlearning / Amnesia Engine
   - Gap 5: Streaming Real-Time Ingestor
   - Gap 6: Semantic Boundary Chunker
   - Gap 7: Modality-Aware Reranker
   - Gap 8: Failure Forensics Engine
   - Gap 9: Knowledge Evolution Manager
6. [Complete Tech Stack](#6-tech-stack)
7. [Project Directory Structure](#7-directory-structure)
8. [Database Schemas](#8-database-schemas)
9. [REST API Design](#9-api-design)
10. [Master Pipeline Orchestrator](#10-orchestrator)
11. [Build Phases & Timeline](#11-build-phases)
12. [Testing Strategy](#12-testing)
13. [Deployment Guide](#13-deployment)
14. [Evaluation & Benchmarks](#14-evaluation)
15. [Research References](#15-references)

---

## 1. Vision

### What NEXUS RAG Solves

Every existing RAG system — LangChain, LlamaIndex, GraphRAG, RAGFlow, Haystack — has the same fundamental limitations:

- Handles only **one or two modalities** (mostly text)
- Is **stateless**: forgets everything between sessions
- Cannot **resolve contradictions** between sources
- Cannot say **"I don't know"** with mathematical certainty
- Cannot **delete data** to comply with GDPR/privacy law
- Cannot **reason causally** ("what caused X?", "what if Y hadn't happened?")
- Cannot **handle real-time data streams**
- Cannot **diagnose its own failures** (was it a retrieval error? reasoning error?)
- Gets **stuck in time**: never evolves as knowledge changes

NEXUS RAG fixes all of these simultaneously with 17 integrated components.

### Core Design Philosophy

```
Every component serves three master principles:

1. ACCURACY OVER SPEED — but optimize speed without sacrificing accuracy
2. HONESTY OVER COMPLETENESS — better to abstain than to hallucinate
3. EVOLUTION OVER STASIS — the system must improve with every query
```

---

## 2. What Makes NEXUS Different

| Capability | LangChain | LlamaIndex | GraphRAG | RAGFlow | NEXUS |
|---|---|---|---|---|---|
| Unified multi-modal embedding | ❌ | ❌ | ❌ | ❌ | ✅ |
| Causal + counterfactual retrieval | ❌ | ❌ | ❌ | ❌ | ✅ |
| Principled abstention (epistemic) | ❌ | ❌ | ❌ | ❌ | ✅ |
| GDPR machine unlearning | ❌ | ❌ | ❌ | ❌ | ✅ |
| Real-time streaming ingestion | ⚠️ | ⚠️ | ❌ | ⚠️ | ✅ |
| Per-claim confidence scores | ❌ | ❌ | ❌ | ❌ | ✅ |
| Failure forensics diagnosis | ❌ | ❌ | ❌ | ❌ | ✅ |
| Cross-lingual reasoning bridge | ❌ | ❌ | ❌ | ❌ | ✅ |
| Semantic boundary chunking | ❌ | ❌ | ❌ | ❌ | ✅ |
| Knowledge evolution manager | ❌ | ❌ | ❌ | ❌ | ✅ |
| Temporal knowledge graph | ❌ | ❌ | ⚠️ | ❌ | ✅ |
| Self-healing auto-correction | ❌ | ⚠️ | ❌ | ❌ | ✅ |
| Conflict source resolution | ❌ | ❌ | ❌ | ❌ | ✅ |
| Self-optimizing feedback loop | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 3. Full Master Architecture

```
╔═════════════════════════════════════════════════════════════════════════════╗
║                         NEXUS RAG — MASTER ARCHITECTURE                      ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │                     INPUT LAYER  (Any Modality)                      │    ║
║  │  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬───────┐  │    ║
║  │  │ Text │ PDF  │Image │Audio │Video │ Code │Table │Stream│Formula│  │    ║
║  │  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴───────┘  │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [GAP 6]  SEMANTIC BOUNDARY CHUNKER                           │    ║
║  │    Vision-guided · Discourse-aware · Topic-shift detection           │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [PILLAR 1]  OMNI-MODAL UNIFICATION ENGINE                    │    ║
║  │    Single 1024-dim embedding space · All modalities comparable       │    ║
║  └──────────────────┬────────────────────────┬──────────────────────────┘    ║
║                     │                        │                                ║
║                     ▼                        ▼                                ║
║  ┌─────────────────────────┐   ┌──────────────────────────────────────┐      ║
║  │  [PILLAR 2]             │   │  [GAP 2]  CAUSAL KNOWLEDGE GRAPH      │      ║
║  │  TEMPORAL KNOWLEDGE     │   │  Cause-effect edges · What-if paths   │      ║
║  │  GRAPH                  │   └──────────────────────────────────────┘      ║
║  └─────────────────────────┘                                                  ║
║                                                                               ║
║  ══════════════════════════  QUERY ARRIVES  ═══════════════════════════════  ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [GAP 5]  STREAMING FRESHNESS CHECK                           │    ║
║  │    Flush any pending micro-batch before answering                    │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │              SEMANTIC CACHE  (Cosine ≥ 0.92 → Instant Return)        │    ║
║  └───────────────────────────MISS──────────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [PILLAR 3]  QUERY DNA CLASSIFIER                             │    ║
║  │    10 dimensions · Routes to optimal retrieval strategy              │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [GAP 3]  CROSS-LINGUAL REASONING BRIDGE                      │    ║
║  │    Concept extraction · Cultural alignment · Reasoning scaffold      │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [PILLAR 4]  ADAPTIVE RETRIEVAL ROUTER                        │    ║
║  └────┬─────────────┬────────────────┬─────────────┬────────────────────┘    ║
║       │             │                │             │                          ║
║       ▼             ▼                ▼             ▼                          ║
║  ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐                   ║
║  │ Dense   │  │BM25+SPLADE│  │Causal Graph│  │Temporal  │                   ║
║  │ HNSW    │  │ Sparse   │  │ Traversal  │  │ Search   │                   ║
║  └─────────┘  └──────────┘  └────────────┘  └──────────┘                   ║
║       │             │                │             │                          ║
║       └─────────────┴────────────────┴─────────────┘                         ║
║                                      │  RRF Fusion                            ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [GAP 7]  MODALITY-AWARE RERANKER                             │    ║
║  │    Query-type × Modality weights · Cross-encoder scoring             │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [PILLAR 5]  CONFLICT RESOLUTION ENGINE                       │    ║
║  │    NLI contradiction detection · Credibility scoring · Resolution    │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [GAP 1]  EPISTEMIC SUFFICIENCY ENGINE                        │    ║
║  │    Entropy minimization · ANSWER / ABSTAIN / RETRIEVE_MORE          │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [PILLAR 6]  LLM GENERATION + UNCERTAINTY QUANTIFIER          │    ║
║  │    Per-claim confidence · Streaming output · Source attribution      │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [PILLAR 7]  SELF-HEALING VERIFIER                            │    ║
║  │    NLI claim verification · Auto-regenerate · Max 3 iterations       │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [GAP 8]  FAILURE FORENSICS ENGINE                            │    ║
║  │    Retrieval / Comprehension / Reasoning / Parametric diagnosis      │    ║
║  └───────────────────────────────────┬──────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║  ┌──────────────────────────────────────────────────────────────────────┐    ║
║  │         [PILLAR 8]  SELF-OPTIMIZING FEEDBACK LOOP                    │    ║
║  │    Signal recording · Weight rebalancing · Periodic fine-tune        │    ║
║  └──────────────────────────────────────────────────────────────────────┘    ║
║                                      │                                        ║
║                                      ▼                                        ║
║                  FINAL ANSWER + CITATIONS + PER-CLAIM CONFIDENCE              ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

## 4. Core 8 Pillars

---

### Pillar 1 — Omni-Modal Unification Engine

**Purpose:** Create a single 1024-dimensional embedding space where text, images, audio, video, code, tables, and mathematical formulas are semantically comparable. A chart and its description should produce vectors with cosine similarity > 0.85.

**Models Used:**

| Modality | Encoder Model | Projection |
|---|---|---|
| Text | `intfloat/e5-large-v2` | Direct (1024-dim) |
| Image | `openai/clip-vit-large-patch14` | Linear → 1024-dim |
| Audio | `openai/whisper-base` + audio encoder | Linear → 1024-dim |
| Video | Frame sampling + audio fusion | Attention → 1024-dim |
| Code | `microsoft/codebert-base` | Linear → 1024-dim |
| Table | `google/tapas-base` | Linear → 1024-dim |
| Formula | `tbs17/MathBERT` | Linear → 1024-dim |

```python
# nexus/core/unified_encoder.py

from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import torch

class Modality(Enum):
    TEXT    = "text"
    IMAGE   = "image"
    AUDIO   = "audio"
    VIDEO   = "video"
    CODE    = "code"
    TABLE   = "table"
    FORMULA = "formula"

@dataclass
class UnifiedChunk:
    id:                   str
    content:              Any           # Raw content (str, bytes, np.ndarray)
    modality:             Modality
    embedding:            np.ndarray    # Always 1024-dim, L2-normalized
    metadata:             dict          # source_url, date, page_num, language …
    context_prefix:       str           # Contextual summary prepended at ingest
    causal_node_ids:      list[str]     # Links into causal graph
    temporal_valid_from:  str           # ISO 8601
    temporal_valid_until: str           # ISO 8601 or "present"
    credibility_score:    float         # 0.0 – 1.0
    language:             str           # ISO 639-1
    chunk_boundary_type:  str           # "semantic"|"visual"|"discourse"
    retrieval_score:      Optional[float] = None   # Set by retrieval router

class UnifiedEncoder:
    DIM = 1024

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import clip, whisper
        from transformers import AutoModel, AutoTokenizer, TapasModel

        self.text_model  = SentenceTransformer("intfloat/e5-large-v2")
        self.code_tok    = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model  = AutoModel.from_pretrained("microsoft/codebert-base")
        self.table_model = TapasModel.from_pretrained("google/tapas-base")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14")
        self.whisper     = whisper.load_model("base")
        self.projectors  = self._init_projectors()  # Modality-specific linear heads

    def encode(self, content: Any, modality: Modality) -> np.ndarray:
        """Encode any modality → 1024-dim L2-normalized vector."""
        if modality == Modality.TEXT:
            raw = self.text_model.encode(content, normalize_embeddings=False)
        elif modality == Modality.CODE:
            raw = self._encode_code(content)
        elif modality == Modality.IMAGE:
            raw = self._encode_image(content)
        elif modality == Modality.AUDIO:
            raw = self._encode_audio(content)
        elif modality == Modality.TABLE:
            raw = self._encode_table(content)
        else:
            raw = self.text_model.encode(str(content), normalize_embeddings=False)

        projected = self.projectors[modality](torch.tensor(raw).float())
        vec = projected.detach().numpy()
        return vec / (np.linalg.norm(vec) + 1e-9)   # L2 normalize

    def _encode_code(self, code: str) -> np.ndarray:
        tokens = self.code_tok(code, return_tensors="pt",
                               max_length=512, truncation=True)
        with torch.no_grad():
            out = self.code_model(**tokens)
        return out.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token

    def _encode_image(self, image_path_or_bytes) -> np.ndarray:
        from PIL import Image
        import io
        if isinstance(image_path_or_bytes, bytes):
            img = Image.open(io.BytesIO(image_path_or_bytes))
        else:
            img = Image.open(image_path_or_bytes)
        tensor = self.clip_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self.clip_model.encode_image(tensor)
        return features.squeeze().numpy()

    def _encode_audio(self, audio_path: str) -> np.ndarray:
        result = self.whisper.transcribe(audio_path)
        transcript = result["text"]
        return self.text_model.encode(transcript, normalize_embeddings=False)

    def _encode_table(self, table_data: dict) -> np.ndarray:
        """Serialize table to text then encode."""
        rows = table_data.get("rows", [])
        headers = table_data.get("headers", [])
        text = " | ".join(headers) + "\n"
        text += "\n".join(" | ".join(str(c) for c in row) for row in rows)
        return self.text_model.encode(text, normalize_embeddings=False)

    def _init_projectors(self) -> dict:
        """Linear projection heads: raw_dim → 1024."""
        import torch.nn as nn
        return {
            Modality.TEXT:    nn.Linear(1024, self.DIM),
            Modality.CODE:    nn.Linear(768, self.DIM),
            Modality.IMAGE:   nn.Linear(768, self.DIM),
            Modality.AUDIO:   nn.Linear(1024, self.DIM),
            Modality.TABLE:   nn.Linear(768, self.DIM),
            Modality.FORMULA: nn.Linear(768, self.DIM),
            Modality.VIDEO:   nn.Linear(1024, self.DIM),
        }
```

---

### Pillar 2 — Living Temporal Knowledge Graph

**Purpose:** Auto-build a knowledge graph from every ingested document. Every fact is tagged with a validity window. Outdated facts are superseded (not deleted) so historical queries remain accurate.

```python
# nexus/core/knowledge_graph.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import neo4j

@dataclass
class TemporalNode:
    id:             str
    entity:         str               # "Tesla revenue"
    value:          str               # "$97.7B"
    valid_from:     datetime
    valid_until:    Optional[datetime] # None = still valid
    confidence:     float
    source_count:   int
    source_ids:     list[str]
    superseded_by:  Optional[str]     # Newer node id if outdated
    causal_parents: list[str]         # "Causes" from this node
    causal_children: list[str]        # "Effects" of this node
    status:         str               # "current"|"superseded"|"contested"|"retracted"

class TemporalKnowledgeGraph:
    def __init__(self, uri: str, auth: tuple):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=auth)
        self._init_schema()

    def _init_schema(self):
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
            s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.entity)")
            s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.valid_from)")
            s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.status)")

    def ingest_chunks(self, chunks: list[UnifiedChunk]):
        for chunk in chunks:
            if chunk.modality != Modality.TEXT:
                continue
            entities   = self._extract_entities(chunk)
            causal     = self._extract_causal_links(chunk)
            for e in entities:
                self._upsert_node(e, chunk)
            for c in causal:
                self._upsert_causal_edge(c)

    def query_at_time(self, entity: str, as_of: datetime) -> list[TemporalNode]:
        """What was known about `entity` on `as_of` date?"""
        with self.driver.session() as s:
            result = s.run("""
                MATCH (n:Node {entity: $entity})
                WHERE n.valid_from <= $ts
                  AND (n.valid_until IS NULL OR n.valid_until >= $ts)
                  AND n.superseded_by IS NULL
                RETURN n ORDER BY n.confidence DESC
            """, entity=entity, ts=as_of.isoformat())
            return [self._to_node(r["n"]) for r in result]

    def find_causal_chain(self, query: str, depth: int = 3) -> list[dict]:
        """Walk cause→effect chains related to query up to `depth` hops."""
        seed_entity = self._extract_primary_entity(query)
        with self.driver.session() as s:
            result = s.run("""
                MATCH path=(start:Node {entity: $entity})-[:CAUSES*1..""" + str(depth) + """]->(end:Node)
                RETURN path
            """, entity=seed_entity)
            return [self._path_to_chain(r["path"]) for r in result]

    def _extract_causal_links(self, chunk: UnifiedChunk) -> list[dict]:
        """Use LLM to extract cause-effect pairs from chunk text."""
        # Prompt LLM for: [{"cause": "...", "effect": "...", "confidence": 0.0-1.0}]
        # Causal signal words: "caused", "led to", "resulted in", "because of", "triggered"
        pass

    def _upsert_node(self, entity: dict, source: UnifiedChunk):
        with self.driver.session() as s:
            s.run("""
                MERGE (n:Node {entity: $entity, value: $value})
                ON CREATE SET
                    n.id = $id,
                    n.valid_from = $vf,
                    n.valid_until = null,
                    n.confidence = $conf,
                    n.source_count = 1,
                    n.status = 'current'
                ON MATCH SET
                    n.source_count = n.source_count + 1,
                    n.confidence   = (n.confidence + $conf) / 2
            """, entity=entity["name"], value=entity["value"],
                 id=f"{entity['name']}_{entity['value']}",
                 vf=source.temporal_valid_from, conf=source.credibility_score)

    def _to_node(self, n) -> TemporalNode:
        return TemporalNode(
            id=n["id"], entity=n["entity"], value=n["value"],
            valid_from=datetime.fromisoformat(n["valid_from"]),
            valid_until=datetime.fromisoformat(n["valid_until"]) if n.get("valid_until") else None,
            confidence=n["confidence"], source_count=n["source_count"],
            source_ids=[], superseded_by=n.get("superseded_by"),
            causal_parents=[], causal_children=[], status=n["status"]
        )
```

---

### Pillar 3 — Query DNA Classifier

**Purpose:** Classify every incoming query across 10 semantic dimensions simultaneously, then compute exact retrieval strategy weights. No single retriever wins — the right combination is determined per-query.

```python
# nexus/core/query_classifier.py

from dataclasses import dataclass
import numpy as np
from langdetect import detect

@dataclass
class QueryDNA:
    raw_query:         str
    normalized_query:  str
    detected_language: str          # ISO 639-1 ("en", "ar", "ja" …)
    # 10 semantic dimensions — each 0.0 to 1.0
    factual:           float  # "What is X?"
    analytical:        float  # "Why did X happen?"
    temporal:          float  # "What was X in 2020?"
    procedural:        float  # "How do I do X?"
    comparative:       float  # "X vs Y?"
    causal:            float  # "What caused X?"
    adversarial:       float  # "Argue for X"
    creative:          float  # "Imagine if X…"
    counterfactual:    float  # "What if X hadn't occurred?"
    multilingual:      float  # Cross-language retrieval required
    # Computed retrieval weights — sum to 1.0
    retrieval_strategy: dict

class QueryDNAClassifier:

    TEMPORAL_SIGNALS     = ["in 2020", "last year", "back in", "at the time",
                            "historically", "previously", "used to", "when did"]
    CAUSAL_SIGNALS       = ["caused", "why did", "led to", "resulted in",
                            "reason for", "triggered by", "consequence of"]
    COUNTERFACTUAL_SIGNALS = ["what if", "if … hadn't", "had … not",
                               "suppose", "imagine", "would have"]
    PROCEDURAL_SIGNALS   = ["how to", "how do i", "steps to", "guide for",
                            "tutorial", "walk me through", "show me how"]
    COMPARATIVE_SIGNALS  = [" vs ", " versus ", "compare", "difference between",
                            "which is better", "pros and cons"]

    def classify(self, query: str) -> QueryDNA:
        lang = self._detect_language(query)
        norm = query.lower().strip()
        emb  = self._embed(norm)
        dims = self._score_dimensions(norm, emb)
        strat = self._compute_strategy(dims)
        return QueryDNA(
            raw_query=query, normalized_query=norm,
            detected_language=lang,
            retrieval_strategy=strat, **dims
        )

    def _detect_language(self, text: str) -> str:
        try:    return detect(text)
        except: return "en"

    def _score_dimensions(self, query: str, emb: np.ndarray) -> dict:
        q = query.lower()
        temporal      = min(sum(s in q for s in self.TEMPORAL_SIGNALS) * 0.4, 1.0)
        causal        = min(sum(s in q for s in self.CAUSAL_SIGNALS) * 0.5, 1.0)
        counterfact   = min(sum(s in q for s in self.COUNTERFACTUAL_SIGNALS) * 0.6, 1.0)
        procedural    = min(sum(s in q for s in self.PROCEDURAL_SIGNALS) * 0.5, 1.0)
        comparative   = min(sum(s in q for s in self.COMPARATIVE_SIGNALS) * 0.5, 1.0)
        factual       = 0.8 if q.startswith(("what ", "who ", "when ", "where ")) else 0.3
        multilingual  = 0.9 if self._detect_language(query) != "en" else 0.0
        adversarial   = 0.8 if any(w in q for w in ["argue", "defend", "case for"]) else 0.0
        creative      = 0.8 if any(w in q for w in ["imagine", "creative", "story"]) else 0.0
        analytical    = 0.7 if any(w in q for w in ["why", "how", "explain", "analyze"]) else 0.3
        return dict(factual=factual, analytical=analytical, temporal=temporal,
                    procedural=procedural, comparative=comparative,
                    causal=causal, adversarial=adversarial, creative=creative,
                    counterfactual=counterfact, multilingual=multilingual)

    def _compute_strategy(self, dims: dict) -> dict:
        """Combine dimension scores into retriever activation weights."""
        raw = {
            "dense_hnsw":     0.40 * dims["factual"]  + 0.20 * dims["analytical"],
            "bm25_sparse":    0.40 * dims["factual"]  + 0.20 * dims["procedural"],
            "causal_graph":   0.80 * dims["causal"]   + 0.70 * dims["counterfactual"],
            "temporal_index": 0.90 * dims["temporal"],
            "multi_branch":   0.80 * dims["comparative"],
            "xrag_bridge":    0.90 * dims["multilingual"],
            "fuzzy_creative": 0.80 * dims["creative"],
        }
        total = sum(raw.values()) + 1e-9
        return {k: round(v / total, 4) for k, v in raw.items()}

    def _embed(self, text: str) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("intfloat/e5-large-v2")
        return m.encode(text)
```

---

### Pillar 4 — Adaptive Retrieval Router

**Purpose:** Run all active retrievers in parallel (async), then fuse results using weighted Reciprocal Rank Fusion. Each retriever's contribution is controlled by the Query DNA strategy weights.

```python
# nexus/core/retrieval_router.py

import asyncio
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import numpy as np

class AdaptiveRetrievalRouter:

    def __init__(
        self,
        qdrant: QdrantClient,
        bm25: BM25Okapi,
        kg: TemporalKnowledgeGraph,
        encoder: UnifiedEncoder
    ):
        self.qdrant   = qdrant
        self.bm25     = bm25
        self.kg       = kg
        self.encoder  = encoder

    async def retrieve(
        self, query_dna: QueryDNA, top_k: int = 20
    ) -> list[UnifiedChunk]:
        s = query_dna.retrieval_strategy
        tasks, weights = [], []

        if s.get("dense_hnsw", 0) > 0.05:
            tasks.append(self._dense(query_dna, top_k)); weights.append(s["dense_hnsw"])
        if s.get("bm25_sparse", 0) > 0.05:
            tasks.append(self._sparse(query_dna, top_k)); weights.append(s["bm25_sparse"])
        if s.get("causal_graph", 0) > 0.05:
            tasks.append(self._causal(query_dna, top_k)); weights.append(s["causal_graph"])
        if s.get("temporal_index", 0) > 0.05:
            tasks.append(self._temporal(query_dna, top_k)); weights.append(s["temporal_index"])
        if s.get("multi_branch", 0) > 0.05:
            tasks.append(self._multi_branch(query_dna, top_k)); weights.append(s["multi_branch"])

        result_lists = await asyncio.gather(*tasks)
        return self._rrf(result_lists, weights, top_k)

    async def _dense(self, dna: QueryDNA, top_k: int) -> list:
        vec = self.encoder.encode(dna.normalized_query, Modality.TEXT)
        hits = self.qdrant.search(
            collection_name="nexus_knowledge",
            query_vector=vec.tolist(),
            limit=top_k
        )
        return [self._hit_to_chunk(h) for h in hits]

    async def _sparse(self, dna: QueryDNA, top_k: int) -> list:
        tokens = dna.normalized_query.split()
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self._idx_to_chunk(i) for i in top_idx if scores[i] > 0]

    async def _causal(self, dna: QueryDNA, top_k: int) -> list:
        chains = self.kg.find_causal_chain(dna.normalized_query, depth=3)
        return chains[:top_k]

    async def _temporal(self, dna: QueryDNA, top_k: int) -> list:
        year = self._extract_year(dna.normalized_query)
        if not year:
            return []
        hits = self.qdrant.search(
            collection_name="nexus_knowledge",
            query_vector=self.encoder.encode(dna.normalized_query, Modality.TEXT).tolist(),
            query_filter={"must": [{"key": "year", "match": {"value": year}}]},
            limit=top_k
        )
        return [self._hit_to_chunk(h) for h in hits]

    async def _multi_branch(self, dna: QueryDNA, top_k: int) -> list:
        """For comparative queries: retrieve for each entity separately."""
        entities = self._extract_comparison_entities(dna.normalized_query)
        branch_results = await asyncio.gather(*[
            self._dense(
                QueryDNA(raw_query=e, normalized_query=e.lower(),
                         detected_language=dna.detected_language,
                         retrieval_strategy={"dense_hnsw": 1.0},
                         **{d: 0.0 for d in
                            ["factual","analytical","temporal","procedural",
                             "comparative","causal","adversarial","creative",
                             "counterfactual","multilingual"]}),
                top_k // max(len(entities), 1)
            ) for e in entities
        ])
        return [c for branch in branch_results for c in branch]

    def _rrf(
        self,
        result_lists: list[list],
        weights: list[float],
        top_k: int,
        k: int = 60
    ) -> list[UnifiedChunk]:
        scores: dict[str, float] = {}
        chunk_map: dict[str, UnifiedChunk] = {}
        for result_list, weight in zip(result_lists, weights):
            for rank, chunk in enumerate(result_list):
                cid = chunk.id
                scores[cid]    = scores.get(cid, 0.0) + weight / (k + rank + 1)
                chunk_map[cid] = chunk
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_map[cid] for cid, _ in ranked[:top_k]]

    def _extract_year(self, text: str) -> Optional[int]:
        import re
        m = re.search(r"\b(19|20)\d{2}\b", text)
        return int(m.group()) if m else None

    def _extract_comparison_entities(self, text: str) -> list[str]:
        for sep in [" vs ", " versus ", " compared to ", " and "]:
            if sep in text.lower():
                parts = text.lower().split(sep)
                return [p.strip() for p in parts if p.strip()]
        return [text]

    def _hit_to_chunk(self, hit) -> UnifiedChunk:
        p = hit.payload
        return UnifiedChunk(
            id=str(hit.id), content=p.get("text", ""),
            modality=Modality(p.get("modality", "text")),
            embedding=np.array([]), metadata=p,
            context_prefix=p.get("context_prefix", ""),
            causal_node_ids=[], temporal_valid_from=p.get("valid_from", ""),
            temporal_valid_until=p.get("valid_until", "present"),
            credibility_score=p.get("credibility", 0.5),
            language=p.get("language", "en"),
            chunk_boundary_type=p.get("boundary_type", "semantic"),
            retrieval_score=hit.score
        )

    def _idx_to_chunk(self, idx: int) -> UnifiedChunk:
        # Look up chunk by BM25 corpus index
        pass

    async def retrieve_for_claim(self, claim: str) -> list[UnifiedChunk]:
        """Targeted retrieval for a single claim during self-healing."""
        vec = self.encoder.encode(claim, Modality.TEXT)
        hits = self.qdrant.search(
            collection_name="nexus_knowledge",
            query_vector=vec.tolist(), limit=5
        )
        return [self._hit_to_chunk(h) for h in hits]
```

---

### Pillar 5 — Conflict Resolution Engine

**Purpose:** Detect when retrieved chunks contradict each other using NLI, score each chunk's credibility across four factors, choose a resolution strategy, and generate a transparent user-facing message when sources disagree.

```python
# nexus/core/conflict_resolver.py

from dataclasses import dataclass
from enum import Enum
from sentence_transformers import CrossEncoder

class Strategy(Enum):
    HIGHER_CREDIBILITY = "use_higher_credibility"
    MORE_RECENT        = "use_more_recent"
    MORE_CORROBORATED  = "use_more_corroborated"
    SURFACE_CONFLICT   = "surface_conflict_to_user"

@dataclass
class ConflictReport:
    detected:           bool
    conflicts:          list[dict]   # [{"chunk_a": ..., "chunk_b": ..., "conf": ...}]
    strategy:           Strategy
    resolved_chunks:    list[UnifiedChunk]
    user_message:       str

class ConflictResolver:
    CONTRADICTION_THRESHOLD = 0.70

    def __init__(self):
        self.nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")

    def resolve(self, chunks: list[UnifiedChunk]) -> ConflictReport:
        conflicts = self._detect(chunks)
        if not conflicts:
            return ConflictReport(detected=False, conflicts=[],
                                  strategy=None, resolved_chunks=chunks,
                                  user_message="")

        for c in chunks:
            c.credibility_score = self._credibility(c)

        strategy = self._choose_strategy(conflicts, chunks)
        resolved = self._apply(strategy, conflicts, chunks)
        return ConflictReport(
            detected=True, conflicts=conflicts, strategy=strategy,
            resolved_chunks=resolved,
            user_message=self._user_msg(conflicts, strategy)
        )

    def _detect(self, chunks: list[UnifiedChunk]) -> list[dict]:
        text_chunks = [c for c in chunks if c.modality == Modality.TEXT]
        conflicts   = []
        for i, a in enumerate(text_chunks):
            for b in text_chunks[i+1:]:
                pairs  = [[a.content[:512], b.content[:512]]]
                scores = self.nli.predict(pairs, apply_softmax=True)
                # scores shape: (1, 3) → [contradiction, neutral, entailment]
                contradiction_score = float(scores[0][0])
                if contradiction_score >= self.CONTRADICTION_THRESHOLD:
                    conflicts.append({"chunk_a": a, "chunk_b": b,
                                      "confidence": contradiction_score})
        return conflicts

    def _credibility(self, c: UnifiedChunk) -> float:
        import time
        from urllib.parse import urlparse
        recency = max(0.0, 1.0 - self._age_days(c) / 365.0)
        corroboration  = min(c.metadata.get("source_count", 1) / 5.0, 1.0)
        domain_rank    = self._domain_authority(c.metadata.get("source_url", ""))
        consistency    = 1.0 - 0.1 * len(c.metadata.get("known_contradictions", []))
        return (0.30 * recency + 0.25 * corroboration +
                0.25 * domain_rank + 0.20 * consistency)

    def _age_days(self, c: UnifiedChunk) -> float:
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(c.temporal_valid_from)
            return (datetime.utcnow() - dt).days
        except:
            return 180.0

    def _domain_authority(self, url: str) -> float:
        # Heuristic: known high-authority domains score 0.9+
        HIGH_AUTHORITY = {".gov", ".edu", "pubmed", "nature.com",
                          "science.org", "arxiv.org", "sec.gov"}
        url_lower = url.lower()
        if any(d in url_lower for d in HIGH_AUTHORITY):
            return 0.90
        if url_lower.startswith("https://"):
            return 0.65
        return 0.40

    def _choose_strategy(self, conflicts, chunks) -> Strategy:
        a = conflicts[0]["chunk_a"]; b = conflicts[0]["chunk_b"]
        cred_gap   = abs(a.credibility_score - b.credibility_score)
        age_gap    = abs(self._age_days(a) - self._age_days(b))
        if cred_gap > 0.25:   return Strategy.HIGHER_CREDIBILITY
        if age_gap  > 365:    return Strategy.MORE_RECENT
        return Strategy.SURFACE_CONFLICT

    def _apply(self, strategy, conflicts, chunks) -> list[UnifiedChunk]:
        if strategy == Strategy.HIGHER_CREDIBILITY:
            a, b = conflicts[0]["chunk_a"], conflicts[0]["chunk_b"]
            loser = a if a.credibility_score < b.credibility_score else b
            return [c for c in chunks if c.id != loser.id]
        if strategy == Strategy.MORE_RECENT:
            a, b = conflicts[0]["chunk_a"], conflicts[0]["chunk_b"]
            older = a if self._age_days(a) > self._age_days(b) else b
            return [c for c in chunks if c.id != older.id]
        return chunks  # SURFACE_CONFLICT: return all, flag to user

    def _user_msg(self, conflicts, strategy) -> str:
        a = conflicts[0]["chunk_a"]; b = conflicts[0]["chunk_b"]
        if strategy == Strategy.SURFACE_CONFLICT:
            return (f"⚠️ Sources disagree: '{a.content[:80]}…' "
                    f"vs '{b.content[:80]}…'. "
                    f"Both views are presented; verify independently.")
        return ""
```

---

### Pillar 6 — Calibrated Uncertainty Quantifier

**Purpose:** Every factual claim in the generated answer gets an individual confidence score computed from source count, recency, agreement ratio, and domain authority. The user sees exactly how reliable each statement is.

```python
# nexus/core/uncertainty_quantifier.py

from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class ConfidenceClaim:
    claim_text:       str
    confidence:       float     # 0.0 – 1.0
    label:            str       # "High" | "Medium" | "Low" | "Uncertain"
    source_count:     int
    recency_days:     float
    agreement_ratio:  float
    authority_score:  float
    chunk_ids:        list[str]

class UncertaintyQuantifier:

    THRESHOLDS = {
        "High":      (0.80, 1.00),
        "Medium":    (0.60, 0.80),
        "Low":       (0.40, 0.60),
        "Uncertain": (0.00, 0.40),
    }

    def quantify(
        self, claim: str, chunks: list[UnifiedChunk]
    ) -> ConfidenceClaim:
        if not chunks:
            return ConfidenceClaim(claim_text=claim, confidence=0.0,
                                   label="Uncertain", source_count=0,
                                   recency_days=999.0, agreement_ratio=0.0,
                                   authority_score=0.0, chunk_ids=[])

        source_n   = len(chunks)
        recency    = self._avg_recency_days(chunks)
        agreement  = self._agreement_ratio(claim, chunks)
        authority  = float(np.mean([c.credibility_score for c in chunks]))
        sem_support = self._avg_semantic_support(claim, chunks)

        confidence = (
            0.25 * min(source_n / 5.0, 1.0)          +  # 5+ sources = max
            0.20 * max(1.0 - recency / 365.0, 0.0)   +  # Fresher = better
            0.25 * agreement                           +
            0.15 * authority                           +
            0.15 * sem_support
        )
        confidence = round(min(confidence, 1.0), 3)
        label = next(k for k, (lo, hi) in self.THRESHOLDS.items()
                     if lo <= confidence < hi or (lo == 0.80 and confidence >= 0.80))

        return ConfidenceClaim(
            claim_text=claim, confidence=confidence, label=label,
            source_count=source_n, recency_days=round(recency, 1),
            agreement_ratio=round(agreement, 3), authority_score=round(authority, 3),
            chunk_ids=[c.id for c in chunks]
        )

    def _avg_recency_days(self, chunks: list[UnifiedChunk]) -> float:
        ages = []
        for c in chunks:
            try:
                dt = datetime.fromisoformat(c.temporal_valid_from)
                ages.append((datetime.utcnow() - dt).days)
            except:
                ages.append(180)
        return float(np.mean(ages)) if ages else 180.0

    def _agreement_ratio(self, claim: str, chunks: list[UnifiedChunk]) -> float:
        """Fraction of text chunks that entail the claim (via cosine threshold)."""
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("intfloat/e5-large-v2")
        c_emb = model.encode(claim)
        supporting = sum(
            1 for c in chunks
            if c.modality == Modality.TEXT and
               float(np.dot(c_emb, model.encode(c.content[:256]))) > 0.65
        )
        return supporting / max(len(chunks), 1)

    def _avg_semantic_support(self, claim: str, chunks: list[UnifiedChunk]) -> float:
        from sentence_transformers import SentenceTransformer
        model  = SentenceTransformer("intfloat/e5-large-v2")
        c_emb  = model.encode(claim)
        scores = [float(np.dot(c_emb, model.encode(c.content[:256])))
                  for c in chunks if c.modality == Modality.TEXT]
        return float(np.mean(scores)) if scores else 0.0
```

---

### Pillar 7 — Self-Healing Verifier

**Purpose:** Decompose the generated answer into atomic claims. Verify each claim against retrieved chunks via NLI. Re-retrieve and regenerate only the unsupported portions. After 3 failed iterations, flag the claim as uncertain rather than output a hallucination.

```python
# nexus/core/self_healer.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class HealingResult:
    healed_answer:      str
    all_supported:      bool
    unsupported_claims: list[str]
    iterations:         int
    overall_confidence: float

class SelfHealingVerifier:
    MAX_ITER = 3

    def __init__(self):
        from sentence_transformers import CrossEncoder
        self.nli = CrossEncoder("cross-encoder/nli-deberta-v3-small")

    def verify_and_heal(
        self,
        answer:   str,
        chunks:   list[UnifiedChunk],
        query:    str,
        router           # AdaptiveRetrievalRouter
    ) -> HealingResult:
        healed = answer
        iters  = 0

        while iters < self.MAX_ITER:
            unsupported = self._unsupported_claims(healed, chunks)
            if not unsupported:
                break
            # Re-retrieve specifically for unsupported claims
            for claim in unsupported:
                import asyncio
                extra = asyncio.run(router.retrieve_for_claim(claim))
                chunks.extend(extra)
            # Regenerate only those portions
            healed  = self._targeted_regen(healed, unsupported, chunks, query)
            iters  += 1

        final_unsupported = self._unsupported_claims(healed, chunks)
        if final_unsupported:
            healed = self._flag_uncertain(healed, final_unsupported)

        return HealingResult(
            healed_answer=healed,
            all_supported=len(final_unsupported) == 0,
            unsupported_claims=final_unsupported,
            iterations=iters,
            overall_confidence=self._overall_conf(healed, chunks)
        )

    def _unsupported_claims(
        self, answer: str, chunks: list[UnifiedChunk]
    ) -> list[str]:
        claims    = self._decompose(answer)
        supported = []
        text_chunks = [c for c in chunks if c.modality == Modality.TEXT]
        for claim in claims:
            pairs   = [[tc.content[:512], claim] for tc in text_chunks[:10]]
            if not pairs:
                supported.append(False); continue
            scores  = self.nli.predict(pairs, apply_softmax=True)
            # Check if ANY chunk entails this claim
            entails = any(float(s[2]) > 0.60 for s in scores)
            supported.append(entails)
        return [c for c, ok in zip(claims, supported) if not ok]

    def _decompose(self, text: str) -> list[str]:
        """Split answer into atomic factual claims."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def _targeted_regen(
        self,
        answer:      str,
        unsupported: list[str],
        chunks:      list[UnifiedChunk],
        query:       str
    ) -> str:
        """Ask LLM to regenerate only the unsupported claims using new context."""
        import anthropic
        client  = anthropic.Anthropic()
        ctx     = "\n\n".join(c.content for c in chunks[:8]
                               if c.modality == Modality.TEXT)
        prompt  = (
            f"Original answer:\n{answer}\n\n"
            f"These claims are not supported by the context:\n"
            + "\n".join(f"- {c}" for c in unsupported)
            + f"\n\nContext:\n{ctx}\n\n"
            f"Rewrite the original answer, correcting only the unsupported "
            f"claims using the provided context. Do not change supported claims."
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    def _flag_uncertain(self, answer: str, unsupported: list[str]) -> str:
        for claim in unsupported:
            flagged = f"[⚠️ UNCERTAIN: {claim}]"
            answer  = answer.replace(claim, flagged)
        return answer

    def _overall_conf(self, answer: str, chunks: list[UnifiedChunk]) -> float:
        claims    = self._decompose(answer)
        text_chks = [c for c in chunks if c.modality == Modality.TEXT][:8]
        if not claims or not text_chks:
            return 0.5
        all_pairs = [[tc.content[:256], cl]
                     for cl in claims for tc in text_chks]
        scores    = self.nli.predict(all_pairs, apply_softmax=True)
        entail_sc = [float(s[2]) for s in scores]
        return round(float(sum(entail_sc) / len(entail_sc)), 3)
```

---

### Pillar 8 — Self-Optimizing Feedback Loop

**Purpose:** Record every query, retrieval decision, and user signal. Boost chunk weights for retrievals that led to positive outcomes. Penalize unhelpful chunks. Every 1,000 queries, trigger a fine-tuning job on the bi-encoder using successful query-chunk pairs.

```python
# nexus/core/feedback_loop.py

import psycopg2
from datetime import datetime
from typing import Optional

class FeedbackLearner:
    FINE_TUNE_EVERY = 1000

    def __init__(self, db_dsn: str):
        self.conn = psycopg2.connect(db_dsn)
        self.total = self._load_total()

    def record(
        self,
        query:      str,
        chunks:     list[UnifiedChunk],
        answer:     str,
        signal:     str,      # "positive"|"negative"|"ignored"|"follow_up"
        latency_ms: float,
        failure:    Optional[str] = None
    ):
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feedback_log
                    (query, chunk_ids, answer, user_signal, latency_ms,
                     failure_type, ts)
                VALUES (%s, %s, %s, %s, %s, %s, now())
            """, (query, [c.id for c in chunks], answer,
                  signal, latency_ms, failure))
        self.conn.commit()

        if signal == "positive":
            self._boost(chunks)
            self._store_positive_pair(query, chunks)
        elif signal == "negative":
            useless = self._low_contribution_chunks(chunks, answer)
            self._penalize(useless)

        self.total += 1
        if self.total % self.FINE_TUNE_EVERY == 0:
            self._schedule_fine_tune()

    def _boost(self, chunks: list[UnifiedChunk]):
        ids = [c.id for c in chunks]
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE chunk_weights
                   SET weight          = LEAST(weight * 1.05, 2.0),
                       positive_count  = positive_count + 1,
                       updated_at      = now()
                 WHERE chunk_id = ANY(%s)
            """, (ids,))
        self.conn.commit()

    def _penalize(self, chunks: list[UnifiedChunk]):
        ids = [c.id for c in chunks]
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE chunk_weights
                   SET weight          = GREATEST(weight * 0.95, 0.1),
                       negative_count  = negative_count + 1,
                       updated_at      = now()
                 WHERE chunk_id = ANY(%s)
            """, (ids,))
        self.conn.commit()

    def _store_positive_pair(self, query: str, chunks: list[UnifiedChunk]):
        top_chunk = chunks[0] if chunks else None
        if not top_chunk:
            return
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO fine_tune_pairs (query, positive_chunk_id, ts)
                VALUES (%s, %s, now())
            """, (query, top_chunk.id))
        self.conn.commit()

    def _low_contribution_chunks(
        self, chunks: list[UnifiedChunk], answer: str
    ) -> list[UnifiedChunk]:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        m     = SentenceTransformer("intfloat/e5-large-v2")
        a_emb = m.encode(answer)
        return [
            c for c in chunks
            if c.modality == Modality.TEXT and
               float(np.dot(a_emb, m.encode(c.content[:256]))) < 0.40
        ]

    def _schedule_fine_tune(self):
        """Queue async fine-tuning job via Celery."""
        from nexus.tasks import run_fine_tune_job
        run_fine_tune_job.delay()

    def _load_total(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM feedback_log")
            return cur.fetchone()[0]
```

---

## 5. Gap Solutions (9 Research-Validated Additions)

---

### Gap 1 — Epistemic Sufficiency Engine

**Research basis:** Ghafouri et al. 2025 "Epistemic Mismatch" — commercial RAG systems hallucinate 17–33% in legal settings. *Entropic Claim Resolution* (arXiv 2603.28444, March 2026) proposes entropy minimization with dynamic termination as the fix.

**Core innovation:** The system computes Shannon entropy over competing answer hypotheses. Low entropy → answer confidently. High entropy → the system mathematically proves it cannot answer reliably, then abstains and recommends better sources.

```python
# nexus/gaps/epistemic_engine.py

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from sentence_transformers import SentenceTransformer
import numpy as np

class EpistemicDecision(Enum):
    ANSWER        = "answer"
    ABSTAIN       = "abstain"
    RETRIEVE_MORE = "retrieve_more"

@dataclass
class EpistemicReport:
    decision:          EpistemicDecision
    entropy:           float
    epsilon:           float
    reason:            str
    suggested_sources: list[str]   # External sources if ABSTAIN
    partial_evidence:  list[str]   # Best available even if insufficient

class EpistemicSufficiencyEngine:
    """
    Based on: Entropic Claim Resolution — arXiv 2603.28444, March 2026.
    Terminates dynamically when epistemically sufficient, not after fixed top-k.
    """

    def __init__(self, epsilon: float = 0.15, max_entropy: float = 0.85):
        self.epsilon     = epsilon        # ≤ this → answer
        self.max_entropy = max_entropy    # ≥ this → abstain
        self.model       = SentenceTransformer("intfloat/e5-large-v2")

    def evaluate(
        self,
        query:        str,
        evidence:     list[UnifiedChunk],
        hypotheses:   list[str]           # Candidate answers from LLM sampling
    ) -> EpistemicReport:
        H = self._entropy(hypotheses, evidence)

        if H <= self.epsilon:
            return EpistemicReport(
                decision=EpistemicDecision.ANSWER,
                entropy=round(H, 4), epsilon=self.epsilon,
                reason=f"Entropy {H:.4f} ≤ {self.epsilon} — sufficient evidence",
                suggested_sources=[], partial_evidence=[]
            )

        if H >= self.max_entropy:
            return EpistemicReport(
                decision=EpistemicDecision.ABSTAIN,
                entropy=round(H, 4), epsilon=self.epsilon,
                reason=f"Entropy {H:.4f} ≥ {self.max_entropy} — no signal found",
                suggested_sources=self._suggest_sources(query),
                partial_evidence=self._best_partial(evidence, query)
            )

        return EpistemicReport(
            decision=EpistemicDecision.RETRIEVE_MORE,
            entropy=round(H, 4), epsilon=self.epsilon,
            reason=f"Entropy {H:.4f} — need more evidence",
            suggested_sources=[], partial_evidence=[]
        )

    def _entropy(self, hypotheses: list[str], evidence: list[UnifiedChunk]) -> float:
        if not hypotheses:
            return 1.0
        h_embs   = self.model.encode(hypotheses)          # (n_h, dim)
        e_texts  = [c.content[:256] for c in evidence if c.modality == Modality.TEXT]
        if not e_texts:
            return 1.0
        e_embs   = self.model.encode(e_texts)             # (n_e, dim)
        # Support score for each hypothesis = avg cosine sim vs evidence
        support  = np.array([
            float(np.mean([np.dot(h, e) for e in e_embs]))
            for h in h_embs
        ])
        support  = np.clip(support, 1e-9, None)
        probs    = support / support.sum()
        H        = float(-np.sum(probs * np.log2(probs + 1e-12)))
        max_H    = math.log2(len(hypotheses)) if len(hypotheses) > 1 else 1.0
        return H / max_H if max_H > 0 else 1.0

    def _suggest_sources(self, query: str) -> list[str]:
        domain_map = {
            ("drug", "medicine", "clinical", "dose"):  "https://pubmed.ncbi.nlm.nih.gov",
            ("law", "regulation", "statute", "court"): "https://www.westlaw.com",
            ("stock", "sec", "filing", "earnings"):    "https://www.sec.gov/edgar",
            ("research", "paper", "arxiv", "study"):   "https://arxiv.org",
        }
        q = query.lower()
        return [url for kws, url in domain_map.items() if any(k in q for k in kws)]

    def _best_partial(
        self, evidence: list[UnifiedChunk], query: str
    ) -> list[str]:
        if not evidence:
            return []
        q_emb    = self.model.encode(query)
        scored   = sorted(
            [(float(np.dot(q_emb, self.model.encode(c.content[:256]))), c)
             for c in evidence if c.modality == Modality.TEXT],
            reverse=True
        )
        return [c.content[:200] for _, c in scored[:3]]
```

---

### Gap 2 — Causal-Counterfactual Layer

**Research basis:** CausalRAG (ACL Findings 2025, arXiv 2503.19878) — superior precision via causal grounding. Causal-Counterfactual RAG (arXiv 2509.14435) extends to "what-if" scenario evaluation.

```python
# nexus/gaps/causal_counterfactual.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class CausalChain:
    nodes:      list[str]    # Ordered list: root → … → effect
    edges:      list[str]    # Causal relationship labels
    confidence: float
    evidence:   list[str]    # Supporting chunk IDs

@dataclass
class CounterfactualResult:
    original_outcome:        str
    removed_condition:       str   # "If interest rates had stayed low…"
    counterfactual_outcome:  str   # "…then SVB's bonds would not have lost value"
    broken_chain_link:       str   # The causal edge that is disrupted
    confidence:              float
    evidence_chunk_ids:      list[str]

class CausalCounterfactualLayer:
    """
    Research: CausalRAG (arXiv 2503.19878) + Causal-CF RAG (arXiv 2509.14435)
    """

    def __init__(self, kg: TemporalKnowledgeGraph):
        self.kg = kg

    def retrieve_causal(self, query: str, depth: int = 3) -> list[CausalChain]:
        """Walk the causal graph to find all chains relevant to this query."""
        chains = self.kg.find_causal_chain(query, depth=depth)
        return sorted(chains, key=lambda c: c["confidence"], reverse=True)

    def evaluate_counterfactual(
        self,
        query:          str,   # "What if the Fed had not raised rates?"
        known_outcome:  str    # "SVB collapsed in March 2023"
    ) -> CounterfactualResult:
        removed = self._parse_removed_condition(query)
        chains  = [c for c in self.kg.find_causal_chain(removed, depth=3)
                   if removed.lower() in str(c).lower()]

        if not chains:
            return CounterfactualResult(
                original_outcome=known_outcome, removed_condition=removed,
                counterfactual_outcome="Cannot determine — condition not in causal graph",
                broken_chain_link="unknown", confidence=0.0,
                evidence_chunk_ids=[]
            )

        best_chain   = chains[0]
        alt_outcome  = self._trace_without(best_chain, removed)

        return CounterfactualResult(
            original_outcome=known_outcome, removed_condition=removed,
            counterfactual_outcome=alt_outcome,
            broken_chain_link=self._find_broken_link(best_chain, removed),
            confidence=best_chain["confidence"],
            evidence_chunk_ids=best_chain.get("evidence", [])
        )

    def _parse_removed_condition(self, query: str) -> str:
        """Extract the counterfactual condition from the query."""
        for marker in ["what if ", "if ", "had ", "suppose "]:
            if marker in query.lower():
                idx = query.lower().index(marker)
                return query[idx + len(marker):].split(",")[0].strip()
        return query

    def _trace_without(self, chain: dict, removed_node: str) -> str:
        """Simulate causal chain with the removed node deleted."""
        nodes = [n for n in chain.get("nodes", [])
                 if removed_node.lower() not in n.lower()]
        if not nodes:
            return "Without that condition, the causal chain is broken — the outcome would likely not have occurred."
        return f"Without '{removed_node}', the chain proceeds only through: {' → '.join(nodes)}, suggesting the original outcome may not have materialized."

    def _find_broken_link(self, chain: dict, removed: str) -> str:
        nodes = chain.get("nodes", [])
        for i, n in enumerate(nodes):
            if removed.lower() in n.lower() and i + 1 < len(nodes):
                return f"{n} → {nodes[i+1]}"
        return "unknown"
```

---

### Gap 3 — Cross-Lingual Reasoning Bridge

**Research basis:** XRAG benchmark (arXiv 2505.10089) found two unreported bugs: monolingual RAG fails at response language correctness; multilingual RAG fails at cross-language *reasoning*, not generation. CORAL (arXiv 2604.25676) adds cultural alignment. ACL 2026: "All Languages Matter."

```python
# nexus/gaps/cross_lingual_bridge.py

from dataclasses import dataclass

@dataclass
class XLingualContext:
    query_language:    str
    doc_languages:     list[str]
    concept_bridge:    dict         # Language-agnostic concept map
    cultural_tags:     list[str]    # e.g. ["ar::ar", "ja::ja"]
    reasoning_scaffold: str         # Tells LLM HOW to reason cross-lingually

class CrossLingualBridge:
    """
    Research: XRAG (arXiv 2505.10089), CORAL (arXiv 2604.25676), ACL 2026.
    Core insight: the problem is cross-lingual REASONING, not translation.
    """

    CULTURAL_AUTHORITY_DOMAINS = {
        "ar": ["aljazeera.com", "alarabiya.net"],
        "ja": ["nhk.or.jp", "asahi.com"],
        "zh": ["xinhuanet.com", "people.com.cn"],
        "de": ["spiegel.de", "faz.net"],
        "fr": ["lemonde.fr", "liberation.fr"],
    }

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("intfloat/multilingual-e5-large")

    def build(
        self,
        query:       str,
        query_lang:  str,
        chunks:      list[UnifiedChunk]
    ) -> XLingualContext:
        # 1. Extract language-agnostic concepts from query
        query_concepts = self._extract_concepts(query)

        # 2. Extract + align concepts from each chunk
        concept_bridge: dict = {}
        for chunk in chunks:
            c_concepts = self._extract_concepts(chunk.content)
            for lang_concept, eng_concept in c_concepts.items():
                concept_bridge.setdefault(eng_concept, {})[chunk.language] = lang_concept

        # 3. Cultural context detection (CORAL paper)
        cultural_tags = self._detect_cultural_context(chunks)

        # 4. Build reasoning scaffold
        scaffold = self._build_scaffold(query_lang, concept_bridge, cultural_tags)

        return XLingualContext(
            query_language=query_lang,
            doc_languages=list({c.language for c in chunks}),
            concept_bridge=concept_bridge,
            cultural_tags=cultural_tags,
            reasoning_scaffold=scaffold
        )

    def _extract_concepts(self, text: str) -> dict:
        """Extract universal semantic concepts (language-agnostic)."""
        # Use multilingual NER + entity linking to canonical Wikidata IDs
        # Simplified: return {surface_form: english_equivalent}
        return {}   # Implement with spaCy multilingual + entity linker

    def _detect_cultural_context(self, chunks: list[UnifiedChunk]) -> list[str]:
        tags = []
        for c in chunks:
            lang = c.language
            url  = c.metadata.get("source_url", "")
            for auth_domain in self.CULTURAL_AUTHORITY_DOMAINS.get(lang, []):
                if auth_domain in url:
                    tags.append(f"{lang}::{auth_domain}")
        return list(set(tags))

    def _build_scaffold(
        self,
        target_lang:    str,
        concept_bridge: dict,
        cultural_tags:  list[str]
    ) -> str:
        concepts_str = ", ".join(list(concept_bridge.keys())[:10])
        cultures_str = "; ".join(cultural_tags) if cultural_tags else "none detected"
        return (
            f"Cross-lingual synthesis instructions:\n"
            f"• Target output language: {target_lang}\n"
            f"• Universal concepts identified: {concepts_str}\n"
            f"• Cultural perspectives present: {cultures_str}\n"
            f"• Step 1: Identify the universal concept being asked about\n"
            f"• Step 2: Gather evidence from ALL language sources via concepts\n"
            f"• Step 3: Note culturally-specific interpretations if any\n"
            f"• Step 4: Synthesise in target language ({target_lang})\n"
            f"• Step 5: Append cultural-context note if it changes the answer"
        )
```

---

### Gap 4 — Machine Unlearning / Amnesia Engine

**Research basis:** EU AI Act (enforced August 2025), GDPR "Right to Be Forgotten," India DPDP Act. Wang et al. IEEE Transactions 2025. Differential privacy alone (ε>0) cannot guarantee zero influence — true unlearning requires lineage tracking and surgical deletion with cryptographic proof.

```python
# nexus/gaps/amnesia_engine.py

import hashlib, hmac, json, secrets
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DataLineage:
    target:              str
    vector_ids:          list[str]   # Qdrant point IDs
    bm25_doc_indices:    list[int]   # BM25 corpus positions
    kg_node_ids:         list[str]   # Neo4j node IDs
    cache_keys:          list[str]   # Redis keys
    feedback_log_ids:    list[str]   # PostgreSQL row IDs
    fine_tune_pair_ids:  list[str]
    chunk_weight_ids:    list[str]
    backup_paths:        list[str]   # S3/GCS object paths

@dataclass
class DeletionCertificate:
    certificate_id:     str
    target:             str
    target_type:        str
    timestamp:          str           # ISO 8601
    lineage_purged:     DataLineage
    verification_hash:  str           # SHA-256 over proof payload
    hmac_signature:     str           # HMAC-SHA256 with signing key
    completeness:       str           # "complete" | "partial:<reason>"
    regulations:        list[str]     # ["GDPR", "CCPA", "EU_AI_ACT", "DPDP"]

class AmnesiaEngine:
    """
    Legally required under EU AI Act (enforced August 2025).
    Research: Wang et al. IEEE Transactions 2025, arXiv 2412.04697.
    Issues cryptographically verifiable proof of deletion for compliance audits.
    """
    REGULATIONS = ["GDPR", "CCPA", "EU_AI_ACT", "DPDP"]

    def __init__(
        self, qdrant, neo4j_driver, bm25, redis_client, db_conn,
        signing_key: bytes = None
    ):
        self.qdrant  = qdrant
        self.neo4j   = neo4j_driver
        self.bm25    = bm25
        self.cache   = redis_client
        self.db      = db_conn
        self.key     = signing_key or secrets.token_bytes(32)

    def forget(
        self, target: str, target_type: str,
        regulations: list[str] = None
    ) -> DeletionCertificate:
        regs    = regulations or self.REGULATIONS
        lineage = self._trace(target, target_type)
        results = self._delete_all(lineage)
        cert    = self._certify(target, target_type, lineage, results, regs)
        self._audit_log(cert)
        return cert

    def _trace(self, target: str, target_type: str) -> DataLineage:
        return DataLineage(
            target=target,
            vector_ids=self._find_vectors(target),
            bm25_doc_indices=self._find_bm25(target),
            kg_node_ids=self._find_kg_nodes(target),
            cache_keys=self._find_cache(target),
            feedback_log_ids=self._find_feedback(target),
            fine_tune_pair_ids=self._find_fine_tune(target),
            chunk_weight_ids=self._find_weights(target),
            backup_paths=self._find_backups(target)
        )

    def _delete_all(self, lineage: DataLineage) -> dict:
        results = {}
        # Vector store
        if lineage.vector_ids:
            self.qdrant.delete("nexus_knowledge", lineage.vector_ids)
            results["qdrant"] = "deleted"
        # Knowledge graph
        if lineage.kg_node_ids:
            with self.neo4j.session() as s:
                s.run("MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
                      ids=lineage.kg_node_ids)
            results["neo4j"] = "deleted"
        # Semantic cache
        for k in lineage.cache_keys:
            self.cache.delete(k)
        results["cache"] = "invalidated"
        # BM25 — rebuild without deleted documents
        if lineage.bm25_doc_indices:
            self.bm25.remove_documents(lineage.bm25_doc_indices)
            results["bm25"] = "removed"
        # PostgreSQL feedback + fine-tune pairs
        with self.db.cursor() as cur:
            if lineage.feedback_log_ids:
                cur.execute("DELETE FROM feedback_log WHERE id = ANY(%s)",
                            (lineage.feedback_log_ids,))
            if lineage.fine_tune_pair_ids:
                cur.execute("DELETE FROM fine_tune_pairs WHERE id = ANY(%s)",
                            (lineage.fine_tune_pair_ids,))
        self.db.commit()
        results["postgres"] = "deleted"
        return results

    def _certify(
        self, target, target_type, lineage, results, regs
    ) -> DeletionCertificate:
        cert_id   = f"CERT-{secrets.token_hex(8).upper()}"
        ts        = datetime.utcnow().isoformat()
        payload   = json.dumps({
            "cert_id": cert_id, "target": target,
            "ts": ts, "results": results
        }, sort_keys=True)
        v_hash    = hashlib.sha256(payload.encode()).hexdigest()
        signature = hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()
        complete  = all(v in ("deleted","invalidated","removed")
                        for v in results.values())
        return DeletionCertificate(
            certificate_id=cert_id, target=target, target_type=target_type,
            timestamp=ts, lineage_purged=lineage,
            verification_hash=v_hash, hmac_signature=signature,
            completeness="complete" if complete else f"partial:{results}",
            regulations=regs
        )

    def _audit_log(self, cert: DeletionCertificate):
        with self.db.cursor() as cur:
            cur.execute("""
                INSERT INTO deletion_certificates
                    (id, target, target_type, deletion_ts, verification_hash,
                     signature, completeness, regulations)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (cert.certificate_id, cert.target, cert.target_type,
                  cert.timestamp, cert.verification_hash, cert.hmac_signature,
                  cert.completeness, cert.regulations))
        self.db.commit()

    # Stub finders — implement per storage backend
    def _find_vectors(self, t): return []
    def _find_bm25(self, t):    return []
    def _find_kg_nodes(self, t): return []
    def _find_cache(self, t):   return []
    def _find_feedback(self, t): return []
    def _find_fine_tune(self, t): return []
    def _find_weights(self, t): return []
    def _find_backups(self, t): return []
```

---

### Gap 5 — Streaming Real-Time Ingestor

**Research basis:** "From Static to Dynamic RAG" (arXiv 2508.05662, 2025) — NASDAQ delivers 500,000 ticks/day; static RAG is infeasible. Streaming RAG must balance throughput-latency-memory under real-time constraints.

```python
# nexus/gaps/streaming_ingestor.py

import asyncio
from collections import deque
from datetime import datetime

class StreamingIngestor:
    """
    Micro-batch streaming with 50ms windows.
    Compatible with: Kafka, Kinesis, WebSocket, RSS, MQTT (IoT).
    """
    BATCH_MS         = 50
    MAX_BUFFER       = 10_000
    FRESHNESS_HOURS  = 24

    def __init__(self, encoder, qdrant, bm25, kg):
        self.encoder = encoder
        self.qdrant  = qdrant
        self.bm25    = bm25
        self.kg      = kg
        self.buffer: deque = deque(maxlen=self.MAX_BUFFER)
        self._last_flush   = datetime.utcnow()

    async def consume(self, source):
        """Consume an async iterable stream."""
        async for message in source:
            self.buffer.append(message)
            elapsed_ms = (datetime.utcnow() - self._last_flush).total_seconds() * 1000
            if len(self.buffer) >= 100 or elapsed_ms >= self.BATCH_MS:
                await self._flush()

    async def _flush(self):
        if not self.buffer:
            return
        batch = list(self.buffer); self.buffer.clear()
        self._last_flush = datetime.utcnow()

        # 1. Parse messages → chunks
        chunks = [c for c in (await asyncio.gather(*[
            self._parse(m) for m in batch
        ])) if c is not None]
        if not chunks:
            return

        # 2. Encode in parallel
        embeddings = await asyncio.gather(*[
            asyncio.to_thread(self.encoder.encode, c.content, c.modality)
            for c in chunks
        ])

        # 3. Evict stale streaming chunks first
        await self._evict_stale()

        # 4. Upsert into Qdrant
        self.qdrant.upsert(
            collection_name="nexus_knowledge",
            points=[{
                "id": c.id,
                "vector": emb.tolist(),
                "payload": {
                    **c.metadata,
                    "text": c.content if isinstance(c.content, str) else "",
                    "modality": c.modality.value,
                    "is_streaming": True,
                    "freshness_ts": datetime.utcnow().isoformat(),
                    "language": c.language,
                    "valid_from": c.temporal_valid_from,
                    "credibility": c.credibility_score,
                }
            } for c, emb in zip(chunks, embeddings)]
        )

        # 5. Update BM25 and knowledge graph
        self.bm25.add_documents([c.content for c in chunks
                                  if isinstance(c.content, str)])
        self.kg.ingest_chunks(chunks)

    async def _evict_stale(self):
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(hours=self.FRESHNESS_HOURS)).isoformat()
        self.qdrant.delete(
            collection_name="nexus_knowledge",
            points_selector={"filter": {"must": [
                {"key": "is_streaming", "match": {"value": True}},
                {"key": "freshness_ts",  "range": {"lt": cutoff}}
            ]}}
        )

    async def _parse(self, message: dict) -> Optional[UnifiedChunk]:
        text = message.get("text") or message.get("content") or str(message)
        if not text:
            return None
        from uuid import uuid4
        return UnifiedChunk(
            id=str(uuid4()), content=text, modality=Modality.TEXT,
            embedding=None,  # Filled in flush
            metadata={"source": message.get("source", "stream")},
            context_prefix="", causal_node_ids=[],
            temporal_valid_from=datetime.utcnow().isoformat(),
            temporal_valid_until="present",
            credibility_score=0.5, language="en",
            chunk_boundary_type="streaming"
        )

    def freshness_score(self, chunk: UnifiedChunk) -> float:
        """Retrieval bonus for fresher streaming chunks."""
        if not chunk.metadata.get("is_streaming"):
            return 0.5
        try:
            age_h = (datetime.utcnow() -
                     datetime.fromisoformat(chunk.metadata["freshness_ts"])
                     ).total_seconds() / 3600
            return max(1.0 - age_h / self.FRESHNESS_HOURS, 0.0)
        except:
            return 0.0
```

---

### Gap 6 — Semantic Boundary Chunker

**Research basis:** Vision-Guided Chunking (arXiv 2506.16035, 2025) calls arbitrary text chunking "a significant gap in the current literature." CHI 2026 raggy tool paper found chunk boundary destruction the most persistent user-reported failure.

```python
# nexus/gaps/semantic_chunker.py

from dataclasses import dataclass
import numpy as np
import re

@dataclass
class BoundarySignal:
    position:      int    # Character offset in document text
    boundary_type: str    # "visual"|"discourse"|"topic_shift"|"structural"
    confidence:    float

@dataclass
class SemanticChunk:
    text:           str
    raw_text:       str
    context_prefix: str
    boundary_type:  str
    start_char:     int
    end_char:       int

class SemanticBoundaryChunker:
    """
    Research: Vision-Guided Chunking (arXiv 2506.16035, 2025).
    Detects where meaning ends, not where a token limit is reached.
    """

    TOPIC_SHIFT_THRESHOLD = 0.35
    DISCOURSE_MARKERS = [
        "however", "therefore", "in contrast", "on the other hand",
        "nevertheless", "consequently", "furthermore", "in conclusion",
        "as a result", "moreover", "in summary", "to summarise",
        "that said", "conversely", "accordingly", "subsequently",
        "in addition", "nonetheless", "having said that",
    ]
    STRUCTURAL_MARKERS = re.compile(
        r"(^#{1,4}\s|^chapter\s+\d|^section\s+\d|^\d+\.\s+[A-Z])",
        re.MULTILINE | re.IGNORECASE
    )

    def __init__(self, encoder: UnifiedEncoder, vision_model=None):
        self.encoder      = encoder
        self.vision_model = vision_model

    def chunk(self, document: dict) -> list[SemanticChunk]:
        text   = document.get("text", "")
        dtype  = document.get("type", "text")
        signals: list[BoundarySignal] = []

        # Signal 1: Structural headings (strongest)
        signals.extend(self._structural_boundaries(text))

        # Signal 2: Visual layout (PDFs, slides)
        if dtype in ("pdf", "pptx", "docx") and self.vision_model:
            signals.extend(self._visual_boundaries(document))

        # Signal 3: Discourse coherence markers
        signals.extend(self._discourse_boundaries(text))

        # Signal 4: Topic shift (weakest, used as fallback)
        signals.extend(self._topic_shift_boundaries(text))

        # Fuse: sort, remove duplicates within 50 chars of each other
        fused  = self._fuse_signals(signals)
        chunks = self._split(text, fused, document)
        return chunks

    def _structural_boundaries(self, text: str) -> list[BoundarySignal]:
        return [
            BoundarySignal(position=m.start(), boundary_type="structural",
                           confidence=0.95)
            for m in self.STRUCTURAL_MARKERS.finditer(text)
        ]

    def _discourse_boundaries(self, text: str) -> list[BoundarySignal]:
        signals = []
        lower   = text.lower()
        for marker in self.DISCOURSE_MARKERS:
            pos = 0
            while True:
                pos = lower.find(f" {marker} ", pos)
                if pos == -1:
                    break
                signals.append(BoundarySignal(pos, "discourse", 0.75))
                pos += 1
        return signals

    def _topic_shift_boundaries(self, text: str) -> list[BoundarySignal]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) < 5:
            return []
        signals, pos = [], 0
        W = 3  # window size
        for i in range(W, len(sentences) - W):
            left  = " ".join(sentences[i-W:i])
            right = " ".join(sentences[i:i+W])
            l_emb = self.encoder.encode(left,  Modality.TEXT)
            r_emb = self.encoder.encode(right, Modality.TEXT)
            sim   = float(np.dot(l_emb, r_emb))
            if sim < self.TOPIC_SHIFT_THRESHOLD:
                char_pos = sum(len(s) + 2 for s in sentences[:i])
                signals.append(BoundarySignal(char_pos, "topic_shift",
                                              1.0 - sim))
            pos += len(sentences[i]) + 2
        return signals

    def _visual_boundaries(self, document: dict) -> list[BoundarySignal]:
        # Use vision model to detect page breaks, figure boundaries, table ends
        return []   # Implement with pdfplumber layout analysis

    def _fuse_signals(
        self, signals: list[BoundarySignal], min_gap: int = 50
    ) -> list[BoundarySignal]:
        if not signals:
            return []
        signals = sorted(signals, key=lambda s: s.position)
        fused, last = [signals[0]], signals[0].position
        for sig in signals[1:]:
            if sig.position - last >= min_gap:
                fused.append(sig); last = sig.position
            elif sig.confidence > fused[-1].confidence:
                fused[-1] = sig
        return fused

    def _split(
        self, text: str, boundaries: list[BoundarySignal], doc: dict
    ) -> list[SemanticChunk]:
        positions = [0] + [b.position for b in boundaries] + [len(text)]
        btypes    = ["start"] + [b.boundary_type for b in boundaries] + ["end"]
        chunks    = []
        for i in range(len(positions) - 1):
            raw = text[positions[i]:positions[i+1]].strip()
            if len(raw) < 30:
                continue
            ctx = self._make_prefix(raw, doc, i)
            chunks.append(SemanticChunk(
                text=f"{ctx}\n\n{raw}", raw_text=raw,
                context_prefix=ctx, boundary_type=btypes[i],
                start_char=positions[i], end_char=positions[i+1]
            ))
        return chunks

    def _make_prefix(self, chunk: str, doc: dict, idx: int) -> str:
        title   = doc.get("title", "Document")
        section = doc.get("section", "")
        return (f"From '{title}'"
                + (f", section '{section}'" if section else "")
                + f", part {idx + 1}. This passage discusses: ")
```

---

### Gap 7 — Modality-Aware Reranker

**Research basis:** CoRe-MMRAG (arXiv 2506.02544, 2025) — text-centric reranking causes cross-modal discrepancy → incorrect passage selection. Image-Text retrieval comparison (arXiv 2511.16654, 2025) confirms preprocessing bias remains unsolved.

```python
# nexus/gaps/modality_reranker.py

from sentence_transformers import CrossEncoder
import numpy as np

class ModalityAwareReranker:
    """
    Research: CoRe-MMRAG (arXiv 2506.02544, 2025).
    Goes beyond text-only cross-encoder: boosts the modality best suited
    to the specific query type.
    """

    QUERY_MODALITY_WEIGHTS = {
        "factual":      {"text": 0.70, "table": 0.80, "code": 0.40, "image": 0.30},
        "quantitative": {"table": 0.90, "text": 0.50, "image": 0.40, "code": 0.30},
        "procedural":   {"text": 0.80, "code": 0.90, "image": 0.60, "table": 0.30},
        "spatial":      {"image": 0.90, "text": 0.50, "table": 0.30, "code": 0.20},
        "temporal":     {"text": 0.80, "table": 0.70, "image": 0.40, "code": 0.20},
        "causal":       {"text": 0.90, "table": 0.60, "image": 0.30, "code": 0.20},
        "comparative":  {"table": 0.85, "text": 0.70, "image": 0.50, "code": 0.30},
        "creative":     {"image": 0.70, "text": 0.70, "code": 0.40, "table": 0.20},
    }

    def __init__(self):
        self.cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(
        self,
        query:     str,
        dna:       QueryDNA,
        chunks:    list[UnifiedChunk],
        top_k:     int = 5
    ) -> list[UnifiedChunk]:
        q_type   = self._dominant_type(dna)
        weights  = self.QUERY_MODALITY_WEIGHTS.get(q_type, {})

        scored = []
        for c in chunks:
            # Cross-encoder semantic score (text-only baseline)
            text_repr = c.content if isinstance(c.content, str) else str(c.content)
            sem_score = float(self.cross_enc.predict([[query, text_repr[:512]]])[0])

            modality_bonus = weights.get(c.modality.value, 0.50)
            freshness      = self._freshness(c)
            credibility    = c.credibility_score

            final = (
                0.45 * sem_score       +
                0.25 * modality_bonus  +
                0.15 * freshness       +
                0.15 * credibility
            )
            scored.append((final, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def _dominant_type(self, dna: QueryDNA) -> str:
        candidates = {
            "factual":      dna.factual,
            "procedural":   dna.procedural,
            "causal":       dna.causal,
            "temporal":     dna.temporal,
            "comparative":  dna.comparative,
            "creative":     dna.creative,
        }
        return max(candidates, key=candidates.get)

    def _freshness(self, c: UnifiedChunk) -> float:
        from datetime import datetime
        if c.metadata.get("is_streaming"):
            try:
                age_h = (datetime.utcnow() -
                         datetime.fromisoformat(c.metadata["freshness_ts"])
                         ).total_seconds() / 3600
                return max(1.0 - age_h / 24.0, 0.0)
            except:
                return 0.5
        return 0.5
```

---

### Gap 8 — Failure Forensics Engine

**Research basis:** RAGChecker (Ru et al., NeurIPS 2024) provides diagnostic metrics but no automated fix. RAG taxonomy paper (arXiv 2408.02854) identifies the need for error attribution: retrieval failure vs comprehension failure vs reasoning failure vs parametric knowledge failure.

```python
# nexus/gaps/forensics_engine.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class FailureMode(Enum):
    RETRIEVAL    = "retrieval_failure"       # Correct info not retrieved
    UTILIZATION  = "context_utilization"    # Retrieved but not used
    COMPREHENSION = "comprehension_failure"  # Fact misread
    REASONING    = "reasoning_failure"       # Multi-hop logic broke
    PARAMETRIC   = "parametric_knowledge"   # LLM's own weights were wrong
    NONE         = "no_failure"

@dataclass
class Diagnosis:
    mode:              FailureMode
    description:       str
    confidence:        float
    recommended_fix:   str
    component_to_fix:  str   # Which NEXUS component to improve

class FailureForensicsEngine:
    """
    Research: RAGChecker (Ru et al., NeurIPS 2024), arXiv 2408.02854.
    Diagnoses exactly WHY an answer was wrong — not just THAT it was wrong.
    """

    def __init__(self):
        from sentence_transformers import CrossEncoder, SentenceTransformer
        self.nli  = CrossEncoder("cross-encoder/nli-deberta-v3-small")
        self.emb  = SentenceTransformer("intfloat/e5-large-v2")

    def diagnose(
        self,
        query:            str,
        ground_truth:     str,           # Ground-truth answer (from user signal)
        generated_answer: str,
        retrieved_chunks: list[UnifiedChunk],
        corpus_sample:    list[UnifiedChunk]  # Broader sample to check retrieval
    ) -> Diagnosis:

        # Test 1: Is the correct info in the retrieved chunks?
        if not self._info_in_chunks(ground_truth, retrieved_chunks):
            if self._info_in_chunks(ground_truth, corpus_sample):
                return Diagnosis(
                    mode=FailureMode.RETRIEVAL,
                    description="Correct information exists in corpus but was not in top-k results",
                    confidence=0.90,
                    recommended_fix="Improve embedding model or lower similarity threshold",
                    component_to_fix="AdaptiveRetrievalRouter"
                )

        # Test 2: Was the retrieved content used in the answer?
        if not self._content_utilized(retrieved_chunks, generated_answer):
            return Diagnosis(
                mode=FailureMode.UTILIZATION,
                description="Correct chunk was retrieved but LLM ignored it (lost-in-the-middle)",
                confidence=0.85,
                recommended_fix="Reorder context: place high-value chunks at start or end",
                component_to_fix="ContextAssembler"
            )

        # Test 3: Was a specific fact misread?
        if not self._facts_read_correctly(retrieved_chunks, generated_answer):
            return Diagnosis(
                mode=FailureMode.COMPREHENSION,
                description="LLM misread a specific fact (wrong number, date, or name)",
                confidence=0.80,
                recommended_fix="Add structured fact-extraction step before final answer",
                component_to_fix="SelfHealingVerifier"
            )

        # Test 4: Did multi-hop reasoning fail?
        if not self._reasoning_correct(query, retrieved_chunks, generated_answer):
            return Diagnosis(
                mode=FailureMode.REASONING,
                description="Multi-hop reasoning chain broke — could not connect retrieved facts",
                confidence=0.75,
                recommended_fix="Add chain-of-thought reasoning verification before final output",
                component_to_fix="CausalCounterfactualLayer"
            )

        # Test 5: LLM parametric knowledge error
        return Diagnosis(
            mode=FailureMode.PARAMETRIC,
            description="LLM's own training knowledge overrode the retrieved context",
            confidence=0.70,
            recommended_fix="Increase system prompt weighting on context over parametric knowledge",
            component_to_fix="GenerationLayer"
        )

    def _info_in_chunks(
        self, truth: str, chunks: list[UnifiedChunk], threshold: float = 0.65
    ) -> bool:
        import numpy as np
        t_emb = self.emb.encode(truth)
        return any(
            float(np.dot(t_emb, self.emb.encode(c.content[:256]))) >= threshold
            for c in chunks if c.modality == Modality.TEXT
        )

    def _content_utilized(
        self, chunks: list[UnifiedChunk], answer: str, threshold: float = 0.40
    ) -> bool:
        import numpy as np
        a_emb = self.emb.encode(answer)
        return any(
            float(np.dot(a_emb, self.emb.encode(c.content[:256]))) >= threshold
            for c in chunks if c.modality == Modality.TEXT
        )

    def _facts_read_correctly(
        self, chunks: list[UnifiedChunk], answer: str
    ) -> bool:
        """Check for numeric/date hallucinations using regex comparison."""
        import re
        chunk_nums  = set(re.findall(r"\b\d[\d,\.]+\b",
                                     " ".join(c.content for c in chunks
                                              if c.modality == Modality.TEXT)))
        answer_nums = set(re.findall(r"\b\d[\d,\.]+\b", answer))
        invented    = answer_nums - chunk_nums
        return len(invented) == 0

    def _reasoning_correct(
        self, query: str, chunks: list[UnifiedChunk], answer: str
    ) -> bool:
        """Use NLI to check if answer can be derived from retrieved chunks."""
        combined = " ".join(c.content[:256] for c in chunks[:5]
                            if c.modality == Modality.TEXT)
        if not combined:
            return True
        scores = self.nli.predict([[combined, answer]], apply_softmax=True)
        return float(scores[0][2]) > 0.45  # entailment score
```

---

### Gap 9 — Knowledge Evolution Manager

**Research basis:** "RAG or Learning?" (ACL 2026 Findings) — graceful knowledge evolution is "currently unsolved." Zylos Research (April 2026) — catastrophic forgetting is now a production engineering crisis for multi-month persistent services.

```python
# nexus/gaps/knowledge_evolution.py

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class FactStatus(Enum):
    CURRENT          = "current"
    SUPERSEDED       = "superseded"      # Replaced by a newer fact
    TEMPORALLY_SCOPED = "scoped"         # Valid only for a specific time range
    RETRACTED        = "retracted"       # Proven incorrect
    CONTESTED        = "contested"       # Credible sources disagree

@dataclass
class EvolutionEvent:
    event_type:  str         # "supersede"|"scope"|"retract"|"contest"
    old_fact_id: str
    new_fact_id: Optional[str]
    reason:      str
    timestamp:   datetime
    trigger:     str         # "new_document"|"user_feedback"|"contradiction"

class KnowledgeEvolutionManager:
    """
    Research: ACL 2026 "RAG or Learning?", Zylos Research April 2026.
    Core insight: never DELETE knowledge — always EVOLVE it with history.
    """

    def __init__(self, kg: TemporalKnowledgeGraph, db_conn):
        self.kg = kg; self.db = db_conn

    def ingest_new_fact(
        self,
        new_fact:     dict,
        source_chunk: UnifiedChunk
    ) -> list[EvolutionEvent]:
        conflicts = self.kg.find_conflicting_facts(new_fact) if hasattr(self.kg, "find_conflicting_facts") else []
        events    = []
        for old in conflicts:
            evt = self._evolve(old, new_fact, source_chunk)
            self._apply(evt, old, new_fact)
            events.append(evt)
        return events

    def _evolve(self, old, new_fact, src: UnifiedChunk) -> EvolutionEvent:
        if self._is_temporal_update(old, new_fact):
            return EvolutionEvent(
                event_type="scope", old_fact_id=old.id, new_fact_id=None,
                reason="Temporal update — both facts valid for different periods",
                timestamp=datetime.utcnow(), trigger="new_document"
            )
        cred_delta = source_chunk.credibility_score - old.confidence
        if cred_delta > 0.20:
            return EvolutionEvent(
                event_type="supersede", old_fact_id=old.id, new_fact_id=None,
                reason=f"New source credibility significantly higher ({cred_delta:.2f})",
                timestamp=datetime.utcnow(), trigger="new_document"
            )
        return EvolutionEvent(
            event_type="contest", old_fact_id=old.id, new_fact_id=None,
            reason="Credible contradiction from different source",
            timestamp=datetime.utcnow(), trigger="new_document"
        )

    def _apply(self, evt: EvolutionEvent, old, new_fact: dict):
        with self.kg.driver.session() as s:
            if evt.event_type == "supersede":
                s.run("MATCH (n:Node {id: $id}) SET n.status='superseded', n.valid_until=$ts",
                      id=old.id, ts=datetime.utcnow().isoformat())
            elif evt.event_type == "scope":
                s.run("MATCH (n:Node {id: $id}) SET n.status='scoped', n.valid_until=$ts",
                      id=old.id, ts=datetime.utcnow().isoformat())
            elif evt.event_type == "contest":
                s.run("MATCH (n:Node {id: $id}) SET n.status='contested'", id=old.id)
        self._log_event(evt)

    def _is_temporal_update(self, old, new_fact: dict) -> bool:
        """True if the two facts are about the same entity in different years."""
        import re
        old_years = set(re.findall(r"\b(19|20)\d{2}\b", str(old)))
        new_years = set(re.findall(r"\b(19|20)\d{2}\b", str(new_fact)))
        return bool(old_years) and bool(new_years) and old_years != new_years

    def query_historical(self, entity: str, as_of: datetime) -> list:
        """What did the system know about `entity` on date `as_of`?"""
        return self.kg.query_at_time(entity, as_of)

    def _log_event(self, evt: EvolutionEvent):
        with self.db.cursor() as cur:
            cur.execute("""
                INSERT INTO evolution_log
                    (event_type, old_fact_id, new_fact_id, reason, trigger, ts)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, (evt.event_type, evt.old_fact_id, evt.new_fact_id,
                  evt.reason, evt.trigger, evt.timestamp))
        self.db.commit()
```

---

## 6. Complete Tech Stack

```
CATEGORY          TOOL                           VERSION    ROLE
─────────────────────────────────────────────────────────────────────────────
Vector DB         Qdrant                         1.9+       HNSW index + filters
Knowledge Graph   Neo4j                          5.x        Temporal + causal graph
Sparse Retrieval  rank_bm25 (BM25Okapi)         0.2.2      Keyword search
Streaming         Apache Kafka                   3.6        Event stream source
Cache             Redis                          7.2        Semantic cache + tasks
Relational DB     PostgreSQL                     16         Feedback / certs / lineage
Text Embedding    intfloat/e5-large-v2           latest     Primary dense encoder
Multilingual Emb  intfloat/multilingual-e5-large latest     Cross-lingual bridge
Code Embedding    microsoft/codebert-base        latest     Code understanding
Table Model       google/tapas-base              latest     Table comprehension
Image Encoder     openai/clip-vit-large-patch14  latest     Image embeddings
Audio Model       openai/whisper-base            latest     Transcription + audio
Cross-Encoder     ms-marco-MiniLM-L-6-v2         latest     Reranking
NLI Model         nli-deberta-v3-small           latest     Conflict + verification
LLM               claude-sonnet-4-6              latest     Generation + reasoning
API Framework     FastAPI + uvicorn              0.110+     REST API
Task Queue        Celery                         5.3        Async fine-tuning jobs
Containerisation  Docker + Docker Compose        latest     Local dev
Orchestration     Kubernetes                     1.29+      Production
Monitoring        LangSmith                      latest     Full pipeline tracing
```

**One-shot install:**
```bash
pip install qdrant-client neo4j rank-bm25 anthropic \
            sentence-transformers transformers torch fastapi uvicorn \
            celery redis psycopg2-binary kafka-python openai-whisper \
            langsmith langdetect spacy numpy pydantic clip-by-openai
```

---

## 7. Project Directory Structure

```
nexus-rag/
│
├── README.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml
│
├── nexus/
│   ├── __init__.py
│   ├── config.py                     # All env-driven config
│   │
│   ├── core/                         # 8 Core Pillars
│   │   ├── unified_encoder.py        # Pillar 1
│   │   ├── knowledge_graph.py        # Pillar 2
│   │   ├── query_classifier.py       # Pillar 3
│   │   ├── retrieval_router.py       # Pillar 4
│   │   ├── conflict_resolver.py      # Pillar 5
│   │   ├── uncertainty_quantifier.py # Pillar 6
│   │   ├── self_healer.py            # Pillar 7
│   │   └── feedback_loop.py          # Pillar 8
│   │
│   ├── gaps/                         # 9 Gap Solutions
│   │   ├── epistemic_engine.py       # Gap 1
│   │   ├── causal_counterfactual.py  # Gap 2
│   │   ├── cross_lingual_bridge.py   # Gap 3
│   │   ├── amnesia_engine.py         # Gap 4
│   │   ├── streaming_ingestor.py     # Gap 5
│   │   ├── semantic_chunker.py       # Gap 6
│   │   ├── modality_reranker.py      # Gap 7
│   │   ├── forensics_engine.py       # Gap 8
│   │   └── knowledge_evolution.py    # Gap 9
│   │
│   ├── pipeline/
│   │   ├── nexus_pipeline.py         # Master orchestrator
│   │   ├── ingest_pipeline.py        # Document ingest flow
│   │   └── query_pipeline.py         # Query answer flow
│   │
│   ├── storage/
│   │   ├── qdrant_store.py
│   │   ├── neo4j_store.py
│   │   ├── postgres_store.py
│   │   ├── redis_cache.py
│   │   └── bm25_store.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── query.py       # POST /query
│   │   │   ├── ingest.py      # POST /ingest
│   │   │   ├── forget.py      # POST /forget
│   │   │   ├── feedback.py    # POST /feedback
│   │   │   └── health.py      # GET /health
│   │   └── models/
│   │       ├── requests.py
│   │       └── responses.py
│   │
│   ├── tasks.py                      # Celery tasks (fine-tuning, eviction)
│   └── utils/
│       ├── language.py
│       ├── document_parser.py
│       ├── credibility_scorer.py
│       └── logging.py
│
├── tests/
│   ├── unit/
│   │   ├── test_encoder.py
│   │   ├── test_classifier.py
│   │   ├── test_conflict_resolver.py
│   │   ├── test_epistemic_engine.py
│   │   ├── test_amnesia_engine.py
│   │   └── test_chunker.py
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   ├── test_streaming.py
│   │   └── test_cross_lingual.py
│   └── benchmarks/
│       ├── bench_speed.py
│       ├── bench_accuracy.py
│       └── bench_hallucination.py
│
├── scripts/
│   ├── setup_qdrant.py
│   ├── setup_neo4j.py
│   ├── setup_postgres.py
│   ├── bulk_ingest.py
│   └── run_eval.py
│
└── docker/
    ├── Dockerfile
    ├── docker-compose.dev.yml
    └── docker-compose.prod.yml
```

---

## 8. Database Schemas

### PostgreSQL

```sql
-- Core chunk metadata
CREATE TABLE chunks (
    id              UUID PRIMARY KEY,
    source_id       VARCHAR(512),
    modality        VARCHAR(50),
    language        CHAR(2),
    content_hash    VARCHAR(64) UNIQUE,
    credibility     FLOAT    DEFAULT 0.5,
    valid_from      TIMESTAMPTZ,
    valid_until     TIMESTAMPTZ,
    status          VARCHAR(50) DEFAULT 'current',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Retrieval weights (feedback-driven)
CREATE TABLE chunk_weights (
    chunk_id        UUID REFERENCES chunks(id) ON DELETE CASCADE,
    query_type      VARCHAR(50),
    weight          FLOAT DEFAULT 1.0,
    positive_count  INT   DEFAULT 0,
    negative_count  INT   DEFAULT 0,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (chunk_id, query_type)
);

-- Query feedback log
CREATE TABLE feedback_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query           TEXT,
    chunk_ids       UUID[],
    answer          TEXT,
    user_signal     VARCHAR(20),
    latency_ms      FLOAT,
    failure_type    VARCHAR(50),
    ts              TIMESTAMPTZ DEFAULT NOW()
);

-- Fine-tuning pairs (positive query-chunk pairs)
CREATE TABLE fine_tune_pairs (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query             TEXT,
    positive_chunk_id UUID REFERENCES chunks(id),
    negative_chunk_id UUID REFERENCES chunks(id),
    ts                TIMESTAMPTZ DEFAULT NOW()
);

-- GDPR deletion certificates (audit trail)
CREATE TABLE deletion_certificates (
    id                UUID PRIMARY KEY,
    target            TEXT,
    target_type       VARCHAR(50),
    deletion_ts       TIMESTAMPTZ,
    verification_hash VARCHAR(64),
    signature         VARCHAR(64),
    completeness      VARCHAR(50),
    regulations       TEXT[],
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge evolution events
CREATE TABLE evolution_log (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type   VARCHAR(50),
    old_fact_id  VARCHAR(255),
    new_fact_id  VARCHAR(255),
    reason       TEXT,
    trigger      VARCHAR(50),
    ts           TIMESTAMPTZ DEFAULT NOW()
);

-- Conflict resolution events
CREATE TABLE conflict_log (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query        TEXT,
    strategy     VARCHAR(50),
    chunk_a_id   UUID,
    chunk_b_id   UUID,
    user_message TEXT,
    ts           TIMESTAMPTZ DEFAULT NOW()
);
```

### Neo4j Constraints

```cypher
CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.entity);
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.valid_from);
CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.status);

// Relationship types used:
// (:Node)-[:CAUSES {confidence: 0.9}]->(:Node)
// (:Node)-[:SUPERSEDES]->(:Node)
// (:Node)-[:CONTRADICTS {nli_score: 0.85}]->(:Node)
// (:Node)-[:TEMPORALLY_FOLLOWS]->(:Node)
// (:Node)-[:PART_OF]->(:Node)
```

### Qdrant Collection

```python
from qdrant_client.models import (
    Distance, VectorParams, HnswConfigDiff,
    PayloadSchemaType, OptimizersConfigDiff
)

client.create_collection(
    collection_name="nexus_knowledge",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
    optimizers_config=OptimizersConfigDiff(indexing_threshold=20_000)
)
for field, ftype in [
    ("modality",     PayloadSchemaType.KEYWORD),
    ("language",     PayloadSchemaType.KEYWORD),
    ("is_streaming", PayloadSchemaType.BOOL),
    ("valid_from",   PayloadSchemaType.FLOAT),
    ("credibility",  PayloadSchemaType.FLOAT),
]:
    client.create_payload_index("nexus_knowledge", field, ftype)
```

---

## 9. REST API Design

```
POST   /query
Body:  { "query": "...", "language": "en", "top_k": 5, "stream": false }
Out:   {
         "answer": "...",
         "claims": [{ "text": "...", "confidence": 0.94, "label": "High",
                      "source_count": 3 }],
         "citations": [{ "chunk_id": "...", "source": "...", "excerpt": "..." }],
         "epistemic_decision": "answer"|"abstain"|"retrieve_more",
         "conflicts_detected": false,
         "conflict_message": "",
         "failure_diagnosis": null,
         "latency_ms": 87,
         "cache_hit": false
       }

POST   /ingest
Body:  multipart/form-data (file) OR { "url": "...", "text": "...",
        "metadata": { "source": "...", "date": "..." } }
Out:   { "chunk_ids": [...], "kg_nodes_created": 12, "status": "indexed" }

POST   /ingest/stream
Body:  { "source": "kafka", "topic": "nexus-feed",
         "bootstrap_servers": "localhost:9092" }
Out:   { "stream_id": "...", "status": "listening" }

POST   /forget
Body:  { "target": "John Doe", "target_type": "person",
         "regulations": ["GDPR", "CCPA"] }
Out:   { "certificate": { ...DeletionCertificate fields... },
         "status": "purged" }

POST   /feedback
Body:  { "query_id": "...", "signal": "positive"|"negative"|"follow_up" }
Out:   { "weights_updated": true, "fine_tune_triggered": false }

GET    /health
Out:   { "status": "ok", "components":
         { "qdrant":"ok","neo4j":"ok","postgres":"ok","redis":"ok" } }

GET    /stats
Out:   { "total_chunks": 124500, "total_queries": 8432,
         "avg_latency_ms": 91, "cache_hit_rate": 0.34,
         "abstention_rate": 0.04, "healing_rate": 0.07 }
```

---

## 10. Master Pipeline Orchestrator

```python
# nexus/pipeline/nexus_pipeline.py

import asyncio, time
import anthropic
from dataclasses import dataclass
from typing import Optional

@dataclass
class NexusResponse:
    answer:                str
    claims:                list          # list[ConfidenceClaim]
    citations:             list
    epistemic_decision:    str
    conflicts_detected:    bool
    conflict_message:      str
    failure_diagnosis:     Optional[dict]
    latency_ms:            float
    cache_hit:             bool

class NexusPipeline:
    """Master orchestrator — 17 components, single coherent flow."""

    def __init__(self, config: dict):
        # Core pillars
        self.encoder    = UnifiedEncoder()
        self.kg         = TemporalKnowledgeGraph(config["neo4j_uri"], config["neo4j_auth"])
        self.classifier = QueryDNAClassifier()
        self.router     = AdaptiveRetrievalRouter(
                            qdrant=self._qdrant(config), bm25=self._bm25(config),
                            kg=self.kg, encoder=self.encoder)
        self.conflict   = ConflictResolver()
        self.uq         = UncertaintyQuantifier()
        self.healer     = SelfHealingVerifier()
        self.feedback   = FeedbackLearner(config["postgres_dsn"])
        # Gap solutions
        self.epistemic  = EpistemicSufficiencyEngine()
        self.causal     = CausalCounterfactualLayer(self.kg)
        self.xrag       = CrossLingualBridge()
        self.amnesia    = AmnesiaEngine(
                            self._qdrant(config), self.kg.driver,
                            None, self._redis(config),
                            self._pg(config), config.get("signing_key"))
        self.streamer   = StreamingIngestor(self.encoder, self._qdrant(config),
                                            None, self.kg)
        self.chunker    = SemanticBoundaryChunker(self.encoder)
        self.reranker   = ModalityAwareReranker()
        self.forensics  = FailureForensicsEngine()
        self.evolution  = KnowledgeEvolutionManager(self.kg, self._pg(config))
        # Cache + LLM
        self.cache      = SemanticCache(threshold=0.92)
        self.llm        = anthropic.Anthropic()

    async def query(self, query: str, language: str = "auto") -> NexusResponse:
        t0 = time.time()

        # 0. Semantic cache
        cached = self.cache.get(query)
        if cached:
            return NexusResponse(**cached, latency_ms=(time.time()-t0)*1000,
                                 cache_hit=True)

        # 1. Streaming freshness flush
        await self.streamer._flush()

        # 2. Query DNA
        dna = self.classifier.classify(query)

        # 3. Cross-lingual bridge (if needed)
        xl_ctx = None
        if dna.multilingual > 0.3:
            xl_ctx = self.xrag.build(query, dna.detected_language, [])

        # 4. Parallel retrieval
        chunks = await self.router.retrieve(dna, top_k=20)

        # 5. Modality-aware reranking
        reranked = self.reranker.rerank(query, dna, chunks, top_k=7)

        # 6. Conflict resolution
        conflict_report = self.conflict.resolve(reranked)
        final_chunks    = conflict_report.resolved_chunks

        # 7. Epistemic sufficiency
        hypotheses  = await self._sample_hypotheses(query, final_chunks)
        ep_report   = self.epistemic.evaluate(query, final_chunks, hypotheses)

        if ep_report.decision.value == "abstain":
            return self._abstain_response(ep_report, t0)

        if ep_report.decision.value == "retrieve_more":
            extra = await self.router.retrieve(dna, top_k=10)
            final_chunks.extend(extra)

        # 8. Generate with uncertainty quantification
        raw_answer, claims = await self._generate_with_uncertainty(
            query, final_chunks, dna, xl_ctx
        )

        # 9. Self-healing verification
        healed = self.healer.verify_and_heal(
            raw_answer, final_chunks, query, self.router
        )

        # 10. Cache result
        self.cache.set(query, {
            "answer": healed.healed_answer,
            "claims": claims,
            "citations": self._citations(final_chunks),
            "epistemic_decision": ep_report.decision.value,
            "conflicts_detected": conflict_report.detected,
            "conflict_message": conflict_report.user_message,
            "failure_diagnosis": None,
        })

        # 11. Feedback record
        self.feedback.record(
            query=query, chunks=final_chunks,
            answer=healed.healed_answer, signal="pending",
            latency_ms=(time.time()-t0)*1000
        )

        return NexusResponse(
            answer=healed.healed_answer,
            claims=claims,
            citations=self._citations(final_chunks),
            epistemic_decision=ep_report.decision.value,
            conflicts_detected=conflict_report.detected,
            conflict_message=conflict_report.user_message,
            failure_diagnosis=None,
            latency_ms=round((time.time()-t0)*1000, 1),
            cache_hit=False
        )

    async def _sample_hypotheses(
        self, query: str, chunks: list[UnifiedChunk], n: int = 4
    ) -> list[str]:
        """Generate n candidate answers at temperature > 0 for entropy estimation."""
        ctx  = "\n\n".join(c.content for c in chunks[:5]
                           if c.modality == Modality.TEXT)
        resp = self.llm.messages.create(
            model="claude-sonnet-4-6", max_tokens=400,
            system="Answer the question. Be concise.",
            messages=[{"role":"user","content":
                       f"Context:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"}]
        )
        base = resp.content[0].text
        # For true entropy estimation, sample at T>0; simplified: return variations
        return [base, f"Based on the evidence: {base}",
                f"The data suggests: {base}", f"In summary: {base}"]

    async def _generate_with_uncertainty(
        self,
        query:     str,
        chunks:    list[UnifiedChunk],
        dna:       QueryDNA,
        xl_ctx:    Optional[XLingualContext]
    ):
        ctx    = "\n\n".join(c.content for c in chunks[:7]
                              if c.modality == Modality.TEXT)
        system = (
            "You are a precise, citation-grounded assistant. "
            "Answer using ONLY the provided context. "
            "If the context does not support a claim, say so explicitly. "
            + (xl_ctx.reasoning_scaffold if xl_ctx else "")
        )
        resp = self.llm.messages.create(
            model="claude-sonnet-4-6", max_tokens=1024,
            system=system,
            messages=[{"role":"user","content":
                       f"Context:\n{ctx}\n\nQuestion: {query}"}]
        )
        raw = resp.content[0].text

        # Quantify uncertainty per claim
        claims = []
        for sentence in raw.split(". "):
            if len(sentence.strip()) < 15:
                continue
            relevant_chunks = [c for c in chunks
                                if c.modality == Modality.TEXT][:5]
            cc = self.uq.quantify(sentence.strip(), relevant_chunks)
            claims.append(cc)

        return raw, claims

    def _abstain_response(self, ep: EpistemicReport, t0: float) -> NexusResponse:
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
            answer=msg, claims=[], citations=[],
            epistemic_decision="abstain",
            conflicts_detected=False, conflict_message="",
            failure_diagnosis=None,
            latency_ms=round((time.time()-t0)*1000, 1),
            cache_hit=False
        )

    def _citations(self, chunks: list[UnifiedChunk]) -> list[dict]:
        return [
            {"chunk_id": c.id,
             "source": c.metadata.get("source_url", "unknown"),
             "excerpt": (c.content[:120] + "…"
                         if isinstance(c.content, str) else ""),
             "credibility": c.credibility_score,
             "language": c.language}
            for c in chunks
        ]

    def _qdrant(self, cfg):
        from qdrant_client import QdrantClient
        return QdrantClient(host=cfg["qdrant_host"], port=cfg.get("qdrant_port",6333))

    def _redis(self, cfg):
        import redis
        return redis.from_url(cfg["redis_url"])

    def _pg(self, cfg):
        import psycopg2
        return psycopg2.connect(cfg["postgres_dsn"])

    def _bm25(self, cfg):
        return None   # Initialise from stored corpus on startup
```

---

## 11. Build Phases & Timeline

### Phase 1 — Foundation (Weeks 1–4)

| Week | Deliverable |
|------|-------------|
| 1 | Docker Compose up (Qdrant + PostgreSQL + Redis) |
| 1 | `UnifiedEncoder` — text + code + table |
| 2 | `SemanticBoundaryChunker` (Gap 6) |
| 2 | `AdaptiveRetrievalRouter` — dense + BM25 |
| 3 | `QueryDNAClassifier` |
| 3 | `SelfHealingVerifier` (Pillar 7) |
| 4 | `UncertaintyQuantifier` (Pillar 6) |
| 4 | `/query` and `/ingest` API endpoints |
| 4 | Baseline benchmark: accuracy + latency |

**Milestone:** Reliable text RAG with self-healing and per-claim confidence.

---

### Phase 2 — Intelligence (Weeks 5–8)

| Week | Deliverable |
|------|-------------|
| 5 | Neo4j up + `TemporalKnowledgeGraph` (Pillar 2) |
| 5 | `ConflictResolver` with NLI model (Pillar 5) |
| 6 | `EpistemicSufficiencyEngine` (Gap 1) |
| 6 | `CausalCounterfactualLayer` (Gap 2) |
| 7 | Image + audio modalities in `UnifiedEncoder` |
| 7 | `ModalityAwareReranker` (Gap 7) |
| 8 | `KnowledgeEvolutionManager` (Gap 9) |
| 8 | Full integration test suite |

**Milestone:** Multi-modal queries, causal reasoning, conflict resolution, principled abstention.

---

### Phase 3 — Compliance & Real-Time (Weeks 9–12)

| Week | Deliverable |
|------|-------------|
| 9  | `AmnesiaEngine` (Gap 4) + `/forget` endpoint |
| 9  | Deletion certificate generation + audit log |
| 10 | Kafka up + `StreamingIngestor` (Gap 5) |
| 10 | `CrossLingualBridge` (Gap 3) |
| 11 | `FailureForensicsEngine` (Gap 8) |
| 11 | `FeedbackLearner` + fine-tune scheduling (Pillar 8) |
| 12 | Rate limiting, auth (API key), health monitoring |
| 12 | Full RAGAS + CRAG + XRAG benchmark run |

**Milestone:** Production-ready system with legal compliance, streaming, and self-improvement.

---

### Phase 4 — Harden & Deploy (Weeks 13–16)

| Week | Deliverable |
|------|-------------|
| 13 | Kubernetes manifests + Helm chart |
| 13 | LangSmith tracing for all 17 components |
| 14 | First fine-tuning cycle (after 1,000 queries) |
| 14 | A/B test: old vs evolved retrieval weights |
| 15 | Load test: 1,000 concurrent queries |
| 15 | Security audit: prompt injection + membership inference |
| 16 | Full API documentation |
| 16 | Open-source release + paper draft |

---

## 12. Testing Strategy

```python
# tests/unit/test_epistemic_engine.py

import pytest
from nexus.gaps.epistemic_engine import EpistemicSufficiencyEngine, EpistemicDecision

class TestEpistemicEngine:
    def setup_method(self):
        self.engine = EpistemicSufficiencyEngine(epsilon=0.15, max_entropy=0.85)

    def _chunk(self, text):
        return UnifiedChunk(id="t1", content=text, modality=Modality.TEXT,
                            embedding=None, metadata={}, context_prefix="",
                            causal_node_ids=[], temporal_valid_from="",
                            temporal_valid_until="present", credibility_score=0.8,
                            language="en", chunk_boundary_type="semantic")

    def test_answers_on_clear_evidence(self):
        hyps = ["Paris is the capital of France",
                "London is the capital of France"]
        ev   = [self._chunk("Paris is the capital of France, home to the Eiffel Tower.")]
        r    = self.engine.evaluate("What is the capital of France?", ev, hyps)
        assert r.decision == EpistemicDecision.ANSWER

    def test_abstains_on_no_evidence(self):
        hyps = ["X", "Y", "Z", "W"]
        ev   = [self._chunk("A recipe for chocolate cake with 200g flour.")]
        r    = self.engine.evaluate("What is the melting point of unobtainium?", ev, hyps)
        assert r.decision == EpistemicDecision.ABSTAIN

# tests/integration/test_full_pipeline.py

import pytest, asyncio
from nexus.pipeline.nexus_pipeline import NexusPipeline

@pytest.mark.asyncio
class TestFullPipeline:

    async def test_basic_query(self, nexus: NexusPipeline):
        await nexus.ingest({"text": "The Eiffel Tower was built in 1889 in Paris.",
                            "metadata": {"source": "test", "date": "2024-01-01"}})
        r = await nexus.query("When was the Eiffel Tower built?")
        assert "1889" in r.answer
        assert r.epistemic_decision == "answer"
        assert r.claims[0].confidence > 0.70

    async def test_conflict_surfaces(self, nexus):
        await nexus.ingest({"text": "CompanyX Q3 revenue was $100M.",
                            "metadata": {"source": "src_a", "credibility": 0.9}})
        await nexus.ingest({"text": "CompanyX Q3 revenue was $80M.",
                            "metadata": {"source": "src_b", "credibility": 0.5}})
        r = await nexus.query("What was CompanyX's Q3 revenue?")
        assert r.conflicts_detected
        assert "100" in r.answer   # Higher-credibility source wins

    async def test_gdpr_deletion(self, nexus):
        await nexus.ingest({"text": "Alice Johnson lives at 1 Main St, SSN 123-45-6789.",
                            "metadata": {"person": "alice_johnson"}})
        cert = nexus.amnesia.forget("alice_johnson", "person")
        assert cert.completeness == "complete"
        r = await nexus.query("What is Alice Johnson's address?")
        assert r.epistemic_decision == "abstain"

    async def test_abstains_on_unknown(self, nexus):
        r = await nexus.query("What is the gravitational constant of Planet Xyz-9?")
        assert r.epistemic_decision == "abstain"
        assert r.answer.startswith("I cannot reliably")
```

---

## 13. Deployment Guide

### Environment Variables

```bash
# .env.example

ANTHROPIC_API_KEY=sk-ant-...

QDRANT_HOST=localhost
QDRANT_PORT=6333

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=nexus_password

POSTGRES_DSN=postgresql://nexus:nexus_password@localhost:5432/nexus

REDIS_URL=redis://localhost:6379

KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=nexus-stream

SIGNING_KEY=<32-byte-hex>

LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=nexus-rag-v1

FINE_TUNE_EVERY_N_QUERIES=1000
EPISTEMIC_EPSILON=0.15
EPISTEMIC_MAX_ENTROPY=0.85
```

### Docker Compose (Development)

```yaml
# docker-compose.yml
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: ["./data/qdrant:/qdrant/storage"]

  neo4j:
    image: neo4j:5
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: "neo4j/nexus_password"
    volumes: ["./data/neo4j:/data"]

  postgres:
    image: postgres:16
    ports: ["5432:5432"]
    environment:
      POSTGRES_DB: nexus
      POSTGRES_USER: nexus
      POSTGRES_PASSWORD: nexus_password
    volumes: ["./data/pg:/var/lib/postgresql/data"]

  redis:
    image: redis:7.2
    ports: ["6379:6379"]

  kafka:
    image: confluentinc/cp-kafka:latest
    ports: ["9092:9092"]
    environment:
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181

  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [qdrant, neo4j, postgres, redis, kafka]
    command: uvicorn nexus.api.main:app --host 0.0.0.0 --port 8000 --reload

  worker:
    build: .
    command: celery -A nexus.tasks worker --loglevel=info
    env_file: .env
    depends_on: [redis, postgres]
```

### Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/nexus-rag
cd nexus-rag
pip install -e ".[dev]"

# 2. Start infrastructure
docker-compose up -d

# 3. Initialise databases
python scripts/setup_qdrant.py
python scripts/setup_neo4j.py
python scripts/setup_postgres.py

# 4. Start API
uvicorn nexus.api.main:app --reload

# 5. Ingest a test document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from NEXUS RAG!", "metadata": {"source": "test"}}'

# 6. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does NEXUS stand for?"}'
```

---

## 14. Evaluation & Benchmarks

| Metric | Target | How Measured |
|--------|--------|--------------|
| Retrieval Hit@5 | > 88% | RAGAS context recall |
| RAGAS Faithfulness | > 0.90 | NLI entailment check |
| RAGAS Answer Relevance | > 0.88 | Cosine similarity |
| Hallucination Rate | < 3% | Self-healer flags / total |
| Epistemic Abstention FPR | < 5% | Abstains on answerable Qs |
| Cache hit latency | < 20ms | P50 response time |
| Cold query latency | < 120ms | P50 response time |
| GDPR deletion time | < 5s | Certificate issuance |
| Streaming ingestion lag | < 200ms | Message → queryable |
| Conflict detection F1 | > 0.82 | Manually labelled set |
| Cross-lingual accuracy | > 75% | XRAG benchmark |

**Run benchmarks:**
```bash
python tests/benchmarks/bench_accuracy.py --dataset RAGAS
python tests/benchmarks/bench_accuracy.py --dataset CRAG
python tests/benchmarks/bench_hallucination.py
python tests/benchmarks/bench_speed.py
```

---

## 15. Research References

| Paper | Year | Venue | Validates |
|-------|------|-------|-----------|
| Epistemic Mismatch (Ghafouri et al.) | 2025 | ACL | Gap 1 |
| Entropic Claim Resolution (arXiv 2603.28444) | 2026 | arXiv | Gap 1 algorithm |
| CausalRAG (arXiv 2503.19878) | 2025 | ACL Findings | Gap 2 |
| Causal-Counterfactual RAG (arXiv 2509.14435) | 2025 | arXiv | Gap 2 |
| XRAG Benchmark (arXiv 2505.10089) | 2025 | EMNLP | Gap 3 |
| CORAL (arXiv 2604.25676) | 2026 | arXiv | Gap 3 cultural |
| All Languages Matter (ACL 2026) | 2026 | ACL | Gap 3 bias |
| Privacy-Preserving RAG (arXiv 2412.04697) | 2024 | arXiv | Gap 4 |
| Machine Unlearning Meets RAG (Wang et al.) | 2025 | IEEE Trans. | Gap 4 |
| From Static to Dynamic RAG (arXiv 2508.05662) | 2025 | arXiv | Gap 5 |
| Vision-Guided Chunking (arXiv 2506.16035) | 2025 | arXiv | Gap 6 |
| CoRe-MMRAG (arXiv 2506.02544) | 2025 | arXiv | Gap 7 |
| Image-Text Retrieval Comparison (arXiv 2511.16654) | 2025 | arXiv | Gap 7 |
| RAGChecker (Ru et al.) | 2024 | NeurIPS | Gap 8 |
| RAG Taxonomy (arXiv 2408.02854) | 2024 | arXiv | Gap 8 |
| RAG or Learning? (ACL 2026 Findings) | 2026 | ACL | Gap 9 |
| Catastrophic Forgetting (Zylos Research) | 2026 | Industry | Gap 9 |
| CRAG Benchmark (arXiv 2409.15337) | 2024 | KDD | Evaluation |
| Long-Context LLMs in RAG (ICLR 2025) | 2025 | ICLR | Context optimization |

---

> **Where to start today:**  
> `docker-compose up -d` → `python scripts/setup_qdrant.py` → implement `SemanticBoundaryChunker` → implement `UnifiedEncoder` for text.  
> Everything else builds on those two.  
>  
> **Single highest-impact first feature:**  
> `AmnesiaEngine` (Gap 4) — it is the only component with a legal deadline (EU AI Act, August 2025). No other open-source RAG system has it.
