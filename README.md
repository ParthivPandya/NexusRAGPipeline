<div align="center">
  <img src="https://via.placeholder.com/150x150.png?text=NEXUS" alt="NEXUS Logo" width="120" />

  # NEXUS RAG 
  **Neural EXtensible Unified Search**

  *Enterprise-Grade Omnimodal RAG Pipeline with Epistemic Abstention & Self-Healing*

  [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Architecture: 17-Component](https://img.shields.io/badge/Architecture-17--Component-success.svg)](#architecture)
  [![EU AI Act Ready](https://img.shields.io/badge/Compliance-EU_AI_Act_Ready-purple.svg)](#compliance--gdpr)

</div>

---

## 🌟 Vision

NEXUS RAG is a production-ready, highly modular retrieval-augmented generation pipeline that prioritizes **accuracy, epistemic honesty, and transparency** over pure speed. 

It implements an advanced **17-component architecture** featuring omnimodal embeddings, a living temporal knowledge graph, causal reasoning, GDPR machine unlearning, and self-healing verification.

Every component serves three master principles:
1. **ACCURACY OVER SPEED** — but optimize speed without sacrificing accuracy
2. **HONESTY OVER COMPLETENESS** — better to abstain than to hallucinate
3. **EVOLUTION OVER STASIS** — the system must improve with every query

---

## ✨ What Makes NEXUS Different

| Capability | LangChain | LlamaIndex | GraphRAG | RAGFlow | **NEXUS** |
|---|---|---|---|---|---|
| **Unified multi-modal embedding** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Causal + counterfactual retrieval** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Principled abstention (epistemic)** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **GDPR machine unlearning** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Real-time streaming ingestion** | ⚠️ | ⚠️ | ❌ | ⚠️ | ✅ |
| **Per-claim confidence scores** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Failure forensics diagnosis** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Cross-lingual reasoning bridge** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Semantic boundary chunking** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Knowledge evolution manager** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Temporal knowledge graph** | ❌ | ❌ | ⚠️ | ❌ | ✅ |
| **Self-healing auto-correction** | ❌ | ⚠️ | ❌ | ❌ | ✅ |
| **Conflict source resolution** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Self-optimizing feedback loop** | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 🏗️ Master Architecture

```text
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

## 🚀 The 8 Core Pillars
1. **Omni-Modal Unification Engine:** Single 1024-dim embedding space for text, images, audio, video, code, and tables.
2. **Living Temporal Knowledge Graph:** Auto-builds from documents, tracking fact supersession and temporal validity.
3. **Query DNA Classifier:** Scores queries across 10 semantic dimensions to dynamically weight retrievers.
4. **Adaptive Retrieval Router:** Parallel async retrieval fusing Dense (HNSW), Sparse (BM25), Causal, and Temporal indexes.
5. **Conflict Resolution Engine:** NLI-based contradiction detection and multi-strategy resolution.
6. **Calibrated Uncertainty Quantifier:** Per-claim confidence scoring (Source Count, Recency, Agreement, Authority).
7. **Self-Healing Verifier:** Iterative fact-checking and targeted regeneration for unsupported claims.
8. **Self-Optimizing Feedback Loop:** Continual learning from user signals with automated fine-tuning.

## 🔬 The 9 Gap Solutions
1. **Epistemic Sufficiency Engine:** Shannon entropy-based dynamic termination (Answer, Abstain, or Retrieve More).
2. **Causal-Counterfactual Layer:** Answers *"What if X hadn't happened?"* using knowledge graph traversal.
3. **Cross-Lingual Reasoning Bridge:** Language-agnostic conceptual alignment and cultural context injection.
4. **Machine Unlearning / Amnesia Engine:** Cryptographically verifiable GDPR/CCPA data deletion certificates.
5. **Streaming Real-Time Ingestor:** 50ms micro-batching for Kafka/WebSocket integration with stale chunk eviction.
6. **Semantic Boundary Chunker:** Splits by meaning (structural, visual, discourse) rather than token limits.
7. **Modality-Aware Reranker:** Cross-encoder reranking tailored to the query's optimal modality.
8. **Failure Forensics Engine:** 5-mode diagnostic cascade for tracing exact failure points.
9. **Knowledge Evolution Manager:** Tracks the full lifecycle of facts (Current → Superseded | Contested | Retracted).

---

## 🛠️ Tech Stack

- **Vector Database:** Qdrant (HNSW, Cosine, 1024-dim)
- **Knowledge Graph:** Neo4j (Temporal & Causal Edges)
- **Sparse Retrieval:** BM25 (Okapi)
- **Relational / Audit:** PostgreSQL
- **Cache / Message Broker:** Redis + Kafka
- **Embeddings:** `intfloat/e5-large-v2`, `CLIP`, `Whisper`
- **Cross-Encoders:** `ms-marco-MiniLM-L-6-v2`, `nli-deberta-v3-small`
- **LLM:** Anthropic Claude (Sonnet 3.5 / Opus)
- **API:** FastAPI + Uvicorn + Celery

---

## 🚦 Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Anthropic API Key

### 2. Installation
```bash
git clone https://github.com/ParthivPandya/NexusRAGPipeline.git
cd NexusRAGPipeline

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 3. Configuration
Copy the environment template and add your API keys:
```bash
cp .env.example .env
# Edit .env to add your ANTHROPIC_API_KEY
```

### 4. Start Infrastructure
Launch Qdrant, Neo4j, PostgreSQL, Redis, and Kafka:
```bash
docker-compose up -d qdrant neo4j postgres redis zookeeper kafka
```

### 5. Initialize Databases
```bash
python scripts/setup_qdrant.py
python scripts/setup_neo4j.py
python scripts/setup_postgres.py
```

### 6. Run the API Server
```bash
docker-compose up -d api worker
# OR run locally:
# uvicorn nexus.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📖 API Documentation

Once the server is running, visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Key Endpoints

- `POST /ingest`: Ingest documents (PDF, Markdown, TXT)
- `POST /query`: Execute the full 17-component query pipeline
- `POST /feedback`: Submit user feedback for the self-optimizing loop
- `POST /forget`: Issue a GDPR-compliant machine unlearning deletion
- `GET /health`: Component health checks

---

## ⚖️ Compliance (GDPR / EU AI Act)

NEXUS RAG is designed to meet strict regulatory standards:
- **Right to Be Forgotten:** The `AmnesiaEngine` traces data lineage across all stores and issues HMAC-signed deletion certificates.
- **Traceability:** The `KnowledgeEvolutionManager` ensures facts are never silently deleted, maintaining a full audit trail of fact supersession.
- **Transparency:** The `SelfHealingVerifier` flags unsupported claims as `[⚠️ UNCERTAIN]` rather than hallucinating.

---

## 🧪 Testing

```bash
# Run unit tests (no external services needed)
pytest tests/unit/ -v

# Run integration tests (requires docker-compose infrastructure)
pytest tests/integration/ -v
```
