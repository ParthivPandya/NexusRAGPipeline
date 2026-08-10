"""
NEXUS RAG — Centralized Configuration
======================================

All environment-driven configuration with Pydantic validation.
Loads from .env file or environment variables.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class NexusConfig(BaseSettings):
    """Master configuration — every tunable knob in one place."""

    # ── LLM ───────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    llm_model: str = Field(default="claude-sonnet-4-6", description="LLM model for generation")

    # ── Vector Database (Qdrant) ──────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="nexus_knowledge")

    # ── Knowledge Graph (Neo4j) ───────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="nexus_password")

    # ── Relational Database (PostgreSQL) ──────────────────────
    postgres_dsn: str = Field(
        default="postgresql://nexus:nexus_password@localhost:5432/nexus"
    )

    # ── Cache (Redis) ─────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379")

    # ── Streaming (Kafka) ─────────────────────────────────────
    kafka_bootstrap_servers: str = Field(default="localhost:9092")
    kafka_topic: str = Field(default="nexus-stream")

    # ── Security ──────────────────────────────────────────────
    signing_key: str = Field(
        default_factory=lambda: secrets.token_hex(32),
        description="HMAC signing key for deletion certificates",
    )
    api_key: str = Field(default="nexus-dev-key", description="API authentication key")

    # ── Monitoring ────────────────────────────────────────────
    langsmith_api_key: str = Field(default="")
    langsmith_project: str = Field(default="nexus-rag-v1")
    langsmith_tracing: bool = Field(default=False)

    # ── Model Configuration ───────────────────────────────────
    text_embedding_model: str = Field(default="intfloat/e5-large-v2")
    multilingual_embedding_model: str = Field(default="intfloat/multilingual-e5-large")
    code_embedding_model: str = Field(default="microsoft/codebert-base")
    table_model: str = Field(default="google/tapas-base")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    nli_model: str = Field(default="cross-encoder/nli-deberta-v3-small")
    whisper_model: str = Field(default="base")
    clip_model: str = Field(default="ViT-L/14")

    # ── Pipeline Tuning ───────────────────────────────────────
    fine_tune_every_n_queries: int = Field(default=1000)
    epistemic_epsilon: float = Field(default=0.15)
    epistemic_max_entropy: float = Field(default=0.85)
    semantic_cache_threshold: float = Field(default=0.92)
    streaming_batch_ms: int = Field(default=50)
    streaming_freshness_hours: int = Field(default=24)
    contradiction_threshold: float = Field(default=0.70)
    embedding_dim: int = Field(default=1024)

    # ── Server ────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @property
    def neo4j_auth(self) -> tuple[str, str]:
        """Return Neo4j auth as a tuple for the driver."""
        return (self.neo4j_user, self.neo4j_password)

    @property
    def signing_key_bytes(self) -> bytes:
        """Return signing key as bytes for HMAC operations."""
        return bytes.fromhex(self.signing_key) if len(self.signing_key) == 64 else self.signing_key.encode()


@lru_cache(maxsize=1)
def get_config() -> NexusConfig:
    """
    Get or create the singleton configuration.
    Uses lru_cache to ensure only one instance exists.
    """
    return NexusConfig()
