"""NEXUS RAG — Storage: PostgreSQL Store with full DDL."""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

SCHEMA_DDL = """
-- Core chunk metadata
CREATE TABLE IF NOT EXISTS chunks (
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
CREATE TABLE IF NOT EXISTS chunk_weights (
    chunk_id        UUID REFERENCES chunks(id) ON DELETE CASCADE,
    query_type      VARCHAR(50),
    weight          FLOAT DEFAULT 1.0,
    positive_count  INT   DEFAULT 0,
    negative_count  INT   DEFAULT 0,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (chunk_id, query_type)
);

-- Query feedback log
CREATE TABLE IF NOT EXISTS feedback_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query           TEXT,
    chunk_ids       UUID[],
    answer          TEXT,
    user_signal     VARCHAR(20),
    latency_ms      FLOAT,
    failure_type    VARCHAR(50),
    ts              TIMESTAMPTZ DEFAULT NOW()
);

-- Fine-tuning pairs
CREATE TABLE IF NOT EXISTS fine_tune_pairs (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query             TEXT,
    positive_chunk_id UUID,
    negative_chunk_id UUID,
    ts                TIMESTAMPTZ DEFAULT NOW()
);

-- GDPR deletion certificates
CREATE TABLE IF NOT EXISTS deletion_certificates (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target            TEXT,
    target_type       VARCHAR(50),
    deletion_ts       TIMESTAMPTZ,
    verification_hash VARCHAR(64),
    signature         VARCHAR(64),
    completeness      VARCHAR(100),
    regulations       TEXT[],
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Knowledge evolution events
CREATE TABLE IF NOT EXISTS evolution_log (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type   VARCHAR(50),
    old_fact_id  VARCHAR(255),
    new_fact_id  VARCHAR(255),
    reason       TEXT,
    trigger      VARCHAR(50),
    ts           TIMESTAMPTZ DEFAULT NOW()
);

-- Conflict resolution events
CREATE TABLE IF NOT EXISTS conflict_log (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query        TEXT,
    strategy     VARCHAR(50),
    chunk_a_id   UUID,
    chunk_b_id   UUID,
    user_message TEXT,
    ts           TIMESTAMPTZ DEFAULT NOW()
);
"""


class PostgresStore:
    """PostgreSQL connection pool with schema management."""

    def __init__(self, dsn: str = ""):
        self.dsn = dsn
        self._conn = None

    @property
    def conn(self):
        if self._conn is None and self.dsn:
            try:
                import psycopg2
                self._conn = psycopg2.connect(self.dsn)
                logger.info("Connected to PostgreSQL")
            except Exception as e:
                logger.error("PostgreSQL connection failed: %s", e)
        return self._conn

    def setup(self):
        """Run all DDL statements."""
        if not self.conn:
            logger.warning("PostgreSQL not available — skipping setup")
            return
        try:
            with self.conn.cursor() as cur:
                cur.execute(SCHEMA_DDL)
            self.conn.commit()
            logger.info("PostgreSQL schema initialized (7 tables)")
        except Exception as e:
            logger.error("PostgreSQL setup failed: %s", e)
            self.conn.rollback()

    def health(self) -> dict:
        if not self.conn:
            return {"status": "unavailable"}
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self):
        if self._conn:
            self._conn.close()
