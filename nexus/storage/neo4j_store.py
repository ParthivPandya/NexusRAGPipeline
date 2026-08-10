"""NEXUS RAG — Storage: Neo4j Knowledge Graph Store."""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


class Neo4jStore:
    """Neo4j driver wrapper with schema initialization."""

    def __init__(self, uri: str = "", user: str = "neo4j", password: str = ""):
        self.uri = uri
        self._driver = None
        self._user = user
        self._password = password

    @property
    def driver(self):
        if self._driver is None and self.uri:
            try:
                import neo4j
                self._driver = neo4j.GraphDatabase.driver(
                    self.uri, auth=(self._user, self._password)
                )
                logger.info("Connected to Neo4j at %s", self.uri)
            except Exception as e:
                logger.error("Neo4j connection failed: %s", e)
        return self._driver

    def setup(self):
        """Create constraints and indexes."""
        if not self.driver:
            logger.warning("Neo4j not available — skipping setup")
            return
        try:
            with self.driver.session() as s:
                s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
                s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.entity)")
                s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.valid_from)")
                s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.status)")
            logger.info("Neo4j schema initialized")
        except Exception as e:
            logger.error("Neo4j setup failed: %s", e)

    def health(self) -> dict:
        if not self.driver:
            return {"status": "unavailable"}
        try:
            with self.driver.session() as s:
                s.run("RETURN 1")
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def close(self):
        if self._driver:
            self._driver.close()
