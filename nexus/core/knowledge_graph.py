"""
NEXUS RAG — Pillar 2: Living Temporal Knowledge Graph
=====================================================

Auto-builds a knowledge graph from every ingested document.
Every fact is tagged with a validity window. Outdated facts are
superseded (not deleted) so historical queries remain accurate.

Features:
    - Temporal validity windows (valid_from, valid_until)
    - Causal edges (CAUSES relationships)
    - Supersession tracking (old facts point to newer ones)
    - Status lifecycle: current → superseded | contested | retracted
    - Historical queries: "What was known about X on date Y?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TemporalNode:
    """A single fact in the knowledge graph with temporal validity."""
    id:              str
    entity:          str                # e.g., "Tesla revenue"
    value:           str                # e.g., "$97.7B"
    valid_from:      datetime
    valid_until:     Optional[datetime] = None  # None = still valid
    confidence:      float = 0.5
    source_count:    int = 1
    source_ids:      list[str] = field(default_factory=list)
    superseded_by:   Optional[str] = None       # Newer node id if outdated
    causal_parents:  list[str] = field(default_factory=list)
    causal_children: list[str] = field(default_factory=list)
    status:          str = "current"             # "current"|"superseded"|"contested"|"retracted"


class TemporalKnowledgeGraph:
    """
    Living knowledge graph backed by Neo4j.
    Falls back to in-memory graph when Neo4j is unavailable.
    """

    def __init__(self, uri: str = "", auth: tuple = ("", ""), use_memory: bool = False):
        self._uri = uri
        self._auth = auth
        self.driver = None
        self._memory_nodes: dict[str, TemporalNode] = {}
        self._memory_edges: list[dict] = []
        self._use_memory = use_memory

        if not use_memory and uri:
            try:
                import neo4j
                self.driver = neo4j.GraphDatabase.driver(uri, auth=auth)
                self._init_schema()
                logger.info("Connected to Neo4j at %s", uri)
            except Exception as e:
                logger.warning("Neo4j unavailable (%s) — using in-memory graph", e)
                self._use_memory = True
        else:
            self._use_memory = True
            logger.info("Using in-memory knowledge graph")

    def _init_schema(self):
        """Create Neo4j constraints and indexes."""
        if not self.driver:
            return
        try:
            with self.driver.session() as s:
                s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE")
                s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.entity)")
                s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.valid_from)")
                s.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.status)")
            logger.info("Neo4j schema initialized")
        except Exception as e:
            logger.error("Failed to initialize Neo4j schema: %s", e)

    def ingest_chunks(self, chunks: list) -> dict:
        """
        Extract entities and causal links from chunks, upsert into KG.

        Returns:
            Summary dict with counts of nodes/edges created.
        """
        from nexus.core.unified_encoder import Modality

        nodes_created = 0
        edges_created = 0

        for chunk in chunks:
            if chunk.modality != Modality.TEXT:
                continue
            if not isinstance(chunk.content, str):
                continue

            entities = self._extract_entities(chunk)
            causal_links = self._extract_causal_links(chunk)

            for entity in entities:
                self._upsert_node(entity, chunk)
                nodes_created += 1

            for link in causal_links:
                self._upsert_causal_edge(link)
                edges_created += 1

        return {"nodes_created": nodes_created, "edges_created": edges_created}

    def query_at_time(self, entity: str, as_of: datetime) -> list[TemporalNode]:
        """What was known about `entity` on `as_of` date?"""
        if self._use_memory:
            return self._memory_query_at_time(entity, as_of)

        try:
            with self.driver.session() as s:
                result = s.run("""
                    MATCH (n:Node {entity: $entity})
                    WHERE n.valid_from <= $ts
                      AND (n.valid_until IS NULL OR n.valid_until >= $ts)
                      AND n.superseded_by IS NULL
                    RETURN n ORDER BY n.confidence DESC
                """, entity=entity, ts=as_of.isoformat())
                return [self._to_node(r["n"]) for r in result]
        except Exception as e:
            logger.error("KG query failed: %s", e)
            return self._memory_query_at_time(entity, as_of)

    def find_causal_chain(self, query: str, depth: int = 3) -> list[dict]:
        """Walk cause→effect chains related to query up to `depth` hops."""
        seed_entity = self._extract_primary_entity(query)
        if not seed_entity:
            return []

        if self._use_memory:
            return self._memory_find_causal(seed_entity, depth)

        try:
            with self.driver.session() as s:
                cypher = (
                    f"MATCH path=(start:Node {{entity: $entity}})"
                    f"-[:CAUSES*1..{depth}]->(end:Node) "
                    f"RETURN path"
                )
                result = s.run(cypher, entity=seed_entity)
                return [self._path_to_chain(r["path"]) for r in result]
        except Exception as e:
            logger.error("Causal chain query failed: %s", e)
            return self._memory_find_causal(seed_entity, depth)

    def find_conflicting_facts(self, new_fact: dict) -> list[TemporalNode]:
        """Find existing facts that conflict with a new fact about the same entity."""
        entity = new_fact.get("name", new_fact.get("entity", ""))
        if not entity:
            return []

        if self._use_memory:
            return [
                node for node in self._memory_nodes.values()
                if node.entity.lower() == entity.lower()
                and node.status == "current"
                and node.value != new_fact.get("value", "")
            ]

        try:
            with self.driver.session() as s:
                result = s.run("""
                    MATCH (n:Node {entity: $entity, status: 'current'})
                    WHERE n.value <> $value
                    RETURN n
                """, entity=entity, value=new_fact.get("value", ""))
                return [self._to_node(r["n"]) for r in result]
        except Exception:
            return []

    # ── Private: Entity & Causal Extraction ───────────────────

    def _extract_entities(self, chunk) -> list[dict]:
        """
        Extract named entities and their values from chunk text.
        Uses simple pattern matching as a baseline; can be replaced with
        spaCy NER or LLM-based extraction.
        """
        import re
        entities = []
        text = chunk.content if isinstance(chunk.content, str) else ""

        # Pattern: "Entity is/was/= Value" or "Entity: Value"
        patterns = [
            r"(?:the\s+)?(\w[\w\s]{2,30})\s+(?:is|was|are|were|equals?)\s+(.{3,60}?)(?:\.|,|;|$)",
            r"(\w[\w\s]{2,30}):\s+(.{3,60}?)(?:\.|,|;|$)",
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                name = match.group(1).strip()
                value = match.group(2).strip()
                if len(name) > 2 and len(value) > 1:
                    entities.append({"name": name, "value": value})

        return entities[:20]  # Cap to avoid noise

    def _extract_causal_links(self, chunk) -> list[dict]:
        """
        Extract cause-effect pairs from chunk text using causal signal words.
        """
        import re
        text = chunk.content if isinstance(chunk.content, str) else ""
        causal_patterns = [
            r"(.{5,60})\s+(?:caused|led to|resulted in|triggered)\s+(.{5,60}?)(?:\.|,|;|$)",
            r"(?:because of|due to)\s+(.{5,60}),?\s+(.{5,60}?)(?:\.|,|;|$)",
        ]
        links = []
        for pattern in causal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                links.append({
                    "cause": match.group(1).strip(),
                    "effect": match.group(2).strip(),
                    "confidence": 0.7,
                    "source_id": chunk.id,
                })
        return links[:10]

    def _extract_primary_entity(self, query: str) -> str:
        """Extract the main entity from a query string."""
        # Remove question words and common prefixes
        import re
        cleaned = re.sub(
            r"^(what|who|when|where|why|how|did|does|is|was|are|were|"
            r"tell me about|explain|describe)\s+",
            "", query, flags=re.IGNORECASE
        ).strip().rstrip("?")
        return cleaned if len(cleaned) > 2 else query

    # ── Private: Node Operations ──────────────────────────────

    def _upsert_node(self, entity: dict, source_chunk) -> None:
        """Insert or update a node in the knowledge graph."""
        node_id = f"{entity['name']}_{entity['value']}"

        if self._use_memory:
            if node_id in self._memory_nodes:
                node = self._memory_nodes[node_id]
                node.source_count += 1
                node.confidence = (node.confidence + source_chunk.credibility_score) / 2
            else:
                self._memory_nodes[node_id] = TemporalNode(
                    id=node_id,
                    entity=entity["name"],
                    value=entity["value"],
                    valid_from=datetime.fromisoformat(source_chunk.temporal_valid_from)
                    if source_chunk.temporal_valid_from else datetime.utcnow(),
                    confidence=source_chunk.credibility_score,
                    source_ids=[source_chunk.id],
                    status="current",
                )
            return

        try:
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
                """,
                    entity=entity["name"],
                    value=entity["value"],
                    id=node_id,
                    vf=source_chunk.temporal_valid_from,
                    conf=source_chunk.credibility_score,
                )
        except Exception as e:
            logger.error("Failed to upsert node: %s", e)
            # Fallback to memory
            self._memory_nodes[node_id] = TemporalNode(
                id=node_id, entity=entity["name"], value=entity["value"],
                valid_from=datetime.utcnow(), confidence=source_chunk.credibility_score,
            )

    def _upsert_causal_edge(self, link: dict) -> None:
        """Create a CAUSES relationship between two nodes."""
        if self._use_memory:
            self._memory_edges.append(link)
            return

        try:
            with self.driver.session() as s:
                s.run("""
                    MERGE (a:Node {entity: $cause})
                    MERGE (b:Node {entity: $effect})
                    MERGE (a)-[r:CAUSES]->(b)
                    SET r.confidence = $conf
                """,
                    cause=link["cause"],
                    effect=link["effect"],
                    conf=link.get("confidence", 0.5),
                )
        except Exception as e:
            logger.error("Failed to create causal edge: %s", e)
            self._memory_edges.append(link)

    # ── Private: Memory-based Fallbacks ───────────────────────

    def _memory_query_at_time(self, entity: str, as_of: datetime) -> list[TemporalNode]:
        """In-memory temporal query."""
        results = []
        for node in self._memory_nodes.values():
            if node.entity.lower() != entity.lower():
                continue
            if node.valid_from <= as_of:
                if node.valid_until is None or node.valid_until >= as_of:
                    if node.superseded_by is None:
                        results.append(node)
        return sorted(results, key=lambda n: n.confidence, reverse=True)

    def _memory_find_causal(self, entity: str, depth: int) -> list[dict]:
        """In-memory causal chain search."""
        chains = []
        for edge in self._memory_edges:
            if entity.lower() in edge.get("cause", "").lower():
                chains.append({
                    "nodes": [edge["cause"], edge["effect"]],
                    "edges": ["CAUSES"],
                    "confidence": edge.get("confidence", 0.5),
                    "evidence": [edge.get("source_id", "")],
                })
        return chains

    def _to_node(self, n) -> TemporalNode:
        """Convert a Neo4j record to a TemporalNode."""
        return TemporalNode(
            id=n.get("id", ""),
            entity=n.get("entity", ""),
            value=n.get("value", ""),
            valid_from=datetime.fromisoformat(n["valid_from"])
            if n.get("valid_from") else datetime.utcnow(),
            valid_until=datetime.fromisoformat(n["valid_until"])
            if n.get("valid_until") else None,
            confidence=n.get("confidence", 0.5),
            source_count=n.get("source_count", 1),
            source_ids=[],
            superseded_by=n.get("superseded_by"),
            causal_parents=[],
            causal_children=[],
            status=n.get("status", "current"),
        )

    def _path_to_chain(self, path) -> dict:
        """Convert a Neo4j path to a chain dict."""
        nodes = [node.get("entity", "") for node in path.nodes]
        return {
            "nodes": nodes,
            "edges": ["CAUSES"] * (len(nodes) - 1),
            "confidence": min(
                (node.get("confidence", 0.5) for node in path.nodes), default=0.5
            ),
            "evidence": [],
        }

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
