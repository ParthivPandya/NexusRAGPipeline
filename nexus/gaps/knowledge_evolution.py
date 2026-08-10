"""
NEXUS RAG — Gap 9: Knowledge Evolution Manager
===============================================

Research: ACL 2026 "RAG or Learning?", Zylos Research April 2026.
Core insight: never DELETE knowledge — always EVOLVE it with history.

Fact lifecycle: current → superseded | scoped | contested | retracted
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class FactStatus(Enum):
    CURRENT           = "current"
    SUPERSEDED        = "superseded"
    TEMPORALLY_SCOPED = "scoped"
    RETRACTED         = "retracted"
    CONTESTED         = "contested"


@dataclass
class EvolutionEvent:
    """A knowledge evolution event."""
    event_type:  str
    old_fact_id: str
    new_fact_id: Optional[str] = None
    reason:      str = ""
    timestamp:   datetime = None
    trigger:     str = "new_document"

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "old_fact_id": self.old_fact_id,
            "new_fact_id": self.new_fact_id,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "trigger": self.trigger,
        }


class KnowledgeEvolutionManager:
    """
    Manages knowledge evolution — never deletes, always evolves.
    Tracks the full lifecycle of facts with audit trail.
    """

    def __init__(self, kg=None, db_conn=None):
        self.kg = kg
        self.db = db_conn

    def ingest_new_fact(
        self,
        new_fact: dict,
        source_chunk=None,
    ) -> list[EvolutionEvent]:
        """
        Check if a new fact conflicts with existing knowledge,
        and evolve accordingly.
        """
        if not self.kg:
            return []

        events = []
        try:
            conflicts = (
                self.kg.find_conflicting_facts(new_fact)
                if hasattr(self.kg, "find_conflicting_facts")
                else []
            )

            for old in conflicts:
                evt = self._evolve(old, new_fact, source_chunk)
                self._apply(evt, old, new_fact)
                events.append(evt)

        except Exception as e:
            logger.error("Knowledge evolution check failed: %s", e)

        return events

    def query_historical(self, entity: str, as_of: datetime) -> list:
        """What did the system know about `entity` on date `as_of`?"""
        if not self.kg:
            return []
        return self.kg.query_at_time(entity, as_of)

    def _evolve(self, old, new_fact: dict, source_chunk=None) -> EvolutionEvent:
        """Determine evolution type based on fact comparison."""
        # Check if this is a temporal update (same entity, different time periods)
        if self._is_temporal_update(old, new_fact):
            return EvolutionEvent(
                event_type="scope",
                old_fact_id=old.id if hasattr(old, 'id') else str(old),
                reason="Temporal update — both facts valid for different periods",
                trigger="new_document",
            )

        # Check credibility gap
        source_cred = source_chunk.credibility_score if source_chunk else 0.5
        old_conf = old.confidence if hasattr(old, 'confidence') else 0.5
        cred_delta = source_cred - old_conf

        if cred_delta > 0.20:
            return EvolutionEvent(
                event_type="supersede",
                old_fact_id=old.id if hasattr(old, 'id') else str(old),
                reason=f"New source credibility significantly higher ({cred_delta:.2f})",
                trigger="new_document",
            )

        return EvolutionEvent(
            event_type="contest",
            old_fact_id=old.id if hasattr(old, 'id') else str(old),
            reason="Credible contradiction from different source",
            trigger="new_document",
        )

    def _apply(self, evt: EvolutionEvent, old, new_fact: dict):
        """Apply evolution event to the knowledge graph."""
        if not self.kg or not self.kg.driver:
            self._log_event(evt)
            return

        try:
            old_id = old.id if hasattr(old, 'id') else str(old)
            with self.kg.driver.session() as s:
                if evt.event_type == "supersede":
                    s.run(
                        "MATCH (n:Node {id: $id}) SET n.status='superseded', n.valid_until=$ts",
                        id=old_id, ts=datetime.utcnow().isoformat(),
                    )
                elif evt.event_type == "scope":
                    s.run(
                        "MATCH (n:Node {id: $id}) SET n.status='scoped', n.valid_until=$ts",
                        id=old_id, ts=datetime.utcnow().isoformat(),
                    )
                elif evt.event_type == "contest":
                    s.run(
                        "MATCH (n:Node {id: $id}) SET n.status='contested'",
                        id=old_id,
                    )
        except Exception as e:
            logger.error("Failed to apply evolution event: %s", e)

        self._log_event(evt)

    def _is_temporal_update(self, old, new_fact: dict) -> bool:
        """True if the two facts are about the same entity in different years."""
        old_str = str(old)
        new_str = str(new_fact)
        old_years = set(re.findall(r"\b(19|20)\d{2}\b", old_str))
        new_years = set(re.findall(r"\b(19|20)\d{2}\b", new_str))
        return bool(old_years) and bool(new_years) and old_years != new_years

    def _log_event(self, evt: EvolutionEvent):
        """Write evolution event to audit log."""
        if not self.db:
            logger.info("Evolution event: %s", evt.to_dict())
            return

        try:
            with self.db.cursor() as cur:
                cur.execute("""
                    INSERT INTO evolution_log
                        (event_type, old_fact_id, new_fact_id, reason, trigger, ts)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    evt.event_type, evt.old_fact_id, evt.new_fact_id,
                    evt.reason, evt.trigger, evt.timestamp,
                ))
            self.db.commit()
        except Exception as e:
            logger.error("Failed to log evolution event: %s", e)
