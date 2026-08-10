"""
NEXUS RAG — Gap 2: Causal-Counterfactual Layer
===============================================

Research: CausalRAG (ACL Findings 2025, arXiv 2503.19878)
         Causal-Counterfactual RAG (arXiv 2509.14435)

Provides:
    - Causal chain retrieval from knowledge graph
    - Counterfactual scenario evaluation ("What if X hadn't happened?")
    - Broken chain link identification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CausalChain:
    """An ordered cause→effect chain."""
    nodes:      list[str] = field(default_factory=list)
    edges:      list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence:   list[str] = field(default_factory=list)


@dataclass
class CounterfactualResult:
    """Result of a counterfactual query evaluation."""
    original_outcome:       str = ""
    removed_condition:      str = ""
    counterfactual_outcome: str = ""
    broken_chain_link:      str = ""
    confidence:             float = 0.0
    evidence_chunk_ids:     list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original_outcome": self.original_outcome,
            "removed_condition": self.removed_condition,
            "counterfactual_outcome": self.counterfactual_outcome,
            "broken_chain_link": self.broken_chain_link,
            "confidence": self.confidence,
        }


class CausalCounterfactualLayer:
    """
    Causal reasoning and counterfactual evaluation layer.
    Uses the temporal knowledge graph's causal edges to answer
    "What caused X?" and "What if Y hadn't happened?" queries.
    """

    def __init__(self, kg=None):
        self.kg = kg

    def retrieve_causal(self, query: str, depth: int = 3) -> list[CausalChain]:
        """Walk the causal graph to find all chains relevant to this query."""
        if not self.kg:
            return []

        try:
            raw_chains = self.kg.find_causal_chain(query, depth=depth)
            chains = []
            for c in raw_chains:
                if isinstance(c, dict):
                    chains.append(CausalChain(
                        nodes=c.get("nodes", []),
                        edges=c.get("edges", []),
                        confidence=c.get("confidence", 0.5),
                        evidence=c.get("evidence", []),
                    ))
            return sorted(chains, key=lambda x: x.confidence, reverse=True)
        except Exception as e:
            logger.error("Causal retrieval failed: %s", e)
            return []

    def evaluate_counterfactual(
        self,
        query: str,
        known_outcome: str,
    ) -> CounterfactualResult:
        """
        Evaluate a counterfactual scenario.

        Args:
            query: "What if the Fed had not raised rates?"
            known_outcome: "SVB collapsed in March 2023"

        Returns:
            CounterfactualResult with analysis
        """
        removed = self._parse_removed_condition(query)

        if not self.kg:
            return CounterfactualResult(
                original_outcome=known_outcome,
                removed_condition=removed,
                counterfactual_outcome="Cannot determine — knowledge graph not available",
                broken_chain_link="unknown",
                confidence=0.0,
            )

        try:
            chains = [
                c for c in self.kg.find_causal_chain(removed, depth=3)
                if isinstance(c, dict) and removed.lower() in str(c).lower()
            ]

            if not chains:
                return CounterfactualResult(
                    original_outcome=known_outcome,
                    removed_condition=removed,
                    counterfactual_outcome=(
                        "Cannot determine — condition not found in causal graph"
                    ),
                    broken_chain_link="unknown",
                    confidence=0.0,
                )

            best_chain = chains[0]
            alt_outcome = self._trace_without(best_chain, removed)

            return CounterfactualResult(
                original_outcome=known_outcome,
                removed_condition=removed,
                counterfactual_outcome=alt_outcome,
                broken_chain_link=self._find_broken_link(best_chain, removed),
                confidence=best_chain.get("confidence", 0.5),
                evidence_chunk_ids=best_chain.get("evidence", []),
            )

        except Exception as e:
            logger.error("Counterfactual evaluation failed: %s", e)
            return CounterfactualResult(
                original_outcome=known_outcome,
                removed_condition=removed,
                counterfactual_outcome=f"Evaluation error: {e}",
                confidence=0.0,
            )

    def _parse_removed_condition(self, query: str) -> str:
        """Extract the counterfactual condition from the query."""
        for marker in ["what if ", "if ", "had ", "suppose ", "assuming "]:
            if marker in query.lower():
                idx = query.lower().index(marker)
                return query[idx + len(marker):].split(",")[0].strip().rstrip("?")
        return query

    def _trace_without(self, chain: dict, removed_node: str) -> str:
        """Simulate causal chain with the removed node deleted."""
        nodes = [
            n for n in chain.get("nodes", [])
            if removed_node.lower() not in n.lower()
        ]
        if not nodes:
            return (
                "Without that condition, the causal chain is broken — "
                "the outcome would likely not have occurred."
            )
        return (
            f"Without '{removed_node}', the chain proceeds only through: "
            f"{' → '.join(nodes)}, suggesting the original outcome "
            f"may not have materialized."
        )

    def _find_broken_link(self, chain: dict, removed: str) -> str:
        """Find the causal link that would be broken."""
        nodes = chain.get("nodes", [])
        for i, n in enumerate(nodes):
            if removed.lower() in n.lower() and i + 1 < len(nodes):
                return f"{n} → {nodes[i + 1]}"
        return "unknown"
