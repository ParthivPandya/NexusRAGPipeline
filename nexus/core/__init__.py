"""NEXUS RAG — Core Pillars (8 components)."""

from nexus.core.unified_encoder import UnifiedEncoder, UnifiedChunk, Modality
from nexus.core.query_classifier import QueryDNAClassifier, QueryDNA
from nexus.core.conflict_resolver import ConflictResolver, ConflictReport
from nexus.core.uncertainty_quantifier import UncertaintyQuantifier, ConfidenceClaim
from nexus.core.self_healer import SelfHealingVerifier, HealingResult
from nexus.core.feedback_loop import FeedbackLearner

__all__ = [
    "UnifiedEncoder", "UnifiedChunk", "Modality",
    "QueryDNAClassifier", "QueryDNA",
    "ConflictResolver", "ConflictReport",
    "UncertaintyQuantifier", "ConfidenceClaim",
    "SelfHealingVerifier", "HealingResult",
    "FeedbackLearner",
]
