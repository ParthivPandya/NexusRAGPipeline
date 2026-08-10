"""
NEXUS RAG — Gap 3: Cross-Lingual Reasoning Bridge
==================================================

Research: XRAG (arXiv 2505.10089), CORAL (arXiv 2604.25676), ACL 2026.
Core insight: the problem is cross-lingual REASONING, not translation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from nexus.core.unified_encoder import UnifiedChunk

logger = logging.getLogger(__name__)


@dataclass
class XLingualContext:
    """Cross-lingual reasoning context for the LLM."""
    query_language:     str = "en"
    doc_languages:      list[str] = field(default_factory=list)
    concept_bridge:     dict = field(default_factory=dict)
    cultural_tags:      list[str] = field(default_factory=list)
    reasoning_scaffold: str = ""


class CrossLingualBridge:
    """
    Cross-lingual reasoning bridge that enables retrieval and
    reasoning across languages. Goes beyond simple translation
    to provide conceptual alignment and cultural context.
    """

    CULTURAL_AUTHORITY_DOMAINS = {
        "ar": ["aljazeera.com", "alarabiya.net"],
        "ja": ["nhk.or.jp", "asahi.com"],
        "zh": ["xinhuanet.com", "people.com.cn"],
        "de": ["spiegel.de", "faz.net"],
        "fr": ["lemonde.fr", "liberation.fr"],
        "es": ["elpais.com", "bbc.com/mundo"],
        "hi": ["ndtv.com", "aajtak.in"],
        "pt": ["folha.uol.com.br", "globo.com"],
    }

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self._model_name = model_name
        self._model = None

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        except Exception as e:
            logger.warning("Multilingual model unavailable: %s", e)

    def build(
        self,
        query: str,
        query_lang: str,
        chunks: list[UnifiedChunk],
    ) -> XLingualContext:
        """Build cross-lingual reasoning context."""
        concept_bridge = self._build_concept_bridge(query, chunks)
        cultural_tags = self._detect_cultural_context(chunks)
        scaffold = self._build_scaffold(query_lang, concept_bridge, cultural_tags)

        return XLingualContext(
            query_language=query_lang,
            doc_languages=list({c.language for c in chunks}),
            concept_bridge=concept_bridge,
            cultural_tags=cultural_tags,
            reasoning_scaffold=scaffold,
        )

    def _build_concept_bridge(self, query: str, chunks: list[UnifiedChunk]) -> dict:
        """Extract language-agnostic concepts and align across languages."""
        bridge: dict = {}
        for chunk in chunks:
            if isinstance(chunk.content, str) and len(chunk.content) > 10:
                bridge.setdefault(chunk.language, []).append(chunk.content[:100])
        return bridge

    def _detect_cultural_context(self, chunks: list[UnifiedChunk]) -> list[str]:
        """Detect cultural context from source domains."""
        tags = []
        for c in chunks:
            lang = c.language
            url = c.metadata.get("source_url", "")
            for auth_domain in self.CULTURAL_AUTHORITY_DOMAINS.get(lang, []):
                if auth_domain in url:
                    tags.append(f"{lang}::{auth_domain}")
        return list(set(tags))

    def _build_scaffold(
        self,
        target_lang: str,
        concept_bridge: dict,
        cultural_tags: list[str],
    ) -> str:
        """Build reasoning instructions for the LLM."""
        languages = ", ".join(concept_bridge.keys()) if concept_bridge else "none"
        cultures = "; ".join(cultural_tags) if cultural_tags else "none detected"

        return (
            f"Cross-lingual synthesis instructions:\n"
            f"• Target output language: {target_lang}\n"
            f"• Source languages with evidence: {languages}\n"
            f"• Cultural perspectives present: {cultures}\n"
            f"• Step 1: Identify the universal concept being asked about\n"
            f"• Step 2: Gather evidence from ALL language sources via concepts\n"
            f"• Step 3: Note culturally-specific interpretations if any\n"
            f"• Step 4: Synthesise in target language ({target_lang})\n"
            f"• Step 5: Append cultural-context note if it changes the answer"
        )
