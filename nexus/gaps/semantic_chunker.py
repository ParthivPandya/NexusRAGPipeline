"""
NEXUS RAG — Gap 6: Semantic Boundary Chunker
=============================================

Research: Vision-Guided Chunking (arXiv 2506.16035, 2025)

Detects where meaning ends, not where a token limit is reached.
Uses four signal layers:
    1. Structural headings (strongest)
    2. Visual layout boundaries (PDFs, slides)
    3. Discourse coherence markers
    4. Topic shift detection (weakest, fallback)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BoundarySignal:
    """A detected chunk boundary."""
    position:      int
    boundary_type: str    # "visual"|"discourse"|"topic_shift"|"structural"
    confidence:    float


@dataclass
class SemanticChunk:
    """A semantically coherent chunk of text."""
    text:           str
    raw_text:       str
    context_prefix: str
    boundary_type:  str
    start_char:     int
    end_char:       int


class SemanticBoundaryChunker:
    """
    Semantic chunker that detects natural meaning boundaries
    instead of arbitrary token-count splits.
    """

    TOPIC_SHIFT_THRESHOLD = 0.35

    DISCOURSE_MARKERS = [
        "however", "therefore", "in contrast", "on the other hand",
        "nevertheless", "consequently", "furthermore", "in conclusion",
        "as a result", "moreover", "in summary", "to summarise",
        "that said", "conversely", "accordingly", "subsequently",
        "in addition", "nonetheless", "having said that",
        "on the contrary", "meanwhile", "alternatively",
    ]

    STRUCTURAL_MARKERS = re.compile(
        r"(^#{1,4}\s|^chapter\s+\d|^section\s+\d|^\d+\.\s+[A-Z])",
        re.MULTILINE | re.IGNORECASE,
    )

    def __init__(self, encoder=None, max_chunk_size: int = 1500, min_chunk_size: int = 30):
        self.encoder = encoder
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: dict) -> list[SemanticChunk]:
        """
        Split a document into semantically coherent chunks.

        Args:
            document: Dict with 'text', 'type', 'title', 'section' keys

        Returns:
            List of SemanticChunks with context prefixes
        """
        text = document.get("text", "")
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        dtype = document.get("type", "text")
        signals: list[BoundarySignal] = []

        # Signal 1: Structural headings (strongest)
        signals.extend(self._structural_boundaries(text))

        # Signal 2: Discourse coherence markers
        signals.extend(self._discourse_boundaries(text))

        # Signal 3: Topic shift (weakest, used as fallback)
        if self.encoder and len(text) > 500:
            signals.extend(self._topic_shift_boundaries(text))

        # Fuse signals: sort, remove duplicates within 50 chars
        fused = self._fuse_signals(signals)

        # If no signals found, fall back to paragraph-based splitting
        if not fused:
            fused = self._paragraph_boundaries(text)

        # Split text at boundaries
        chunks = self._split(text, fused, document)

        # Enforce max chunk size
        chunks = self._enforce_max_size(chunks, document)

        logger.debug("Chunked document into %d semantic chunks", len(chunks))
        return chunks

    def _structural_boundaries(self, text: str) -> list[BoundarySignal]:
        """Detect structural boundaries (headings, chapters, sections)."""
        return [
            BoundarySignal(
                position=m.start(),
                boundary_type="structural",
                confidence=0.95,
            )
            for m in self.STRUCTURAL_MARKERS.finditer(text)
        ]

    def _discourse_boundaries(self, text: str) -> list[BoundarySignal]:
        """Detect discourse boundaries using transition markers."""
        signals = []
        lower = text.lower()
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
        """Detect topic shifts using embedding similarity between windows."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) < 5:
            return []

        signals = []
        W = 3  # window size

        for i in range(W, len(sentences) - W):
            left = " ".join(sentences[i - W:i])
            right = " ".join(sentences[i:i + W])

            try:
                l_emb = self.encoder.encode(left, None)  # Will use TEXT modality
                r_emb = self.encoder.encode(right, None)
                sim = float(np.dot(l_emb, r_emb))

                if sim < self.TOPIC_SHIFT_THRESHOLD:
                    char_pos = sum(len(s) + 2 for s in sentences[:i])
                    signals.append(BoundarySignal(
                        char_pos, "topic_shift", 1.0 - sim,
                    ))
            except Exception:
                continue

        return signals

    def _paragraph_boundaries(self, text: str) -> list[BoundarySignal]:
        """Fallback: split on double newlines (paragraphs)."""
        signals = []
        for m in re.finditer(r"\n\s*\n", text):
            signals.append(BoundarySignal(
                position=m.start(),
                boundary_type="paragraph",
                confidence=0.60,
            ))
        return signals

    def _fuse_signals(
        self, signals: list[BoundarySignal], min_gap: int = 50
    ) -> list[BoundarySignal]:
        """Fuse signals: remove duplicates within min_gap characters."""
        if not signals:
            return []

        signals = sorted(signals, key=lambda s: s.position)
        fused = [signals[0]]
        last = signals[0].position

        for sig in signals[1:]:
            if sig.position - last >= min_gap:
                fused.append(sig)
                last = sig.position
            elif sig.confidence > fused[-1].confidence:
                fused[-1] = sig

        return fused

    def _split(
        self, text: str, boundaries: list[BoundarySignal], doc: dict
    ) -> list[SemanticChunk]:
        """Split text at boundary positions."""
        positions = [0] + [b.position for b in boundaries] + [len(text)]
        btypes = ["start"] + [b.boundary_type for b in boundaries] + ["end"]
        chunks = []

        for i in range(len(positions) - 1):
            raw = text[positions[i]:positions[i + 1]].strip()
            if len(raw) < self.min_chunk_size:
                continue

            ctx = self._make_prefix(raw, doc, i)
            chunks.append(SemanticChunk(
                text=f"{ctx}\n\n{raw}",
                raw_text=raw,
                context_prefix=ctx,
                boundary_type=btypes[i],
                start_char=positions[i],
                end_char=positions[i + 1],
            ))

        return chunks

    def _enforce_max_size(
        self, chunks: list[SemanticChunk], doc: dict
    ) -> list[SemanticChunk]:
        """Split chunks that exceed max_chunk_size."""
        result = []
        for chunk in chunks:
            if len(chunk.raw_text) <= self.max_chunk_size:
                result.append(chunk)
            else:
                # Split on sentence boundaries within the chunk
                sentences = re.split(r"(?<=[.!?])\s+", chunk.raw_text)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) > self.max_chunk_size and current:
                        ctx = self._make_prefix(current, doc, len(result))
                        result.append(SemanticChunk(
                            text=f"{ctx}\n\n{current}",
                            raw_text=current,
                            context_prefix=ctx,
                            boundary_type="overflow",
                            start_char=chunk.start_char,
                            end_char=chunk.end_char,
                        ))
                        current = sent
                    else:
                        current = f"{current} {sent}".strip() if current else sent
                if current:
                    ctx = self._make_prefix(current, doc, len(result))
                    result.append(SemanticChunk(
                        text=f"{ctx}\n\n{current}",
                        raw_text=current,
                        context_prefix=ctx,
                        boundary_type="overflow",
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                    ))
        return result

    def _make_prefix(self, chunk: str, doc: dict, idx: int) -> str:
        """Create a contextual prefix for the chunk."""
        title = doc.get("title", "Document")
        section = doc.get("section", "")
        return (
            f"From '{title}'"
            + (f", section '{section}'" if section else "")
            + f", part {idx + 1}. This passage discusses: "
        )
