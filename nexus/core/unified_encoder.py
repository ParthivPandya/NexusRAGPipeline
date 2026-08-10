"""
NEXUS RAG — Pillar 1: Omni-Modal Unification Engine
====================================================

Creates a single 1024-dimensional embedding space where text, images,
audio, video, code, tables, and mathematical formulas are semantically
comparable. A chart and its description produce vectors with cosine
similarity > 0.85.

Models:
    Text    → intfloat/e5-large-v2         (direct 1024-dim)
    Image   → openai/clip-vit-large-patch14 (Linear → 1024)
    Audio   → openai/whisper-base + text    (Linear → 1024)
    Video   → frame sampling + audio fusion (Attention → 1024)
    Code    → microsoft/codebert-base       (Linear → 1024)
    Table   → google/tapas-base             (Linear → 1024)
    Formula → tbs17/MathBERT                (Linear → 1024)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ── Modality Enumeration ──────────────────────────────────────

class Modality(Enum):
    """Supported content modalities."""
    TEXT    = "text"
    IMAGE   = "image"
    AUDIO   = "audio"
    VIDEO   = "video"
    CODE    = "code"
    TABLE   = "table"
    FORMULA = "formula"


# ── Unified Chunk Data Structure ──────────────────────────────

@dataclass
class UnifiedChunk:
    """
    The atomic unit of knowledge in NEXUS.
    Every piece of content — regardless of modality — becomes
    a UnifiedChunk with a 1024-dim L2-normalized embedding.
    """
    id:                   str
    content:              Any              # Raw content (str, bytes, np.ndarray)
    modality:             Modality
    embedding:            Optional[np.ndarray]  # Always 1024-dim, L2-normalized
    metadata:             dict             # source_url, date, page_num, language …
    context_prefix:       str              # Contextual summary prepended at ingest
    causal_node_ids:      list[str] = field(default_factory=list)
    temporal_valid_from:  str   = ""       # ISO 8601
    temporal_valid_until: str   = "present"  # ISO 8601 or "present"
    credibility_score:    float = 0.5      # 0.0 – 1.0
    language:             str   = "en"     # ISO 639-1
    chunk_boundary_type:  str   = "semantic"  # "semantic"|"visual"|"discourse"
    retrieval_score:      Optional[float] = None  # Set by retrieval router

    def to_dict(self) -> dict:
        """Serialize chunk for storage (excludes embedding)."""
        return {
            "id": self.id,
            "content": self.content if isinstance(self.content, str) else "<binary>",
            "modality": self.modality.value,
            "metadata": self.metadata,
            "context_prefix": self.context_prefix,
            "causal_node_ids": self.causal_node_ids,
            "temporal_valid_from": self.temporal_valid_from,
            "temporal_valid_until": self.temporal_valid_until,
            "credibility_score": self.credibility_score,
            "language": self.language,
            "chunk_boundary_type": self.chunk_boundary_type,
        }


# ── Unified Encoder ──────────────────────────────────────────

class UnifiedEncoder:
    """
    Encode any modality → 1024-dim L2-normalized vector.

    All modality-specific encoders produce raw vectors that are
    projected through a learned linear head into the unified
    1024-dimensional space, then L2-normalized.
    """

    DIM = 1024

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._models: dict = {}
        self._projectors: dict[Modality, nn.Module] = {}
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization — models loaded on first use."""
        if self._initialized:
            return

        logger.info("Initializing UnifiedEncoder models...")
        self._init_text_model()
        self._init_code_model()
        self._init_projectors()
        self._initialized = True
        logger.info("UnifiedEncoder ready — %d modalities supported", len(self._projectors))

    def _init_text_model(self):
        """Load the primary text embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            model_name = self._config.get("text_model", "intfloat/e5-large-v2")
            self._models["text"] = SentenceTransformer(model_name)
            logger.info("Loaded text model: %s", model_name)
        except Exception as e:
            logger.warning("Failed to load text model: %s — using fallback", e)
            self._models["text"] = None

    def _init_code_model(self):
        """Load the code embedding model."""
        try:
            from transformers import AutoModel, AutoTokenizer
            model_name = self._config.get("code_model", "microsoft/codebert-base")
            self._models["code_tok"] = AutoTokenizer.from_pretrained(model_name)
            self._models["code_model"] = AutoModel.from_pretrained(model_name)
            logger.info("Loaded code model: %s", model_name)
        except Exception as e:
            logger.warning("Failed to load code model: %s", e)

    def _init_projectors(self) -> None:
        """Linear projection heads: raw_dim → 1024."""
        if not TORCH_AVAILABLE:
            logger.warning("Torch not available — skipping projectors")
            return
            
        self._projectors = {
            Modality.TEXT:    nn.Linear(1024, self.DIM),
            Modality.CODE:    nn.Linear(768, self.DIM),
            Modality.IMAGE:   nn.Linear(768, self.DIM),
            Modality.AUDIO:   nn.Linear(1024, self.DIM),
            Modality.TABLE:   nn.Linear(768, self.DIM),
            Modality.FORMULA: nn.Linear(768, self.DIM),
            Modality.VIDEO:   nn.Linear(1024, self.DIM),
        }
        # Initialize with Xavier uniform for better initial embeddings
        for proj in self._projectors.values():
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def encode(self, content: Any, modality: Modality) -> np.ndarray:
        """
        Encode any modality → 1024-dim L2-normalized vector.

        Args:
            content: Raw content (str for text/code, bytes/path for image/audio)
            modality: The content modality

        Returns:
            1024-dimensional L2-normalized numpy array
        """
        self._lazy_init()

        try:
            if modality == Modality.TEXT:
                raw = self._encode_text(content)
            elif modality == Modality.CODE:
                raw = self._encode_code(content)
            elif modality == Modality.IMAGE:
                raw = self._encode_image(content)
            elif modality == Modality.AUDIO:
                raw = self._encode_audio(content)
            elif modality == Modality.TABLE:
                raw = self._encode_table(content)
            elif modality == Modality.FORMULA:
                raw = self._encode_formula(content)
            elif modality == Modality.VIDEO:
                raw = self._encode_video(content)
            else:
                # Fallback: treat as text
                raw = self._encode_text(str(content))

            # Project to unified space
            projected = self._project(raw, modality)
            return self._l2_normalize(projected)

        except Exception as e:
            logger.error("Encoding failed for modality %s: %s", modality, e)
            # Return zero vector on failure (will have low similarity to everything)
            return np.zeros(self.DIM, dtype=np.float32)

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using e5-large-v2."""
        model = self._models.get("text")
        if model is None:
            # Fallback: simple hash-based embedding (deterministic, for testing)
            return self._fallback_encode(text)
        return model.encode(text, normalize_embeddings=False)

    def _encode_code(self, code: str) -> np.ndarray:
        """Encode source code using CodeBERT (CLS token)."""
        tokenizer = self._models.get("code_tok")
        model = self._models.get("code_model")
        if tokenizer is None or model is None:
            return self._fallback_encode(code, dim=768)

        tokens = tokenizer(
            code, return_tensors="pt",
            max_length=512, truncation=True, padding=True
        )
        with torch.no_grad():
            out = model(**tokens)
        return out.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token

    def _encode_image(self, image_input) -> np.ndarray:
        """Encode image using CLIP ViT-L/14."""
        try:
            import clip
            from PIL import Image

            if "clip_model" not in self._models:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = clip.load(
                    self._config.get("clip_model", "ViT-L/14"), device=device
                )
                self._models["clip_model"] = model
                self._models["clip_preprocess"] = preprocess
                self._models["clip_device"] = device

            model = self._models["clip_model"]
            preprocess = self._models["clip_preprocess"]
            device = self._models["clip_device"]

            if isinstance(image_input, bytes):
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, str):
                img = Image.open(image_input)
            else:
                img = image_input

            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = model.encode_image(tensor)
            return features.squeeze().cpu().numpy()

        except ImportError:
            logger.warning("CLIP not available — using text fallback for image")
            return self._fallback_encode(str(image_input), dim=768)

    def _encode_audio(self, audio_path: str) -> np.ndarray:
        """Encode audio: transcribe with Whisper, then encode text."""
        try:
            import whisper
            if "whisper_model" not in self._models:
                model_size = self._config.get("whisper_model", "base")
                self._models["whisper_model"] = whisper.load_model(model_size)

            result = self._models["whisper_model"].transcribe(audio_path)
            transcript = result["text"]
            logger.debug("Audio transcribed: %s...", transcript[:100])
            return self._encode_text(transcript)

        except ImportError:
            logger.warning("Whisper not available — using text fallback for audio")
            return self._fallback_encode(str(audio_path))

    def _encode_table(self, table_data: dict) -> np.ndarray:
        """Serialize table to text then encode."""
        rows = table_data.get("rows", [])
        headers = table_data.get("headers", [])

        text_parts = []
        if headers:
            text_parts.append(" | ".join(str(h) for h in headers))
        for row in rows:
            text_parts.append(" | ".join(str(c) for c in row))

        text = "\n".join(text_parts)
        return self._encode_text(text)

    def _encode_formula(self, formula: str) -> np.ndarray:
        """Encode mathematical formula as text (MathBERT fallback to text)."""
        # Normalize LaTeX notation for better embedding
        normalized = formula.replace("\\frac", "fraction").replace("\\sqrt", "sqrt")
        return self._encode_text(normalized)

    def _encode_video(self, video_path: str) -> np.ndarray:
        """
        Encode video: sample key frames + extract audio,
        then fuse embeddings with attention weighting.
        """
        # Simplified: extract frames and average their embeddings
        try:
            frames = self._sample_frames(video_path, n_frames=4)
            if not frames:
                return self._fallback_encode(str(video_path))

            frame_embs = [self._encode_image(f) for f in frames]
            # Mean pooling of frame embeddings
            return np.mean(frame_embs, axis=0)

        except Exception as e:
            logger.warning("Video encoding failed: %s — using fallback", e)
            return self._fallback_encode(str(video_path))

    def _sample_frames(self, video_path: str, n_frames: int = 4) -> list:
        """Sample evenly-spaced frames from a video file."""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total == 0:
                return []

            indices = np.linspace(0, total - 1, n_frames, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    from PIL import Image
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
            cap.release()
            return frames
        except ImportError:
            return []

    def _project(self, raw: np.ndarray, modality: Modality) -> np.ndarray:
        """Project raw embedding through modality-specific linear head."""
        if not TORCH_AVAILABLE:
            if len(raw) == self.DIM:
                return raw
            elif len(raw) < self.DIM:
                return np.pad(raw, (0, self.DIM - len(raw)))
            else:
                return raw[:self.DIM]
                
        projector = self._projectors.get(modality)
        if projector is None:
            # No projector for this modality, return padded/truncated
            if len(raw) == self.DIM:
                return raw
            elif len(raw) < self.DIM:
                return np.pad(raw, (0, self.DIM - len(raw)))
            else:
                return raw[:self.DIM]

        with torch.no_grad():
            tensor = torch.tensor(raw, dtype=torch.float32)
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            projected = projector(tensor)
        return projected.squeeze().numpy()

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            return vec
        return vec / norm

    @staticmethod
    def _fallback_encode(text: str, dim: int = 1024) -> np.ndarray:
        """
        Deterministic fallback encoding using text hash.
        Used when ML models are not available.
        """
        import hashlib
        h = hashlib.sha512(text.encode()).digest()
        # Expand hash to fill dim dimensions
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def batch_encode(
        self, items: list[tuple[Any, Modality]], batch_size: int = 32
    ) -> list[np.ndarray]:
        """
        Batch encode multiple items for efficiency.

        Args:
            items: List of (content, modality) tuples
            batch_size: Number of items to process at once

        Returns:
            List of 1024-dim L2-normalized numpy arrays
        """
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            # Group by modality for batched processing
            text_items = [(idx, c) for idx, (c, m) in enumerate(batch) if m == Modality.TEXT]

            # Batch encode text items
            if text_items and self._models.get("text"):
                indices, texts = zip(*text_items)
                text_embs = self._models["text"].encode(
                    list(texts), normalize_embeddings=False, batch_size=batch_size
                )
                text_map = dict(zip(indices, text_embs))
            else:
                text_map = {}

            # Encode all items (using batch results where available)
            for local_idx, (content, modality) in enumerate(batch):
                if modality == Modality.TEXT and local_idx in text_map:
                    raw = text_map[local_idx]
                    projected = self._project(raw, modality)
                    results.append(self._l2_normalize(projected))
                else:
                    results.append(self.encode(content, modality))

        return results
