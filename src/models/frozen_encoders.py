from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer

if TYPE_CHECKING:
    from PIL import Image as PILImage

# BGE-M3 main branch only has pytorch_model.bin; this PR adds safetensors.
_BGEM3_SAFETENSORS_REVISION = "refs/pr/130"


def _infer_dtype(device: str) -> torch.dtype:
    """float16 on CUDA (faster, less VRAM); float32 everywhere else (cpu, mps)."""
    return torch.float16 if device == "cuda" else torch.float32


class SigLIP2Encoder:
    """Frozen SigLIP2-SO400M vision encoder.

    Encodes PIL images → L2-normalised float32 embeddings.
    Use as a context manager to automatically free VRAM after use.
    """

    MODEL_ID = "google/siglip2-so400m-patch16-512"
    EMBED_DIM = 1152
    MAX_TEXT_LENGTH = 256

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model: AutoModel | None = None
        self._processor: AutoProcessor | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        self._processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self._model = (
            AutoModel.from_pretrained(self.MODEL_ID, dtype=_infer_dtype(self._device))
            .to(self._device)
            .eval()
        )

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> SigLIP2Encoder:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode(
        self, images: list[PILImage.Image], batch_size: int = 16
    ) -> np.ndarray:
        """Return (N, EMBED_DIM) float32 L2-normalised embeddings."""
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() or use as a context manager first.")
        batches: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            inputs = self._processor(
                images=chunk, return_tensors="pt", padding=True
            ).to(self._device)
            features = self._model.get_image_features(**inputs)
            embs = features.cpu().float().numpy()
            batches.append(_l2_normalize(embs))
        return np.concatenate(batches, axis=0)

    @torch.inference_mode()
    def encode_text(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return (N, EMBED_DIM) float32 L2-normalised text embeddings.

        Text is encoded in the same shared space as images, enabling
        direct cosine-similarity comparison for ranking.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() or use as a context manager first.")
        batches: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            inputs = self._processor(
                text=chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_TEXT_LENGTH,
            ).to(self._device)
            features = self._model.get_text_features(**inputs)
            embs = features.cpu().float().numpy()
            batches.append(_l2_normalize(embs))
        return np.concatenate(batches, axis=0)


class BGEM3Encoder:
    """Frozen BGE-M3 multilingual text encoder.

    Encodes strings → L2-normalised float32 embeddings via CLS pooling.
    Use as a context manager to automatically free VRAM after use.
    """

    MODEL_ID = "BAAI/bge-m3"
    EMBED_DIM = 1024
    MAX_LENGTH = 512

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model: AutoModel | None = None
        self._tokenizer: AutoTokenizer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, revision=_BGEM3_SAFETENSORS_REVISION
        )
        self._model = (
            AutoModel.from_pretrained(
                self.MODEL_ID,
                revision=_BGEM3_SAFETENSORS_REVISION,
                dtype=_infer_dtype(self._device),
            )
            .to(self._device)
            .eval()
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> BGEM3Encoder:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return (N, EMBED_DIM) float32 L2-normalised embeddings."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call load() or use as a context manager first.")
        batches: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_LENGTH,
            ).to(self._device)
            outputs = self._model(**inputs)
            # CLS token as dense embedding (BGE-M3 convention)
            cls = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
            batches.append(_l2_normalize(cls))
        return np.concatenate(batches, axis=0)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)
