"""Unit tests for frozen encoders.

All tests here are fast (no model downloads). Heavy integration tests
that require actual GPU models are marked ``slow`` and skipped by default.
Run them with: pytest -m slow
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BatchEncoding
from transformers.image_processing_base import BatchFeature

from src.models.frozen_encoders import BGEM3Encoder, SigLIP2Encoder, _l2_normalize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_siglip(embed_dim: int = SigLIP2Encoder.EMBED_DIM, batch_size: int = 1):
    """Return (fake_processor, fake_model) mocks for SigLIP2."""
    fake_proc = MagicMock()
    fake_proc.return_value = BatchFeature(
        {"pixel_values": torch.zeros(batch_size, 3, 512, 512)}
    )

    fake_model = MagicMock()
    fake_model.get_image_features.return_value = torch.randn(batch_size, embed_dim)

    return fake_proc, fake_model


def _make_fake_bgem3(embed_dim: int = BGEM3Encoder.EMBED_DIM, batch_size: int = 1):
    """Return (fake_tokenizer, fake_model) mocks for BGE-M3."""
    fake_tok = MagicMock()
    fake_tok.return_value = BatchEncoding(
        {"input_ids": torch.zeros(batch_size, 10, dtype=torch.long)}
    )

    hidden = torch.randn(batch_size, 10, embed_dim)
    fake_out = MagicMock()
    fake_out.last_hidden_state = hidden

    fake_model = MagicMock()
    fake_model.return_value = fake_out

    return fake_tok, fake_model


# ---------------------------------------------------------------------------
# _l2_normalize
# ---------------------------------------------------------------------------


class TestL2Normalize:
    def test_unit_vectors(self):
        arr = np.array([[3.0, 4.0], [1.0, 0.0]])
        result = _l2_normalize(arr)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_zero_vector_no_nan(self):
        arr = np.zeros((1, 4), dtype=np.float32)
        result = _l2_normalize(arr)
        assert not np.any(np.isnan(result))

    def test_shape_preserved(self):
        arr = np.random.randn(5, 16).astype(np.float32)
        assert _l2_normalize(arr).shape == (5, 16)


# ---------------------------------------------------------------------------
# SigLIP2Encoder
# ---------------------------------------------------------------------------


class TestSigLIP2Encoder:
    @pytest.fixture()
    def encoder_and_mocks(self):
        fake_proc, fake_model = _make_fake_siglip()
        with (
            patch("src.models.frozen_encoders.AutoProcessor") as mock_proc_cls,
            patch("src.models.frozen_encoders.AutoModel") as mock_model_cls,
        ):
            mock_proc_cls.from_pretrained.return_value = fake_proc
            mock_model_cls.from_pretrained.return_value = fake_model
            enc = SigLIP2Encoder(device="cpu")
            enc.load()
            # Swap out the real (mocked) model with one that returns right shape
            enc._model = fake_model
            enc._processor = fake_proc
            yield enc, fake_proc, fake_model

    # --- shape & dtype ---

    def test_encode_returns_correct_shape(self, encoder_and_mocks):
        enc, fake_proc, fake_model = encoder_and_mocks
        n = 3
        fake_model.get_image_features.return_value = torch.randn(n, enc.EMBED_DIM)
        images = [Image.new("RGB", (512, 512)) for _ in range(n)]
        result = enc.encode(images)
        assert result.shape == (n, enc.EMBED_DIM)
        assert result.dtype == np.float32

    def test_embeddings_are_l2_normalised(self, encoder_and_mocks):
        enc, _, fake_model = encoder_and_mocks
        fake_model.get_image_features.return_value = torch.randn(2, enc.EMBED_DIM)
        images = [Image.new("RGB", (512, 512)) for _ in range(2)]
        result = enc.encode(images)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # --- batching ---

    def test_batch_processing_concatenates_correctly(self):
        """Encoding 7 images with batch_size=3 should return shape (7, dim)."""
        n_images = 7
        batch_size = 3

        fake_proc = MagicMock()
        fake_proc.return_value = {"pixel_values": torch.zeros(1, 3, 512, 512)}

        call_sizes: list[int] = []

        def side_effect(**kwargs):
            # Determine how many images were in this batch from pixel_values
            pv = kwargs.get("pixel_values", fake_proc.return_value["pixel_values"])
            b = pv.shape[0]
            call_sizes.append(b)
            return torch.randn(b, SigLIP2Encoder.EMBED_DIM)

        fake_model = MagicMock()
        fake_model.get_image_features.side_effect = side_effect

        with (
            patch("src.models.frozen_encoders.AutoProcessor") as mp,
            patch("src.models.frozen_encoders.AutoModel") as mm,
        ):
            mp.from_pretrained.return_value = fake_proc
            mm.from_pretrained.return_value = fake_model
            enc = SigLIP2Encoder(device="cpu")
            enc.load()
            enc._model = fake_model

            # Make processor return correct batch size each call
            def proc_side_effect(images=None, **kwargs):
                b = len(images) if images else 1
                return BatchFeature({"pixel_values": torch.zeros(b, 3, 512, 512)})

            enc._processor = MagicMock(side_effect=proc_side_effect)

            images = [Image.new("RGB", (512, 512)) for _ in range(n_images)]
            result = enc.encode(images, batch_size=batch_size)

        assert result.shape == (n_images, SigLIP2Encoder.EMBED_DIM)

    # --- lifecycle ---

    def test_raises_if_not_loaded(self):
        enc = SigLIP2Encoder(device="cpu")
        with pytest.raises(RuntimeError, match="load()"):
            enc.encode([Image.new("RGB", (64, 64))])

    def test_context_manager_loads_and_unloads(self):
        fake_proc, fake_model = _make_fake_siglip()
        with (
            patch("src.models.frozen_encoders.AutoProcessor") as mp,
            patch("src.models.frozen_encoders.AutoModel") as mm,
        ):
            mp.from_pretrained.return_value = fake_proc
            mm.from_pretrained.return_value = fake_model
            enc = SigLIP2Encoder(device="cpu")
            with enc:
                assert enc._model is not None
            assert enc._model is None
            assert enc._processor is None


# ---------------------------------------------------------------------------
# BGEM3Encoder
# ---------------------------------------------------------------------------


class TestBGEM3Encoder:
    @pytest.fixture()
    def encoder_and_mocks(self):
        fake_tok, fake_model = _make_fake_bgem3()
        with (
            patch("src.models.frozen_encoders.AutoTokenizer") as mock_tok_cls,
            patch("src.models.frozen_encoders.AutoModel") as mock_model_cls,
        ):
            mock_tok_cls.from_pretrained.return_value = fake_tok
            mock_model_cls.from_pretrained.return_value = fake_model
            enc = BGEM3Encoder(device="cpu")
            enc.load()
            enc._tokenizer = fake_tok
            enc._model = fake_model
            yield enc, fake_tok, fake_model

    # --- shape & dtype ---

    def test_encode_returns_correct_shape(self, encoder_and_mocks):
        enc, fake_tok, fake_model = encoder_and_mocks
        n = 4
        hidden = torch.randn(n, 10, enc.EMBED_DIM)
        out = MagicMock()
        out.last_hidden_state = hidden
        fake_model.return_value = out
        fake_tok.return_value = BatchEncoding({
            "input_ids": torch.zeros(n, 10, dtype=torch.long),
            "attention_mask": torch.ones(n, 10, dtype=torch.long),
        })
        texts = [f"sentence {i}" for i in range(n)]
        result = enc.encode(texts)
        assert result.shape == (n, enc.EMBED_DIM)
        assert result.dtype == np.float32

    def test_embeddings_are_l2_normalised(self, encoder_and_mocks):
        enc, fake_tok, fake_model = encoder_and_mocks
        n = 3
        hidden = torch.randn(n, 5, enc.EMBED_DIM)
        out = MagicMock()
        out.last_hidden_state = hidden
        fake_model.return_value = out
        fake_tok.return_value = BatchEncoding({
            "input_ids": torch.zeros(n, 5, dtype=torch.long),
        })
        result = enc.encode(["a", "b", "c"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # --- batching ---

    def test_batch_processing_concatenates_correctly(self):
        n_texts = 5
        batch_size = 2

        def make_output(n):
            hidden = torch.randn(n, 10, BGEM3Encoder.EMBED_DIM)
            out = MagicMock()
            out.last_hidden_state = hidden
            return out

        call_count = [0]

        def model_side_effect(**kwargs):
            b = kwargs["input_ids"].shape[0]
            call_count[0] += 1
            return make_output(b)

        fake_tok = MagicMock()

        def tok_side_effect(texts, **kwargs):
            b = len(texts)
            return BatchEncoding({
                "input_ids": torch.zeros(b, 10, dtype=torch.long),
                "attention_mask": torch.ones(b, 10, dtype=torch.long),
            })

        fake_tok.side_effect = tok_side_effect
        fake_model = MagicMock(side_effect=model_side_effect)

        with (
            patch("src.models.frozen_encoders.AutoTokenizer") as mt,
            patch("src.models.frozen_encoders.AutoModel") as mm,
        ):
            mt.from_pretrained.return_value = fake_tok
            mm.from_pretrained.return_value = fake_model
            enc = BGEM3Encoder(device="cpu")
            enc.load()
            enc._tokenizer = fake_tok
            enc._model = fake_model

            texts = [f"text {i}" for i in range(n_texts)]
            result = enc.encode(texts, batch_size=batch_size)

        assert result.shape == (n_texts, BGEM3Encoder.EMBED_DIM)
        expected_calls = (n_texts + batch_size - 1) // batch_size
        assert call_count[0] == expected_calls

    # --- lifecycle ---

    def test_raises_if_not_loaded(self):
        enc = BGEM3Encoder(device="cpu")
        with pytest.raises(RuntimeError, match="load()"):
            enc.encode(["hello"])

    def test_context_manager_loads_and_unloads(self):
        fake_tok, fake_model = _make_fake_bgem3()
        with (
            patch("src.models.frozen_encoders.AutoTokenizer") as mt,
            patch("src.models.frozen_encoders.AutoModel") as mm,
        ):
            mt.from_pretrained.return_value = fake_tok
            mm.from_pretrained.return_value = fake_model
            enc = BGEM3Encoder(device="cpu")
            with enc:
                assert enc._model is not None
            assert enc._model is None
            assert enc._tokenizer is None
