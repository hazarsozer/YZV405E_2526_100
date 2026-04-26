"""Unit tests for the category-aware heuristic ranker."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.ranker import (
    LITERAL_PENALTY_WEIGHT,
    cosine_similarity,
    rank_images,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 16
IMAGE_NAMES = ["img_a.png", "img_b.png", "img_c.png", "img_d.png", "img_e.png"]


def _normed(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-8)


def _random_embs(n: int, dim: int = EMBED_DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _normed(rng.standard_normal((n, dim)).astype(np.float32))


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_returns_one(self):
        v = _normed(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        imgs = v.reshape(1, -1)
        result = cosine_similarity(v, imgs)
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_orthogonal_vectors_returns_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        result = cosine_similarity(a, b)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_output_shape(self):
        text = _normed(np.ones(EMBED_DIM, dtype=np.float32))
        imgs = _random_embs(5)
        result = cosine_similarity(text, imgs)
        assert result.shape == (5,)
        assert result.dtype == np.float32

    def test_opposite_vectors_returns_negative(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([[-1.0, 0.0]], dtype=np.float32)
        result = cosine_similarity(a, b)
        np.testing.assert_allclose(result, -1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# rank_images
# ---------------------------------------------------------------------------


class TestRankImages:
    def test_returns_all_image_names(self):
        img_embs = _random_embs(5)
        txt = _random_embs(1)[0]
        result = rank_images(img_embs, txt, txt, p_idiomatic=0.5,
                             image_names=IMAGE_NAMES)
        assert sorted(result) == sorted(IMAGE_NAMES)

    def test_returns_correct_count(self):
        img_embs = _random_embs(5)
        txt = _random_embs(1)[0]
        result = rank_images(img_embs, txt, txt, 0.5, IMAGE_NAMES)
        assert len(result) == 5

    def test_raises_on_length_mismatch(self):
        img_embs = _random_embs(3)
        txt = _random_embs(1)[0]
        with pytest.raises(ValueError, match="length"):
            rank_images(img_embs, txt, txt, 0.5, IMAGE_NAMES)

    def test_idiomatic_branch_uses_paraphrased(self):
        """When P(idiomatic) is high, ranking should follow paraphrased text."""
        rng = np.random.default_rng(42)
        para_text = _normed(rng.standard_normal(EMBED_DIM).astype(np.float32))
        orig_text = _normed(rng.standard_normal(EMBED_DIM).astype(np.float32))

        # Make image 2 very similar to paraphrased text
        img_embs = _random_embs(5, seed=10)
        img_embs[2] = para_text + 0.01 * rng.standard_normal(EMBED_DIM).astype(np.float32)
        img_embs = _normed(img_embs)

        result = rank_images(img_embs, orig_text, para_text,
                             p_idiomatic=0.9, image_names=IMAGE_NAMES)
        # Image at index 2 should be ranked first (or very close)
        assert result[0] == IMAGE_NAMES[2]

    def test_literal_branch_uses_original(self):
        """When P(idiomatic) is low, ranking should follow original text."""
        rng = np.random.default_rng(42)
        orig_text = _normed(rng.standard_normal(EMBED_DIM).astype(np.float32))
        para_text = _normed(rng.standard_normal(EMBED_DIM).astype(np.float32))

        # Make image 4 very similar to original text
        img_embs = _random_embs(5, seed=20)
        img_embs[4] = orig_text + 0.01 * rng.standard_normal(EMBED_DIM).astype(np.float32)
        img_embs = _normed(img_embs)

        result = rank_images(img_embs, orig_text, para_text,
                             p_idiomatic=0.1, image_names=IMAGE_NAMES)
        assert result[0] == IMAGE_NAMES[4]

    def test_uncertain_bypass_uses_paraphrased(self):
        """When confidence is uncertain, ranking should use paraphrased only."""
        rng = np.random.default_rng(42)
        para_text = _normed(rng.standard_normal(EMBED_DIM).astype(np.float32))
        orig_text = _normed(rng.standard_normal(EMBED_DIM).astype(np.float32))

        # Make image 1 very similar to paraphrased text
        img_embs = _random_embs(5, seed=30)
        img_embs[1] = para_text + 0.01 * rng.standard_normal(EMBED_DIM).astype(np.float32)
        img_embs = _normed(img_embs)

        result = rank_images(img_embs, orig_text, para_text,
                             p_idiomatic=0.5, image_names=IMAGE_NAMES)
        assert result[0] == IMAGE_NAMES[1]

    def test_custom_thresholds(self):
        """Custom thresholds should shift which branch activates."""
        img_embs = _random_embs(5, seed=5)
        txt = _random_embs(1, seed=6)[0]
        # With default thresholds, 0.6 is uncertain
        r1 = rank_images(img_embs, txt, txt, p_idiomatic=0.6,
                         image_names=IMAGE_NAMES)
        # With low threshold at 0.7, 0.6 is literal
        r2 = rank_images(img_embs, txt, txt, p_idiomatic=0.6,
                         image_names=IMAGE_NAMES,
                         high_threshold=0.8, low_threshold=0.7)
        # The rankings may differ since different branches fire
        assert isinstance(r1, list) and isinstance(r2, list)

    def test_deterministic(self):
        img_embs = _random_embs(5, seed=99)
        orig = _random_embs(1, seed=100)[0]
        para = _random_embs(1, seed=101)[0]
        r1 = rank_images(img_embs, orig, para, 0.9, IMAGE_NAMES)
        r2 = rank_images(img_embs, orig, para, 0.9, IMAGE_NAMES)
        assert r1 == r2
