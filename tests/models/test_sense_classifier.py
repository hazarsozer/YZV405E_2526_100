"""Unit tests for LRSenseClassifier.

All tests run without GPU. Persistence tests use a tmp_path fixture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.sense_classifier import (
    HIGH_CONFIDENCE_THRESHOLD,
    IDIOMATIC_LABEL,
    LITERAL_LABEL,
    LOW_CONFIDENCE_THRESHOLD,
    LRSenseClassifier,
)

EMBED_DIM = 1024
N_TRAIN = 40  # enough for cv=5 with balanced classes


def _make_embeddings(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, EMBED_DIM)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


def _make_labels(n: int, ratio: float = 0.5) -> list[str]:
    n_idiomatic = int(n * ratio)
    return [IDIOMATIC_LABEL] * n_idiomatic + [LITERAL_LABEL] * (n - n_idiomatic)


@pytest.fixture()
def fitted_clf() -> LRSenseClassifier:
    embs = _make_embeddings(N_TRAIN)
    labels = _make_labels(N_TRAIN)
    clf = LRSenseClassifier()
    clf.fit(embs, labels)
    return clf


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


class TestFit:
    def test_fit_succeeds_with_valid_inputs(self):
        clf = LRSenseClassifier()
        clf.fit(_make_embeddings(N_TRAIN), _make_labels(N_TRAIN))
        assert clf._clf is not None

    def test_fit_raises_on_length_mismatch(self):
        clf = LRSenseClassifier()
        with pytest.raises(ValueError, match="length"):
            clf.fit(_make_embeddings(10), _make_labels(5))

    def test_fit_accepts_all_supported_labels(self):
        clf = LRSenseClassifier()
        labels = [IDIOMATIC_LABEL] * 20 + [LITERAL_LABEL] * 20
        clf.fit(_make_embeddings(40), labels)
        assert clf._clf is not None


# ---------------------------------------------------------------------------
# predict_proba_idiomatic
# ---------------------------------------------------------------------------


class TestPredictProbaIdiomatic:
    def test_output_shape(self, fitted_clf: LRSenseClassifier):
        embs = _make_embeddings(7, seed=99)
        probas = fitted_clf.predict_proba_idiomatic(embs)
        assert probas.shape == (7,)

    def test_output_dtype_is_float32(self, fitted_clf: LRSenseClassifier):
        embs = _make_embeddings(3, seed=42)
        probas = fitted_clf.predict_proba_idiomatic(embs)
        assert probas.dtype == np.float32

    def test_probabilities_in_unit_interval(self, fitted_clf: LRSenseClassifier):
        embs = _make_embeddings(20, seed=7)
        probas = fitted_clf.predict_proba_idiomatic(embs)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_raises_if_not_fitted(self):
        clf = LRSenseClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba_idiomatic(_make_embeddings(2))

    def test_single_sample(self, fitted_clf: LRSenseClassifier):
        emb = _make_embeddings(1, seed=55)
        proba = fitted_clf.predict_proba_idiomatic(emb)
        assert proba.shape == (1,)
        assert 0.0 <= proba[0] <= 1.0


# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_high_threshold_above_low(self):
        assert HIGH_CONFIDENCE_THRESHOLD > LOW_CONFIDENCE_THRESHOLD

    def test_thresholds_in_unit_interval(self):
        assert 0.0 < LOW_CONFIDENCE_THRESHOLD < HIGH_CONFIDENCE_THRESHOLD < 1.0


# ---------------------------------------------------------------------------
# Persistence: save / from_path
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_creates_file(self, fitted_clf: LRSenseClassifier, tmp_path: Path):
        out = tmp_path / "clf.joblib"
        fitted_clf.save(out)
        assert out.exists()

    def test_save_raises_if_not_fitted(self, tmp_path: Path):
        clf = LRSenseClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.save(tmp_path / "clf.joblib")

    def test_from_path_produces_same_probas(
        self, fitted_clf: LRSenseClassifier, tmp_path: Path
    ):
        out = tmp_path / "clf.joblib"
        fitted_clf.save(out)
        loaded = LRSenseClassifier.from_path(out)
        embs = _make_embeddings(10, seed=123)
        np.testing.assert_allclose(
            fitted_clf.predict_proba_idiomatic(embs),
            loaded.predict_proba_idiomatic(embs),
            atol=1e-6,
        )

    def test_from_path_raises_if_file_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            LRSenseClassifier.from_path(tmp_path / "nonexistent.joblib")

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        clf = LRSenseClassifier()
        clf.fit(_make_embeddings(N_TRAIN), _make_labels(N_TRAIN))
        nested = tmp_path / "a" / "b" / "clf.joblib"
        clf.save(nested)
        assert nested.exists()
