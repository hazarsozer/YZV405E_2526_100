from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

IDIOMATIC_LABEL = "idiomatic"
LITERAL_LABEL = "literal"

# Regularisation constant — strong enough to prevent overfit on ~few-hundred samples
LR_C = 1.0
LR_MAX_ITER = 1000

# Calibration settings
CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV = 5

# Decision thresholds used by the downstream ranker
HIGH_CONFIDENCE_THRESHOLD = 0.75
LOW_CONFIDENCE_THRESHOLD = 0.25


class LRSenseClassifier:
    """Calibrated Logistic Regression classifier for idiomatic/literal sense detection.

    Input: BGE-M3 L2-normalised embeddings (1024-d float32).
    Output: calibrated P(idiomatic) in [0, 1].

    Wraps sklearn's CalibratedClassifierCV so that predict_proba returns
    well-calibrated probabilities rather than raw LR scores.
    """

    def __init__(self) -> None:
        self._clf: CalibratedClassifierCV | None = None
        self._label_encoder: LabelEncoder = LabelEncoder()
        self._idiomatic_col: int | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray, labels: Sequence[str]) -> None:
        """Fit the classifier on labelled BGE-M3 embeddings.

        Args:
            embeddings: Shape (N, 1024) float32 L2-normalised embeddings.
            labels: N strings, each "idiomatic" or "literal".
        """
        if len(embeddings) != len(labels):
            raise ValueError(
                f"embeddings length {len(embeddings)} != labels length {len(labels)}"
            )
        label_array = self._label_encoder.fit_transform(labels)
        base = LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER, random_state=0)
        self._clf = CalibratedClassifierCV(
            estimator=base, method=CALIBRATION_METHOD, cv=CALIBRATION_CV
        )
        self._clf.fit(embeddings, label_array)
        self._idiomatic_col = self._resolve_idiomatic_col()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba_idiomatic(self, embeddings: np.ndarray) -> np.ndarray:
        """Return P(idiomatic) for each row, shape (N,).

        Raises RuntimeError if the model has not been fitted yet.
        """
        if self._clf is None or self._idiomatic_col is None:
            raise RuntimeError(
                "Classifier is not fitted. Call fit() or load from_path() first."
            )
        proba = self._clf.predict_proba(embeddings)
        return proba[:, self._idiomatic_col].astype(np.float32)

    def _resolve_idiomatic_col(self) -> int:
        # sklearn labels are sorted lexicographically: "idiomatic" < "literal"
        # so column 0 = P(idiomatic) when both classes are present
        classes: list[str] = list(
            self._label_encoder.inverse_transform(self._clf.classes_)  # type: ignore[union-attr]
        )
        return classes.index(IDIOMATIC_LABEL)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialise classifier to *path* using joblib."""
        if self._clf is None:
            raise RuntimeError("Classifier is not fitted. Call fit() first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"clf": self._clf, "label_encoder": self._label_encoder}, path
        )

    @classmethod
    def from_path(cls, path: Path) -> LRSenseClassifier:
        """Load a previously saved classifier from *path*."""
        if not path.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {path}")
        payload = joblib.load(path)
        instance = cls()
        instance._clf = payload["clf"]
        instance._label_encoder = payload["label_encoder"]
        instance._idiomatic_col = instance._resolve_idiomatic_col()
        return instance
