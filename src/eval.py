"""Evaluation metrics for the AdMIRe 2.0 image-ranking subtask.

Metrics
-------
* **Top-1 accuracy** — fraction of instances where the predicted top image
  matches the gold top image.
* **DCG (Discounted Cumulative Gain)** — measures how well the full
  predicted ranking agrees with the gold ranking, using position-based
  relevance weights discounted by rank.
* **nDCG** — DCG normalised by the ideal (gold) DCG so the score ∈ [0, 1].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Default relevance weights for 5 ranked positions (best → worst).
# Image at gold-rank 0 gets weight 5, gold-rank 1 gets 4, …, gold-rank 4 gets 1.
DEFAULT_N_IMAGES = 5


# ------------------------------------------------------------------
# Low-level metric functions
# ------------------------------------------------------------------


def _relevance_from_gold(
    predicted: list[str], gold: list[str]
) -> list[float]:
    """Map each predicted position to a relevance score based on gold rank.

    The top gold image gets relevance N, second gets N-1, … last gets 1.
    If a predicted image is not in the gold list, relevance = 0.
    """
    n = len(gold)
    gold_rank = {name: n - i for i, name in enumerate(gold)}
    return [float(gold_rank.get(name, 0)) for name in predicted]


def dcg(predicted: list[str], gold: list[str]) -> float:
    """Discounted Cumulative Gain for a single instance.

    DCG = Σ rel_i / log₂(i + 2)   for i = 0 … N-1
    (using i+2 so position 0 gets log₂(2) = 1).
    """
    rels = _relevance_from_gold(predicted, gold)
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ideal_dcg(gold: list[str]) -> float:
    """DCG of the perfect ranking (= IDCG)."""
    return dcg(gold, gold)


def ndcg(predicted: list[str], gold: list[str]) -> float:
    """Normalised DCG ∈ [0, 1]. Returns 0.0 if IDCG is 0."""
    idcg = ideal_dcg(gold)
    if idcg == 0.0:
        return 0.0
    return dcg(predicted, gold) / idcg


def top1_match(predicted: list[str], gold: list[str]) -> bool:
    """True if the top-ranked predicted image matches the gold top image."""
    if not predicted or not gold:
        return False
    return predicted[0] == gold[0]


# ------------------------------------------------------------------
# Aggregate evaluation
# ------------------------------------------------------------------


@dataclass
class LanguageResult:
    """Aggregated metrics for one language."""

    language_code: str
    n_instances: int = 0
    top1_correct: int = 0
    dcg_sum: float = 0.0
    ndcg_sum: float = 0.0

    @property
    def top1_accuracy(self) -> float:
        return self.top1_correct / max(self.n_instances, 1)

    @property
    def mean_dcg(self) -> float:
        return self.dcg_sum / max(self.n_instances, 1)

    @property
    def mean_ndcg(self) -> float:
        return self.ndcg_sum / max(self.n_instances, 1)


@dataclass
class EvalReport:
    """Full evaluation report across all languages."""

    per_language: dict[str, LanguageResult] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def add(
        self,
        language_code: str,
        predicted: list[str],
        gold: list[str],
    ) -> None:
        """Score one instance and accumulate into the language bucket."""
        if language_code not in self.per_language:
            self.per_language[language_code] = LanguageResult(
                language_code=language_code
            )
        lr = self.per_language[language_code]
        lr.n_instances += 1
        lr.top1_correct += int(top1_match(predicted, gold))
        lr.dcg_sum += dcg(predicted, gold)
        lr.ndcg_sum += ndcg(predicted, gold)

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    @property
    def overall_top1_accuracy(self) -> float:
        total = sum(lr.n_instances for lr in self.per_language.values())
        correct = sum(lr.top1_correct for lr in self.per_language.values())
        return correct / max(total, 1)

    @property
    def overall_mean_ndcg(self) -> float:
        total = sum(lr.n_instances for lr in self.per_language.values())
        ndcg_sum = sum(lr.ndcg_sum for lr in self.per_language.values())
        return ndcg_sum / max(total, 1)

    def summary_table(self) -> str:
        """Return a human-readable summary table."""
        lines = [
            f"{'Lang':<8} {'N':>5} {'Top-1':>7} {'nDCG':>7} {'DCG':>7}",
            "-" * 38,
        ]
        for code in sorted(self.per_language):
            lr = self.per_language[code]
            lines.append(
                f"{code:<8} {lr.n_instances:>5} "
                f"{lr.top1_accuracy:>7.3f} "
                f"{lr.mean_ndcg:>7.3f} "
                f"{lr.mean_dcg:>7.3f}"
            )
        lines.append("-" * 38)
        lines.append(
            f"{'ALL':<8} "
            f"{sum(lr.n_instances for lr in self.per_language.values()):>5} "
            f"{self.overall_top1_accuracy:>7.3f} "
            f"{self.overall_mean_ndcg:>7.3f}"
        )
        return "\n".join(lines)
