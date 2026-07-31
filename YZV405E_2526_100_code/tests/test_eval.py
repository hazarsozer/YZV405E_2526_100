"""Unit tests for evaluation metrics."""
from __future__ import annotations

import math

import pytest

from src.eval import (
    EvalReport,
    LanguageResult,
    dcg,
    ideal_dcg,
    ndcg,
    top1_match,
    _relevance_from_gold,
)


GOLD = ["a.png", "b.png", "c.png", "d.png", "e.png"]


# ---------------------------------------------------------------------------
# _relevance_from_gold
# ---------------------------------------------------------------------------


class TestRelevance:
    def test_perfect_order(self):
        assert _relevance_from_gold(GOLD, GOLD) == [5.0, 4.0, 3.0, 2.0, 1.0]

    def test_reversed_order(self):
        pred = list(reversed(GOLD))
        assert _relevance_from_gold(pred, GOLD) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_unknown_image(self):
        pred = ["x.png"] + GOLD[1:]
        rels = _relevance_from_gold(pred, GOLD)
        assert rels[0] == 0.0  # unknown gets 0


# ---------------------------------------------------------------------------
# top1_match
# ---------------------------------------------------------------------------


class TestTop1Match:
    def test_match(self):
        assert top1_match(["a.png", "b.png"], ["a.png", "c.png"]) is True

    def test_no_match(self):
        assert top1_match(["b.png", "a.png"], ["a.png", "b.png"]) is False

    def test_empty_predicted(self):
        assert top1_match([], GOLD) is False

    def test_empty_gold(self):
        assert top1_match(GOLD, []) is False


# ---------------------------------------------------------------------------
# DCG / nDCG
# ---------------------------------------------------------------------------


class TestDCG:
    def test_perfect_ranking_equals_ideal(self):
        assert dcg(GOLD, GOLD) == pytest.approx(ideal_dcg(GOLD))

    def test_reversed_ranking_lower_than_ideal(self):
        pred = list(reversed(GOLD))
        assert dcg(pred, GOLD) < ideal_dcg(GOLD)

    def test_ndcg_perfect_is_one(self):
        assert ndcg(GOLD, GOLD) == pytest.approx(1.0)

    def test_ndcg_in_unit_interval(self):
        pred = ["c.png", "a.png", "e.png", "b.png", "d.png"]
        score = ndcg(pred, GOLD)
        assert 0.0 <= score <= 1.0

    def test_ndcg_reversed_less_than_one(self):
        pred = list(reversed(GOLD))
        assert ndcg(pred, GOLD) < 1.0

    def test_ideal_dcg_value(self):
        # For 5 items with rels [5,4,3,2,1]:
        # IDCG = 5/log2(2) + 4/log2(3) + 3/log2(4) + 2/log2(5) + 1/log2(6)
        expected = (
            5 / math.log2(2)
            + 4 / math.log2(3)
            + 3 / math.log2(4)
            + 2 / math.log2(5)
            + 1 / math.log2(6)
        )
        assert ideal_dcg(GOLD) == pytest.approx(expected)

    def test_ndcg_empty_gold_returns_zero(self):
        assert ndcg(["a.png"], []) == 0.0


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------


class TestEvalReport:
    def test_add_and_top1_accuracy(self):
        report = EvalReport()
        report.add("TR", GOLD, GOLD)  # correct
        report.add("TR", list(reversed(GOLD)), GOLD)  # wrong
        assert report.per_language["TR"].top1_accuracy == pytest.approx(0.5)

    def test_overall_top1_across_languages(self):
        report = EvalReport()
        report.add("TR", GOLD, GOLD)
        report.add("EL", GOLD, GOLD)
        report.add("ZH", list(reversed(GOLD)), GOLD)
        assert report.overall_top1_accuracy == pytest.approx(2 / 3)

    def test_overall_mean_ndcg_perfect(self):
        report = EvalReport()
        report.add("TR", GOLD, GOLD)
        report.add("EL", GOLD, GOLD)
        assert report.overall_mean_ndcg == pytest.approx(1.0)

    def test_summary_table_has_header(self):
        report = EvalReport()
        report.add("TR", GOLD, GOLD)
        table = report.summary_table()
        assert "Lang" in table
        assert "TR" in table
        assert "ALL" in table

    def test_language_result_properties(self):
        lr = LanguageResult(language_code="TR", n_instances=4,
                            top1_correct=2, dcg_sum=10.0, ndcg_sum=3.0)
        assert lr.top1_accuracy == pytest.approx(0.5)
        assert lr.mean_dcg == pytest.approx(2.5)
        assert lr.mean_ndcg == pytest.approx(0.75)
