"""Evaluate any submission directory against EN and PT-BR training splits.

Reads gold labels from the training TSVs (which include expected_order and
sentence_type) and computes Top-1 accuracy + mean NDCG for each strategy
folder found in the given submissions directory.

Usage:
    # Evaluate all strategy subdirs under data/submissions/ablations/
    uv run python -m scripts.eval_train

    # Evaluate a specific strategy
    uv run python -m scripts.eval_train --strategy bge_clf_caption

    # Point at a custom submissions root
    uv run python -m scripts.eval_train --submissions-dir data/submissions/ablations
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.eval import EvalReport

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

EVAL_LANGS = {"EN": "English", "PT-BR": "Portuguese-Brazil"}
TRAIN_FILENAME = "subtask_a_train.tsv"


def _load_gold(train_root: Path) -> dict[str, pd.DataFrame]:
    """Return {lang_code: DataFrame} with columns compound, sentence, expected_order."""
    gold: dict[str, pd.DataFrame] = {}
    for lang_code, lang_name in EVAL_LANGS.items():
        path = train_root / lang_name / TRAIN_FILENAME
        if not path.exists():
            log.warning("Training TSV not found: %s — skipping %s.", path, lang_code)
            continue
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        gold[lang_code] = df
        log.info("Loaded %d gold rows for %s.", len(df), lang_code)
    return gold


def _eval_strategy(
    strategy_dir: Path,
    gold: dict[str, pd.DataFrame],
) -> dict[str, tuple[float, float]]:
    """Return {lang_code: (top1_acc, mean_ndcg)} for each available language."""
    results: dict[str, tuple[float, float]] = {}

    for lang_code, lang_name in EVAL_LANGS.items():
        if lang_code not in gold:
            continue

        pred_path = strategy_dir / f"submission_{lang_code}.tsv"
        if not pred_path.exists():
            log.warning("Prediction file not found: %s", pred_path)
            continue

        pred_df = pd.read_csv(pred_path, sep="\t", dtype=str, keep_default_na=False)
        gold_df = gold[lang_code]

        merged = pred_df.merge(
            gold_df[["compound", "sentence", "expected_order"]],
            on=["compound", "sentence"],
            suffixes=("_pred", "_gold"),
        )
        if merged.empty:
            log.warning("%s: no rows matched after join — skipping.", lang_code)
            continue
        log.info("%s: %d train rows matched in submission.", lang_code, len(merged))

        report = EvalReport()
        for _, row in merged.iterrows():
            pred_order_raw = row.get("expected_order_pred", "")
            gold_order_raw = row.get("expected_order_gold", "")

            if not pred_order_raw or not gold_order_raw:
                continue

            pred_order = _parse_order(pred_order_raw)
            gold_order = _parse_order(gold_order_raw)

            if len(pred_order) != 5 or len(gold_order) != 5:
                continue

            report.add(lang_code, pred_order, gold_order)

        n_total = sum(lr.n_instances for lr in report.per_language.values())
        if n_total > 0:
            top1_acc = report.overall_top1_accuracy
            mean_ndcg = report.overall_mean_ndcg
            results[lang_code] = (top1_acc, mean_ndcg)

    return results


def _parse_order(raw: str) -> list[str]:
    """Parse "['a.png', 'b.png', ...]" into a list of filenames."""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1]
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    return []


def main(args: argparse.Namespace) -> None:
    submissions_root = Path(args.submissions_dir)
    train_root = Path(args.train_root)

    gold = _load_gold(train_root)
    if not gold:
        log.error(
            "No gold data found under %s. "
            "Make sure EN and PT-BR training TSVs exist.",
            train_root,
        )
        return

    if args.strategy:
        strategy_dirs = [submissions_root / args.strategy]
    else:
        strategy_dirs = sorted(
            d for d in submissions_root.iterdir() if d.is_dir()
        )

    if not strategy_dirs:
        log.error("No strategy directories found under %s.", submissions_root)
        return

    header = f"{'Strategy':<35} {'Lang':<8} {'Top-1':>6} {'NDCG':>6}"
    print("\n" + header)
    print("-" * len(header))

    summary: list[tuple[str, float, float]] = []

    for strat_dir in strategy_dirs:
        results = _eval_strategy(strat_dir, gold)
        if not results:
            continue

        for lang_code, (top1, ndcg) in sorted(results.items()):
            print(f"{strat_dir.name:<35} {lang_code:<8} {top1:>6.4f} {ndcg:>6.4f}")

        avg_top1 = sum(v[0] for v in results.values()) / len(results)
        avg_ndcg = sum(v[1] for v in results.values()) / len(results)
        summary.append((strat_dir.name, avg_top1, avg_ndcg))
        print(f"{'  → avg':<35} {'':8} {avg_top1:>6.4f} {avg_ndcg:>6.4f}")
        print()

    if len(summary) > 1:
        print("=" * len(header))
        print(f"\n{'Strategy':<35} {'Avg Top-1':>9} {'Avg NDCG':>8}")
        for name, top1, ndcg in sorted(summary, key=lambda x: -x[1]):
            print(f"{name:<35} {top1:>9.4f} {ndcg:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--submissions-dir",
        default="data/submissions/ablations",
        help="Root dir containing per-strategy subdirs (default: data/submissions/ablations).",
    )
    parser.add_argument(
        "--train-root",
        default="data/raw/admire2_data",
        help="Root of the raw data containing Language/subtask_a_train.tsv files.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        metavar="NAME",
        help="Evaluate only this strategy subdir (default: all).",
    )
    main(parser.parse_args())
