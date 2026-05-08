"""Audit the LR sense classifier on the EN dev split.

Loads cached BGE-M3 embeddings (no model needed) and reports accuracy,
precision, recall, and a per-instance prediction table.

Usage:
    uv run python -m scripts.audit_classifier
    uv run python -m scripts.audit_classifier --tsv data/raw/train_data/EN/dev/dev/subtask_a_dev.tsv
    uv run python -m scripts.audit_classifier --tsv data/raw/train_data/EN/dev/dev/subtask_a_dev.tsv \\
        data/raw/train_data/PT/dev/dev/subtask_a_dev.tsv --ptbr
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.models.frozen_encoders import BGEM3Encoder
from src.models.sense_classifier import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    LRSenseClassifier,
)
from src.utils.cache import EmbeddingCache
from src.utils.io import sha256_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_labeled_tsv(tsv_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (sentences, compounds, labels) from a labeled TSV."""
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"sentence", "sentence_type", "compound"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TSV {tsv_path} missing columns: {missing}")
    mask = df["sentence_type"].isin({"idiomatic", "literal"})
    df = df[mask]
    return (
        df["sentence"].tolist(),
        df["compound"].tolist(),
        df["sentence_type"].tolist(),
    )


def get_embeddings(
    sentences: list[str],
    cache: EmbeddingCache,
    model_name: str,
    device: str,
) -> np.ndarray:
    """Return BGE-M3 embeddings, loading model only for cache misses."""
    keys = [sha256_string(s) for s in sentences]
    cached = cache.get_batch(model_name, keys)
    miss_idx = [i for i, k in enumerate(keys) if k not in cached]

    if miss_idx:
        log.info("Cache miss for %d sentences — loading BGE-M3 …", len(miss_idx))
        with BGEM3Encoder(device=device) as enc:
            miss_texts = [sentences[i] for i in miss_idx]
            new_embs = enc.encode(miss_texts)
            new_cache = {keys[i]: new_embs[j] for j, i in enumerate(miss_idx)}
            cache.set_batch(model_name, new_cache)
            cached.update(new_cache)
    else:
        log.info("All %d embeddings found in cache.", len(sentences))

    return np.stack([cached[k] for k in keys], axis=0)


def print_confusion(tp: int, fp: int, tn: int, fn: int) -> None:
    print("\nConfusion matrix (positive = idiomatic):")
    print(f"  {'':12} {'Pred idiom':>12} {'Pred literal':>12}")
    print(f"  {'True idiom':12} {tp:>12} {fn:>12}")
    print(f"  {'True literal':12} {fp:>12} {tn:>12}")


def main(args: argparse.Namespace) -> None:
    cache = EmbeddingCache(Path(args.cache_dir))
    clf = LRSenseClassifier.from_path(Path(args.classifier))
    model_name = BGEM3Encoder.MODEL_ID

    all_sentences: list[str] = []
    all_labels: list[str] = []
    all_compounds: list[str] = []
    tsv_tag: list[str] = []

    for tsv_str in args.tsvs:
        tsv_path = Path(tsv_str)
        sents, comps, labs = load_labeled_tsv(tsv_path)
        log.info("%s: %d labeled instances", tsv_path.name, len(sents))
        all_sentences.extend(sents)
        all_labels.extend(labs)
        all_compounds.extend(comps)
        tsv_tag.extend([tsv_path.stem] * len(sents))

    if not all_sentences:
        raise SystemExit("No labeled instances found.")

    embs = get_embeddings(all_sentences, cache, model_name, args.device)
    p_idiom = clf.predict_proba_idiomatic(embs)
    pred_labels = ["idiomatic" if p > 0.5 else "literal" for p in p_idiom]

    # --- Per-instance table ---
    header = f"{'#':>4}  {'Compound':<30}  {'True':>9}  {'P(idiom)':>9}  {'Pred':>9}  {'OK':>3}"
    print("\n" + header)
    print("-" * len(header))
    correct = 0
    tp = fp = tn = fn = 0
    branch_counts = {"idiomatic": 0, "literal": 0, "uncertain": 0}

    for i, (sent, comp, gold, pred, p) in enumerate(
        zip(all_sentences, all_compounds, all_labels, pred_labels, p_idiom)
    ):
        ok = pred == gold
        if ok:
            correct += 1
        mark = "✓" if ok else "✗"

        if gold == "idiomatic":
            if pred == "idiomatic":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "literal":
                tn += 1
            else:
                fp += 1

        if p > HIGH_CONFIDENCE_THRESHOLD:
            branch_counts["idiomatic"] += 1
        elif p < LOW_CONFIDENCE_THRESHOLD:
            branch_counts["literal"] += 1
        else:
            branch_counts["uncertain"] += 1

        print(
            f"{i+1:>4}  {comp:<30}  {gold:>9}  {p:>9.3f}  {pred:>9}  {mark:>3}"
        )

    n = len(all_sentences)
    acc = correct / n
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print("\n" + "=" * 60)
    print(f"Instances evaluated : {n}")
    print(f"Accuracy            : {acc:.3f}  ({correct}/{n} correct)")
    print(f"Precision (idiom)   : {precision:.3f}")
    print(f"Recall    (idiom)   : {recall:.3f}")
    print(f"F1 (idiom)          : {f1:.3f}")
    print_confusion(tp, fp, tn, fn)

    print("\nRanker branch distribution (at inference thresholds):")
    print(f"  P > {HIGH_CONFIDENCE_THRESHOLD} → idiomatic branch : {branch_counts['idiomatic']} ({100*branch_counts['idiomatic']/n:.1f}%)")
    print(f"  P < {LOW_CONFIDENCE_THRESHOLD} → literal branch   : {branch_counts['literal']} ({100*branch_counts['literal']/n:.1f}%)")
    print(f"  else → uncertain bypass      : {branch_counts['uncertain']} ({100*branch_counts['uncertain']/n:.1f}%)")

    # P distribution summary
    print(f"\nP(idiomatic) stats:")
    print(f"  mean={p_idiom.mean():.3f}  std={p_idiom.std():.3f}  "
          f"min={p_idiom.min():.3f}  max={p_idiom.max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tsvs", nargs="+",
        default=[
            "data/raw/train_data/EN/dev/dev/subtask_a_dev.tsv",
            "data/raw/train_data/EN/train/train/subtask_a_train.tsv",
        ],
        metavar="TSV",
        help="One or more labeled TSV files to evaluate on.",
    )
    parser.add_argument(
        "--classifier", default="models/lr_sense_classifier.joblib",
        help="Path to the trained classifier checkpoint.",
    )
    parser.add_argument(
        "--cache-dir", default="data/processed",
        help="Embedding cache directory.",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="Device for BGE-M3 if cache miss (default: cpu).",
    )
    main(parser.parse_args())
