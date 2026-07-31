"""Train the LR sense classifier on EN and PT-BR labelled splits.

Usage:
    python scripts/train_lr.py \\
        --en-tsvs  data/raw/train_data/EN/train/subtask_a_train.tsv \\
                   data/raw/train_data/EN/dev/subtask_a_dev.tsv \\
                   data/raw/train_data/EN/test/subtask_a_test.tsv \\
                   data/raw/train_data/EN/xeval/subtask_a_xe.tsv \\
        --ptbr-tsvs data/raw/train_data/PT/train/subtask_a_train.tsv \\
                    data/raw/train_data/PT/dev/subtask_a_dev.tsv \\
                    data/raw/train_data/PT/test/subtask_a_test.tsv \\
                    data/raw/train_data/PT/xeval/subtask_a_xp.tsv \\
        [--data-root data/raw/admire2_data] \\
        [--output models/lr_sense_classifier.joblib] \\
        [--cache-dir data/processed] \\
        [--device cpu]

Pass as many TSV files per language as you have (train/dev/test/xeval).
Duplicate sentences across splits are deduplicated automatically.

The script embeds all sentences with BGE-M3 (results cached to parquet),
fits CalibratedClassifierCV(LogisticRegression, isotonic, cv=5), and saves
the checkpoint to --output.
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.loader import AdMIReRepository
from src.models.frozen_encoders import BGEM3Encoder
from src.models.sense_classifier import LRSenseClassifier
from src.utils.cache import EmbeddingCache
from src.utils.io import sha256_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_labelled_instances(
    repo: AdMIReRepository, tsv_path: Path, language_name: str
) -> tuple[list[str], list[str]]:
    """Return (sentences, labels) for all labelled rows in *tsv_path*."""
    dataset = repo.load_training_tsv(tsv_path, language_name)
    sentences: list[str] = []
    labels: list[str] = []
    for inst in dataset.instances:
        if inst.sentence_type is None:
            log.warning("Row %d has no sentence_type — skipped.", inst.row_index)
            continue
        sentences.append(inst.sentence)
        labels.append(inst.sentence_type)
    return sentences, labels


def _embed_with_cache(
    sentences: list[str],
    encoder: BGEM3Encoder,
    cache: EmbeddingCache,
    model_name: str,
) -> np.ndarray:
    """Return BGE-M3 embeddings for *sentences*, using *cache* where available."""
    keys = [sha256_string(s) for s in sentences]
    cached = cache.get_batch(model_name, keys)

    missing_idx = [i for i, k in enumerate(keys) if k not in cached]
    if missing_idx:
        missing_texts = [sentences[i] for i in missing_idx]
        log.info("Embedding %d new sentences with BGE-M3 …", len(missing_texts))
        new_embs = encoder.encode(missing_texts)
        new_cache_entries = {
            keys[i]: new_embs[j] for j, i in enumerate(missing_idx)
        }
        cache.set_batch(model_name, new_cache_entries)
        cached.update(new_cache_entries)

    return np.stack([cached[k] for k in keys], axis=0)


def main(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    cache_dir = Path(args.cache_dir)
    device: str = args.device

    repo = AdMIReRepository(
        data_root=Path(args.data_root),
        templates_root=Path("data/submissions/templates"),
    )

    # ------------------------------------------------------------------
    # Collect labelled data — deduplicate by sentence text
    # ------------------------------------------------------------------
    seen: set[str] = set()
    all_sentences: list[str] = []
    all_labels: list[str] = []

    lang_tsvs: list[tuple[list[str], str]] = [
        (args.en_tsvs or [], "English"),
        (args.ptbr_tsvs or [], "Portuguese-Brazil"),
    ]

    if not any(paths for paths, _ in lang_tsvs):
        raise ValueError("Provide at least one TSV via --en-tsvs or --ptbr-tsvs.")

    for tsv_paths, lang_name in lang_tsvs:
        lang_total = 0
        for tsv_str in tsv_paths:
            tsv_path = Path(tsv_str)
            if not tsv_path.exists():
                raise FileNotFoundError(f"Training TSV not found: {tsv_path}")
            sents, labs = _load_labelled_instances(repo, tsv_path, lang_name)
            added = 0
            for s, label in zip(sents, labs):
                if s not in seen:
                    seen.add(s)
                    all_sentences.append(s)
                    all_labels.append(label)
                    added += 1
            log.info("  %s  →  %d new instances (%d total in file)",
                     tsv_path.name, added, len(sents))
            lang_total += added
        if lang_total:
            log.info("%s: %d unique labelled instances loaded.", lang_name, lang_total)

    log.info("Total training instances: %d", len(all_sentences))
    label_counts = Counter(all_labels)
    log.info("Label distribution: %s", label_counts)

    # ------------------------------------------------------------------
    # Embed with BGE-M3 (cached)
    # ------------------------------------------------------------------
    cache = EmbeddingCache(cache_dir)
    model_name = BGEM3Encoder.MODEL_ID
    keys_all = [sha256_string(s) for s in all_sentences]
    pre_cached = cache.get_batch(model_name, keys_all)
    missing_idx = [i for i, k in enumerate(keys_all) if k not in pre_cached]

    if missing_idx:
        with BGEM3Encoder(device=device) as encoder:
            embeddings = _embed_with_cache(all_sentences, encoder, cache, model_name)
    else:
        log.info("All embeddings already in cache — BGE-M3 not loaded.")
        embeddings = np.stack([pre_cached[k] for k in keys_all], axis=0)

    log.info("Embeddings shape: %s", embeddings.shape)

    # ------------------------------------------------------------------
    # Train and save
    # ------------------------------------------------------------------
    clf = LRSenseClassifier()
    log.info("Fitting LRSenseClassifier …")
    clf.fit(embeddings, all_labels)
    clf.save(output_path)
    log.info("Saved to %s", output_path)

    probas = clf.predict_proba_idiomatic(embeddings[:5])
    log.info("P(idiomatic) for first 5 training samples: %s", probas.round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--en-tsvs",
        nargs="*",
        default=[],
        metavar="TSV",
        help="One or more English labelled TSV files (train/dev/test/xeval).",
    )
    parser.add_argument(
        "--ptbr-tsvs",
        nargs="*",
        default=[],
        metavar="TSV",
        help="One or more PT-BR labelled TSV files (train/dev/test/xeval).",
    )
    parser.add_argument(
        "--data-root",
        default="data/raw/admire2_data",
        help="Root directory of the AdMIRe 2 image data (default: data/raw/admire2_data).",
    )
    parser.add_argument(
        "--output",
        default="models/lr_sense_classifier.joblib",
        help="Where to save the trained classifier checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/processed",
        help="Directory for BGE-M3 embedding cache.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Torch device for BGE-M3.",
    )
    main(parser.parse_args())
