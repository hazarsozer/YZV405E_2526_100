"""Train the LR sense classifier on EN and PT-BR labelled splits.

Usage:
    python scripts/train_lr.py \\
        --en-tsv  data/raw/admire2_data/English/train.tsv \\
        --ptbr-tsv data/raw/admire2_data/Portuguese-Brazil/train.tsv \\
        [--output models/lr_sense_classifier.joblib] \\
        [--cache-dir data/processed] \\
        [--device cuda]

The script embeds all sentences with BGE-M3 (results cached to parquet),
fits CalibratedClassifierCV(LogisticRegression, isotonic, cv=5), and saves
the checkpoint to --output.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from src.data.loader import AdMIReRepository
from src.models.frozen_encoders import BGEM3Encoder
from src.models.sense_classifier import LRSenseClassifier
from src.utils.cache import EmbeddingCache
from src.utils.io import sha256_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Language name → code mapping for the two supervised splits
SUPERVISED_LANGS: dict[str, str] = {
    "English": "EN",
    "Portuguese-Brazil": "PT-BR",
}


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

    # ------------------------------------------------------------------
    # Collect labelled data from EN and PT-BR splits
    # ------------------------------------------------------------------
    all_sentences: list[str] = []
    all_labels: list[str] = []

    repo = AdMIReRepository(
        data_root=Path(args.en_tsv).parent.parent,
        templates_root=Path("data/submissions/templates"),
    )

    splits: list[tuple[Path, str]] = []
    if args.en_tsv:
        splits.append((Path(args.en_tsv), "English"))
    if args.ptbr_tsv:
        splits.append((Path(args.ptbr_tsv), "Portuguese-Brazil"))

    if not splits:
        raise ValueError("At least one of --en-tsv or --ptbr-tsv must be provided.")

    for tsv_path, lang_name in splits:
        if not tsv_path.exists():
            raise FileNotFoundError(f"Training TSV not found: {tsv_path}")
        sents, labs = _load_labelled_instances(repo, tsv_path, lang_name)
        log.info("%s: %d labelled instances.", lang_name, len(sents))
        all_sentences.extend(sents)
        all_labels.extend(labs)

    log.info("Total training instances: %d", len(all_sentences))
    label_counts = {
        label: all_labels.count(label) for label in set(all_labels)
    }
    log.info("Label distribution: %s", label_counts)

    # ------------------------------------------------------------------
    # Embed with BGE-M3 (cached)
    # ------------------------------------------------------------------
    cache = EmbeddingCache(cache_dir)
    model_name = BGEM3Encoder.MODEL_ID

    with BGEM3Encoder(device=device) as encoder:
        embeddings = _embed_with_cache(all_sentences, encoder, cache, model_name)

    log.info("Embeddings shape: %s", embeddings.shape)

    # ------------------------------------------------------------------
    # Train and save
    # ------------------------------------------------------------------
    clf = LRSenseClassifier()
    log.info("Fitting LRSenseClassifier …")
    clf.fit(embeddings, all_labels)
    clf.save(output_path)
    log.info("Saved to %s", output_path)

    # Quick sanity check
    probas = clf.predict_proba_idiomatic(embeddings[:5])
    log.info("P(idiomatic) for first 5 training samples: %s", probas.round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--en-tsv",
        default=None,
        help="Path to the English labelled training TSV.",
    )
    parser.add_argument(
        "--ptbr-tsv",
        default=None,
        help="Path to the PT-BR labelled training TSV.",
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
