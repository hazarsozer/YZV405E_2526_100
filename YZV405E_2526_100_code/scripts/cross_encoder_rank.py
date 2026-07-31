"""Cross-encoder image ranking using BAAI/bge-reranker-v2-m3.

Instead of bi-encoder cosine similarity (embed query and caption separately,
then dot-product), this scores each (query, caption) pair jointly.  The
cross-encoder can attend across both texts, catching semantic relationships
that cosine similarity misses.

Strategies
----------
* xenc_clf   — classifier-gated: paraphrase query for idiomatic, original for literal
* xenc_para  — always use paraphrase as query
* xenc_orig  — always use original (translated) sentence as query

Usage:
    uv run python -m scripts.cross_encoder_rank
    uv run python -m scripts.cross_encoder_rank --strategies xenc_clf
    uv run python -m scripts.cross_encoder_rank --paraphrase-ns qwen25_paraphrase
    uv run python -m scripts.cross_encoder_rank --device cuda --batch-size 64
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.loader import AdMIReRepository, LANG_NAME_TO_CODE, Instance
from src.models.sense_classifier import (
    HIGH_CONFIDENCE_THRESHOLD,
    LRSenseClassifier,
)
from src.models.translator import NO_OP_LANGS
from src.utils.cache import EmbeddingCache, TextCache
from src.utils.io import sha256_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_ID = "BAAI/bge-reranker-v2-m3"
CACHE_MODEL_TAG = "bge-reranker-v2-m3"


# ---------------------------------------------------------------------------
# Cross-encoder wrapper
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """BAAI/bge-reranker-v2-m3 loaded in fp16."""

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model: AutoModelForSequenceClassification | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        ).to(self._device)
        self._model.train(False)
        log.info("bge-reranker-v2-m3 loaded (fp16) on %s.", self._device)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> CrossEncoderReranker:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()

    @torch.inference_mode()
    def score_batch(
        self, pairs: list[tuple[str, str]], batch_size: int = 64
    ) -> np.ndarray:
        """Score a list of (query, passage) pairs. Returns float32 logits."""
        assert self._model is not None and self._tokenizer is not None

        all_scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            queries = [p[0] for p in chunk]
            passages = [p[1] for p in chunk]

            enc = self._tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            logits = self._model(**enc).logits.squeeze(-1)
            all_scores.extend(logits.float().cpu().tolist())

        return np.array(all_scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Cache helpers  (scores stored as shape-(1,) arrays in EmbeddingCache)
# ---------------------------------------------------------------------------


def _score_key(paraphrase_ns: str, query: str, caption: str) -> str:
    return sha256_string(paraphrase_ns, query, caption)


def load_cached_scores(
    keys: list[str], emb_cache: EmbeddingCache
) -> dict[str, float]:
    raw = emb_cache.get_batch(CACHE_MODEL_TAG, keys)
    return {k: float(v[0]) for k, v in raw.items()}


def save_scores(
    key_score: dict[str, float], emb_cache: EmbeddingCache
) -> None:
    arr_map = {k: np.array([v], dtype=np.float32) for k, v in key_score.items()}
    emb_cache.set_batch(CACHE_MODEL_TAG, arr_map)


# ---------------------------------------------------------------------------
# Pipeline helpers (shared with caption_rank.py)
# ---------------------------------------------------------------------------


def load_translated(
    instances: list[tuple[str, Instance]], text_cache: TextCache
) -> list[str]:
    ns = "nllb200_translation"
    translated = [""] * len(instances)
    real_idx, real_keys = [], []
    for i, (lang, inst) in enumerate(instances):
        if lang in NO_OP_LANGS:
            translated[i] = inst.sentence
        else:
            real_idx.append(i)
            real_keys.append(sha256_string(lang, inst.sentence))

    if real_idx:
        cached = text_cache.get_batch(ns, real_keys)
        missing = [k for k in real_keys if k not in cached]
        if missing:
            raise RuntimeError(f"{len(missing)} translation cache misses — run full pipeline first.")
        for j, i in enumerate(real_idx):
            translated[i] = cached[real_keys[j]]

    return translated


def load_paraphrased(
    instances: list[tuple[str, Instance]],
    translated: list[str],
    text_cache: TextCache,
    paraphrase_ns: str,
) -> list[str]:
    keys = [
        sha256_string(translated[i], inst.compound)
        for i, (_, inst) in enumerate(instances)
    ]
    cached = text_cache.get_batch(paraphrase_ns, keys)
    missing = [i for i, k in enumerate(keys) if k not in cached]
    if missing:
        raise RuntimeError(
            f"{len(missing)} paraphrase cache misses in namespace '{paraphrase_ns}'."
        )
    return [cached[k] for k in keys]


def load_bge_embeddings_for_classifier(
    translated: list[str], emb_cache: EmbeddingCache, device: str
) -> np.ndarray:
    from src.models.frozen_encoders import BGEM3Encoder

    model_name = BGEM3Encoder.MODEL_ID
    keys = [sha256_string(t) for t in translated]
    cached = emb_cache.get_batch(model_name, keys)
    miss_idx = [i for i, k in enumerate(keys) if k not in cached]

    if miss_idx:
        log.info("BGE-M3 cache miss for %d sentences — loading model …", len(miss_idx))
        with BGEM3Encoder(device=device) as enc:
            new_embs = enc.encode([translated[i] for i in miss_idx])
            new_cache = {keys[i]: new_embs[j] for j, i in enumerate(miss_idx)}
            emb_cache.set_batch(model_name, new_cache)
            cached.update(new_cache)

    return np.stack([cached[k] for k in keys], axis=0)


# ---------------------------------------------------------------------------
# Submission writer
# ---------------------------------------------------------------------------


def write_submissions(
    strategy_name: str,
    instances: list[tuple[str, Instance]],
    rankings: list[list[str]],
    out_dir: Path,
    templates_root: Path,
) -> None:
    code_to_name = {v: k for k, v in LANG_NAME_TO_CODE.items()}
    results_by_lang: dict[str, list[list[str]]] = {}
    for (lang, _), order in zip(instances, rankings):
        results_by_lang.setdefault(lang, []).append(order)

    strat_dir = out_dir / strategy_name
    strat_dir.mkdir(parents=True, exist_ok=True)

    for lang_code, orders in results_by_lang.items():
        lang_name = code_to_name[lang_code]
        template_path = templates_root / f"submission_{lang_name}.tsv"
        df = pd.read_csv(template_path, sep="\t", dtype=str, keep_default_na=False)
        df["expected_order"] = [
            "[" + ", ".join(f"'{n}'" for n in o) + "]" for o in orders
        ]
        df.to_csv(strat_dir / f"submission_{lang_code}.tsv", sep="\t", index=False)

    log.info("Strategy %-20s → %s", f"'{strategy_name}'", strat_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    templates_root = Path(args.templates_root)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)

    repo = AdMIReRepository(data_root, templates_root)
    all_datasets = repo.load_all_templates()
    instances: list[tuple[str, Instance]] = [
        (lang, inst)
        for lang, ds in all_datasets.items()
        for inst in ds.instances
    ]
    log.info("Loaded %d instances across %d languages.", len(instances), len(all_datasets))

    emb_cache = EmbeddingCache(cache_dir / "embeddings")
    text_cache = TextCache(cache_dir / "text")

    log.info("Loading translations …")
    translated = load_translated(instances, text_cache)
    log.info("Loading paraphrases (namespace: %s) …", args.paraphrase_ns)
    paraphrased = load_paraphrased(instances, translated, text_cache, args.paraphrase_ns)

    log.info("Loading BGE-M3 embeddings for classifier …")
    bge_embs = load_bge_embeddings_for_classifier(translated, emb_cache, args.device)

    log.info("Running LR classifier …")
    clf = LRSenseClassifier.from_path(Path(args.classifier))
    p_idiom = clf.predict_proba_idiomatic(bge_embs)

    selected = set(args.strategies) if args.strategies else {"xenc_clf", "xenc_para", "xenc_orig"}

    # --- Build per-instance queries for each strategy ------------------
    queries_by_strategy: dict[str, list[str]] = {}
    for strat in selected:
        qs: list[str] = []
        for i in range(len(instances)):
            p = float(p_idiom[i])
            if strat == "xenc_para":
                qs.append(paraphrased[i])
            elif strat == "xenc_orig":
                qs.append(translated[i])
            else:  # xenc_clf
                qs.append(paraphrased[i] if p > HIGH_CONFIDENCE_THRESHOLD else translated[i])
        queries_by_strategy[strat] = qs

    # --- Check score cache, collect all pairs that need scoring --------
    all_needed: dict[str, tuple[str, str]] = {}  # key → (query, caption)
    for strat, queries in queries_by_strategy.items():
        for i, (_, inst) in enumerate(instances):
            for img in inst.images:
                k = _score_key(f"{args.paraphrase_ns}:{strat}", queries[i], img.caption)
                if k not in all_needed:
                    all_needed[k] = (queries[i], img.caption)

    existing = load_cached_scores(list(all_needed.keys()), emb_cache)
    miss_keys = [k for k in all_needed if k not in existing]

    if miss_keys:
        log.info(
            "%d / %d (query, caption) pairs need scoring — loading cross-encoder …",
            len(miss_keys), len(all_needed),
        )
        pairs_to_score = [all_needed[k] for k in miss_keys]

        with CrossEncoderReranker(device=args.device) as reranker:
            scores_arr = reranker.score_batch(pairs_to_score, batch_size=args.batch_size)

        new_scores = {k: float(scores_arr[j]) for j, k in enumerate(miss_keys)}
        save_scores(new_scores, emb_cache)
        existing.update(new_scores)
        log.info("Cross-encoder scoring done. Scores cached.")
    else:
        log.info("All %d scores found in cache.", len(all_needed))

    # --- Rank and write submissions ------------------------------------
    for strat in sorted(selected):
        queries = queries_by_strategy[strat]
        rankings: list[list[str]] = []

        for i, (_, inst) in enumerate(instances):
            names = [img.name for img in inst.images]
            scores = np.array([
                existing[_score_key(f"{args.paraphrase_ns}:{strat}", queries[i], img.caption)]
                for img in inst.images
            ])
            rankings.append([names[j] for j in np.argsort(-scores)])

        write_submissions(strat, instances, rankings, out_dir, templates_root)

    print("\n" + "=" * 60)
    print("Cross-encoder strategies generated:")
    for s in sorted(selected):
        print(f"  {out_dir / s}")
    print(f"\nParaphrase namespace : {args.paraphrase_ns}")
    print("To zip: cd data/submissions/ablations/<name> && zip -j ../<name>.zip *.tsv")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", default="data/raw/admire2_data")
    parser.add_argument("--templates-root", default="data/submissions/templates")
    parser.add_argument("--classifier", default="models/lr_sense_classifier.joblib")
    parser.add_argument("--cache-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/submissions/ablations")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Number of (query, caption) pairs per cross-encoder forward pass.",
    )
    parser.add_argument(
        "--strategies", nargs="+", metavar="NAME",
        choices=["xenc_clf", "xenc_para", "xenc_orig"],
        help="Strategies to run (default: all 3).",
    )
    parser.add_argument(
        "--paraphrase-ns", default="qwen25_paraphrase",
        help="Paraphrase cache namespace (default: qwen25_paraphrase).",
    )
    main(parser.parse_args())
