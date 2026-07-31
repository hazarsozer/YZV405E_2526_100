"""Replay ranking with multiple strategies using cached embeddings.

Re-runs ONLY the ranking step over the cached SigLIP2/BGE-M3 embeddings from
a previous full pipeline run.  No models are loaded if the cache is warm
(which it will be after a full inference run).

Outputs one submission directory per strategy, each containing 15 TSV files
ready to zip and submit to Codabench.

Usage:
    uv run python -m scripts.replay_ranking               # all strategies
    uv run python -m scripts.replay_ranking --strategies orig para current
    uv run python -m scripts.replay_ranking --alpha-sweep # 10-point alpha sweep
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import AdMIReRepository, LANG_NAME_TO_CODE, Instance
from src.models.frozen_encoders import BGEM3Encoder, SigLIP2Encoder
from src.models.ranker import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    cosine_similarity,
)
from src.models.sense_classifier import LRSenseClassifier
from src.models.translator import NO_OP_LANGS
from src.utils.cache import EmbeddingCache, TextCache
from src.utils.io import sha256_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankStrategy:
    name: str
    description: str
    # If not None, overrides the classifier's p_idiomatic for every instance
    force_p: float | None = None
    # Literal penalty weight (only used in the idiomatic branch)
    alpha: float = 0.3
    # If True, use average of sim_orig and sim_para
    blend: bool = False


DEFAULT_STRATEGIES: list[RankStrategy] = [
    RankStrategy(
        "orig_only",
        "Pure sim(original_text, image) — approximates vanilla SigLIP2 baseline",
        force_p=0.0,
        alpha=0.0,
    ),
    RankStrategy(
        "para_only",
        "Pure sim(paraphrased_text, image) — no literal penalty, always uses paraphrase",
        force_p=1.0,
        alpha=0.0,
    ),
    RankStrategy(
        "para_a0.1",
        "Always-idiomatic with alpha=0.1 literal penalty",
        force_p=1.0,
        alpha=0.1,
    ),
    RankStrategy(
        "para_a0.3",
        "Always-idiomatic with alpha=0.3 literal penalty (original idiomatic branch logic)",
        force_p=1.0,
        alpha=0.3,
    ),
    RankStrategy(
        "para_a0.5",
        "Always-idiomatic with alpha=0.5 literal penalty",
        force_p=1.0,
        alpha=0.5,
    ),
    RankStrategy(
        "classifier_a0.0",
        "Classifier-gated, no literal penalty (alpha=0.0)",
        force_p=None,
        alpha=0.0,
    ),
    RankStrategy(
        "classifier_a0.3",
        "Classifier-gated, alpha=0.3 — this is our submitted version (score: 0.30)",
        force_p=None,
        alpha=0.3,
    ),
    RankStrategy(
        "blend",
        "Average of sim_orig and sim_para — simple ensemble without classifier",
        blend=True,
    ),
]


def make_alpha_sweep_strategies(
    alphas: list[float], force_p: float | None = None
) -> list[RankStrategy]:
    tag = "clf" if force_p is None else f"p{force_p:.1f}"
    return [
        RankStrategy(
            f"{tag}_a{a:.2f}",
            f"{'Classifier' if force_p is None else f'P={force_p}'} with alpha={a:.2f}",
            force_p=force_p,
            alpha=a,
        )
        for a in alphas
    ]


# ---------------------------------------------------------------------------
# Ranking logic
# ---------------------------------------------------------------------------


def apply_strategy(
    image_embs: np.ndarray,
    orig_emb: np.ndarray,
    para_emb: np.ndarray,
    p_idiomatic: float,
    image_names: list[str],
    strategy: RankStrategy,
) -> list[str]:
    """Rank images under *strategy* and return image names best-to-worst."""
    sim_orig = cosine_similarity(orig_emb, image_embs)
    sim_para = cosine_similarity(para_emb, image_embs)

    if strategy.blend:
        scores = 0.5 * sim_orig + 0.5 * sim_para
    else:
        p = strategy.force_p if strategy.force_p is not None else p_idiomatic
        if p > HIGH_CONFIDENCE_THRESHOLD:
            scores = sim_para - strategy.alpha * sim_orig
        elif p < LOW_CONFIDENCE_THRESHOLD:
            scores = sim_orig
        else:
            scores = sim_para

    ranked = np.argsort(-scores)
    return [image_names[int(i)] for i in ranked]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_translated(
    instances: list[tuple[str, Instance]],
    text_cache: TextCache,
) -> list[str]:
    """Reconstruct translated sentences from cache or use originals for NO_OP langs."""
    ns = "nllb200_translation"
    translated = [""] * len(instances)
    real_idx: list[int] = []
    real_keys: list[str] = []

    for i, (lang, inst) in enumerate(instances):
        if lang in NO_OP_LANGS:
            translated[i] = inst.sentence
        else:
            real_idx.append(i)
            real_keys.append(sha256_string(lang, inst.sentence))

    if real_idx:
        cached = text_cache.get_batch(ns, real_keys)
        missing = [real_keys[j] for j, k in enumerate(real_keys) if k not in cached]
        if missing:
            raise RuntimeError(
                f"{len(missing)} translation cache misses — run full pipeline first.\n"
                f"First missing key: {missing[0]}"
            )
        for j, i in enumerate(real_idx):
            translated[i] = cached[real_keys[j]]

    return translated


def load_paraphrased(
    instances: list[tuple[str, Instance]],
    translated: list[str],
    text_cache: TextCache,
) -> list[str]:
    """Reconstruct paraphrased sentences from cache."""
    ns = "phi35_paraphrase"
    keys = [
        sha256_string(translated[i], inst.compound)
        for i, (_, inst) in enumerate(instances)
    ]
    cached = text_cache.get_batch(ns, keys)
    missing_idx = [i for i, k in enumerate(keys) if k not in cached]
    if missing_idx:
        raise RuntimeError(
            f"{len(missing_idx)} paraphrase cache misses — run full pipeline first.\n"
            f"First missing index: {missing_idx[0]} ({instances[missing_idx[0]][1].sentence[:60]})"
        )
    return [cached[k] for k in keys]


def load_bge_embeddings(
    translated: list[str],
    emb_cache: EmbeddingCache,
    device: str,
) -> np.ndarray:
    """Return BGE-M3 embeddings, loading model only on cache miss."""
    model_name = BGEM3Encoder.MODEL_ID
    keys = [sha256_string(t) for t in translated]
    cached = emb_cache.get_batch(model_name, keys)
    miss_idx = [i for i, k in enumerate(keys) if k not in cached]

    if miss_idx:
        log.info("BGE-M3 cache miss for %d sentences — loading model …", len(miss_idx))
        with BGEM3Encoder(device=device) as enc:
            miss_texts = [translated[i] for i in miss_idx]
            new_embs = enc.encode(miss_texts)
            new_cache = {keys[i]: new_embs[j] for j, i in enumerate(miss_idx)}
            emb_cache.set_batch(model_name, new_cache)
            cached.update(new_cache)

    return np.stack([cached[k] for k in keys], axis=0)


def load_siglip2_embeddings(
    instances: list[tuple[str, Instance]],
    translated: list[str],
    paraphrased: list[str],
    emb_cache: EmbeddingCache,
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Return (img_embs, orig_embs, para_embs) from cache or model."""
    model_name = SigLIP2Encoder.MODEL_ID

    # Build all keys
    img_keys: dict[str, Path] = {}
    img_keys_per_inst: list[list[str]] = []
    for _, inst in instances:
        ks = [sha256_string(str(img.absolute_path)) for img in inst.images]
        img_keys_per_inst.append(ks)
        for k, img in zip(ks, inst.images):
            img_keys[k] = img.absolute_path

    text_set: dict[str, str] = {}
    orig_keys: list[str] = []
    para_keys: list[str] = []
    for i in range(len(instances)):
        ko = sha256_string("text", translated[i])
        kp = sha256_string("text", paraphrased[i])
        orig_keys.append(ko)
        para_keys.append(kp)
        text_set[ko] = translated[i]
        text_set[kp] = paraphrased[i]

    all_keys = list(img_keys.keys()) + list(text_set.keys())
    cached = emb_cache.get_batch(model_name, all_keys)
    missing_img = [k for k in img_keys if k not in cached]
    missing_txt = [k for k in text_set if k not in cached]

    if missing_img or missing_txt:
        log.info(
            "SigLIP2 cache miss: %d images, %d texts — loading model …",
            len(missing_img), len(missing_txt),
        )
        from PIL import Image as PILImage

        with SigLIP2Encoder(device=device) as enc:
            if missing_img:
                pil_imgs = [PILImage.open(img_keys[k]).convert("RGB") for k in missing_img]
                new_img_embs = enc.encode(pil_imgs)
                new_cache = {k: new_img_embs[j] for j, k in enumerate(missing_img)}
                emb_cache.set_batch(model_name, new_cache)
                cached.update(new_cache)
            if missing_txt:
                texts = [text_set[k] for k in missing_txt]
                new_txt_embs = enc.encode_text(texts)
                new_cache = {k: new_txt_embs[j] for j, k in enumerate(missing_txt)}
                emb_cache.set_batch(model_name, new_cache)
                cached.update(new_cache)

    img_embs_list = [
        np.stack([cached[k] for k in ks], axis=0)
        for ks in img_keys_per_inst
    ]
    orig_embs = [cached[k] for k in orig_keys]
    para_embs = [cached[k] for k in para_keys]

    return img_embs_list, orig_embs, para_embs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_strategy(
    strategy: RankStrategy,
    instances: list[tuple[str, Instance]],
    img_embs: list[np.ndarray],
    orig_embs: list[np.ndarray],
    para_embs: list[np.ndarray],
    p_idiom: np.ndarray,
    out_dir: Path,
    templates_root: Path,
) -> None:
    code_to_name = {v: k for k, v in LANG_NAME_TO_CODE.items()}
    results_by_lang: dict[str, list[tuple[int, list[str]]]] = {}

    for idx, (lang, inst) in enumerate(instances):
        names = [img.name for img in inst.images]
        order = apply_strategy(
            img_embs[idx],
            orig_embs[idx],
            para_embs[idx],
            float(p_idiom[idx]),
            names,
            strategy,
        )
        results_by_lang.setdefault(lang, []).append((idx, order))

    strat_dir = out_dir / strategy.name
    strat_dir.mkdir(parents=True, exist_ok=True)

    for lang_code, ranked_list in results_by_lang.items():
        lang_name = code_to_name[lang_code]
        template_path = templates_root / f"submission_{lang_name}.tsv"
        df = pd.read_csv(template_path, sep="\t", dtype=str, keep_default_na=False)
        orders = [
            "[" + ", ".join(f"'{n}'" for n in order) + "]"
            for _, order in ranked_list
        ]
        df["expected_order"] = orders
        out_path = strat_dir / f"submission_{lang_code}.tsv"
        df.to_csv(out_path, sep="\t", index=False)

    log.info("Strategy %-28s → %s", f"'{strategy.name}'", strat_dir)


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

    # Reconstruct pipeline intermediates from cache
    log.info("Loading translations from cache …")
    translated = load_translated(instances, text_cache)

    log.info("Loading paraphrases from cache …")
    paraphrased = load_paraphrased(instances, translated, text_cache)

    log.info("Loading BGE-M3 embeddings …")
    bge_embs = load_bge_embeddings(translated, emb_cache, args.device)

    log.info("Loading SigLIP2 embeddings …")
    img_embs, orig_embs, para_embs = load_siglip2_embeddings(
        instances, translated, paraphrased, emb_cache, args.device
    )

    log.info("Running LR sense classifier …")
    clf = LRSenseClassifier.from_path(Path(args.classifier))
    p_idiom = clf.predict_proba_idiomatic(bge_embs)
    idiom_frac = (p_idiom > 0.5).mean()
    log.info(
        "P(idiomatic): mean=%.3f  std=%.3f  >0.5: %.1f%%",
        p_idiom.mean(), p_idiom.std(), 100 * idiom_frac,
    )

    # Select strategies
    if args.alpha_sweep:
        alphas = [round(a * 0.1, 1) for a in range(0, 11)]
        strategies = (
            make_alpha_sweep_strategies(alphas, force_p=1.0)
            + make_alpha_sweep_strategies(alphas, force_p=None)
        )
    elif args.strategies:
        name_map = {s.name: s for s in DEFAULT_STRATEGIES}
        strategies = [name_map[n] for n in args.strategies if n in name_map]
        unknown = [n for n in args.strategies if n not in name_map]
        if unknown:
            log.warning("Unknown strategy names (skipped): %s", unknown)
    else:
        strategies = DEFAULT_STRATEGIES

    log.info("Running %d strategies …", len(strategies))
    for strat in strategies:
        run_strategy(
            strat, instances, img_embs, orig_embs, para_embs, p_idiom,
            out_dir, templates_root,
        )

    print("\n" + "=" * 60)
    print("Strategies generated (each dir has 15 TSVs to zip+submit):")
    for strat in strategies:
        print(f"  {out_dir / strat.name}  —  {strat.description}")
    print("\nTo zip and submit a strategy:")
    print(f"  cd {out_dir}/<strategy_name> && zip -j ../submission_<strategy>.zip *.tsv")
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
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="Device for any model that hits a cache miss (default: cpu).",
    )
    parser.add_argument(
        "--strategies", nargs="+", metavar="NAME",
        help="Specific strategy names to run (default: all).",
    )
    parser.add_argument(
        "--alpha-sweep", action="store_true",
        help="Generate a 0.0-1.0 alpha sweep instead of default strategies.",
    )
    main(parser.parse_args())
