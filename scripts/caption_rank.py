"""Caption-based image ranking using BGE-M3 text similarity.

Instead of relying on SigLIP2 image-text alignment, this approach:
1. Uses the English image captions already provided in the dataset.
2. Encodes captions + sentences with BGE-M3 (strong multilingual semantic model).
3. Ranks images by caption-to-sentence cosine similarity.

Strategies
----------
* bge_para_caption   — BGE-M3 sim(paraphrase, caption) — always uses paraphrase
* bge_orig_caption   — BGE-M3 sim(original_translated, caption)
* bge_clf_caption    — classifier-gated: paraphrase for idiomatic, original for literal
* siglip_bge_blend   — 50/50 SigLIP2 best-strategy + BGE caption
* siglip_bge_clf     — classifier-gated ensemble (SigLIP2 + BGE)

Usage:
    uv run python -m scripts.caption_rank
    uv run python -m scripts.caption_rank --strategies bge_clf_caption siglip_bge_clf
    uv run python -m scripts.caption_rank --siglip-weight 0.4  # tune blend weight
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
from src.models.ranker import cosine_similarity
from src.models.sense_classifier import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    LRSenseClassifier,
)
from src.models.translator import NO_OP_LANGS
from src.utils.cache import EmbeddingCache, TextCache
from src.utils.io import sha256_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caption embedding helpers
# ---------------------------------------------------------------------------


def get_caption_embeddings(
    instances: list[tuple[str, Instance]],
    emb_cache: EmbeddingCache,
    device: str,
) -> list[np.ndarray]:
    """Return BGE-M3 embeddings for all 5 captions per instance.

    Returns a list of (5, 1024) arrays, one per instance.
    Cache key = sha256("caption", caption_text).
    """
    model_name = BGEM3Encoder.MODEL_ID

    # Build unique caption set
    cap_key_map: dict[str, str] = {}  # key → caption text
    cap_keys_per_inst: list[list[str]] = []
    for _, inst in instances:
        ks: list[str] = []
        for img in inst.images:
            k = sha256_string("caption", img.caption)
            ks.append(k)
            cap_key_map[k] = img.caption
        cap_keys_per_inst.append(ks)

    all_keys = list(cap_key_map.keys())
    cached = emb_cache.get_batch(model_name, all_keys)
    miss_keys = [k for k in all_keys if k not in cached]

    if miss_keys:
        log.info(
            "Embedding %d unique captions with BGE-M3 (cache miss) …",
            len(miss_keys),
        )
        with BGEM3Encoder(device=device) as enc:
            miss_texts = [cap_key_map[k] for k in miss_keys]
            new_embs = enc.encode(miss_texts)
            new_cache = {k: new_embs[j] for j, k in enumerate(miss_keys)}
            emb_cache.set_batch(model_name, new_cache)
            cached.update(new_cache)
    else:
        log.info("All %d unique caption embeddings found in cache.", len(all_keys))

    return [
        np.stack([cached[k] for k in ks], axis=0)
        for ks in cap_keys_per_inst
    ]


def get_sentence_bge_embeddings(
    sentences: list[str],
    emb_cache: EmbeddingCache,
    device: str,
) -> list[np.ndarray]:
    """Return BGE-M3 embeddings for a list of sentences (1024-d each)."""
    model_name = BGEM3Encoder.MODEL_ID
    keys = [sha256_string(s) for s in sentences]
    cached = emb_cache.get_batch(model_name, keys)
    miss_idx = [i for i, k in enumerate(keys) if k not in cached]

    if miss_idx:
        log.info(
            "BGE-M3 cache miss for %d sentences — loading model …", len(miss_idx)
        )
        with BGEM3Encoder(device=device) as enc:
            miss_texts = [sentences[i] for i in miss_idx]
            new_embs = enc.encode(miss_texts)
            new_cache = {keys[i]: new_embs[j] for j, i in enumerate(miss_idx)}
            emb_cache.set_batch(model_name, new_cache)
            cached.update(new_cache)
    else:
        log.info("All %d BGE-M3 sentence embeddings from cache.", len(sentences))

    return [cached[k] for k in keys]


# ---------------------------------------------------------------------------
# Cached pipeline reconstruction
# ---------------------------------------------------------------------------


def load_translated(
    instances: list[tuple[str, Instance]],
    text_cache: TextCache,
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
            raise RuntimeError(
                f"{len(missing)} translation cache misses — run full pipeline first."
            )
        for j, i in enumerate(real_idx):
            translated[i] = cached[real_keys[j]]
    return translated


def load_paraphrased(
    instances: list[tuple[str, Instance]],
    translated: list[str],
    text_cache: TextCache,
    paraphrase_ns: str = "phi35_paraphrase",
) -> list[str]:
    keys = [
        sha256_string(translated[i], inst.compound)
        for i, (_, inst) in enumerate(instances)
    ]
    cached = text_cache.get_batch(paraphrase_ns, keys)
    missing = [i for i, k in enumerate(keys) if k not in cached]
    if missing:
        raise RuntimeError(
            f"{len(missing)} paraphrase cache misses in namespace '{paraphrase_ns}' "
            "— run regen_paraphrases.py first."
        )
    return [cached[k] for k in keys]


def load_siglip2_embeddings(
    instances: list[tuple[str, Instance]],
    translated: list[str],
    paraphrased: list[str],
    emb_cache: EmbeddingCache,
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    model_name = SigLIP2Encoder.MODEL_ID
    img_keys: dict[str, Path] = {}
    img_keys_per_inst: list[list[str]] = []
    for _, inst in instances:
        ks = [sha256_string(str(img.absolute_path)) for img in inst.images]
        img_keys_per_inst.append(ks)
        for k, img in zip(ks, inst.images):
            img_keys[k] = img.absolute_path

    text_set: dict[str, str] = {}
    orig_keys, para_keys = [], []
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
                new_embs = enc.encode(pil_imgs)
                new_cache = {k: new_embs[j] for j, k in enumerate(missing_img)}
                emb_cache.set_batch(model_name, new_cache)
                cached.update(new_cache)
            if missing_txt:
                texts = [text_set[k] for k in missing_txt]
                new_embs = enc.encode_text(texts)
                new_cache = {k: new_embs[j] for j, k in enumerate(missing_txt)}
                emb_cache.set_batch(model_name, new_cache)
                cached.update(new_cache)

    img_embs = [np.stack([cached[k] for k in ks], axis=0) for ks in img_keys_per_inst]
    orig_embs = [cached[k] for k in orig_keys]
    para_embs = [cached[k] for k in para_keys]
    return img_embs, orig_embs, para_embs


# ---------------------------------------------------------------------------
# Ranking functions
# ---------------------------------------------------------------------------


def rank_bge_caption(
    cap_embs: np.ndarray,
    sent_emb: np.ndarray,
    image_names: list[str],
) -> list[str]:
    """Rank images by BGE-M3 cosine_sim(sentence, caption)."""
    scores = cosine_similarity(sent_emb, cap_embs)
    return [image_names[int(i)] for i in np.argsort(-scores)]


def rank_siglip_bge_blend(
    cap_embs: np.ndarray,
    bge_sent_emb: np.ndarray,
    siglip_img_embs: np.ndarray,
    siglip_orig_emb: np.ndarray,
    siglip_para_emb: np.ndarray,
    image_names: list[str],
    p_idiomatic: float,
    siglip_weight: float,
    use_classifier: bool,
) -> list[str]:
    """Ensemble: SigLIP2 best-strategy + BGE caption similarity."""
    bge_scores = cosine_similarity(bge_sent_emb, cap_embs)

    if use_classifier:
        # SigLIP2 side: classifier-gated (orig for literal, para for idiomatic)
        if p_idiomatic > HIGH_CONFIDENCE_THRESHOLD:
            sig_scores = cosine_similarity(siglip_para_emb, siglip_img_embs)
        elif p_idiomatic < LOW_CONFIDENCE_THRESHOLD:
            sig_scores = cosine_similarity(siglip_orig_emb, siglip_img_embs)
        else:
            sig_scores = cosine_similarity(siglip_orig_emb, siglip_img_embs)
    else:
        # SigLIP2 side: plain orig (best baseline)
        sig_scores = cosine_similarity(siglip_orig_emb, siglip_img_embs)

    # Normalise both score vectors to [0,1] before blending
    def _norm(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        span = hi - lo
        if span < 1e-8:
            return np.full_like(v, 0.5)
        return (v - lo) / span

    scores = siglip_weight * _norm(sig_scores) + (1 - siglip_weight) * _norm(bge_scores)
    return [image_names[int(i)] for i in np.argsort(-scores)]


# ---------------------------------------------------------------------------
# Write submission TSVs
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
        out_path = strat_dir / f"submission_{lang_code}.tsv"
        df.to_csv(out_path, sep="\t", index=False)

    log.info("Strategy %-28s → %s", f"'{strategy_name}'", strat_dir)


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

    log.info("Loading BGE-M3 sentence embeddings …")
    bge_translated = get_sentence_bge_embeddings(translated, emb_cache, args.device)
    bge_paraphrased = get_sentence_bge_embeddings(paraphrased, emb_cache, args.device)

    log.info("Loading BGE-M3 caption embeddings …")
    cap_embs = get_caption_embeddings(instances, emb_cache, args.device)

    log.info("Loading SigLIP2 embeddings …")
    sig_img_embs, sig_orig_embs, sig_para_embs = load_siglip2_embeddings(
        instances, translated, paraphrased, emb_cache, args.device
    )

    log.info("Running LR classifier …")
    clf = LRSenseClassifier.from_path(Path(args.classifier))
    p_idiom = clf.predict_proba_idiomatic(
        np.stack(bge_translated, axis=0)
    )

    selected = set(args.strategies) if args.strategies else {
        "bge_para_caption",
        "bge_orig_caption",
        "bge_clf_caption",
        "siglip_bge_blend",
        "siglip_bge_clf",
    }

    # Threshold sweep: add one bge_clf_t{N} strategy per threshold value
    sweep_thresholds: list[float] = args.threshold_sweep or []
    sweep_strategies = {f"bge_clf_t{int(t*100):03d}": t for t in sweep_thresholds}

    w = args.siglip_weight  # SigLIP2 weight in blend

    for strat_name in sorted(selected | set(sweep_strategies)):
        rankings: list[list[str]] = []
        for i, (_, inst) in enumerate(instances):
            names = [img.name for img in inst.images]
            p = float(p_idiom[i])

            if strat_name == "bge_para_caption":
                order = rank_bge_caption(cap_embs[i], bge_paraphrased[i], names)

            elif strat_name == "bge_orig_caption":
                order = rank_bge_caption(cap_embs[i], bge_translated[i], names)

            elif strat_name == "bge_clf_caption":
                sent_emb = (
                    bge_paraphrased[i]
                    if p > HIGH_CONFIDENCE_THRESHOLD
                    else bge_translated[i]
                )
                order = rank_bge_caption(cap_embs[i], sent_emb, names)

            elif strat_name in sweep_strategies:
                threshold = sweep_strategies[strat_name]
                sent_emb = bge_paraphrased[i] if p > threshold else bge_translated[i]
                order = rank_bge_caption(cap_embs[i], sent_emb, names)

            elif strat_name == "bge_soft_blend":
                # Soft blend: p × sim(paraphrase, caption) + (1−p) × sim(original, caption)
                para_scores = cosine_similarity(bge_paraphrased[i], cap_embs[i])
                orig_scores = cosine_similarity(bge_translated[i], cap_embs[i])
                blended = p * para_scores + (1 - p) * orig_scores
                order = [names[int(j)] for j in np.argsort(-blended)]

            elif strat_name == "siglip_bge_blend":
                order = rank_siglip_bge_blend(
                    cap_embs[i], bge_paraphrased[i],
                    sig_img_embs[i], sig_orig_embs[i], sig_para_embs[i],
                    names, p,
                    siglip_weight=w, use_classifier=False,
                )

            elif strat_name == "siglip_bge_clf":
                order = rank_siglip_bge_blend(
                    cap_embs[i], bge_paraphrased[i],
                    sig_img_embs[i], sig_orig_embs[i], sig_para_embs[i],
                    names, p,
                    siglip_weight=w, use_classifier=True,
                )

            else:
                log.warning("Unknown strategy: %s — skipped.", strat_name)
                break

            rankings.append(order)
        else:
            write_submissions(strat_name, instances, rankings, out_dir, templates_root)

    print("\n" + "=" * 60)
    print("Caption-based strategies generated:")
    for s in sorted(selected):
        print(f"  {out_dir / s}")
    print(f"\nBlend weight: SigLIP2={w:.2f}, BGE_caption={1-w:.2f}")
    print("\nTo zip: cd data/submissions/ablations/<name> && zip -j ../<name>.zip *.tsv")
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
        help="Device for model if cache miss (default: cpu, since captions need BGE-M3).",
    )
    parser.add_argument(
        "--strategies", nargs="+", metavar="NAME",
        help="Strategy names to run (default: all 5).",
    )
    parser.add_argument(
        "--siglip-weight", type=float, default=0.3,
        help="Weight for SigLIP2 scores in the ensemble blend (default: 0.3).",
    )
    parser.add_argument(
        "--paraphrase-ns", default="phi35_paraphrase",
        help="Text cache namespace to load paraphrases from (default: phi35_paraphrase).",
    )
    parser.add_argument(
        "--threshold-sweep", nargs="+", type=float, metavar="T",
        help="Run bge_clf_caption at each threshold value, e.g. --threshold-sweep 0.5 0.55 0.6 0.65 0.7 0.8",
    )
    main(parser.parse_args())
