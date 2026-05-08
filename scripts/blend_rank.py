"""3-way channel blend ranker using cached embeddings.

Combines three independent ranking signals via per-instance min-max
normalisation and a weighted sum:

  C1  = BGE-M3 cos(query, original_caption)
  C2  = BGE-M3 cos(query, enriched_caption)   [--use-enriched-captions]
  C3  = SigLIP2 cos(siglip_text(query), siglip_image)

  score(image_i) = w1*norm(C1[i]) + w2*norm(C2[i]) + w3*norm(C3[i])

The 'query' embedding is classifier-gated:
  P(idiomatic) > HIGH threshold  → use paraphrase embedding
  P(idiomatic) < LOW  threshold  → use translated sentence embedding
  otherwise                      → use translated sentence embedding

Run enrich_captions.py before using --use-enriched-captions.

Usage:
    # Reproduce best existing result (bge_clf_caption equivalent, C1-only):
    uv run python -m scripts.blend_rank --weights 1.0 0.0 0.0

    # Enriched-only:
    uv run python -m scripts.blend_rank --weights 0.0 1.0 0.0 --use-enriched-captions

    # 3-way blend (equal):
    uv run python -m scripts.blend_rank --weights 0.33 0.33 0.34 --use-enriched-captions

    # Grid search over weights:
    uv run python -m scripts.blend_rank --weight-grid --use-enriched-captions
"""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import json

import numpy as np
import pandas as pd

from scripts.caption_rank import (
    get_caption_embeddings,
    get_sentence_bge_embeddings,
    load_paraphrased,
    load_siglip2_embeddings,
    load_translated,
    write_submissions,
)
from src.data.loader import AdMIReRepository, Instance
from src.models.frozen_encoders import BGEM3Encoder
from src.models.ranker import cosine_similarity
from src.models.sense_classifier import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    LRSenseClassifier,
)
from src.utils.cache import EmbeddingCache, TextCache
from src.utils.io import sha256_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ENRICH_NS = "phi35_caption_enrich"


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Normalise a score vector to [0, 1] per instance."""
    lo, hi = scores.min(), scores.max()
    span = hi - lo
    if span < 1e-8:
        return np.full_like(scores, 0.5)
    return (scores - lo) / span


def get_enriched_caption_embeddings(
    instances: list[tuple[str, Instance]],
    emb_cache: EmbeddingCache,
    text_cache: TextCache,
    device: str,
) -> list[np.ndarray]:
    """Return BGE-M3 embeddings for enriched captions (5 per instance).

    Reads enriched text from TextCache ns 'phi35_caption_enrich', then
    encodes with BGE-M3.  Missing enrichments fall back silently to the
    original caption text.
    """
    model_name = BGEM3Encoder.MODEL_ID

    enrich_key_map: dict[str, str] = {}
    enrich_keys_per_inst: list[list[str]] = []

    for _, inst in instances:
        ks: list[str] = []
        for img in inst.images:
            raw_key = sha256_string(img.caption, inst.compound)
            emb_key = sha256_string("caption_enriched", raw_key)
            ks.append(emb_key)
            enrich_key_map[emb_key] = (img.caption, inst.compound, raw_key)
        enrich_keys_per_inst.append(ks)

    all_emb_keys = list(enrich_key_map.keys())
    cached_embs = emb_cache.get_batch(model_name, all_emb_keys)
    miss_emb_keys = [k for k in all_emb_keys if k not in cached_embs]

    if miss_emb_keys:
        # Load enriched text from TextCache
        raw_keys_needed = [enrich_key_map[k][2] for k in miss_emb_keys]
        cached_text = text_cache.get_batch(ENRICH_NS, raw_keys_needed)

        texts_to_encode: list[str] = []
        for emb_key, raw_key in zip(miss_emb_keys, raw_keys_needed):
            caption, _compound, _ = enrich_key_map[emb_key]
            texts_to_encode.append(cached_text.get(raw_key, caption))

        log.info(
            "Encoding %d enriched captions with BGE-M3 (%d cache miss) …",
            len(texts_to_encode),
            sum(1 for rk in raw_keys_needed if rk not in cached_text),
        )
        with BGEM3Encoder(device=device) as enc:
            new_embs = enc.encode(texts_to_encode)
            new_cache = {k: new_embs[j] for j, k in enumerate(miss_emb_keys)}
            emb_cache.set_batch(model_name, new_cache)
            cached_embs.update(new_cache)
    else:
        log.info("All %d enriched caption embeddings from cache.", len(all_emb_keys))

    return [
        np.stack([cached_embs[k] for k in ks], axis=0)
        for ks in enrich_keys_per_inst
    ]


def get_mean_para_embeddings(
    instances: list[tuple[str, Instance]],
    translated: list[str],
    emb_cache: EmbeddingCache,
    text_cache: TextCache,
    device: str,
    para_ns: str,
    k: int,
) -> list[np.ndarray]:
    """Return L2-normalised mean BGE-M3 embedding across k paraphrase variants per instance.

    Reads JSON-encoded list[str] from TextCache (key=sha256(translated, compound, str(k))),
    encodes each unique variant with BGE-M3, then returns the L2-normalised mean for
    each instance.  Falls back to the translated sentence embedding for any cache miss.
    """
    model_name = BGEM3Encoder.MODEL_ID

    # Build per-instance keys and load JSON lists
    keys = [
        sha256_string(translated[i], inst.compound, str(k))
        for i, (_, inst) in enumerate(instances)
    ]
    cached_lists = text_cache.get_batch(para_ns, keys)

    variants_per_inst: list[list[str]] = []
    for i, (_, inst) in enumerate(instances):
        raw = cached_lists.get(keys[i], "")
        if raw:
            try:
                variants = json.loads(raw)
                if isinstance(variants, list) and variants:
                    variants_per_inst.append(variants)
                    continue
            except json.JSONDecodeError:
                pass
        # Fallback: single translated sentence
        variants_per_inst.append([translated[i]])

    n_miss = sum(1 for k_raw, raw in zip(keys, cached_lists.values()) if not raw)
    log.info(
        "Loaded k-para lists (%d ns, %d instances, %d cache misses).",
        k, len(instances), sum(1 for key in keys if key not in cached_lists),
    )

    # Collect all unique strings to encode
    all_texts: list[str] = []
    text_to_idx: dict[str, int] = {}
    for variants in variants_per_inst:
        for v in variants:
            if v not in text_to_idx:
                text_to_idx[v] = len(all_texts)
                all_texts.append(v)

    # Encode with BGE-M3 (cache-aware)
    emb_keys = [sha256_string(t) for t in all_texts]
    cached_embs = emb_cache.get_batch(model_name, emb_keys)
    miss_emb_idx = [i for i, ek in enumerate(emb_keys) if ek not in cached_embs]

    if miss_emb_idx:
        log.info("BGE-M3 cache miss for %d k-para strings — encoding …", len(miss_emb_idx))
        with BGEM3Encoder(device=device) as enc:
            miss_texts = [all_texts[i] for i in miss_emb_idx]
            new_embs = enc.encode(miss_texts)
            new_cache = {emb_keys[i]: new_embs[j] for j, i in enumerate(miss_emb_idx)}
            emb_cache.set_batch(model_name, new_cache)
            cached_embs.update(new_cache)

    # Compute L2-normalised mean per instance
    mean_embs: list[np.ndarray] = []
    for variants in variants_per_inst:
        emb_stack = np.stack(
            [cached_embs[emb_keys[text_to_idx[v]]] for v in variants], axis=0
        )
        mean_vec = emb_stack.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        mean_embs.append(mean_vec / norm if norm > 1e-8 else mean_vec)

    return mean_embs


def _rank_blend(
    bge_query: np.ndarray,
    cap_c1: np.ndarray,
    cap_c2: np.ndarray,
    sig_img: np.ndarray,
    sig_text: np.ndarray,
    image_names: list[str],
    w1: float,
    w2: float,
    w3: float,
) -> list[str]:
    scores = (
        w1 * _minmax(cosine_similarity(bge_query, cap_c1))
        + w2 * _minmax(cosine_similarity(bge_query, cap_c2))
        + w3 * _minmax(cosine_similarity(sig_text, sig_img))
    )
    return [image_names[int(i)] for i in np.argsort(-scores)]


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

    log.info("Loading BGE-M3 translated sentence embeddings …")
    bge_translated = get_sentence_bge_embeddings(translated, emb_cache, args.device)

    use_mean = args.use_mean_emb
    if use_mean:
        para_k = args.para_k
        para_ns = args.paraphrase_ns if args.paraphrase_ns != "phi35_paraphrase" else f"phi35_para_k{para_k}"
        log.info("Loading k=%d mean BGE-M3 paraphrase embeddings (ns: %s) …", para_k, para_ns)
        bge_paraphrased = get_mean_para_embeddings(
            instances, translated, emb_cache, text_cache, args.device, para_ns, para_k
        )
        # For SigLIP2 gating we still need single paraphrases — fall back to translated
        paraphrased = translated
    else:
        log.info("Loading paraphrases (ns: %s) …", args.paraphrase_ns)
        paraphrased = load_paraphrased(instances, translated, text_cache, args.paraphrase_ns)
        log.info("Loading BGE-M3 paraphrase embeddings …")
        bge_paraphrased = get_sentence_bge_embeddings(paraphrased, emb_cache, args.device)

    log.info("Loading BGE-M3 original caption embeddings (C1) …")
    cap_c1_embs = get_caption_embeddings(instances, emb_cache, args.device)

    use_enriched = args.use_enriched_captions
    if use_enriched:
        log.info("Loading BGE-M3 enriched caption embeddings (C2) …")
        cap_c2_embs = get_enriched_caption_embeddings(
            instances, emb_cache, text_cache, args.device
        )
    else:
        cap_c2_embs = cap_c1_embs  # C2 = C1 when not enriched

    log.info("Loading SigLIP2 embeddings (C3) …")
    sig_img_embs, sig_orig_embs, sig_para_embs = load_siglip2_embeddings(
        instances, translated, paraphrased, emb_cache, args.device
    )

    log.info("Loading LR classifier …")
    clf = LRSenseClassifier.from_path(Path(args.classifier))
    p_idiom = clf.predict_proba_idiomatic(np.stack(bge_translated, axis=0))

    # Build weight configs to run
    if args.weight_grid:
        steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        weight_configs = [
            (w1, w2, w3)
            for w1, w2, w3 in itertools.product(steps, repeat=3)
            if abs(w1 + w2 + w3 - 1.0) < 1e-6
        ]
        log.info("Running weight grid: %d configurations.", len(weight_configs))
    else:
        w1, w2, w3 = args.weights
        total = w1 + w2 + w3
        if abs(total - 1.0) > 1e-3:
            log.warning("Weights sum to %.3f, normalising to 1.", total)
            w1, w2, w3 = w1 / total, w2 / total, w3 / total
        weight_configs = [(w1, w2, w3)]

    for w1, w2, w3 in weight_configs:
        enr_tag = "_enr" if use_enriched else ""
        mean_tag = f"_mk{args.para_k}" if use_mean else ""
        strat_name = f"blend{enr_tag}{mean_tag}_c1{w1:.2f}_c2{w2:.2f}_c3{w3:.2f}"

        rankings: list[list[str]] = []
        for i, (_, inst) in enumerate(instances):
            names = [img.name for img in inst.images]
            p = float(p_idiom[i])

            bge_query = (
                bge_paraphrased[i]
                if p > HIGH_CONFIDENCE_THRESHOLD
                else bge_translated[i]
            )
            sig_text = (
                sig_para_embs[i]
                if p > HIGH_CONFIDENCE_THRESHOLD
                else sig_orig_embs[i]
            )

            order = _rank_blend(
                bge_query,
                cap_c1_embs[i],
                cap_c2_embs[i],
                sig_img_embs[i],
                sig_text,
                names,
                w1,
                w2,
                w3,
            )
            rankings.append(order)

        write_submissions(strat_name, instances, rankings, out_dir, templates_root)

    log.info("Done. %d strategy/ies written to %s.", len(weight_configs), out_dir)


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
        help="Device for model inference on cache miss (default: cpu).",
    )
    parser.add_argument(
        "--paraphrase-ns",
        default="phi35_paraphrase",
        help="Text cache namespace for paraphrases (default: phi35_paraphrase).",
    )
    parser.add_argument(
        "--use-enriched-captions",
        action="store_true",
        help="Load C2 from phi35_caption_enrich namespace (requires enrich_captions.py).",
    )
    parser.add_argument(
        "--use-mean-emb",
        action="store_true",
        help="Use L2-normalised mean of k paraphrase embeddings instead of single (requires regen_paraphrases.py --k).",
    )
    parser.add_argument(
        "--para-k",
        type=int,
        default=4,
        help="Number of paraphrase variants for mean embedding (default: 4, used with --use-mean-emb).",
    )
    parser.add_argument(
        "--weights",
        nargs=3,
        type=float,
        metavar=("W1", "W2", "W3"),
        default=[1.0, 0.0, 0.0],
        help="Weights for C1 (original caption), C2 (enriched), C3 (SigLIP2). "
             "Auto-normalised to sum=1 (default: 1.0 0.0 0.0).",
    )
    parser.add_argument(
        "--weight-grid",
        action="store_true",
        help="Ignore --weights and run full grid w ∈ {0,0.2,0.4,0.6,0.8,1.0}^3, sum=1.",
    )
    main(parser.parse_args())
