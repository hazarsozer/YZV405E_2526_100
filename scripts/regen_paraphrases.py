"""Regenerate paraphrases for all 2663 instances.

--k 1 (default): Phi-3.5-mini-instruct greedy → namespace 'phi35_paraphrase'.
--k > 1:         Phi-3.5-mini-instruct sampling → namespace 'phi35_para_k{k}'.
                 Cache value is a JSON-encoded list[str] of k variants.
                 Cache key is sha256(translated, compound, str(k)).

Usage:
    uv run python -m scripts.regen_paraphrases                            # Phi35 k=1
    uv run python -m scripts.regen_paraphrases --k 4                      # Phi35 k=4
    uv run python -m scripts.regen_paraphrases --k 4 --checkpoint-every 25
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.data.loader import AdMIReRepository, LANG_NAME_TO_CODE
from src.models.translator import NO_OP_LANGS
from src.utils.cache import TextCache
from src.utils.io import sha256_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_NS_SINGLE = "phi35_paraphrase"
DEFAULT_NS_K = "phi35_para_k{k}"
SRC_NS = "nllb200_translation"


def _load_translations(
    instances: list[tuple[str, object]],
    text_cache: TextCache,
    device: str,
) -> list[str]:
    """Load translated sentences from cache, running NLLB-200 on misses."""
    translated: list[str] = [""] * len(instances)
    miss_translate: list[tuple[int, str, str, str]] = []

    for i, (lang, inst) in enumerate(instances):
        if lang in NO_OP_LANGS:
            translated[i] = inst.sentence
        else:
            key = sha256_string(lang, inst.sentence)
            result = text_cache.get_batch(SRC_NS, [key])
            if key in result:
                translated[i] = result[key]
            else:
                miss_translate.append((i, lang, inst.sentence, key))

    if miss_translate:
        log.info("Translation cache miss for %d sentences — running NLLB-200 …", len(miss_translate))
        from src.models.translator import NLLB200Translator

        by_lang: dict[str, list[tuple[int, str, str]]] = {}
        for i, lang, sentence, key in miss_translate:
            by_lang.setdefault(lang, []).append((i, sentence, key))

        with NLLB200Translator(device=device) as translator:
            new_trans: dict[str, str] = {}
            for lang, items in by_lang.items():
                texts = [item[1] for item in items]
                results = translator.translate(texts, lang)
                for (i, _, key), result in zip(items, results):
                    translated[i] = result
                    new_trans[key] = result
            text_cache.set_batch(SRC_NS, new_trans)
        log.info("NLLB-200 done for %d sentences.", len(miss_translate))

    return translated


def _run_single(
    instances: list[tuple[str, object]],
    translated: list[str],
    text_cache: TextCache,
    cache_ns: str,
    device: str,
    checkpoint_every: int,
) -> None:
    """Generate one Phi-3.5-mini-instruct greedy paraphrase per instance."""
    from src.models.slm_paraphraser import Phi35Paraphraser

    keys = [
        sha256_string(translated[i], inst.compound)
        for i, (_, inst) in enumerate(instances)
    ]
    existing = text_cache.get_batch(cache_ns, keys)
    miss_idx = [i for i, k in enumerate(keys) if k not in existing]

    if not miss_idx:
        log.info("All %d paraphrases already cached under '%s'.", len(instances), cache_ns)
        return

    log.info("%d / %d need generation.", len(miss_idx), len(instances))
    total = len(miss_idx)

    with Phi35Paraphraser(device=device) as para:
        new_cache: dict[str, str] = {}
        for done, mi in enumerate(miss_idx, 1):
            _, inst = instances[mi]
            result = para.paraphrase(translated[mi], inst.compound)
            new_cache[keys[mi]] = result
            if done % checkpoint_every == 0 or done == total:
                text_cache.set_batch(cache_ns, new_cache)
                new_cache = {}
                log.info("  %d / %d done", done, total)

    log.info("Done. %d paraphrases → namespace '%s'.", total, cache_ns)


def _run_k(
    instances: list[tuple[str, object]],
    translated: list[str],
    text_cache: TextCache,
    cache_ns: str,
    k: int,
    device: str,
    checkpoint_every: int,
) -> None:
    """Generate k Phi-3.5 sampled paraphrases per instance, stored as JSON list."""
    from src.models.slm_paraphraser import Phi35Paraphraser

    keys = [
        sha256_string(translated[i], inst.compound, str(k))
        for i, (_, inst) in enumerate(instances)
    ]
    existing = text_cache.get_batch(cache_ns, keys)
    miss_idx = [i for i, key in enumerate(keys) if key not in existing]

    if not miss_idx:
        log.info(
            "All %d k-paraphrases (k=%d) already cached under '%s'.",
            len(instances), k, cache_ns,
        )
        return

    log.info("%d / %d instances need k=%d paraphrases.", len(miss_idx), len(instances), k)
    total = len(miss_idx)

    with Phi35Paraphraser(device=device) as para:
        new_cache: dict[str, str] = {}
        for done, mi in enumerate(miss_idx, 1):
            _, inst = instances[mi]
            variants = para.paraphrase_k(translated[mi], inst.compound, k=k)
            new_cache[keys[mi]] = json.dumps(variants)
            if done % checkpoint_every == 0 or done == total:
                text_cache.set_batch(cache_ns, new_cache)
                new_cache = {}
                log.info("  %d / %d done", done, total)

    log.info("Done. %d k-paraphrase lists (k=%d) → namespace '%s'.", total, k, cache_ns)


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    templates_root = Path(args.templates_root)
    cache_dir = Path(args.cache_dir)
    k: int = args.k

    cache_ns = args.namespace or (
        DEFAULT_NS_SINGLE if k == 1 else DEFAULT_NS_K.format(k=k)
    )

    repo = AdMIReRepository(data_root, templates_root)
    all_datasets = repo.load_all_templates()
    instances = [
        (lang, inst)
        for lang, ds in all_datasets.items()
        for inst in ds.instances
    ]
    log.info("Loaded %d instances across %d languages.", len(instances), len(all_datasets))

    text_cache = TextCache(cache_dir / "text")

    log.info("Loading translations from cache …")
    translated = _load_translations(instances, text_cache, args.device)

    log.info("Generating paraphrases (k=%d) → namespace '%s' …", k, cache_ns)
    if k == 1:
        _run_single(instances, translated, text_cache, cache_ns, args.device, args.checkpoint_every)
    else:
        _run_k(instances, translated, text_cache, cache_ns, k, args.device, args.checkpoint_every)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/raw/admire2_data")
    parser.add_argument("--templates-root", default="data/submissions/templates")
    parser.add_argument("--cache-dir", default="data/processed")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--k", type=int, default=1,
        help="Number of paraphrase variants per instance (1=Qwen greedy, >1=Phi35 sampling).",
    )
    parser.add_argument(
        "--namespace", default=None,
        help="TextCache namespace (default: 'phi35_paraphrase' for k=1, 'phi35_para_k{k}' for k>1).",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Flush cache to disk every N items (default: 50).")
    main(parser.parse_args())
