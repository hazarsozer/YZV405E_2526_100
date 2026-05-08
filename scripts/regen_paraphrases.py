"""Regenerate paraphrases for all 2663 instances using Qwen2.5-7B-Instruct.

Reads translated sentences from the existing text cache (requires a prior
full pipeline run), generates new paraphrases with Qwen, and stores them
under the namespace 'qwen25_paraphrase'.

The original 'phi35_paraphrase' cache entries are untouched so both
strategies can be compared via caption_rank.py --paraphrase-ns.

Usage:
    uv run python -m scripts.regen_paraphrases
    uv run python -m scripts.regen_paraphrases --device cuda --checkpoint-every 25
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.loader import AdMIReRepository, LANG_NAME_TO_CODE
from src.models.qwen_paraphraser import Qwen25Paraphraser
from src.models.translator import NO_OP_LANGS
from src.utils.cache import TextCache
from src.utils.io import sha256_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_NS = "qwen25_paraphrase"
SRC_NS = "nllb200_translation"


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    templates_root = Path(args.templates_root)
    cache_dir = Path(args.cache_dir)

    repo = AdMIReRepository(data_root, templates_root)
    all_datasets = repo.load_all_templates()
    instances = [
        (lang, inst)
        for lang, ds in all_datasets.items()
        for inst in ds.instances
    ]
    log.info("Loaded %d instances across %d languages.", len(instances), len(all_datasets))

    text_cache = TextCache(cache_dir / "text")

    # --- Load translations from cache (run NLLB for any misses) ---
    log.info("Loading translations from cache …")
    translated: list[str] = [""] * len(instances)
    miss_translate: list[tuple[int, str, str, str]] = []  # (idx, lang, sentence, key)

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

        with NLLB200Translator(device=args.device) as translator:
            new_trans: dict[str, str] = {}
            for lang, items in by_lang.items():
                texts = [item[1] for item in items]
                results = translator.translate(texts, lang)
                for (i, _, key), result in zip(items, results):
                    translated[i] = result
                    new_trans[key] = result
            text_cache.set_batch(SRC_NS, new_trans)
        log.info("NLLB-200 translation done for %d sentences.", len(miss_translate))

    # --- Find cache misses for qwen25_paraphrase ---
    keys = [
        sha256_string(translated[i], inst.compound)
        for i, (_, inst) in enumerate(instances)
    ]
    existing = text_cache.get_batch(CACHE_NS, keys)
    miss_idx = [i for i, k in enumerate(keys) if k not in existing]

    if not miss_idx:
        log.info("All %d paraphrases already cached under '%s'. Nothing to do.", len(instances), CACHE_NS)
        return

    log.info(
        "%d / %d paraphrases need generation (already cached: %d).",
        len(miss_idx), len(instances), len(instances) - len(miss_idx),
    )

    # --- Generate with Qwen ---
    checkpoint_every = args.checkpoint_every
    total = len(miss_idx)

    with Qwen25Paraphraser(device=args.device) as para:
        new_cache: dict[str, str] = {}

        for done, mi in enumerate(miss_idx, 1):
            _, inst = instances[mi]
            result = para.paraphrase(translated[mi], inst.compound)
            new_cache[keys[mi]] = result

            if done % checkpoint_every == 0 or done == total:
                text_cache.set_batch(CACHE_NS, new_cache)
                new_cache = {}
                log.info("  %d / %d paraphrased", done, total)

    log.info("Done. %d paraphrases written to cache namespace '%s'.", total, CACHE_NS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/raw/admire2_data")
    parser.add_argument("--templates-root", default="data/submissions/templates")
    parser.add_argument("--cache-dir", default="data/processed")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Flush cache to disk every N paraphrases (default: 50).")
    main(parser.parse_args())
