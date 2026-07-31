"""Generate compound-aware enriched captions for all images using Phi-3.5.

For each unique (caption, compound) pair, appends one sentence describing the
abstract/symbolic concept the image could represent for the idiom.  The result
is stored in the TextCache namespace 'phi35_caption_enrich'.

The original 'caption' embeddings in EmbeddingCache are untouched so this is
purely additive.  Run before blend_rank.py --use-enriched-captions.

Usage:
    uv run python -m scripts.enrich_captions
    uv run python -m scripts.enrich_captions --device cuda --checkpoint-every 50
    uv run python -m scripts.enrich_captions --dry-run   # print counts only
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.data.loader import AdMIReRepository
from src.models.slm_paraphraser import Phi35Paraphraser
from src.utils.cache import TextCache
from src.utils.io import sha256_string

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_NS = "phi35_caption_enrich"


def _collect_unique_pairs(
    instances_by_lang: dict,
) -> list[tuple[str, str]]:
    """Return sorted unique (caption, compound) pairs across all instances."""
    seen: set[tuple[str, str]] = set()
    for lang_datasets in instances_by_lang.values():
        for inst in lang_datasets.instances:
            for img in inst.images:
                pair = (img.caption, inst.compound)
                seen.add(pair)
    return sorted(seen)


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    templates_root = Path(args.templates_root)
    cache_dir = Path(args.cache_dir)

    repo = AdMIReRepository(data_root, templates_root)
    all_datasets = repo.load_all_templates()
    log.info("Loaded templates for %d languages.", len(all_datasets))

    pairs = _collect_unique_pairs(all_datasets)
    log.info("Unique (caption, compound) pairs: %d", len(pairs))

    if args.dry_run:
        log.info("Dry-run: exiting without generation.")
        return

    text_cache = TextCache(cache_dir / "text")

    keys = [sha256_string(caption, compound) for caption, compound in pairs]
    existing = text_cache.get_batch(CACHE_NS, keys)
    miss_idx = [i for i, k in enumerate(keys) if k not in existing]

    if not miss_idx:
        log.info(
            "All %d enriched captions already cached under '%s'. Nothing to do.",
            len(pairs),
            CACHE_NS,
        )
        return

    log.info(
        "%d / %d pairs need enrichment (already cached: %d).",
        len(miss_idx),
        len(pairs),
        len(pairs) - len(miss_idx),
    )

    checkpoint_every = args.checkpoint_every
    total = len(miss_idx)

    with Phi35Paraphraser(device=args.device) as para:
        new_cache: dict[str, str] = {}

        for done, mi in enumerate(miss_idx, 1):
            caption, compound = pairs[mi]
            enriched = para.enrich_caption(caption, compound)
            new_cache[keys[mi]] = enriched

            if done % checkpoint_every == 0 or done == total:
                text_cache.set_batch(CACHE_NS, new_cache)
                new_cache = {}
                log.info("  %d / %d enriched", done, total)

    log.info(
        "Done. %d enriched captions written to namespace '%s'.",
        total,
        CACHE_NS,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", default="data/raw/admire2_data")
    parser.add_argument("--templates-root", default="data/submissions/templates")
    parser.add_argument("--cache-dir", default="data/processed")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Flush cache to disk every N enrichments (default: 50).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts and exit without running the model.",
    )
    main(parser.parse_args())
