"""Run the full AdMIRe 2.0 inference pipeline and write submission TSVs.

Usage:
    python scripts/run_inference.py \\
        --data-root data/raw/admire2_data \\
        --templates-root data/submissions/templates \\
        --classifier models/lr_sense_classifier.joblib \\
        [--output-dir data/submissions/output] \\
        [--cache-dir data/processed] \\
        [--device cuda] \\
        [--languages TR EL ZH] \\
        [--skip-translation] \\
        [--skip-paraphrase]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.pipeline import AdMIRePipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        data_root=Path(args.data_root),
        templates_root=Path(args.templates_root),
        classifier_path=Path(args.classifier),
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        device=args.device,
        languages=args.languages or None,
        skip_translation=args.skip_translation,
        skip_paraphrase=args.skip_paraphrase,
    )

    pipeline = AdMIRePipeline(config)
    results = pipeline.run()
    paths = pipeline.write_submissions(results)

    log.info("Done — %d submission files written.", len(paths))
    for p in paths:
        log.info("  %s", p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the AdMIRe 2.0 inference pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory of the AdMIRe 2 image data.",
    )
    parser.add_argument(
        "--templates-root",
        required=True,
        help="Directory containing submission_*.tsv template files.",
    )
    parser.add_argument(
        "--classifier",
        required=True,
        help="Path to trained LR classifier (.joblib).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/submissions/output",
        help="Directory to write submission TSVs.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/processed",
        help="Directory for embedding and text caches.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Torch device.",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Language codes to process (e.g. TR EL ZH). Default: all 15.",
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip NLLB-200 translation (use original sentences).",
    )
    parser.add_argument(
        "--skip-paraphrase",
        action="store_true",
        help="Skip Phi-3.5 paraphrasing (use translated sentences as-is).",
    )
    main(parser.parse_args())
