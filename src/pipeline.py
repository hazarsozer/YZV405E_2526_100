"""End-to-end AdMIRe 2.0 inference pipeline.

Orchestrates: translate → paraphrase → classify → encode → rank.

Models are loaded and unloaded sequentially so only one large model is
in VRAM at a time.  All intermediate results are cached to disk.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.loader import (
    AdMIReRepository,
    Dataset,
    Instance,
    LANG_NAME_TO_CODE,
)
from src.models.frozen_encoders import BGEM3Encoder, SigLIP2Encoder
from src.models.ranker import rank_images
from src.models.sense_classifier import LRSenseClassifier
from src.models.slm_paraphraser import IdentityParaphraser, Phi35Paraphraser
from src.models.translator import (
    NLLB200Translator,
    NO_OP_LANGS,
    NoOpTranslator,
)
from src.utils.cache import EmbeddingCache, TextCache
from src.utils.io import sha256_file, sha256_string

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclasses.dataclass
class PipelineConfig:
    """All knobs for a pipeline run."""

    data_root: Path
    templates_root: Path
    classifier_path: Path
    output_dir: Path
    cache_dir: Path = Path("data/processed")
    device: str = "cuda"
    languages: list[str] | None = None  # None = all 15
    skip_translation: bool = False
    skip_paraphrase: bool = False


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclasses.dataclass
class RankedInstance:
    """One instance with its predicted image ranking."""

    instance: Instance
    translated_sentence: str
    paraphrased_sentence: str
    p_idiomatic: float
    predicted_order: list[str]


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------


class AdMIRePipeline:
    """Run the full 3-step pipeline and write submission TSVs."""

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self._repo = AdMIReRepository(config.data_root, config.templates_root)
        self._emb_cache = EmbeddingCache(config.cache_dir / "embeddings")
        self._text_cache = TextCache(config.cache_dir / "text")

    # ==================================================================
    # Public entry point
    # ==================================================================

    def run(self) -> dict[str, list[RankedInstance]]:
        """Execute pipeline, return results keyed by language code."""
        datasets = self._load_datasets()
        all_instances = [
            (lang, inst)
            for lang, ds in datasets.items()
            for inst in ds.instances
        ]
        logger.info("Total instances: %d across %d languages",
                     len(all_instances), len(datasets))

        # --- Phase 1: Translate -----------------------------------------
        translated = self._translate_phase(all_instances)

        # --- Phase 2: Paraphrase ----------------------------------------
        paraphrased = self._paraphrase_phase(all_instances, translated)

        # --- Phase 3: Sense classification (BGE-M3 → LR) ---------------
        p_idiomatic = self._classify_phase(translated)

        # --- Phase 4: Encode images + text with SigLIP2 -----------------
        img_embs, txt_orig_embs, txt_para_embs = self._siglip2_phase(
            all_instances, translated, paraphrased
        )

        # --- Phase 5: Rank ----------------------------------------------
        results: dict[str, list[RankedInstance]] = {}
        for idx, (lang, inst) in enumerate(all_instances):
            names = [img.name for img in inst.images]
            order = rank_images(
                image_embeddings=img_embs[idx],
                original_text_emb=txt_orig_embs[idx],
                paraphrased_text_emb=txt_para_embs[idx],
                p_idiomatic=float(p_idiomatic[idx]),
                image_names=names,
            )
            results.setdefault(lang, []).append(
                RankedInstance(
                    instance=inst,
                    translated_sentence=translated[idx],
                    paraphrased_sentence=paraphrased[idx],
                    p_idiomatic=float(p_idiomatic[idx]),
                    predicted_order=order,
                )
            )

        return results

    def write_submissions(
        self, results: dict[str, list[RankedInstance]]
    ) -> list[Path]:
        """Write one submission TSV per language, return file paths."""
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        code_to_name = {v: k for k, v in LANG_NAME_TO_CODE.items()}

        for lang_code, ranked_list in results.items():
            lang_name = code_to_name[lang_code]
            template_path = (
                self.cfg.templates_root / f"submission_{lang_name}.tsv"
            )
            df = pd.read_csv(
                template_path, sep="\t", dtype=str, keep_default_na=False
            )

            orders: list[str] = []
            for ri in ranked_list:
                order_str = "[" + ", ".join(
                    f"'{n}'" for n in ri.predicted_order
                ) + "]"
                orders.append(order_str)

            df["expected_order"] = orders
            out_path = self.cfg.output_dir / f"submission_{lang_code}.tsv"
            df.to_csv(out_path, sep="\t", index=False)
            paths.append(out_path)
            logger.info("Wrote %s (%d rows)", out_path.name, len(df))

        return paths

    # ==================================================================
    # Internal phases
    # ==================================================================

    def _load_datasets(self) -> dict[str, Dataset]:
        if self.cfg.languages:
            code_to_name = {v: k for k, v in LANG_NAME_TO_CODE.items()}
            return {
                lang: self._repo.load_submission_template(code_to_name[lang])
                for lang in self.cfg.languages
            }
        return self._repo.load_all_templates()

    # --- Translation ---------------------------------------------------

    def _translate_phase(
        self, instances: list[tuple[str, Instance]]
    ) -> list[str]:
        """Translate all sentences → English.  Returns list parallel to *instances*."""
        logger.info("=== Phase 1: Translation ===")
        translated: list[str] = [""] * len(instances)

        # Separate no-op vs real translation
        real_indices: list[int] = []
        real_texts: list[str] = []
        real_langs: list[str] = []

        for idx, (lang, inst) in enumerate(instances):
            if lang in NO_OP_LANGS or self.cfg.skip_translation:
                translated[idx] = inst.sentence
            else:
                real_indices.append(idx)
                real_texts.append(inst.sentence)
                real_langs.append(lang)

        if not real_indices:
            logger.info("No sentences need translation.")
            return translated

        # Check text cache
        cache_ns = "nllb200_translation"
        keys = [sha256_string(lang, txt) for lang, txt in zip(real_langs, real_texts)]
        cached = self._text_cache.get_batch(cache_ns, keys)

        miss_idx = [i for i, k in enumerate(keys) if k not in cached]
        if miss_idx:
            logger.info("Translating %d sentences with NLLB-200 …", len(miss_idx))
            with NLLB200Translator(device=self.cfg.device) as translator:
                # Group by source language for efficient batching
                by_lang: dict[str, list[int]] = {}
                for mi in miss_idx:
                    by_lang.setdefault(real_langs[mi], []).append(mi)

                new_cache: dict[str, str] = {}
                for lang, indices in by_lang.items():
                    texts = [real_texts[i] for i in indices]
                    results = translator.translate(texts, lang)
                    for i, result in zip(indices, results):
                        cached[keys[i]] = result
                        new_cache[keys[i]] = result

                self._text_cache.set_batch(cache_ns, new_cache)
        else:
            logger.info("All translations found in cache.")

        for i, ri in enumerate(real_indices):
            translated[ri] = cached[keys[i]]

        return translated

    # --- Paraphrasing --------------------------------------------------

    def _paraphrase_phase(
        self,
        instances: list[tuple[str, Instance]],
        translated: list[str],
    ) -> list[str]:
        """Paraphrase all translated sentences. Returns list parallel to *instances*."""
        logger.info("=== Phase 2: Paraphrasing ===")
        if self.cfg.skip_paraphrase:
            logger.info("Paraphrasing skipped (--skip-paraphrase).")
            return list(translated)

        cache_ns = "phi35_paraphrase"
        keys = [
            sha256_string(translated[i], inst.compound)
            for i, (_, inst) in enumerate(instances)
        ]
        cached = self._text_cache.get_batch(cache_ns, keys)
        miss_idx = [i for i, k in enumerate(keys) if k not in cached]

        if miss_idx:
            total = len(miss_idx)
            logger.info("Paraphrasing %d sentences with Phi-3.5 …", total)
            _CHECKPOINT_EVERY = 50
            with Phi35Paraphraser(device=self.cfg.device) as para:
                new_cache: dict[str, str] = {}
                for done, mi in enumerate(miss_idx, 1):
                    _, inst = instances[mi]
                    result = para.paraphrase(translated[mi], inst.compound)
                    cached[keys[mi]] = result
                    new_cache[keys[mi]] = result
                    if done % _CHECKPOINT_EVERY == 0 or done == total:
                        self._text_cache.set_batch(cache_ns, new_cache)
                        new_cache = {}
                        logger.info("  paraphrased %d / %d", done, total)
        else:
            logger.info("All paraphrases found in cache.")

        return [cached.get(k, translated[i]) for i, k in enumerate(keys)]

    # --- Sense classification ------------------------------------------

    def _classify_phase(self, translated: list[str]) -> np.ndarray:
        """Embed with BGE-M3 → run LR classifier → P(idiomatic)."""
        logger.info("=== Phase 3: Sense classification ===")

        # Embed with BGE-M3
        model_name = BGEM3Encoder.MODEL_ID
        keys = [sha256_string(t) for t in translated]
        cached = self._emb_cache.get_batch(model_name, keys)
        miss_idx = [i for i, k in enumerate(keys) if k not in cached]

        if miss_idx:
            logger.info("Embedding %d sentences with BGE-M3 …", len(miss_idx))
            with BGEM3Encoder(device=self.cfg.device) as enc:
                miss_texts = [translated[i] for i in miss_idx]
                new_embs = enc.encode(miss_texts)
                new_cache = {keys[i]: new_embs[j] for j, i in enumerate(miss_idx)}
                self._emb_cache.set_batch(model_name, new_cache)
                cached.update(new_cache)
        else:
            logger.info("All BGE-M3 embeddings found in cache.")

        embeddings = np.stack([cached[k] for k in keys], axis=0)

        # Classify
        clf = LRSenseClassifier.from_path(self.cfg.classifier_path)
        p_idiomatic = clf.predict_proba_idiomatic(embeddings)
        logger.info(
            "Sense distribution: %.1f%% idiomatic (mean P=%.3f)",
            100 * (p_idiomatic > 0.5).mean(),
            p_idiomatic.mean(),
        )
        return p_idiomatic

    # --- SigLIP2 encoding (images + text for ranking) -------------------

    def _siglip2_phase(
        self,
        instances: list[tuple[str, Instance]],
        translated: list[str],
        paraphrased: list[str],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Encode images and text with SigLIP2 for the ranking step.

        Returns:
            img_embs:      list of (5, D) arrays, one per instance
            txt_orig_embs: list of (D,) arrays (original translated text)
            txt_para_embs: list of (D,) arrays (paraphrased text)
        """
        logger.info("=== Phase 4: SigLIP2 encoding ===")
        model_name = SigLIP2Encoder.MODEL_ID

        # --- Collect all unique images; store per-instance key lists ---
        image_keys: dict[str, Path] = {}  # cache_key → path
        img_keys_per_instance: list[list[str]] = []
        for _, inst in instances:
            inst_keys = [sha256_string(str(img.absolute_path)) for img in inst.images]
            img_keys_per_instance.append(inst_keys)
            for ik, img in zip(inst_keys, inst.images):
                image_keys[ik] = img.absolute_path

        # --- Collect all unique texts; store per-instance key pairs ---
        text_set: dict[str, str] = {}  # cache_key → text
        orig_keys: list[str] = []
        para_keys: list[str] = []
        for i in range(len(instances)):
            tk_orig = sha256_string("text", translated[i])
            tk_para = sha256_string("text", paraphrased[i])
            orig_keys.append(tk_orig)
            para_keys.append(tk_para)
            text_set[tk_orig] = translated[i]
            text_set[tk_para] = paraphrased[i]

        # --- Check caches ---
        all_keys = list(image_keys.keys()) + list(text_set.keys())
        cached = self._emb_cache.get_batch(model_name, all_keys)
        missing_img_keys = [k for k in image_keys if k not in cached]
        missing_txt_keys = [k for k in text_set if k not in cached]

        if missing_img_keys or missing_txt_keys:
            logger.info(
                "Encoding %d images + %d texts with SigLIP2 …",
                len(missing_img_keys), len(missing_txt_keys),
            )
            with SigLIP2Encoder(device=self.cfg.device) as enc:
                # Encode missing images
                if missing_img_keys:
                    pil_images = [
                        Image.open(image_keys[k]).convert("RGB")
                        for k in missing_img_keys
                    ]
                    img_embs_new = enc.encode(pil_images)
                    new_cache = {
                        k: img_embs_new[j]
                        for j, k in enumerate(missing_img_keys)
                    }
                    self._emb_cache.set_batch(model_name, new_cache)
                    cached.update(new_cache)

                # Encode missing texts
                if missing_txt_keys:
                    texts = [text_set[k] for k in missing_txt_keys]
                    txt_embs_new = enc.encode_text(texts)
                    new_cache = {
                        k: txt_embs_new[j]
                        for j, k in enumerate(missing_txt_keys)
                    }
                    self._emb_cache.set_batch(model_name, new_cache)
                    cached.update(new_cache)
        else:
            logger.info("All SigLIP2 embeddings found in cache.")

        # --- Assemble per-instance arrays ---
        img_embs_list: list[np.ndarray] = []
        txt_orig_list: list[np.ndarray] = []
        txt_para_list: list[np.ndarray] = []

        for i in range(len(instances)):
            ie = np.stack([cached[k] for k in img_keys_per_instance[i]], axis=0)
            img_embs_list.append(ie)
            txt_orig_list.append(cached[orig_keys[i]])
            txt_para_list.append(cached[para_keys[i]])

        return img_embs_list, txt_orig_list, txt_para_list
