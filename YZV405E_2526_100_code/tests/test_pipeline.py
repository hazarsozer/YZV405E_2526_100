"""Unit tests for the pipeline orchestrator (mocked — no GPU required)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.loader import Dataset, ImageCandidate, Instance
from src.pipeline import AdMIRePipeline, PipelineConfig, RankedInstance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIGLIP_DIM = 1152
BGEM3_DIM = 1024


def _dummy_instance(lang: str = "TR", idx: int = 0) -> Instance:
    images = tuple(
        ImageCandidate(
            name=f"img{i}.png",
            absolute_path=Path(f"/fake/{lang}/compound/img{i}.png"),
            caption=f"caption {i}",
        )
        for i in range(1, 6)
    )
    return Instance(
        compound="test_compound",
        sentence="Bu bir test cümlesi.",
        language_code=lang,
        language_name="Turkish",
        images=images,
        row_index=idx,
    )


def _dummy_dataset(lang: str = "TR", n: int = 2) -> Dataset:
    return Dataset(
        language_code=lang,
        instances=tuple(_dummy_instance(lang, i) for i in range(n)),
        source_tsv=Path(f"/fake/submission_{lang}.tsv"),
    )


def _normed(arr: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(n, 1e-8)


@pytest.fixture()
def config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        data_root=tmp_path / "data",
        templates_root=tmp_path / "templates",
        classifier_path=tmp_path / "clf.joblib",
        output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
        device="cpu",
        languages=["TR"],
        skip_translation=True,
        skip_paraphrase=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_defaults(self, tmp_path: Path):
        cfg = PipelineConfig(
            data_root=tmp_path,
            templates_root=tmp_path,
            classifier_path=tmp_path / "clf.joblib",
            output_dir=tmp_path / "out",
        )
        assert cfg.device == "cuda"
        assert cfg.languages is None
        assert cfg.skip_translation is False


class TestRankedInstance:
    def test_fields(self):
        inst = _dummy_instance()
        ri = RankedInstance(
            instance=inst,
            translated_sentence="This is a test.",
            paraphrased_sentence="This is a test.",
            p_idiomatic=0.8,
            predicted_order=["img1.png", "img2.png", "img3.png",
                             "img4.png", "img5.png"],
        )
        assert ri.p_idiomatic == 0.8
        assert len(ri.predicted_order) == 5


class TestPipelineRun:
    """Test the pipeline with internal phases mocked out."""

    def test_run_produces_results(self, config: PipelineConfig):
        n_instances = 2

        def fake_siglip_phase(instances, translated, paraphrased):
            img_embs = [
                _normed(np.random.randn(5, SIGLIP_DIM).astype(np.float32))
                for _ in instances
            ]
            txt_orig = [
                _normed(np.random.randn(SIGLIP_DIM).astype(np.float32))
                for _ in instances
            ]
            txt_para = [
                _normed(np.random.randn(SIGLIP_DIM).astype(np.float32))
                for _ in instances
            ]
            return img_embs, txt_orig, txt_para

        def fake_classify_phase(translated):
            return np.array([0.8, 0.2], dtype=np.float32)

        with (
            patch.object(
                AdMIRePipeline, "_load_datasets",
                return_value={"TR": _dummy_dataset("TR", n_instances)},
            ),
            patch.object(
                AdMIRePipeline, "_siglip2_phase",
                side_effect=fake_siglip_phase,
            ),
            patch.object(
                AdMIRePipeline, "_classify_phase",
                side_effect=fake_classify_phase,
            ),
        ):
            pipeline = AdMIRePipeline(config)
            results = pipeline.run()

        assert "TR" in results
        assert len(results["TR"]) == n_instances
        for ri in results["TR"]:
            assert len(ri.predicted_order) == 5
            assert set(ri.predicted_order) == {
                f"img{i}.png" for i in range(1, 6)
            }

    def test_first_instance_idiomatic_second_literal(
        self, config: PipelineConfig
    ):
        """Verify that different P(idiomatic) values lead to valid rankings."""
        n_instances = 2

        # Deterministic embeddings so results are reproducible
        rng = np.random.default_rng(0)

        def fake_siglip(instances, translated, paraphrased):
            return (
                [_normed(rng.standard_normal((5, SIGLIP_DIM)).astype(np.float32))
                 for _ in instances],
                [_normed(rng.standard_normal(SIGLIP_DIM).astype(np.float32))
                 for _ in instances],
                [_normed(rng.standard_normal(SIGLIP_DIM).astype(np.float32))
                 for _ in instances],
            )

        def fake_classify(translated):
            # First instance = idiomatic, second = literal
            return np.array([0.95, 0.05], dtype=np.float32)

        with (
            patch.object(AdMIRePipeline, "_load_datasets",
                         return_value={"TR": _dummy_dataset("TR", n_instances)}),
            patch.object(AdMIRePipeline, "_siglip2_phase", side_effect=fake_siglip),
            patch.object(AdMIRePipeline, "_classify_phase", side_effect=fake_classify),
        ):
            pipeline = AdMIRePipeline(config)
            results = pipeline.run()

        assert results["TR"][0].p_idiomatic == pytest.approx(0.95)
        assert results["TR"][1].p_idiomatic == pytest.approx(0.05)

