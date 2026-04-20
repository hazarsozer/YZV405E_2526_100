"""Unit tests for translator module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import BatchEncoding

from src.models.translator import (
    LANG_CODE_TO_FLORES,
    NO_OP_LANGS,
    NLLB200Translator,
    NoOpTranslator,
)


# ---------------------------------------------------------------------------
# NoOpTranslator
# ---------------------------------------------------------------------------


class TestNoOpTranslator:
    def test_returns_input_unchanged(self):
        t = NoOpTranslator()
        texts = ["Hello world", "How are you?"]
        assert t.translate(texts, "EN") == texts

    def test_returns_new_list(self):
        t = NoOpTranslator()
        texts = ["a", "b"]
        result = t.translate(texts, "EN")
        assert result is not texts

    def test_empty_list(self):
        assert NoOpTranslator().translate([], "EN") == []

    def test_no_op_langs_constant(self):
        assert "EN" in NO_OP_LANGS
        assert "PT-BR" in NO_OP_LANGS


# ---------------------------------------------------------------------------
# LANG_CODE_TO_FLORES mapping
# ---------------------------------------------------------------------------


class TestLangMapping:
    def test_all_blind_langs_mapped(self):
        blind_langs = {"ZH", "KA", "EL", "IG", "KK", "NO", "PT-PT", "RU", "SR",
                       "SK", "SL", "ES-EC", "TR", "UZ"}
        for lang in blind_langs:
            assert lang in LANG_CODE_TO_FLORES, f"{lang} missing from FLORES mapping"

    def test_supervised_langs_not_in_map(self):
        # EN and PT-BR are NoOp — they shouldn't need FLORES codes
        assert "EN" not in LANG_CODE_TO_FLORES
        assert "PT-BR" not in LANG_CODE_TO_FLORES

    def test_flores_codes_have_correct_format(self):
        for code in LANG_CODE_TO_FLORES.values():
            lang, script = code.split("_")
            assert len(lang) == 3
            assert len(script) == 4


# ---------------------------------------------------------------------------
# NLLB200Translator
# ---------------------------------------------------------------------------


def _make_nllb_mocks(decoded_output: list[str]):
    """Return (fake_tokenizer, fake_model) that produce *decoded_output*."""
    fake_tok = MagicMock()
    fake_tok.src_lang = None
    fake_tok.convert_tokens_to_ids.return_value = 256047  # eng_Latn token id
    fake_tok.return_value = BatchEncoding(
        {"input_ids": torch.zeros(len(decoded_output), 10, dtype=torch.long),
         "attention_mask": torch.ones(len(decoded_output), 10, dtype=torch.long)}
    )
    fake_tok.batch_decode.return_value = decoded_output

    fake_model = MagicMock()
    fake_model.generate.return_value = torch.zeros(
        len(decoded_output), 15, dtype=torch.long
    )

    return fake_tok, fake_model


class TestNLLB200Translator:
    @pytest.fixture()
    def translator_and_mocks(self):
        decoded = ["Hello world"]
        fake_tok, fake_model = _make_nllb_mocks(decoded)
        with (
            patch("src.models.translator.AutoTokenizer") as mt,
            patch("src.models.translator.AutoModelForSeq2SeqLM") as mm,
        ):
            mt.from_pretrained.return_value = fake_tok
            mm.from_pretrained.return_value = fake_model
            tr = NLLB200Translator(device="cpu")
            tr.load()
            tr._tokenizer = fake_tok
            tr._model = fake_model
            yield tr, fake_tok, fake_model

    # --- basic translation ---

    def test_translate_returns_list_of_strings(self, translator_and_mocks):
        tr, fake_tok, _ = translator_and_mocks
        fake_tok.batch_decode.return_value = ["Translated text"]
        result = tr.translate(["Merhaba dünya"], "TR")
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_translate_sets_src_lang_on_tokenizer(self, translator_and_mocks):
        tr, fake_tok, _ = translator_and_mocks
        fake_tok.batch_decode.return_value = ["Hello"]
        tr.translate(["Merhaba"], "TR")
        assert fake_tok.src_lang == LANG_CODE_TO_FLORES["TR"]

    def test_translate_length_matches_input(self, translator_and_mocks):
        tr, fake_tok, fake_model = translator_and_mocks
        n = 5
        fake_tok.batch_decode.return_value = [f"out {i}" for i in range(n)]
        fake_model.generate.return_value = torch.zeros(n, 10, dtype=torch.long)
        fake_tok.return_value = BatchEncoding(
            {"input_ids": torch.zeros(n, 10, dtype=torch.long),
             "attention_mask": torch.ones(n, 10, dtype=torch.long)}
        )
        result = tr.translate([f"text {i}" for i in range(n)], "TR")
        assert len(result) == n

    def test_unknown_lang_raises_value_error(self, translator_and_mocks):
        tr, _, _ = translator_and_mocks
        with pytest.raises(ValueError, match="Unknown language code"):
            tr.translate(["text"], "XX")

    def test_raises_if_not_loaded(self):
        tr = NLLB200Translator(device="cpu")
        with pytest.raises(RuntimeError, match="load()"):
            tr.translate(["text"], "TR")

    # --- batching ---

    def test_batching_splits_correctly(self):
        n = 7
        batch_size = 3
        call_sizes: list[int] = []

        def tokenizer_side_effect(texts, **kwargs):
            b = len(texts)
            call_sizes.append(b)
            return BatchEncoding({
                "input_ids": torch.zeros(b, 5, dtype=torch.long),
                "attention_mask": torch.ones(b, 5, dtype=torch.long),
            })

        fake_tok = MagicMock()
        fake_tok.src_lang = None
        fake_tok.convert_tokens_to_ids.return_value = 256047
        fake_tok.side_effect = tokenizer_side_effect
        fake_tok.batch_decode.side_effect = lambda t, **kw: [f"out_{i}" for i in range(t.shape[0])]

        fake_model = MagicMock()
        fake_model.generate.side_effect = lambda **kw: torch.zeros(
            kw["input_ids"].shape[0], 5, dtype=torch.long
        )

        with (
            patch("src.models.translator.AutoTokenizer") as mt,
            patch("src.models.translator.AutoModelForSeq2SeqLM") as mm,
        ):
            mt.from_pretrained.return_value = fake_tok
            mm.from_pretrained.return_value = fake_model
            tr = NLLB200Translator(device="cpu")
            tr.load()
            tr._tokenizer = fake_tok
            tr._model = fake_model
            result = tr.translate([f"t{i}" for i in range(n)], "TR", batch_size=batch_size)

        assert len(result) == n
        assert call_sizes == [3, 3, 1]

    # --- lifecycle ---

    def test_context_manager_loads_and_unloads(self):
        fake_tok, fake_model = _make_nllb_mocks(["out"])
        with (
            patch("src.models.translator.AutoTokenizer") as mt,
            patch("src.models.translator.AutoModelForSeq2SeqLM") as mm,
        ):
            mt.from_pretrained.return_value = fake_tok
            mm.from_pretrained.return_value = fake_model
            tr = NLLB200Translator(device="cpu")
            with tr:
                assert tr._model is not None
            assert tr._model is None
            assert tr._tokenizer is None
