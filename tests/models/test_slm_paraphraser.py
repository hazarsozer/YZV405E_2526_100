"""Unit tests for SLM paraphraser module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.slm_paraphraser import IdentityParaphraser, Phi35Paraphraser


# ---------------------------------------------------------------------------
# IdentityParaphraser
# ---------------------------------------------------------------------------


class TestIdentityParaphraser:
    def test_returns_sentence_unchanged(self):
        p = IdentityParaphraser()
        sentence = "The project was a real can of worms."
        assert p.paraphrase(sentence, "can of worms") == sentence

    def test_compound_arg_ignored(self):
        p = IdentityParaphraser()
        sentence = "She spilled the beans."
        assert p.paraphrase(sentence, "spilled the beans") == sentence

    def test_empty_sentence(self):
        assert IdentityParaphraser().paraphrase("", "compound") == ""


# ---------------------------------------------------------------------------
# Phi35Paraphraser helpers
# ---------------------------------------------------------------------------


def _make_phi_mocks(generated_text: str = "She revealed the secret."):
    """Return (fake_tokenizer, fake_model) producing *generated_text*."""
    prompt_len = 20
    gen_len = 10

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 2
    # apply_chat_template → tensor of shape (1, prompt_len)
    fake_tok.apply_chat_template.return_value = torch.zeros(
        1, prompt_len, dtype=torch.long
    )
    # decode → the generated text
    fake_tok.decode.return_value = generated_text

    fake_model = MagicMock()
    # generate → tensor of shape (1, prompt_len + gen_len)
    fake_model.generate.return_value = torch.zeros(
        1, prompt_len + gen_len, dtype=torch.long
    )

    return fake_tok, fake_model, prompt_len


# ---------------------------------------------------------------------------
# Phi35Paraphraser
# ---------------------------------------------------------------------------


class TestPhi35Paraphraser:
    @pytest.fixture()
    def paraphraser_and_mocks(self):
        generated = "She revealed the secret."
        fake_tok, fake_model, prompt_len = _make_phi_mocks(generated)
        with (
            patch("src.models.slm_paraphraser.AutoTokenizer") as mt,
            patch("src.models.slm_paraphraser.AutoModelForCausalLM") as mm,
            patch("src.models.slm_paraphraser.BitsAndBytesConfig"),
        ):
            mt.from_pretrained.return_value = fake_tok
            mm.from_pretrained.return_value = fake_model
            p = Phi35Paraphraser(device="cpu")
            p.load()
            p._tokenizer = fake_tok
            p._model = fake_model
            yield p, fake_tok, fake_model, prompt_len

    # --- basic paraphrasing ---

    def test_paraphrase_returns_string(self, paraphraser_and_mocks):
        p, *_ = paraphraser_and_mocks
        result = p.paraphrase("She spilled the beans.", "spilled the beans")
        assert isinstance(result, str)

    def test_paraphrase_returns_model_output(self, paraphraser_and_mocks):
        p, fake_tok, *_ = paraphraser_and_mocks
        fake_tok.decode.return_value = "She told everyone."
        result = p.paraphrase("She spilled the beans.", "spilled the beans")
        assert result == "She told everyone."

    def test_paraphrase_strips_whitespace(self, paraphraser_and_mocks):
        p, fake_tok, *_ = paraphraser_and_mocks
        fake_tok.decode.return_value = "  She told everyone.  "
        result = p.paraphrase("She spilled the beans.", "spilled the beans")
        assert result == "She told everyone."

    def test_compound_appears_in_prompt(self, paraphraser_and_mocks):
        p, fake_tok, *_ = paraphraser_and_mocks
        p.paraphrase("He kicked the bucket.", "kicked the bucket")
        call_args = fake_tok.apply_chat_template.call_args
        messages = call_args[0][0]
        assert any("kicked the bucket" in m["content"] for m in messages)

    def test_sentence_appears_in_prompt(self, paraphraser_and_mocks):
        p, fake_tok, *_ = paraphraser_and_mocks
        sentence = "He kicked the bucket last year."
        p.paraphrase(sentence, "kicked the bucket")
        call_args = fake_tok.apply_chat_template.call_args
        messages = call_args[0][0]
        assert any(sentence in m["content"] for m in messages)

    # --- new tokens slicing ---

    def test_only_new_tokens_are_decoded(self, paraphraser_and_mocks):
        p, fake_tok, fake_model, prompt_len = paraphraser_and_mocks
        total_len = prompt_len + 8
        fake_model.generate.return_value = torch.arange(total_len).unsqueeze(0)
        p.paraphrase("test sentence", "compound")
        decoded_tokens = fake_tok.decode.call_args[0][0]
        # Should only contain the generated tokens, not the prompt
        assert decoded_tokens.shape[0] == 8

    # --- empty output fallback ---

    def test_falls_back_to_original_on_empty_output(self, paraphraser_and_mocks):
        p, fake_tok, *_ = paraphraser_and_mocks
        fake_tok.decode.return_value = "   "  # whitespace-only → empty after strip
        sentence = "She spilled the beans."
        result = p.paraphrase(sentence, "spilled the beans")
        assert result == sentence

    # --- lifecycle ---

    def test_raises_if_not_loaded(self):
        p = Phi35Paraphraser(device="cpu")
        with pytest.raises(RuntimeError, match="load()"):
            p.paraphrase("test", "compound")

    def test_context_manager_loads_and_unloads(self):
        fake_tok, fake_model, _ = _make_phi_mocks()
        with (
            patch("src.models.slm_paraphraser.AutoTokenizer") as mt,
            patch("src.models.slm_paraphraser.AutoModelForCausalLM") as mm,
            patch("src.models.slm_paraphraser.BitsAndBytesConfig"),
        ):
            mt.from_pretrained.return_value = fake_tok
            mm.from_pretrained.return_value = fake_model
            p = Phi35Paraphraser(device="cpu")
            with p:
                assert p._model is not None
            assert p._model is None
            assert p._tokenizer is None

    def test_unload_clears_model_and_tokenizer(self, paraphraser_and_mocks):
        p, *_ = paraphraser_and_mocks
        p.unload()
        assert p._model is None
        assert p._tokenizer is None
