"""Unit tests for SLM paraphraser module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.slm_paraphraser import (
    IdentityParaphraser,
    Phi35Paraphraser,
    _filter_paraphrase_variants,
)


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


class _FakeInputs(dict):
    """Dict with .to(device) so it behaves like a tokenizer output (return_dict=True)."""

    def to(self, device: str) -> "_FakeInputs":
        return self


def _make_phi_mocks(generated_text: str = "She revealed the secret."):
    """Return (fake_tokenizer, fake_model) producing *generated_text*."""
    prompt_len = 20
    gen_len = 10

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 2
    # apply_chat_template → dict-like with input_ids (matches return_dict=True)
    fake_tok.apply_chat_template.return_value = _FakeInputs(
        input_ids=torch.zeros(1, prompt_len, dtype=torch.long)
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

    # --- enrich_caption ---

    def test_enrich_caption_returns_string(self, paraphraser_and_mocks):
        p, fake_tok, *_ = paraphraser_and_mocks
        fake_tok.decode.return_value = "A person causes trouble for others. This symbolizes spreading corruption."
        result = p.enrich_caption("A person causes trouble for others.", "bad apple")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_enrich_caption_compound_in_prompt(self, paraphraser_and_mocks):
        p, fake_tok, *_ = paraphraser_and_mocks
        fake_tok.decode.return_value = "enriched"
        p.enrich_caption("A rotten apple on a table.", "bad apple")
        call_args = fake_tok.apply_chat_template.call_args
        messages = call_args[0][0]
        assert any("bad apple" in m["content"] for m in messages)

    # --- paraphrase_k ---

    def test_paraphrase_k_returns_k_variants(self, paraphraser_and_mocks):
        p, fake_tok, *_ = paraphraser_and_mocks
        outputs = ["She told everyone.", "She disclosed the secret.", "She blabbed.", "She confessed."]
        fake_tok.decode.side_effect = outputs
        results = p.paraphrase_k("She spilled the beans.", "spilled the beans", k=4)
        assert len(results) == 4

    def test_paraphrase_k_falls_back_on_empty(self, paraphraser_and_mocks):
        p, fake_tok, *_ = paraphraser_and_mocks
        fake_tok.decode.side_effect = ["", "", "", "She told everyone."]
        results = p.paraphrase_k("She spilled the beans.", "spilled the beans", k=4)
        assert all(isinstance(r, str) for r in results)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# _filter_paraphrase_variants
# ---------------------------------------------------------------------------


class TestFilterParaphraseVariants:
    def test_removes_compound_present(self):
        original = "She spilled the beans at the meeting."
        variants = [
            "She revealed the secret.",       # good
            "She spilled the beans again.",    # compound still present → filtered
            "She disclosed everything.",       # good
            "She told the truth.",             # good
        ]
        filtered = _filter_paraphrase_variants(variants, original, "spilled the beans")
        assert not any("spilled the beans" in v for v in filtered)

    def test_removes_too_short(self):
        original = "The project was going well but then hit a real obstacle."
        # Two long variants survive → no fallback → short one is filtered out
        variants = [
            "OK.",
            "Things were fine but then they encountered a serious problem.",
            "The work progressed well until they ran into a significant obstacle.",
        ]
        filtered = _filter_paraphrase_variants(variants, original, "hit a wall")
        assert not any(len(v.split()) < 0.5 * len(original.split()) for v in filtered)

    def test_fallback_if_too_few_survive(self):
        original = "She spilled the beans."
        # All have compound — normally all filtered, but fallback kicks in
        variants = ["She spilled the beans again.", "She spilled the beans loudly."]
        result = _filter_paraphrase_variants(variants, original, "spilled the beans")
        assert len(result) == 2  # fallback to original list
