from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    'Rewrite the following sentence by replacing the expression "{compound}" '
    "with a plain-English phrase that captures its meaning in this context. "
    "Output only the rewritten sentence, no explanation.\n\n"
    "Sentence: {sentence}"
)

_QWEN_SYSTEM = (
    "You are a language expert specialising in idiomatic expressions. "
    "When asked to rewrite a sentence, you replace the given expression with "
    "clear, literal language that conveys exactly what it means in context. "
    "You return only the rewritten sentence — no explanation, no quotes, no preamble."
)

_QWEN_PROMPT_TEMPLATE = (
    'Rewrite the sentence below by replacing the expression "{compound}" with '
    "plain English that captures its meaning in this specific context. "
    "Return ONLY the rewritten sentence.\n\n"
    "Sentence: {sentence}\n"
    "Rewritten:"
)

_ENRICH_TEMPLATE = (
    'The expression "{compound}" is an idiom. Below is a literal description of '
    "an image. Rewrite the description by appending ONE concise sentence at the "
    "end that states what abstract concept this image could symbolize for the "
    'idiom "{compound}". Keep the original description intact.\n\n'
    "Image description: {caption}\n"
    "Enriched description:"
)


def _filter_paraphrase_variants(
    variants: list[str], original: str, compound: str
) -> list[str]:
    """Remove degenerate paraphrase variants (too short/long, compound still present).

    Falls back to the unfiltered list if fewer than 2 variants survive.
    """
    orig_words = len(original.split())
    filtered = [
        v
        for v in variants
        if (
            0.5 * orig_words <= len(v.split()) <= 2.5 * orig_words
            and compound.lower() not in v.lower()
        )
    ]
    return filtered if len(filtered) >= 2 else variants


class IdentityParaphraser:
    """Fallback paraphraser — returns the sentence unchanged."""

    def paraphrase(self, sentence: str, compound: str) -> str:
        return sentence


class Phi35Paraphraser:
    """Phi-3.5-mini-instruct paraphraser loaded in 4-bit for ~2.5 GB VRAM.

    Rewrites a sentence so the idiomatic compound is replaced by its
    plain-English meaning.  Use as a context manager to free VRAM after use.
    """

    MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
    MAX_NEW_TOKENS = 256

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            quantization_config=bnb_config,
            device_map=self._device,
        )
        self._model.eval()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> Phi35Paraphraser:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Paraphrasing
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def paraphrase(self, sentence: str, compound: str) -> str:
        """Return *sentence* with *compound* replaced by its plain meaning.

        Falls back to the original sentence if generation fails or produces
        an empty output.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call load() or use as a context manager first.")

        prompt = _PROMPT_TEMPLATE.format(sentence=sentence, compound=compound)
        messages = [{"role": "user", "content": prompt}]

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._device)

        generated = self._model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        # Strip the prompt tokens; keep only the newly generated part
        new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
        output = self._tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

        if not output:
            logger.warning(
                "Phi-3.5 returned empty output for compound=%r; using original.",
                compound,
            )
            return sentence

        return output

    @torch.inference_mode()
    def enrich_caption(self, caption: str, compound: str) -> str:
        """Return *caption* with one sentence appended describing the abstract/symbolic concept.

        Uses the idiomatic meaning of *compound* as context. Falls back to the
        original caption on empty or failed generation.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call load() or use as a context manager first.")

        prompt = _ENRICH_TEMPLATE.format(caption=caption, compound=compound)
        messages = [{"role": "user", "content": prompt}]

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._device)

        generated = self._model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_tokens = generated[:, inputs["input_ids"].shape[-1] :]
        output = self._tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

        if not output:
            logger.warning(
                "Phi-3.5 returned empty enrichment for compound=%r; using original caption.",
                compound,
            )
            return caption

        return output

    @torch.inference_mode()
    def paraphrase_k(self, sentence: str, compound: str, k: int = 4) -> list[str]:
        """Return *k* diverse paraphrases of *sentence* via sampling.

        Each variant replaces *compound* with its plain-English meaning.
        Degenerate outputs (wrong length or compound still present) are
        filtered; falls back to unfiltered list if fewer than 2 survive.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call load() or use as a context manager first.")

        prompt = _PROMPT_TEMPLATE.format(sentence=sentence, compound=compound)
        messages = [{"role": "user", "content": prompt}]

        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._device)
        prompt_len = inputs["input_ids"].shape[-1]

        results: list[str] = []
        for _ in range(k):
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            new_tokens = generated[:, prompt_len:]
            output = self._tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
            results.append(output if output else sentence)

        return _filter_paraphrase_variants(results, sentence, compound)
