from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Main branch only has pytorch_model.bin; safetensors available via this PR.
_NLLB200_SAFETENSORS_REVISION = "refs/pr/45"

# FLORES-200 codes used by NLLB-200
LANG_CODE_TO_FLORES: dict[str, str] = {
    "ZH": "zho_Hans",
    "KA": "kat_Geor",
    "EL": "ell_Grek",
    "IG": "ibo_Latn",
    "KK": "kaz_Cyrl",
    "NO": "nob_Latn",
    "PT-PT": "por_Latn",
    "RU": "rus_Cyrl",
    "SR": "srp_Cyrl",
    "SK": "slk_Latn",
    "SL": "slv_Latn",
    "ES-EC": "spa_Latn",
    "TR": "tur_Latn",
    "UZ": "uzn_Latn",
}

# Languages already in English — translation is a no-op
NO_OP_LANGS: frozenset[str] = frozenset({"EN", "PT-BR"})


class NoOpTranslator:
    """Pass-through translator for sentences already in English."""

    def translate(self, texts: list[str], src_lang: str) -> list[str]:
        return list(texts)


class NLLB200Translator:
    """NLLB-200-distilled-600M translator: any supported language → English.

    Use as a context manager to automatically free VRAM after use.
    """

    MODEL_ID = "facebook/nllb-200-distilled-600M"
    TARGET_LANG = "eng_Latn"
    MAX_INPUT_LENGTH = 512
    MAX_OUTPUT_LENGTH = 512

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._model: AutoModelForSeq2SeqLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, revision=_NLLB200_SAFETENSORS_REVISION
        )
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                self.MODEL_ID,
                revision=_NLLB200_SAFETENSORS_REVISION,
                dtype=dtype,
            )
            .to(self._device)
            .eval()
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> NLLB200Translator:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def translate(
        self, texts: list[str], src_lang: str, batch_size: int = 16
    ) -> list[str]:
        """Translate *texts* from *src_lang* (our code, e.g. 'ZH') to English."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call load() or use as a context manager first.")

        flores_src = LANG_CODE_TO_FLORES.get(src_lang)
        if flores_src is None:
            raise ValueError(
                f"Unknown language code '{src_lang}'. "
                f"Supported: {sorted(LANG_CODE_TO_FLORES)}"
            )

        self._tokenizer.src_lang = flores_src
        target_id = self._tokenizer.convert_tokens_to_ids(self.TARGET_LANG)

        results: list[str] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.MAX_INPUT_LENGTH,
            ).to(self._device)
            generated = self._model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_length=self.MAX_OUTPUT_LENGTH,
            )
            decoded = self._tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            results.extend(decoded)

        return results
