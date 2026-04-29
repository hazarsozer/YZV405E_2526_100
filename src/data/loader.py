from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import pandas as pd

SentenceType = Literal["idiomatic", "literal"]

LANG_NAME_TO_CODE: dict[str, str] = {
    "English": "EN",
    "Chinese": "ZH",
    "Georgian": "KA",
    "Greek": "EL",
    "Igbo": "IG",
    "Kazakh": "KK",
    "Norwegian": "NO",
    "Portuguese-Brazil": "PT-BR",
    "Portuguese-Portugal": "PT-PT",
    "Russian": "RU",
    "Serbian": "SR",
    "Slovak": "SK",
    "Slovenian": "SL",
    "Spanish-Ecuador": "ES-EC",
    "Turkish": "TR",
    "Uzbek": "UZ",
}


@dataclasses.dataclass(frozen=True)
class ImageCandidate:
    name: str
    absolute_path: Path
    caption: str
    caption_pt: str | None = None


@dataclasses.dataclass(frozen=True)
class Instance:
    compound: str
    sentence: str
    language_code: str
    language_name: str
    images: tuple[
        ImageCandidate,
        ImageCandidate,
        ImageCandidate,
        ImageCandidate,
        ImageCandidate,
    ]
    sentence_type: SentenceType | None = None
    expected_order: tuple[str, ...] | None = None
    row_index: int = -1


@dataclasses.dataclass(frozen=True)
class Dataset:
    language_code: str
    instances: tuple[Instance, ...]
    source_tsv: Path


class AdMIReRepository:
    def __init__(self, data_root: Path, templates_root: Path) -> None:
        self._data_root = data_root
        self._templates_root = templates_root

    def resolve_image_path(
        self, language_name: str, compound: str, image_name: str
    ) -> Path:
        return self._data_root / language_name / compound / image_name

    def validate_images(self, dataset: Dataset) -> list[str]:
        missing: list[str] = []
        for inst in dataset.instances:
            for img in inst.images:
                if not img.absolute_path.exists():
                    missing.append(str(img.absolute_path))
        return missing

    def load_submission_template(self, language_name: str) -> Dataset:
        tsv_path = self._templates_root / f"submission_{language_name}.tsv"
        return self._load_tsv(tsv_path, language_name, has_labels=False)

    def load_training_tsv(self, tsv_path: Path, language_name: str) -> Dataset:
        return self._load_tsv(tsv_path, language_name, has_labels=True)

    def load_all_templates(self) -> dict[str, Dataset]:
        return {
            LANG_NAME_TO_CODE[lang]: self.load_submission_template(lang)
            for lang in LANG_NAME_TO_CODE
            if (self._templates_root / f"submission_{lang}.tsv").exists()
        }

    def _load_tsv(
        self, tsv_path: Path, language_name: str, *, has_labels: bool
    ) -> Dataset:
        lang_code = LANG_NAME_TO_CODE[language_name]
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)

        instances: list[Instance] = []
        for row_index, row in enumerate(df.itertuples(index=False)):
            images = tuple(
                ImageCandidate(
                    name=getattr(row, f"image{i}_name"),
                    absolute_path=self.resolve_image_path(
                        language_name,
                        row.compound,
                        getattr(row, f"image{i}_name"),
                    ),
                    caption=getattr(row, f"image{i}_caption"),
                )
                for i in range(1, 6)
            )

            sentence_type: SentenceType | None = None
            if has_labels and hasattr(row, "sentence_type") and row.sentence_type:
                sentence_type = row.sentence_type  # type: ignore[assignment]

            expected_order: tuple[str, ...] | None = None
            raw_order = getattr(row, "expected_order", "")
            if has_labels and raw_order:
                expected_order = tuple(
                    name.strip().strip("'\"")
                    for name in raw_order.strip("[]").split(",")
                    if name.strip()
                )

            instances.append(
                Instance(
                    compound=row.compound,
                    sentence=row.sentence,
                    language_code=lang_code,
                    language_name=language_name,
                    images=images,  # type: ignore[arg-type]
                    sentence_type=sentence_type,
                    expected_order=expected_order,
                    row_index=row_index,
                )
            )

        return Dataset(
            language_code=lang_code,
            instances=tuple(instances),
            source_tsv=tsv_path,
        )
