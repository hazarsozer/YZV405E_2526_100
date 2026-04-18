import dataclasses

import pytest

from src.data.loader import AdMIReRepository, Dataset, Instance


ALL_LANGUAGES = [
    "Chinese", "Georgian", "Greek", "Igbo", "Kazakh", "Norwegian",
    "Portuguese-Brazil", "Portuguese-Portugal", "Russian", "Serbian",
    "Slovak", "Slovenian", "Spanish-Ecuador", "Turkish", "Uzbek",
]

EXPECTED_ROW_COUNTS = {
    "Chinese": 179, "Georgian": 113, "Greek": 208, "Igbo": 115,
    "Kazakh": 156, "Norwegian": 202, "Portuguese-Brazil": 228,
    "Portuguese-Portugal": 220, "Russian": 140, "Serbian": 363,
    "Slovak": 151, "Slovenian": 240, "Spanish-Ecuador": 48,
    "Turkish": 180, "Uzbek": 120,
}

EXPECTED_LANG_CODES = {
    "Chinese": "ZH", "Georgian": "KA", "Greek": "EL", "Igbo": "IG",
    "Kazakh": "KK", "Norwegian": "NO", "Portuguese-Brazil": "PT-BR",
    "Portuguese-Portugal": "PT-PT", "Russian": "RU", "Serbian": "SR",
    "Slovak": "SK", "Slovenian": "SL", "Spanish-Ecuador": "ES-EC",
    "Turkish": "TR", "Uzbek": "UZ",
}


@pytest.mark.unit
def test_template_loads_all_15_languages(repo: AdMIReRepository):
    datasets = repo.load_all_templates()
    assert set(datasets.keys()) == set(EXPECTED_LANG_CODES.values())
    assert len(datasets) == 15


@pytest.mark.unit
def test_template_has_no_expected_order(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    for instance in ds.instances:
        assert instance.expected_order is None


@pytest.mark.unit
def test_template_has_no_sentence_type(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    for instance in ds.instances:
        assert instance.sentence_type is None


@pytest.mark.unit
def test_each_instance_has_exactly_5_images(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    for instance in ds.instances:
        assert len(instance.images) == 5


@pytest.mark.unit
@pytest.mark.parametrize("lang_name,expected_count", EXPECTED_ROW_COUNTS.items())
def test_row_counts_match_spec(repo: AdMIReRepository, lang_name: str, expected_count: int):
    ds = repo.load_submission_template(lang_name)
    assert len(ds.instances) == expected_count, (
        f"{lang_name}: expected {expected_count} rows, got {len(ds.instances)}"
    )


@pytest.mark.unit
def test_turkish_has_54_unique_compounds(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    assert len({i.compound for i in ds.instances}) == 54


@pytest.mark.unit
def test_serbian_has_95_unique_compounds(repo: AdMIReRepository):
    ds = repo.load_submission_template("Serbian")
    assert len({i.compound for i in ds.instances}) == 95


@pytest.mark.unit
def test_compound_shares_same_5_image_names_across_sentences(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    by_compound: dict[str, list] = {}
    for inst in ds.instances:
        by_compound.setdefault(inst.compound, []).append(inst)
    for compound, instances in by_compound.items():
        if len(instances) < 2:
            continue
        first_names = {img.name for img in instances[0].images}
        for other in instances[1:]:
            assert {img.name for img in other.images} == first_names, (
                f"Compound '{compound}' has inconsistent image sets across sentences"
            )


@pytest.mark.unit
@pytest.mark.parametrize("lang_name,expected_code", EXPECTED_LANG_CODES.items())
def test_language_code_assigned_correctly(repo: AdMIReRepository, lang_name: str, expected_code: str):
    ds = repo.load_submission_template(lang_name)
    assert ds.language_code == expected_code
    for inst in ds.instances:
        assert inst.language_code == expected_code


@pytest.mark.unit
def test_instance_is_immutable(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    instance = ds.instances[0]
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        instance.sentence = "mutated"  # type: ignore[misc]


@pytest.mark.unit
def test_row_index_is_sequential(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    for i, inst in enumerate(ds.instances):
        assert inst.row_index == i


@pytest.mark.unit
def test_image_names_end_with_png(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    for inst in ds.instances:
        for img in inst.images:
            assert img.name.endswith(".png"), f"Unexpected image name: {img.name}"


@pytest.mark.unit
def test_captions_are_non_empty(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    for inst in ds.instances:
        for img in inst.images:
            assert len(img.caption) > 0


@pytest.mark.integration
def test_validate_images_returns_empty_for_all_languages(repo: AdMIReRepository):
    """Confirms all image references in TSVs resolve to actual files on disk."""
    datasets = repo.load_all_templates()
    for lang_code, ds in datasets.items():
        missing = repo.validate_images(ds)
        assert missing == [], f"Missing images for {lang_code}: {missing[:5]}"


@pytest.mark.unit
def test_resolve_image_path_returns_existing_file(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    inst = ds.instances[0]
    img = inst.images[0]
    path = repo.resolve_image_path("Turkish", inst.compound, img.name)
    assert path.exists(), f"Image path does not exist: {path}"


@pytest.mark.unit
def test_dataset_source_tsv_is_set(repo: AdMIReRepository):
    ds = repo.load_submission_template("Turkish")
    assert ds.source_tsv.exists()
    assert ds.source_tsv.suffix == ".tsv"
