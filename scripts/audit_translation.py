"""Quick translation-quality audit.

Pull cached NLLB-200 translations for sample rows in several languages and
print them alongside the source sentence + compound so we can eyeball whether
the idiom survives translation.

Run with:
    uv run python -m scripts.audit_translation [--lang TR] [--n 20]
"""
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path

from src.utils.cache import TextCache
from src.utils.io import sha256_string

# Language code -> submission TSV filename
LANG_FILES: dict[str, str] = {
    "ZH": "submission_Chinese.tsv",
    "KA": "submission_Georgian.tsv",
    "EL": "submission_Greek.tsv",
    "IG": "submission_Igbo.tsv",
    "KK": "submission_Kazakh.tsv",
    "NO": "submission_Norwegian.tsv",
    "PT-PT": "submission_Portuguese-Portugal.tsv",
    "RU": "submission_Russian.tsv",
    "SR": "submission_Serbian.tsv",
    "SK": "submission_Slovak.tsv",
    "SL": "submission_Slovenian.tsv",
    "ES-EC": "submission_Spanish-Ecuador.tsv",
    "TR": "submission_Turkish.tsv",
    "UZ": "submission_Uzbek.tsv",
}


def load_rows_for_lang(lang: str, base: Path) -> pd.DataFrame:
    """Load TSV rows for a language."""
    tsv = base / LANG_FILES[lang]
    if not tsv.exists():
        raise FileNotFoundError(f"No TSV for {lang}: {tsv}")
    return pd.read_csv(tsv, sep="\t")


def audit(lang: str, n: int, text_cache: TextCache) -> None:
    base = Path("data/raw/admire_file/admire_file")
    df = load_rows_for_lang(lang, base)
    print(f"\n{'='*80}")
    print(f"  LANGUAGE: {lang}  ({LANG_FILES[lang]})   |  rows: {len(df)}   |  unique compounds: {df['compound'].nunique()}")
    print(f"{'='*80}\n")

    # Pick first n unique compounds and one sentence each for breadth
    seen_compounds: set[str] = set()
    picked: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        cmp = row["compound"]
        if cmp in seen_compounds:
            continue
        seen_compounds.add(cmp)
        picked.append((cmp, row["sentence"]))
        if len(picked) >= n:
            break

    # Look up cached translations
    keys = [sha256_string(lang, sent) for _, sent in picked]
    cached = text_cache.get_batch("nllb200_translation", keys)

    hits = 0
    for (compound, sentence), key in zip(picked, keys):
        translation = cached.get(key, "[MISS]")
        if key in cached:
            hits += 1
        print(f"  COMPOUND : {compound}")
        print(f"  SOURCE   : {sentence}")
        print(f"  NLLB EN  : {translation}")
        print()
    print(f"  -- cache hits: {hits}/{len(picked)} --")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", nargs="+", default=["TR", "RU", "ZH", "ES-EC", "EL"])
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()

    text_cache = TextCache(Path("data/processed/text"))
    for lang in args.lang:
        if lang not in LANG_FILES:
            print(f"[skip] unknown lang: {lang}")
            continue
        audit(lang, args.n, text_cache)


if __name__ == "__main__":
    main()
