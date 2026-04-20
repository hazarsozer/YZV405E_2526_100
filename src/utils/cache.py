from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import sha256_string


class EmbeddingCache:
    """Parquet-backed cache for model embeddings.

    One parquet file per model, keyed by sha256 of input identifier(s).
    Supports batch get/set so callers can minimise disk I/O.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_batch(self, model_name: str, keys: list[str]) -> dict[str, np.ndarray]:
        """Return cached embeddings for *keys* that exist in cache."""
        df = self._load_df(model_name)
        if df.empty:
            return {}
        found = df[df["key"].isin(keys)]
        return {
            row["key"]: np.array(row["embedding"], dtype=np.float32)
            for _, row in found.iterrows()
        }

    def set_batch(self, model_name: str, data: dict[str, np.ndarray]) -> None:
        """Upsert *data* into the cache for *model_name*."""
        if not data:
            return
        existing = self._load_df(model_name)
        new_rows = pd.DataFrame(
            [{"key": k, "embedding": v.tolist()} for k, v in data.items()]
        )
        merged = pd.concat(
            [existing[~existing["key"].isin(data)], new_rows], ignore_index=True
        )
        merged.to_parquet(self._path(model_name), index=False)

    @staticmethod
    def make_key(*parts: str) -> str:
        return sha256_string(*parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, model_name: str) -> Path:
        safe = model_name.replace("/", "__").replace("\\", "__")
        return self._cache_dir / f"{safe}.parquet"

    def _load_df(self, model_name: str) -> pd.DataFrame:
        path = self._path(model_name)
        if not path.exists():
            return pd.DataFrame(columns=["key", "embedding"])
        return pd.read_parquet(path)


class TextCache:
    """Parquet-backed cache for text outputs (translations, paraphrases).

    Same interface as EmbeddingCache but stores plain strings.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get_batch(self, namespace: str, keys: list[str]) -> dict[str, str]:
        """Return cached strings for *keys* that exist in cache."""
        df = self._load_df(namespace)
        if df.empty:
            return {}
        found = df[df["key"].isin(keys)]
        return {row["key"]: row["value"] for _, row in found.iterrows()}

    def set_batch(self, namespace: str, data: dict[str, str]) -> None:
        """Upsert *data* into the cache for *namespace*."""
        if not data:
            return
        existing = self._load_df(namespace)
        new_rows = pd.DataFrame(
            [{"key": k, "value": v} for k, v in data.items()]
        )
        merged = pd.concat(
            [existing[~existing["key"].isin(data)], new_rows], ignore_index=True
        )
        merged.to_parquet(self._path(namespace), index=False)

    @staticmethod
    def make_key(*parts: str) -> str:
        return sha256_string(*parts)

    def _path(self, namespace: str) -> Path:
        safe = namespace.replace("/", "__").replace("\\", "__")
        return self._cache_dir / f"{safe}.parquet"

    def _load_df(self, namespace: str) -> pd.DataFrame:
        path = self._path(namespace)
        if not path.exists():
            return pd.DataFrame(columns=["key", "value"])
        return pd.read_parquet(path)
