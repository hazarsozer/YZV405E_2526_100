from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.utils.cache import EmbeddingCache, TextCache


@pytest.fixture()
def cache(tmp_path: Path) -> EmbeddingCache:
    return EmbeddingCache(tmp_path / "embeddings")


class TestEmbeddingCache:
    def test_cache_dir_created(self, tmp_path: Path):
        cache_dir = tmp_path / "deep" / "nested"
        EmbeddingCache(cache_dir)
        assert cache_dir.exists()

    def test_get_missing_returns_empty(self, cache: EmbeddingCache):
        result = cache.get_batch("model", ["nonexistent"])
        assert result == {}

    def test_set_and_get_roundtrip(self, cache: EmbeddingCache):
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache.set_batch("model", {"key1": emb})
        result = cache.get_batch("model", ["key1"])
        assert "key1" in result
        np.testing.assert_array_almost_equal(result["key1"], emb)

    def test_get_batch_partial_hit(self, cache: EmbeddingCache):
        emb = np.ones(4, dtype=np.float32)
        cache.set_batch("model", {"exists": emb})
        result = cache.get_batch("model", ["exists", "missing"])
        assert "exists" in result
        assert "missing" not in result

    def test_set_batch_multiple(self, cache: EmbeddingCache):
        data = {f"k{i}": np.full(3, float(i), dtype=np.float32) for i in range(5)}
        cache.set_batch("model", data)
        result = cache.get_batch("model", list(data.keys()))
        assert len(result) == 5
        for k, emb in data.items():
            np.testing.assert_array_almost_equal(result[k], emb)

    def test_upsert_overwrites_existing_key(self, cache: EmbeddingCache):
        old = np.zeros(3, dtype=np.float32)
        new = np.ones(3, dtype=np.float32)
        cache.set_batch("model", {"k": old})
        cache.set_batch("model", {"k": new})
        result = cache.get_batch("model", ["k"])
        np.testing.assert_array_almost_equal(result["k"], new)

    def test_different_models_isolated(self, cache: EmbeddingCache):
        emb = np.ones(4, dtype=np.float32)
        cache.set_batch("model_a", {"key": emb})
        result_b = cache.get_batch("model_b", ["key"])
        assert result_b == {}

    def test_model_name_with_slash(self, cache: EmbeddingCache):
        emb = np.array([0.5, 0.5], dtype=np.float32)
        cache.set_batch("BAAI/bge-m3", {"k": emb})
        result = cache.get_batch("BAAI/bge-m3", ["k"])
        np.testing.assert_array_almost_equal(result["k"], emb)

    def test_make_key_deterministic(self):
        k1 = EmbeddingCache.make_key("model", "input")
        k2 = EmbeddingCache.make_key("model", "input")
        assert k1 == k2

    def test_make_key_differs_on_different_parts(self):
        assert EmbeddingCache.make_key("a", "b") != EmbeddingCache.make_key("b", "a")

    def test_set_empty_dict_is_noop(self, cache: EmbeddingCache):
        cache.set_batch("model", {})
        result = cache.get_batch("model", ["k"])
        assert result == {}


class TestTextCache:
    @pytest.fixture()
    def cache(self, tmp_path: Path) -> TextCache:
        return TextCache(tmp_path / "text_cache")

    def test_cache_dir_created(self, tmp_path: Path):
        TextCache(tmp_path / "deep" / "path")
        assert (tmp_path / "deep" / "path").exists()

    def test_get_missing_returns_empty(self, cache: TextCache):
        assert cache.get_batch("ns", ["nope"]) == {}

    def test_set_and_get_roundtrip(self, cache: TextCache):
        cache.set_batch("translations", {"k1": "Hello world"})
        result = cache.get_batch("translations", ["k1"])
        assert result == {"k1": "Hello world"}

    def test_get_batch_partial_hit(self, cache: TextCache):
        cache.set_batch("ns", {"exists": "value"})
        result = cache.get_batch("ns", ["exists", "missing"])
        assert "exists" in result
        assert "missing" not in result

    def test_set_batch_multiple(self, cache: TextCache):
        data = {f"k{i}": f"text {i}" for i in range(5)}
        cache.set_batch("ns", data)
        result = cache.get_batch("ns", list(data.keys()))
        assert result == data

    def test_upsert_overwrites_existing_key(self, cache: TextCache):
        cache.set_batch("ns", {"k": "old"})
        cache.set_batch("ns", {"k": "new"})
        result = cache.get_batch("ns", ["k"])
        assert result["k"] == "new"

    def test_different_namespaces_isolated(self, cache: TextCache):
        cache.set_batch("translations", {"k": "hello"})
        assert cache.get_batch("paraphrases", ["k"]) == {}

    def test_namespace_with_slash(self, cache: TextCache):
        cache.set_batch("facebook/nllb-200", {"k": "text"})
        result = cache.get_batch("facebook/nllb-200", ["k"])
        assert result["k"] == "text"

    def test_make_key_deterministic(self):
        assert TextCache.make_key("a", "b") == TextCache.make_key("a", "b")

    def test_make_key_order_sensitive(self):
        assert TextCache.make_key("a", "b") != TextCache.make_key("b", "a")

    def test_set_empty_dict_is_noop(self, cache: TextCache):
        cache.set_batch("ns", {})
        assert cache.get_batch("ns", ["k"]) == {}
