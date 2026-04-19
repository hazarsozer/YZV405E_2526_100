from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.io import sha256_file, sha256_string


class TestSha256String:
    def test_returns_64_hex_chars(self):
        result = sha256_string("hello")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self):
        assert sha256_string("foo") == sha256_string("foo")

    def test_different_inputs_differ(self):
        assert sha256_string("a") != sha256_string("b")

    def test_multiple_parts_joined(self):
        combined = sha256_string("a", "b")
        individual = sha256_string("a|b")
        assert combined == individual

    def test_order_matters(self):
        assert sha256_string("a", "b") != sha256_string("b", "a")


class TestSha256File:
    def test_returns_64_hex_chars(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = sha256_file(f)
        assert len(result) == 64

    def test_deterministic(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"consistent content")
        assert sha256_file(f) == sha256_file(f)

    def test_different_contents_differ(self, tmp_path: Path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"aaa")
        f2.write_bytes(b"bbb")
        assert sha256_file(f1) != sha256_file(f2)

    def test_known_hash(self, tmp_path: Path):
        # sha256("") == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert sha256_file(f) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
