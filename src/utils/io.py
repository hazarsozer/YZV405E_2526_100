from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_string(*parts: str) -> str:
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()
