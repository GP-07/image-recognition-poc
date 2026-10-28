"""Content-addressed artifact store.

Each stage output is saved as JSON under ``<root>/<stage>/<key[:2]>/<key>.json``. The key hashes
the stage input and the stage fingerprint, so a re-run with the same input and settings reuses
the stored output (e.g. re-running only the LLM with another model reuses the OCR).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def make_key(*parts: Any) -> str:
    """Stable sha256 over JSON-serialisable parts (dict keys sorted)."""
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, stage: str, key: str) -> Path:
        return self.root / stage / key[:2] / f"{key}.json"

    def get(self, stage: str, key: str) -> dict[str, Any] | None:
        path = self._path(stage, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, stage: str, key: str, data: dict[str, Any]) -> Path:
        path = self._path(stage, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file and rename, so an interrupted run never leaves a half-written entry.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return path
