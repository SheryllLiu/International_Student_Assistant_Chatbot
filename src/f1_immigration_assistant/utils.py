"""Small shared helpers. Do not turn this into a junk drawer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_id(*parts: str) -> str:
    """Deterministic short id built from the given parts."""

    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return h[:12]


def read_json(path: Path) -> Any:
    """Read a JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """Write a JSON file, creating parent directories if needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_files(directory: Path, suffix: str) -> list[Path]:
    """Return files with ``suffix`` directly inside ``directory``, sorted."""

    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix == suffix)
