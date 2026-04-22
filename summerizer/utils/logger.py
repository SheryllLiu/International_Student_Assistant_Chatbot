"""Write simple timestamped log messages to ``logs.txt``.

This helper keeps logging logic in one place so the other pipeline files can
just say ``write_log("some message")``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = PROJECT_ROOT / "logs.txt"


def write_log(message: str) -> None:
    """Append one timestamped message to ``logs.txt``."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")
