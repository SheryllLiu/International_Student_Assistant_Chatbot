"""One-call logging setup for the package.

Keep this intentionally small: a single ``setup_logging`` that callers
(scripts, the CLI, notebooks) can invoke once. Library modules should never
configure logging themselves — they just call ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once with a readable format.

    Safe to call multiple times; re-running replaces existing handlers so
    the format does not get duplicated when the CLI and a script both call it.
    """

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
