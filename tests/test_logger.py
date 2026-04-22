from pathlib import Path

from summerizer.utils.logger import write_log


def test_write_log_appends_timestamped_message(tmp_path: Path):
    log_path = tmp_path / "logs.txt"

    # Reuse the same logging behavior with a local file path.
    from summerizer.utils import logger as logger_module

    original_log_path = logger_module.LOG_PATH
    logger_module.LOG_PATH = log_path
    try:
        write_log("test message")
    finally:
        logger_module.LOG_PATH = original_log_path

    contents = log_path.read_text(encoding="utf-8")
    assert "test message" in contents
    assert contents.startswith("[")
