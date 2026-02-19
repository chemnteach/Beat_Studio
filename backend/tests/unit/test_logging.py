"""Tests for structured logging."""
import logging
import json
from pathlib import Path

import pytest

from backend.services.shared.logging import get_logger, setup_logging


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = get_logger("beat_studio.audio")
        assert logger.name == "beat_studio.audio"

    def test_same_name_returns_same_logger(self):
        l1 = get_logger("test.same")
        l2 = get_logger("test.same")
        assert l1 is l2


class TestSetupLogging:
    def test_setup_with_level(self):
        setup_logging(level="DEBUG")
        root = logging.getLogger("beat_studio")
        assert root.level == logging.DEBUG

    def test_setup_file_handler(self, tmp_dir):
        log_file = tmp_dir / "test.log"
        setup_logging(level="INFO", log_file=str(log_file))
        logger = get_logger("beat_studio.test_file")
        logger.info("test message")
        assert log_file.exists()
        content = log_file.read_text()
        assert "test message" in content

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            setup_logging(level="INVALID_LEVEL")
