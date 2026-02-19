"""Tests for the unified ConfigManager."""
import os
from pathlib import Path

import pytest
import yaml

from backend.services.shared.config import Config, get_config, reset_config


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_config()
    yield
    reset_config()


class TestConfigDotNotation:
    def test_get_top_level_key(self, sample_settings):
        cfg = Config(str(sample_settings))
        assert cfg.get("audio") is not None

    def test_get_nested_key(self, sample_settings):
        cfg = Config(str(sample_settings))
        assert cfg.get("audio.sample_rate") == 44100

    def test_get_deeply_nested(self, sample_settings):
        cfg = Config(str(sample_settings))
        assert cfg.get("mashup.weight_bpm") == 0.35

    def test_get_missing_key_returns_default(self, sample_settings):
        cfg = Config(str(sample_settings))
        assert cfg.get("nonexistent.key") is None
        assert cfg.get("nonexistent.key", "fallback") == "fallback"

    def test_get_partial_missing_returns_default(self, sample_settings):
        cfg = Config(str(sample_settings))
        assert cfg.get("audio.nonexistent", 99) == 99


class TestConfigPaths:
    def test_get_path_returns_path_object(self, sample_settings):
        cfg = Config(str(sample_settings))
        result = cfg.get_path("paths.uploads")
        assert isinstance(result, Path)

    def test_get_path_string_value(self, sample_settings):
        cfg = Config(str(sample_settings))
        result = cfg.get_path("paths.uploads")
        assert "uploads" in str(result)

    def test_get_path_missing_key_raises(self, sample_settings):
        cfg = Config(str(sample_settings))
        with pytest.raises(KeyError):
            cfg.get_path("paths.nonexistent_path_key_xyz")


class TestConfigEnvOverride:
    def test_env_overrides_yaml(self, sample_settings, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-abc")
        cfg = Config(str(sample_settings))
        assert cfg.get_env("ANTHROPIC_API_KEY") == "test-key-abc"

    def test_get_env_missing_returns_none(self, sample_settings):
        cfg = Config(str(sample_settings))
        assert cfg.get_env("NONEXISTENT_ENV_VAR_XYZ") is None

    def test_get_env_with_default(self, sample_settings):
        cfg = Config(str(sample_settings))
        assert cfg.get_env("NONEXISTENT_ENV_VAR_XYZ", "default") == "default"


class TestConfigSingleton:
    def test_get_config_returns_same_instance(self, sample_settings):
        cfg1 = get_config(str(sample_settings))
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_reset_clears_singleton(self, sample_settings):
        cfg1 = get_config(str(sample_settings))
        reset_config()
        cfg2 = get_config(str(sample_settings))
        assert cfg1 is not cfg2

    def test_get_config_without_path_raises_if_not_initialized(self):
        with pytest.raises(RuntimeError):
            get_config()


class TestConfigValidation:
    def test_missing_file_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            Config(str(tmp_dir / "nonexistent.yaml"))

    def test_invalid_yaml_raises(self, tmp_dir):
        bad = tmp_dir / "bad.yaml"
        bad.write_text("key: [unclosed bracket\n")
        with pytest.raises(Exception):
            Config(str(bad))
