"""Unified configuration manager for Beat Studio.

Merges AI_Mixer's dot-notation singleton pattern with BeatCanvas's .env
priority chain:
  1. Global ~/.claude/.env  (lowest priority)
  2. Local backend/.env     (overrides global)
  3. Environment variables  (highest priority)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

_config_instance: Optional["Config"] = None


class Config:
    """Unified configuration with dot-notation access and env override."""

    def __init__(self, config_path: str):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path) as f:
            self._data: dict = yaml.safe_load(f)
        if not isinstance(self._data, dict):
            raise ValueError(f"Config file must be a YAML mapping, got: {type(self._data)}")
        self._load_env()

    # ── private ──────────────────────────────────────────────────────────────

    def _load_env(self) -> None:
        """Load .env files in priority order (global → local)."""
        global_env = Path.home() / ".claude" / ".env"
        local_env = Path(__file__).parent.parent.parent / ".env"
        if global_env.exists():
            load_dotenv(global_env, override=False)
        if local_env.exists():
            load_dotenv(local_env, override=True)

    # ── public ───────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation access into the YAML tree.

        Example::

            config.get("mashup.weight_bpm")        # 0.35
            config.get("video.backends.svd.fps")    # 8
            config.get("missing.key", "fallback")   # "fallback"
        """
        keys = key.split(".")
        val: Any = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def get_path(self, key: str) -> Path:
        """Return a config value as a Path object.

        Raises KeyError if the key does not exist.
        """
        val = self.get(key)
        if val is None:
            raise KeyError(f"Config key not found: {key}")
        return Path(str(val))

    def get_env(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """Return an environment variable value."""
        return os.environ.get(name, default)


# ── module-level singleton ────────────────────────────────────────────────────


def get_config(config_path: Optional[str] = None) -> Config:
    """Return the singleton Config instance.

    On first call, ``config_path`` is required.  Subsequent calls may omit it
    and will return the existing instance.

    Raises:
        RuntimeError: If called before the singleton is initialised.
    """
    global _config_instance
    if _config_instance is None:
        if config_path is None:
            raise RuntimeError(
                "Config not yet initialised — call get_config(config_path) first."
            )
        _config_instance = Config(config_path)
    return _config_instance


def reset_config() -> None:
    """Clear the singleton (mainly for testing)."""
    global _config_instance
    _config_instance = None
