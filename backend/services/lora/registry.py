"""LoRA Registry — YAML-backed inventory of available LoRAs."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import yaml

from backend.services.lora.types import LoRAEntry, LoRAValidation

logger = logging.getLogger("beat_studio.lora.registry")


class LoRARegistry:
    """Central registry for all LoRAs, backed by a YAML file.

    Usage::

        reg = LoRARegistry("backend/config/loras.yaml", base_path="output/loras")
        entries = reg.list_all(type_filter="style")
        reg.register(LoRAEntry(...))
        result = reg.validate("my_lora")
    """

    def __init__(self, registry_path: str, base_path: str = "output/loras"):
        self._path = Path(registry_path)
        self._base = Path(base_path)
        self._entries: List[LoRAEntry] = []
        self._load()

    # ── public ────────────────────────────────────────────────────────────────

    def list_all(self, type_filter: Optional[str] = None) -> List[LoRAEntry]:
        """Return all registered LoRAs, optionally filtered by type."""
        if type_filter is None:
            return list(self._entries)
        return [e for e in self._entries if e.type == type_filter]

    def get(self, name: str) -> Optional[LoRAEntry]:
        """Return a LoRAEntry by name, or None if not found."""
        for e in self._entries:
            if e.name == name:
                return e
        return None

    def register(self, entry: LoRAEntry) -> None:
        """Add or update a LoRA entry and persist to disk."""
        for i, existing in enumerate(self._entries):
            if existing.name == entry.name:
                self._entries[i] = entry
                self._save()
                return
        self._entries.append(entry)
        self._save()

    def validate(self, name: str) -> LoRAValidation:
        """Check that a LoRA is registered and its file exists on disk."""
        entry = self.get(name)
        if entry is None:
            return LoRAValidation(
                name=name, valid=False, file_exists=False,
                error=f"LoRA '{name}' not found in registry",
            )
        file_path = self._base / entry.file_path
        exists = file_path.exists()
        if not exists:
            return LoRAValidation(
                name=name, valid=False, file_exists=False,
                error=f"File not found: {file_path}",
            )
        return LoRAValidation(name=name, valid=True, file_exists=True)

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning("Registry file not found: %s — starting empty", self._path)
            return
        with open(self._path) as f:
            data = yaml.safe_load(f) or {}
        raw_entries = data.get("loras", []) or []
        self._entries = [self._dict_to_entry(d) for d in raw_entries]
        logger.debug("Loaded %d LoRAs from %s", len(self._entries), self._path)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"loras": [self._entry_to_dict(e) for e in self._entries]}
        with open(self._path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def _dict_to_entry(d: dict) -> LoRAEntry:
        return LoRAEntry(
            name=d["name"],
            type=d["type"],
            trigger_token=d["trigger_token"],
            file_path=d["file_path"],
            weight=float(d.get("weight", 0.8)),
            status=d.get("status", "available"),
            description=d.get("description", ""),
            tags=d.get("tags", []) or [],
            source=d.get("source", "local"),
            source_url=d.get("source_url"),
        )

    @staticmethod
    def _entry_to_dict(e: LoRAEntry) -> dict:
        return {
            "name": e.name,
            "type": e.type,
            "trigger_token": e.trigger_token,
            "file_path": e.file_path,
            "weight": e.weight,
            "status": e.status,
            "description": e.description,
            "tags": e.tags,
            "source": e.source,
            "source_url": e.source_url,
        }
