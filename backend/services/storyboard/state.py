"""StoryboardStateStore — JSON-backed persistence for storyboard keyframe sessions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from backend.services.storyboard.types import (
    StoryboardScene,
    StoryboardState,
    VersionEntry,
)

logger = logging.getLogger("beat_studio.storyboard.state")

_DEFAULT_BASE_DIR = Path(__file__).parent.parent.parent.parent / "output" / "storyboard"


class StoryboardStateStore:
    """Read/write storyboard session state from disk.

    Each session lives under ``{base_dir}/{storyboard_id}/state.json``.
    Images live under ``{base_dir}/{storyboard_id}/scene_{idx}/v{n}.png``.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base = Path(base_dir) if base_dir else _DEFAULT_BASE_DIR

    # ── public ────────────────────────────────────────────────────────────────

    def create(self, state: StoryboardState) -> None:
        """Persist a fresh StoryboardState to disk. Overwrites if exists."""
        session_dir = self._base / state.storyboard_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._write(state)

    def load(self, storyboard_id: str) -> Optional[StoryboardState]:
        """Return the StoryboardState for *storyboard_id*, or None if not found."""
        path = self._state_path(storyboard_id)
        if not path.exists():
            return None
        return self._read(path)

    def append_version(
        self,
        storyboard_id: str,
        scene_idx: int,
        entry: VersionEntry,
    ) -> None:
        """Add a new version entry to *scene_idx*, evicting the oldest if >MAX_VERSIONS.

        The physical PNG must already exist at ``scene_dir(storyboard_id, scene_idx) / entry.filename``
        before calling this method. Eviction deletes the oldest PNG from disk.
        """
        state = self._require(storyboard_id)
        scene = self._get_scene(state, scene_idx)

        scene.versions.append(entry)

        if len(scene.versions) > scene.MAX_VERSIONS:
            oldest = scene.versions.pop(0)
            old_file = self.scene_dir(storyboard_id, scene_idx) / oldest.filename
            if old_file.exists():
                old_file.unlink()
                logger.debug("Evicted storyboard version: %s", old_file)

        self._write(state)

    def update_status(
        self,
        storyboard_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Update the top-level status (and optional error message)."""
        state = self._require(storyboard_id)
        state.status = status
        if error is not None:
            state.error = error
        self._write(state)

    def set_approved(
        self,
        storyboard_id: str,
        selections: Dict[int, int],
    ) -> Dict[int, str]:
        """Record approved version per scene and return {scene_idx: abs_path}.

        Args:
            storyboard_id: The storyboard session ID.
            selections: ``{scene_idx: version_number}`` — 1-indexed version.

        Returns:
            ``{scene_idx: absolute_path_to_approved_png}``

        Raises:
            LookupError: If storyboard_id is not found.
            ValueError: If a scene_idx in *selections* does not exist.
        """
        state = self._require(storyboard_id)
        approved_paths: Dict[int, str] = {}

        for scene_idx, version_num in selections.items():
            scene = self._get_scene(state, scene_idx)
            scene.approved_version = version_num
            approved_paths[scene_idx] = str(
                self.scene_dir(storyboard_id, scene_idx) / f"v{version_num}.png"
            )

        self._write(state)
        return approved_paths

    def scene_dir(
        self,
        storyboard_id: str,
        scene_idx: int,
        create: bool = False,
    ) -> Path:
        """Return (and optionally create) the directory for scene images."""
        d = self._base / storyboard_id / f"scene_{scene_idx}"
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    # ── internal ──────────────────────────────────────────────────────────────

    def _state_path(self, storyboard_id: str) -> Path:
        return self._base / storyboard_id / "state.json"

    def _require(self, storyboard_id: str) -> StoryboardState:
        state = self.load(storyboard_id)
        if state is None:
            raise LookupError(
                f"Storyboard '{storyboard_id}' not found — call create() first."
            )
        return state

    def _get_scene(self, state: StoryboardState, scene_idx: int) -> StoryboardScene:
        for scene in state.scenes:
            if scene.scene_idx == scene_idx:
                return scene
        raise ValueError(
            f"scene_idx {scene_idx} not found in storyboard '{state.storyboard_id}'. "
            f"Available: {[s.scene_idx for s in state.scenes]}"
        )

    def _write(self, state: StoryboardState) -> None:
        path = self._state_path(state.storyboard_id)
        data = {
            "storyboard_id": state.storyboard_id,
            "style": state.style,
            "base_checkpoint": state.base_checkpoint,
            "lora_names": state.lora_names,
            "status": state.status,
            "error": state.error,
            "scenes": [
                {
                    "scene_idx": s.scene_idx,
                    "storyboard_prompt": s.storyboard_prompt,
                    "positive_prompt": s.positive_prompt,
                    "approved_version": s.approved_version,
                    "versions": [
                        {
                            "version": v.version,
                            "filename": v.filename,
                            "seed": v.seed,
                            "timestamp": v.timestamp,
                            "lora_weights": v.lora_weights,
                        }
                        for v in s.versions
                    ],
                }
                for s in state.scenes
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    def _read(self, path: Path) -> StoryboardState:
        data = json.loads(path.read_text())
        scenes = [
            StoryboardScene(
                scene_idx=s["scene_idx"],
                storyboard_prompt=s["storyboard_prompt"],
                positive_prompt=s["positive_prompt"],
                approved_version=s.get("approved_version"),
                versions=[
                    VersionEntry(
                        version=v["version"],
                        filename=v["filename"],
                        seed=v["seed"],
                        timestamp=v["timestamp"],
                        lora_weights=v.get("lora_weights", {}),
                    )
                    for v in s.get("versions", [])
                ],
            )
            for s in data.get("scenes", [])
        ]
        return StoryboardState(
            storyboard_id=data["storyboard_id"],
            style=data["style"],
            base_checkpoint=data["base_checkpoint"],
            lora_names=data.get("lora_names", []),
            status=data["status"],
            error=data.get("error"),
            scenes=scenes,
        )
