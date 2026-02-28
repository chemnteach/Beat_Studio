"""Tests for storyboard snapshot creation/restore and ZIP upload.

TDD: these tests define the expected behaviour for:
  - VersionEntry.source field (round-trip, backward compat)
  - StoryboardStateStore.create_snapshot / list_snapshots
  - StoryboardService.restore_snapshot
  - StoryboardService.upload_zip (including validation errors)
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# PIL is a core dependency; import lazily so tests don't fail during collection
# if Pillow is not installed in a lightweight CI environment.
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from backend.services.storyboard.service import StoryboardService
from backend.services.storyboard.state import StoryboardStateStore
from backend.services.storyboard.types import (
    SceneInput,
    StoryboardScene,
    StoryboardState,
    VersionEntry,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_png_bytes(width: int = 8, height: int = 8) -> bytes:
    """Return a minimal valid PNG as bytes."""
    if not _PIL_AVAILABLE:
        pytest.skip("Pillow not installed")
    img = Image.new("RGB", (width, height), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_zip(scenes: dict[int, bytes]) -> bytes:
    """Create a ZIP with scene_NN.png entries. scenes = {1: bytes, 2: bytes, ...}"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for scene_num, img_bytes in scenes.items():
            zf.writestr(f"scene_{scene_num:02d}.png", img_bytes)
    buf.seek(0)
    return buf.read()


def make_state(storyboard_id: str, num_scenes: int) -> StoryboardState:
    scenes = [
        StoryboardScene(
            scene_idx=i,
            storyboard_prompt=f"Scene {i} storyboard prompt",
            positive_prompt=f"scene {i} positive prompt",
            approved_version=None,
        )
        for i in range(num_scenes)
    ]
    return StoryboardState(
        storyboard_id=storyboard_id,
        style="cinematic",
        base_checkpoint="stabilityai/stable-diffusion-xl-base-1.0",
        lora_names=[],
        status="complete",
        scenes=scenes,
    )


def populate_scene_images(
    store: StoryboardStateStore,
    storyboard_id: str,
    num_scenes: int,
    versions_per_scene: int = 1,
) -> None:
    """Write fake PNG files and version entries for all scenes."""
    for i in range(num_scenes):
        scene_dir = store.scene_dir(storyboard_id, i, create=True)
        for v in range(1, versions_per_scene + 1):
            (scene_dir / f"v{v}.png").write_bytes(b"fake_png_data")
            store.append_version(
                storyboard_id, i,
                VersionEntry(version=v, filename=f"v{v}.png", seed=v * 100, timestamp=f"ts{v}"),
            )


def make_service(store: StoryboardStateStore) -> StoryboardService:
    """Create a StoryboardService with a real store and mocked VRAM manager."""
    mock_vm = MagicMock()
    return StoryboardService(state_store=store, vram_manager=mock_vm)


# ── VersionEntry.source field ──────────────────────────────────────────────────

class TestVersionEntrySourceField:
    def test_default_source_is_generated(self):
        entry = VersionEntry(version=1, filename="v1.png", seed=42, timestamp="ts")
        assert entry.source == "generated"

    def test_source_field_round_trips_through_state(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 1)
        store.create(state)
        scene_dir = store.scene_dir("sb1", 0, create=True)
        (scene_dir / "v1.png").write_bytes(b"fake")
        store.append_version(
            "sb1", 0,
            VersionEntry(version=1, filename="v1.png", seed=None, timestamp="ts", source="upload"),
        )
        loaded = store.load("sb1")
        assert loaded.scenes[0].versions[0].source == "upload"

    def test_backward_compat_missing_source_defaults_to_generated(self, tmp_path):
        """State JSON written before this feature (no 'source' key) reads as 'generated'."""
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 1)
        store.create(state)
        scene_dir = store.scene_dir("sb1", 0, create=True)
        (scene_dir / "v1.png").write_bytes(b"fake")
        store.append_version("sb1", 0, VersionEntry(1, "v1.png", 42, "ts"))

        # Manually remove 'source' from state.json (simulate old format)
        state_path = tmp_path / "sb1" / "state.json"
        data = json.loads(state_path.read_text())
        for s in data["scenes"]:
            for v in s["versions"]:
                v.pop("source", None)
        state_path.write_text(json.dumps(data))

        loaded = store.load("sb1")
        assert loaded.scenes[0].versions[0].source == "generated"


# ── StoryboardStateStore snapshot methods ─────────────────────────────────────

class TestCreateSnapshot:
    def test_snapshot_creates_directory_with_current_images(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 2)
        store.create(state)
        populate_scene_images(store, "sb1", 2)

        result = store.create_snapshot("sb1")

        snap_dir = tmp_path / "sb1" / "snapshots" / result["snapshot_id"]
        assert snap_dir.exists()
        assert (snap_dir / "scene_00.png").exists()
        assert (snap_dir / "scene_01.png").exists()
        assert result["scene_count"] == 2

    def test_snapshot_copies_state_json(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 1)
        store.create(state)
        populate_scene_images(store, "sb1", 1)

        result = store.create_snapshot("sb1")

        snap_dir = tmp_path / "sb1" / "snapshots" / result["snapshot_id"]
        assert (snap_dir / "state.json").exists()
        saved = json.loads((snap_dir / "state.json").read_text())
        assert saved["storyboard_id"] == "sb1"

    def test_list_snapshots_returns_all_snapshots_sorted_newest_first(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 1)
        store.create(state)
        populate_scene_images(store, "sb1", 1)

        result1 = store.create_snapshot("sb1")
        result2 = store.create_snapshot("sb1")

        snapshots = store.list_snapshots("sb1")
        assert len(snapshots) == 2
        # Newest first (timestamps sort lexicographically)
        assert snapshots[0]["snapshot_id"] >= snapshots[1]["snapshot_id"]
        assert all("scene_count" in s for s in snapshots)
        assert all("timestamp" in s for s in snapshots)

    def test_list_snapshots_empty_when_none_exist(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 1)
        store.create(state)

        assert store.list_snapshots("sb1") == []


# ── StoryboardService.restore_snapshot ────────────────────────────────────────

class TestRestoreSnapshot:
    def test_restore_snapshot_creates_backup_first(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 1)
        store.create(state)
        populate_scene_images(store, "sb1", 1)

        svc = make_service(store)
        snap_result = store.create_snapshot("sb1")

        result = svc.restore_snapshot("sb1", snap_result["snapshot_id"])

        assert "backup_snapshot_id" in result
        assert result["backup_snapshot_id"] != snap_result["snapshot_id"]
        backup_dir = tmp_path / "sb1" / "snapshots" / result["backup_snapshot_id"]
        assert backup_dir.exists()

    def test_restore_snapshot_adds_new_versions_with_upload_source(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 1)
        store.create(state)
        populate_scene_images(store, "sb1", 1)

        svc = make_service(store)
        snap_result = store.create_snapshot("sb1")

        result = svc.restore_snapshot("sb1", snap_result["snapshot_id"])

        loaded = store.load("sb1")
        # Original v1 + new restored version
        assert len(loaded.scenes[0].versions) == 2
        assert loaded.scenes[0].versions[-1].source == "upload"
        assert result["restored_from"] == snap_result["snapshot_id"]

    def test_restore_snapshot_raises_for_unknown_snapshot(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 1)
        store.create(state)

        svc = make_service(store)

        with pytest.raises(LookupError, match="[Ss]napshot"):
            svc.restore_snapshot("sb1", "nonexistent_snapshot")


# ── StoryboardService.upload_zip ───────────────────────────────────────────────

class TestUploadZip:
    def test_upload_zip_creates_snapshot_before_applying(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 2)
        store.create(state)
        populate_scene_images(store, "sb1", 2)

        svc = make_service(store)
        assert len(store.list_snapshots("sb1")) == 0

        zip_bytes = make_zip({1: make_png_bytes(), 2: make_png_bytes()})
        result = svc.upload_zip("sb1", zip_bytes)

        assert "snapshot_id" in result
        assert len(store.list_snapshots("sb1")) == 1

    def test_upload_zip_adds_versions_with_source_upload(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 2)
        store.create(state)
        populate_scene_images(store, "sb1", 2)

        svc = make_service(store)
        zip_bytes = make_zip({1: make_png_bytes(), 2: make_png_bytes()})
        result = svc.upload_zip("sb1", zip_bytes)

        assert result["uploaded"] == 2
        loaded = store.load("sb1")
        for scene in loaded.scenes:
            latest = scene.versions[-1]
            assert latest.source == "upload"
            assert latest.seed is None

    def test_upload_zip_scene_count_mismatch_raises_value_error(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 3)
        store.create(state)
        populate_scene_images(store, "sb1", 3)

        svc = make_service(store)
        # ZIP has only 2 scenes, storyboard expects 3
        zip_bytes = make_zip({1: make_png_bytes(), 2: make_png_bytes()})

        with pytest.raises(ValueError, match=r"2.*3|3.*2"):
            svc.upload_zip("sb1", zip_bytes)

    def test_upload_zip_missing_scene_file_raises_value_error(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 3)
        store.create(state)
        populate_scene_images(store, "sb1", 3)

        svc = make_service(store)
        # 3 files, but scene_03 is missing (has scene_04 instead)
        zip_bytes = make_zip({1: make_png_bytes(), 2: make_png_bytes(), 4: make_png_bytes()})

        with pytest.raises(ValueError, match="[Mm]issing"):
            svc.upload_zip("sb1", zip_bytes)

    def test_upload_zip_invalid_png_raises_value_error(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 2)
        store.create(state)
        populate_scene_images(store, "sb1", 2)

        svc = make_service(store)
        zip_bytes = make_zip({1: make_png_bytes(), 2: b"this is not a png"})

        with pytest.raises(ValueError, match=r"[Pp][Nn][Gg]|[Ii]nvalid"):
            svc.upload_zip("sb1", zip_bytes)

    def test_upload_zip_non_zip_file_raises_value_error(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 2)
        store.create(state)
        populate_scene_images(store, "sb1", 2)

        svc = make_service(store)

        with pytest.raises(ValueError, match=r"[Zz][Ii][Pp]"):
            svc.upload_zip("sb1", b"this is not a zip file at all")

    def test_upload_zip_ignores_macosx_directory(self, tmp_path):
        """__MACOSX/ entries from macOS zipping should be ignored, not counted as scenes."""
        store = StoryboardStateStore(base_dir=tmp_path)
        state = make_state("sb1", 2)
        store.create(state)
        populate_scene_images(store, "sb1", 2)

        svc = make_service(store)

        # Add a __MACOSX entry that looks like a scene file
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w") as zf:
            zf.writestr("scene_01.png", make_png_bytes())
            zf.writestr("scene_02.png", make_png_bytes())
            zf.writestr("__MACOSX/scene_01.png", b"macos junk")
        buf.seek(0)
        zip_bytes = buf.read()

        # Should succeed — __MACOSX/scene_01.png is ignored
        result = svc.upload_zip("sb1", zip_bytes)
        assert result["uploaded"] == 2
