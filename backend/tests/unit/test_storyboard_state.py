"""TDD tests for StoryboardStateStore — written before the implementation."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from backend.services.storyboard.state import StoryboardStateStore
from backend.services.storyboard.types import StoryboardScene, StoryboardState, VersionEntry


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_state(storyboard_id: str = "test-id-123", num_scenes: int = 2) -> StoryboardState:
    scenes = [
        StoryboardScene(
            scene_idx=i,
            storyboard_prompt=f"Scene {i} cinematic description",
            positive_prompt=f"cinematic, scene {i}",
            approved_version=None,
        )
        for i in range(num_scenes)
    ]
    return StoryboardState(
        storyboard_id=storyboard_id,
        style="cinematic",
        base_checkpoint="emilianJR/epiCRealism",
        lora_names=["nova_fade_id_v1"],
        status="generating",
        scenes=scenes,
    )


def _make_version(version: int, seed: int) -> VersionEntry:
    return VersionEntry(
        version=version,
        filename=f"v{version}.png",
        seed=seed,
        timestamp=_ts(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Create and load
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateAndLoad:
    def test_create_writes_state_json(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = _make_state("abc-123")
        store.create(state)
        assert (tmp_path / "abc-123" / "state.json").exists()

    def test_load_returns_none_for_unknown_id(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        assert store.load("does-not-exist") is None

    def test_load_roundtrip_preserves_fields(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        original = _make_state("roundtrip-id", num_scenes=3)
        store.create(original)

        loaded = store.load("roundtrip-id")
        assert loaded is not None
        assert loaded.storyboard_id == "roundtrip-id"
        assert loaded.style == "cinematic"
        assert loaded.base_checkpoint == "emilianJR/epiCRealism"
        assert loaded.lora_names == ["nova_fade_id_v1"]
        assert loaded.status == "generating"
        assert len(loaded.scenes) == 3

    def test_load_roundtrip_preserves_scene_fields(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = _make_state("scene-check", num_scenes=1)
        state.scenes[0].storyboard_prompt = "Ocean at dusk, dramatic waves"
        state.scenes[0].positive_prompt = "cinematic film still, ocean at dusk"
        store.create(state)

        loaded = store.load("scene-check")
        scene = loaded.scenes[0]
        assert scene.scene_idx == 0
        assert scene.storyboard_prompt == "Ocean at dusk, dramatic waves"
        assert scene.positive_prompt == "cinematic film still, ocean at dusk"
        assert scene.approved_version is None
        assert scene.versions == []


# ═══════════════════════════════════════════════════════════════════════════════
# Version append and eviction
# ═══════════════════════════════════════════════════════════════════════════════


class TestVersionHistory:
    def test_append_version_adds_entry(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        store.create(_make_state("v-append"))

        store.append_version("v-append", scene_idx=0, entry=_make_version(1, seed=0))
        loaded = store.load("v-append")
        assert len(loaded.scenes[0].versions) == 1
        assert loaded.scenes[0].versions[0].version == 1
        assert loaded.scenes[0].versions[0].seed == 0

    def test_append_multiple_versions_in_order(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        store.create(_make_state("v-multi"))

        for i in range(1, 4):
            store.append_version("v-multi", scene_idx=0, entry=_make_version(i, seed=i * 137))

        loaded = store.load("v-multi")
        versions = loaded.scenes[0].versions
        assert len(versions) == 3
        assert [v.version for v in versions] == [1, 2, 3]

    def test_eviction_at_max_drops_oldest(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        state = _make_state("v-evict")
        store.create(state)

        # Add MAX_VERSIONS (5) versions
        for i in range(1, 6):
            # Create the PNG so eviction can delete it
            scene_dir = tmp_path / "v-evict" / "scene_0"
            scene_dir.mkdir(parents=True, exist_ok=True)
            (scene_dir / f"v{i}.png").write_bytes(b"fake png")
            store.append_version("v-evict", scene_idx=0, entry=_make_version(i, seed=i))

        # Add a 6th — should evict v1
        scene_dir = tmp_path / "v-evict" / "scene_0"
        (scene_dir / "v6.png").write_bytes(b"fake png")
        store.append_version("v-evict", scene_idx=0, entry=_make_version(6, seed=6))

        loaded = store.load("v-evict")
        versions = loaded.scenes[0].versions
        assert len(versions) == 5
        assert versions[0].version == 2, "v1 should have been evicted"
        assert versions[-1].version == 6

    def test_eviction_deletes_oldest_file_from_disk(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        store.create(_make_state("v-del"))

        scene_dir = tmp_path / "v-del" / "scene_0"
        scene_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, 7):
            (scene_dir / f"v{i}.png").write_bytes(b"fake png")
            store.append_version("v-del", scene_idx=0, entry=_make_version(i, seed=i))

        assert not (scene_dir / "v1.png").exists(), "v1.png should be deleted after eviction"
        assert (scene_dir / "v2.png").exists(), "v2.png should still exist"

    def test_append_only_affects_specified_scene(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        store.create(_make_state("v-isolate", num_scenes=3))

        store.append_version("v-isolate", scene_idx=1, entry=_make_version(1, seed=42))

        loaded = store.load("v-isolate")
        assert len(loaded.scenes[0].versions) == 0
        assert len(loaded.scenes[1].versions) == 1
        assert len(loaded.scenes[2].versions) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Status update
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatusUpdate:
    def test_update_status_persists(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        store.create(_make_state("status-test"))

        store.update_status("status-test", status="complete")
        loaded = store.load("status-test")
        assert loaded.status == "complete"

    def test_update_status_with_error(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        store.create(_make_state("err-test"))

        store.update_status("err-test", status="failed", error="CUDA OOM")
        loaded = store.load("err-test")
        assert loaded.status == "failed"
        assert loaded.error == "CUDA OOM"


# ═══════════════════════════════════════════════════════════════════════════════
# Approval
# ═══════════════════════════════════════════════════════════════════════════════


class TestApproval:
    def _make_state_with_versions(self, tmp_path: Path, storyboard_id: str) -> StoryboardState:
        store = StoryboardStateStore(base_dir=tmp_path)
        state = _make_state(storyboard_id, num_scenes=3)
        store.create(state)

        for scene_idx in range(3):
            scene_dir = tmp_path / storyboard_id / f"scene_{scene_idx}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            for ver in range(1, 3):
                (scene_dir / f"v{ver}.png").write_bytes(b"fake png")
                store.append_version(
                    storyboard_id, scene_idx=scene_idx,
                    entry=_make_version(ver, seed=ver)
                )
        return store

    def test_set_approved_records_version_per_scene(self, tmp_path):
        store = self._make_state_with_versions(tmp_path, "approve-basic")
        store.set_approved("approve-basic", selections={0: 1, 1: 2, 2: 1})

        loaded = store.load("approve-basic")
        assert loaded.scenes[0].approved_version == 1
        assert loaded.scenes[1].approved_version == 2
        assert loaded.scenes[2].approved_version == 1

    def test_set_approved_returns_correct_abs_paths(self, tmp_path):
        store = self._make_state_with_versions(tmp_path, "approve-paths")
        paths = store.set_approved("approve-paths", selections={0: 2, 1: 1, 2: 2})

        assert paths[0] == str(tmp_path / "approve-paths" / "scene_0" / "v2.png")
        assert paths[1] == str(tmp_path / "approve-paths" / "scene_1" / "v1.png")
        assert paths[2] == str(tmp_path / "approve-paths" / "scene_2" / "v2.png")

    def test_set_approved_raises_for_missing_scene(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        store.create(_make_state("approve-bad", num_scenes=2))

        with pytest.raises(ValueError, match="scene_idx 99"):
            store.set_approved("approve-bad", selections={99: 1})

    def test_set_approved_raises_for_unknown_storyboard(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)

        with pytest.raises(LookupError, match="does-not-exist"):
            store.set_approved("does-not-exist", selections={0: 1})

    def test_approved_paths_keyed_by_int_scene_idx(self, tmp_path):
        store = self._make_state_with_versions(tmp_path, "approve-keys")
        paths = store.set_approved("approve-keys", selections={0: 1, 1: 1, 2: 1})

        assert all(isinstance(k, int) for k in paths.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Scene directory helpers
# ═══════════════════════════════════════════════════════════════════════════════


class TestSceneDirectory:
    def test_scene_dir_returns_correct_path(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        d = store.scene_dir("my-id", scene_idx=3)
        assert d == tmp_path / "my-id" / "scene_3"

    def test_scene_dir_created_on_demand(self, tmp_path):
        store = StoryboardStateStore(base_dir=tmp_path)
        d = store.scene_dir("new-id", scene_idx=0, create=True)
        assert d.exists()
