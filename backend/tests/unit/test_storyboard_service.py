"""TDD tests for StoryboardService — written before the implementation.

All tests run without a GPU:
  - StableDiffusionXLPipeline.from_pretrained is monkeypatched.
  - StoryboardService is constructed with device="cpu" so torch.Generator("cpu")
    is used instead of torch.Generator("cuda").
  - VRAMManager is replaced with a MagicMock.
"""
from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from backend.services.storyboard.service import StoryboardService
from backend.services.storyboard.state import StoryboardStateStore
from backend.services.storyboard.types import SceneInput, VersionEntry


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_scenes(n: int = 2) -> list[SceneInput]:
    return [
        SceneInput(
            scene_idx=i,
            storyboard_prompt=f"Scene {i}: dramatic ocean at dusk",
            positive_prompt=f"cinematic film still, scene {i}, ocean",
        )
        for i in range(n)
    ]


@pytest.fixture
def mock_sdxl(monkeypatch):
    """Replace StableDiffusionXLPipeline with a mock that returns fake images."""
    mock_image = MagicMock()
    mock_image.save = MagicMock()

    mock_pipe = MagicMock()
    mock_pipe.return_value = MagicMock(images=[mock_image])
    mock_pipe.load_lora_weights = MagicMock()
    mock_pipe.set_adapters = MagicMock()

    # from_pretrained(...).to(...) → mock_pipe
    mock_cls = MagicMock()
    mock_cls.from_pretrained.return_value.to.return_value = mock_pipe

    monkeypatch.setattr(
        "backend.services.storyboard.service.StableDiffusionXLPipeline",
        mock_cls,
    )
    return mock_cls, mock_pipe, mock_image


@pytest.fixture
def mock_vm():
    """Fake VRAMManager — tracks kill() and set_pipeline() calls."""
    vm = MagicMock()
    return vm


@pytest.fixture
def store(tmp_path):
    return StoryboardStateStore(base_dir=tmp_path)


def make_service(store, mock_vm, tmp_path):
    return StoryboardService(
        state_store=store,
        vram_manager=mock_vm,
        loras_yaml=str(tmp_path / "loras.yaml"),
        lora_base=tmp_path,
        device="cpu",  # avoid CUDA requirement in tests
    )


# ═══════════════════════════════════════════════════════════════════════════════
# generate_all_scenes — pipeline lifecycle
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateAllScenesLifecycle:
    def test_pipeline_loaded_from_sdxl_base_checkpoint(self, mock_sdxl, mock_vm, store, tmp_path):
        mock_cls, mock_pipe, _ = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-1", _make_scenes(2), "cinematic", [])

        mock_cls.from_pretrained.assert_called_once()
        call_kwargs = mock_cls.from_pretrained.call_args
        assert call_kwargs[0][0] == StoryboardService.SDXL_BASE

    def test_pipeline_loaded_once_for_all_scenes(self, mock_sdxl, mock_vm, store, tmp_path):
        mock_cls, mock_pipe, _ = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-2", _make_scenes(4), "cinematic", [])

        mock_cls.from_pretrained.assert_called_once()  # 1 load for N scenes
        assert mock_pipe.call_count == 4               # but N inference calls

    def test_pipe_called_with_1024x576_resolution(self, mock_sdxl, mock_vm, store, tmp_path):
        mock_cls, mock_pipe, _ = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-3", _make_scenes(1), "cinematic", [])

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["width"] == StoryboardService.PREVIEW_WIDTH
        assert call_kwargs["height"] == StoryboardService.PREVIEW_HEIGHT

    def test_vram_set_pipeline_called(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-4", _make_scenes(1), "cinematic", [])

        mock_vm.set_pipeline.assert_called_once_with(mock_pipe, "sdxl_storyboard")

    def test_vram_killed_after_all_scenes(self, mock_sdxl, mock_vm, store, tmp_path):
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-5", _make_scenes(2), "cinematic", [])

        mock_vm.kill.assert_called_once()

    def test_vram_killed_even_on_exception(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        mock_pipe.side_effect = RuntimeError("CUDA OOM")
        svc = make_service(store, mock_vm, tmp_path)

        with pytest.raises(RuntimeError):
            svc.generate_all_scenes("id-6", _make_scenes(1), "cinematic", [])

        mock_vm.kill.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# generate_all_scenes — prompts and style
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateAllScenesPrompts:
    def test_positive_prompt_includes_style_prefix(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        scenes = [SceneInput(0, "Ocean at dusk", "ocean at dusk, waves")]
        svc.generate_all_scenes("id-p1", scenes, "cinematic", [])

        prompt_arg = mock_pipe.call_args[1]["prompt"]
        # style.prefix for "cinematic" is prepended
        assert "cinematic" in prompt_arg.lower()
        assert "ocean at dusk" in prompt_arg

    def test_negative_prompt_includes_style_negatives(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-p2", _make_scenes(1), "cinematic", [])

        neg = mock_pipe.call_args[1]["negative_prompt"]
        # cinematic style negatives: "cartoon, anime, flat..."
        assert "cartoon" in neg or "anime" in neg

    def test_uses_style_steps_and_cfg(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-p3", _make_scenes(1), "cinematic", [])

        kwargs = mock_pipe.call_args[1]
        # cinematic style: steps=30, guidance_scale=7.5
        assert kwargs["num_inference_steps"] == 30
        assert kwargs["guidance_scale"] == 7.5

    def test_deterministic_seed_per_scene(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-p4", _make_scenes(3), "cinematic", [])

        # Inspect generator seeds across all three calls
        seeds_used = []
        for c in mock_pipe.call_args_list:
            gen = c[1]["generator"]
            # generator is a real torch.Generator("cpu") — check seed via initial_seed()
            seeds_used.append(gen.initial_seed())

        assert seeds_used[0] == 0                             # scene 0 → seed 0
        assert seeds_used[1] == StoryboardService.SEED_STRIDE  # scene 1 → 137
        assert seeds_used[2] == StoryboardService.SEED_STRIDE * 2


# ═══════════════════════════════════════════════════════════════════════════════
# generate_all_scenes — state and disk
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateAllScenesState:
    def test_state_created_with_generating_status_initially(
        self, mock_sdxl, mock_vm, store, tmp_path, monkeypatch
    ):
        # Capture status at the moment pipe() is called (i.e. during generation)
        captured_status = []
        _, mock_pipe, _ = mock_sdxl

        def pipe_side_effect(*args, **kwargs):
            state = store.load("id-s1")
            captured_status.append(state.status if state else None)
            return MagicMock(images=[MagicMock(save=MagicMock())])

        mock_pipe.side_effect = pipe_side_effect
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-s1", _make_scenes(1), "cinematic", [])

        assert captured_status[0] == "generating"

    def test_state_marked_complete_after_success(self, mock_sdxl, mock_vm, store, tmp_path):
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-s2", _make_scenes(2), "cinematic", [])

        state = store.load("id-s2")
        assert state.status == "complete"

    def test_state_marked_failed_on_exception(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        mock_pipe.side_effect = RuntimeError("GPU gone")
        svc = make_service(store, mock_vm, tmp_path)

        with pytest.raises(RuntimeError):
            svc.generate_all_scenes("id-s3", _make_scenes(1), "cinematic", [])

        state = store.load("id-s3")
        assert state.status == "failed"
        assert "GPU gone" in state.error

    def test_state_stores_sdxl_base_as_checkpoint(self, mock_sdxl, mock_vm, store, tmp_path):
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-s4", _make_scenes(1), "cinematic", [])

        state = store.load("id-s4")
        assert state.base_checkpoint == StoryboardService.SDXL_BASE

    def test_each_scene_gets_v1_entry(self, mock_sdxl, mock_vm, store, tmp_path):
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-s5", _make_scenes(3), "cinematic", [])

        state = store.load("id-s5")
        for scene in state.scenes:
            assert len(scene.versions) == 1
            assert scene.versions[0].version == 1
            assert scene.versions[0].filename == "v1.png"

    def test_image_save_called_per_scene(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, mock_image = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-s6", _make_scenes(3), "cinematic", [])

        assert mock_image.save.call_count == 3

    def test_image_saved_to_correct_scene_dir(self, mock_sdxl, mock_vm, store, tmp_path):
        _, _, mock_image = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-s7", _make_scenes(2), "cinematic", [])

        saved_paths = [str(c[0][0]) for c in mock_image.save.call_args_list]
        assert any("scene_0" in p and "v1.png" in p for p in saved_paths)
        assert any("scene_1" in p and "v1.png" in p for p in saved_paths)


# ═══════════════════════════════════════════════════════════════════════════════
# generate_all_scenes — LoRA loading
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateAllScenesLoRA:
    def _make_lora_yaml(self, tmp_path: Path, names: list[str]) -> Path:
        """Write a minimal loras.yaml to tmp_path for testing."""
        import yaml
        loras = [
            {
                "name": n,
                "file_path": f"scene/{n}.safetensors",
                "trigger_token": n,
                "type": "scene",
                "weight": 0.75,
                "status": "available",
                "tags": [],
                "source": "local",
                "source_url": None,
                "description": f"Test LoRA {n}",
            }
            for n in names
        ]
        # Create dummy safetensors files so registry validates them
        for n in names:
            p = tmp_path / "scene" / f"{n}.safetensors"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00" * 8)  # minimal bytes to exist on disk

        yaml_path = tmp_path / "loras.yaml"
        yaml_path.write_text(yaml.dump({"loras": loras}))
        return yaml_path

    def test_no_lora_calls_when_lora_names_empty(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        svc = make_service(store, mock_vm, tmp_path)

        svc.generate_all_scenes("id-l1", _make_scenes(1), "cinematic", [])

        mock_pipe.load_lora_weights.assert_not_called()

    def test_load_lora_weights_called_for_each_lora(
        self, mock_sdxl, mock_vm, store, tmp_path
    ):
        _, mock_pipe, _ = mock_sdxl
        yaml_path = self._make_lora_yaml(tmp_path, ["lora_a", "lora_b"])
        svc = StoryboardService(
            state_store=store,
            vram_manager=mock_vm,
            loras_yaml=str(yaml_path),
            lora_base=tmp_path,
            device="cpu",
        )

        svc.generate_all_scenes("id-l2", _make_scenes(1), "cinematic", ["lora_a", "lora_b"])

        assert mock_pipe.load_lora_weights.call_count == 2

    def test_unknown_lora_is_skipped_with_warning(
        self, mock_sdxl, mock_vm, store, tmp_path
    ):
        _, mock_pipe, _ = mock_sdxl
        yaml_path = self._make_lora_yaml(tmp_path, [])
        svc = StoryboardService(
            state_store=store,
            vram_manager=mock_vm,
            loras_yaml=str(yaml_path),
            lora_base=tmp_path,
            device="cpu",
        )

        with patch("backend.services.storyboard.service.logger") as mock_logger:
            svc.generate_all_scenes("id-l3", _make_scenes(1), "cinematic", ["ghost_lora"])

        warning_msgs = " ".join(str(c) for c in mock_logger.warning.call_args_list)
        assert "ghost_lora" in warning_msgs
        mock_pipe.load_lora_weights.assert_not_called()

    def test_set_adapters_called_when_multiple_loras(
        self, mock_sdxl, mock_vm, store, tmp_path
    ):
        _, mock_pipe, _ = mock_sdxl
        yaml_path = self._make_lora_yaml(tmp_path, ["lora_a", "lora_b"])
        svc = StoryboardService(
            state_store=store,
            vram_manager=mock_vm,
            loras_yaml=str(yaml_path),
            lora_base=tmp_path,
            device="cpu",
        )

        svc.generate_all_scenes("id-l4", _make_scenes(1), "cinematic", ["lora_a", "lora_b"])

        mock_pipe.set_adapters.assert_called_once()

    def test_set_adapters_not_called_for_single_lora(
        self, mock_sdxl, mock_vm, store, tmp_path
    ):
        _, mock_pipe, _ = mock_sdxl
        yaml_path = self._make_lora_yaml(tmp_path, ["lora_a"])
        svc = StoryboardService(
            state_store=store,
            vram_manager=mock_vm,
            loras_yaml=str(yaml_path),
            lora_base=tmp_path,
            device="cpu",
        )

        svc.generate_all_scenes("id-l5", _make_scenes(1), "cinematic", ["lora_a"])

        mock_pipe.set_adapters.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# generate_single_scene
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateSingleScene:
    def _setup_existing_storyboard(self, store, tmp_path, storyboard_id="regen-id"):
        """Create a storyboard with one scene already at v1."""
        svc = StoryboardService(
            state_store=store,
            vram_manager=MagicMock(),
            loras_yaml=str(tmp_path / "loras.yaml"),
            lora_base=tmp_path,
            device="cpu",
        )
        from backend.services.storyboard.types import StoryboardState, StoryboardScene, VersionEntry
        from datetime import datetime, timezone
        state = StoryboardState(
            storyboard_id=storyboard_id,
            style="cinematic",
            base_checkpoint=StoryboardService.SDXL_BASE,
            lora_names=[],
            status="complete",
            scenes=[
                StoryboardScene(
                    scene_idx=0,
                    storyboard_prompt="Ocean at dusk",
                    positive_prompt="cinematic film still, ocean at dusk",
                    approved_version=None,
                    versions=[
                        VersionEntry(version=1, filename="v1.png", seed=0,
                                     timestamp=datetime.now(timezone.utc).isoformat())
                    ],
                )
            ],
        )
        store.create(state)
        # Write fake v1.png so eviction code can find it
        scene_dir = store.scene_dir(storyboard_id, 0, create=True)
        (scene_dir / "v1.png").write_bytes(b"fake")
        return svc

    def test_appends_new_version_after_existing(self, mock_sdxl, mock_vm, store, tmp_path):
        svc = self._setup_existing_storyboard(store, tmp_path, "sg-1")
        svc._vm = mock_vm

        svc.generate_single_scene("sg-1", scene_idx=0, prompt_override=None, seed=42, lora_names=[])

        state = store.load("sg-1")
        assert len(state.scenes[0].versions) == 2
        assert state.scenes[0].versions[1].version == 2

    def test_new_version_uses_provided_seed(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        svc = self._setup_existing_storyboard(store, tmp_path, "sg-2")
        svc._vm = mock_vm

        svc.generate_single_scene("sg-2", scene_idx=0, prompt_override=None, seed=999, lora_names=[])

        gen = mock_pipe.call_args[1]["generator"]
        assert gen.initial_seed() == 999

    def test_none_seed_uses_random_seed(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        svc = self._setup_existing_storyboard(store, tmp_path, "sg-3")
        svc._vm = mock_vm

        svc.generate_single_scene("sg-3", scene_idx=0, prompt_override=None, seed=None, lora_names=[])

        gen = mock_pipe.call_args[1]["generator"]
        # Any seed is valid — just check the generator was created
        assert gen is not None

    def test_prompt_override_replaces_stored_prompt(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        svc = self._setup_existing_storyboard(store, tmp_path, "sg-4")
        svc._vm = mock_vm

        svc.generate_single_scene(
            "sg-4", scene_idx=0,
            prompt_override="moonlit beach, waves crashing",
            seed=1,
            lora_names=[],
        )

        prompt = mock_pipe.call_args[1]["prompt"]
        assert "moonlit beach" in prompt

    def test_none_prompt_uses_stored_positive_prompt(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        svc = self._setup_existing_storyboard(store, tmp_path, "sg-5")
        svc._vm = mock_vm

        svc.generate_single_scene("sg-5", scene_idx=0, prompt_override=None, seed=1, lora_names=[])

        prompt = mock_pipe.call_args[1]["prompt"]
        assert "ocean at dusk" in prompt

    def test_vram_killed_after_single_scene(self, mock_sdxl, mock_vm, store, tmp_path):
        svc = self._setup_existing_storyboard(store, tmp_path, "sg-6")
        svc._vm = mock_vm

        svc.generate_single_scene("sg-6", scene_idx=0, prompt_override=None, seed=1, lora_names=[])

        mock_vm.kill.assert_called_once()

    def test_vram_killed_on_exception(self, mock_sdxl, mock_vm, store, tmp_path):
        _, mock_pipe, _ = mock_sdxl
        mock_pipe.side_effect = RuntimeError("OOM")
        svc = self._setup_existing_storyboard(store, tmp_path, "sg-7")
        svc._vm = mock_vm

        with pytest.raises(RuntimeError):
            svc.generate_single_scene("sg-7", scene_idx=0, prompt_override=None, seed=1, lora_names=[])

        mock_vm.kill.assert_called_once()

    def test_returns_version_entry(self, mock_sdxl, mock_vm, store, tmp_path):
        svc = self._setup_existing_storyboard(store, tmp_path, "sg-8")
        svc._vm = mock_vm

        result = svc.generate_single_scene("sg-8", scene_idx=0, prompt_override=None, seed=7, lora_names=[])

        assert isinstance(result, VersionEntry)
        assert result.version == 2
        assert result.filename == "v2.png"
        assert result.seed == 7


# ═══════════════════════════════════════════════════════════════════════════════
# get_state and approve (thin delegates — quick sanity checks)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDelegates:
    def test_get_state_returns_none_for_unknown(self, mock_vm, store, tmp_path):
        svc = make_service(store, mock_vm, tmp_path)
        assert svc.get_state("unknown") is None

    def test_approve_delegates_to_store(self, mock_sdxl, mock_vm, store, tmp_path):
        """approve() calls store.set_approved and returns paths."""
        svc = make_service(store, mock_vm, tmp_path)
        # Set up a storyboard with one scene and one version on disk
        svc.generate_all_scenes("approve-id", _make_scenes(1), "cinematic", [])

        paths = svc.approve("approve-id", {0: 1})

        assert 0 in paths
        assert "v1.png" in paths[0]


# ═══════════════════════════════════════════════════════════════════════════════
# generate_single_scene — lora_weights override
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateSingleSceneLoraWeights:
    """Tests that per-scene weight overrides reach the SDXL pipeline and are persisted."""

    def _make_lora_yaml(self, tmp_path: Path, names: list[str]) -> Path:
        import yaml
        loras = [
            {
                "name": n,
                "file_path": f"scene/{n}.safetensors",
                "trigger_token": n.replace("-", "_"),
                "type": "scene",
                "weight": 0.75,   # registry default
                "status": "available",
                "tags": [],
                "source": "local",
                "source_url": None,
                "description": f"Test LoRA {n}",
            }
            for n in names
        ]
        for n in names:
            p = tmp_path / "scene" / f"{n}.safetensors"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\x00" * 8)
        yaml_path = tmp_path / "loras.yaml"
        yaml_path.write_text(yaml.dump({"loras": loras}))
        return yaml_path

    def _setup_storyboard_with_loras(
        self, store, tmp_path, storyboard_id: str, lora_names: list[str]
    ) -> StoryboardService:
        from backend.services.storyboard.types import (
            StoryboardState, StoryboardScene, VersionEntry as VE,
        )
        from datetime import datetime, timezone
        yaml_path = self._make_lora_yaml(tmp_path, lora_names)
        svc = StoryboardService(
            state_store=store,
            vram_manager=MagicMock(),
            loras_yaml=str(yaml_path),
            lora_base=tmp_path,
            device="cpu",
        )
        state = StoryboardState(
            storyboard_id=storyboard_id,
            style="cinematic",
            base_checkpoint=StoryboardService.SDXL_BASE,
            lora_names=lora_names,
            status="complete",
            scenes=[
                StoryboardScene(
                    scene_idx=0,
                    storyboard_prompt="Ocean at dusk",
                    positive_prompt="cinematic film still, ocean at dusk",
                    approved_version=None,
                    versions=[
                        VE(version=1, filename="v1.png", seed=0,
                           timestamp=datetime.now(timezone.utc).isoformat()),
                    ],
                )
            ],
        )
        store.create(state)
        return svc

    def test_lora_weights_override_calls_set_adapters(
        self, mock_sdxl, mock_vm, store, tmp_path
    ):
        """Weight override triggers pipe.set_adapters with the caller-provided weight."""
        _, mock_pipe, _ = mock_sdxl
        svc = self._setup_storyboard_with_loras(store, tmp_path, "wt-1", ["lora-a"])
        svc._vm = mock_vm

        svc.generate_single_scene(
            "wt-1", scene_idx=0,
            prompt_override=None, seed=1,
            lora_names=["lora-a"],
            lora_weights={"lora-a": 0.3},
        )

        # set_adapters should be called with overridden weight 0.3
        mock_pipe.set_adapters.assert_called()
        call_kwargs = mock_pipe.set_adapters.call_args
        weights_arg = call_kwargs[1].get("adapter_weights") or call_kwargs[0][1]
        assert weights_arg == [0.3], f"Expected [0.3], got {weights_arg}"

    def test_lora_weights_for_unknown_name_falls_back_to_registry_default(
        self, mock_sdxl, mock_vm, store, tmp_path
    ):
        """Weight dict with key not in adapter_names uses registry default (0.75)."""
        _, mock_pipe, _ = mock_sdxl
        svc = self._setup_storyboard_with_loras(store, tmp_path, "wt-2", ["lora-a"])
        svc._vm = mock_vm

        svc.generate_single_scene(
            "wt-2", scene_idx=0,
            prompt_override=None, seed=1,
            lora_names=["lora-a"],
            lora_weights={"totally-different-lora": 0.9},   # key not in adapters
        )

        mock_pipe.set_adapters.assert_called()
        call_kwargs = mock_pipe.set_adapters.call_args
        weights_arg = call_kwargs[1].get("adapter_weights") or call_kwargs[0][1]
        # should fall back to registry weight 0.75
        assert weights_arg == [0.75], f"Expected [0.75], got {weights_arg}"

    def test_no_lora_weights_arg_skips_override_set_adapters(
        self, mock_sdxl, mock_vm, store, tmp_path
    ):
        """When lora_weights is None, set_adapters is not called (single LoRA, no override)."""
        _, mock_pipe, _ = mock_sdxl
        svc = self._setup_storyboard_with_loras(store, tmp_path, "wt-3", ["lora-a"])
        svc._vm = mock_vm

        svc.generate_single_scene(
            "wt-3", scene_idx=0,
            prompt_override=None, seed=1,
            lora_names=["lora-a"],
            # no lora_weights kwarg → None
        )

        # single LoRA, no override → set_adapters not called
        mock_pipe.set_adapters.assert_not_called()

    def test_lora_weights_stored_in_version_entry(
        self, mock_sdxl, mock_vm, store, tmp_path
    ):
        """Weights used for a regen are persisted in the VersionEntry."""
        svc = self._setup_storyboard_with_loras(store, tmp_path, "wt-4", ["lora-a"])
        svc._vm = mock_vm

        svc.generate_single_scene(
            "wt-4", scene_idx=0,
            prompt_override=None, seed=1,
            lora_names=["lora-a"],
            lora_weights={"lora-a": 0.35},
        )

        loaded = store.load("wt-4")
        # v2 is the newly generated version
        latest = loaded.scenes[0].versions[-1]
        assert latest.lora_weights == {"lora-a": 0.35}

    def test_empty_lora_weights_dict_stores_empty(
        self, mock_sdxl, mock_vm, store, tmp_path
    ):
        """Passing lora_weights={} stores {} in the version entry."""
        svc = self._setup_storyboard_with_loras(store, tmp_path, "wt-5", [])
        svc._vm = mock_vm

        svc.generate_single_scene(
            "wt-5", scene_idx=0,
            prompt_override=None, seed=1,
            lora_names=[],
            lora_weights={},
        )

        loaded = store.load("wt-5")
        assert loaded.scenes[0].versions[-1].lora_weights == {}

    def test_two_loras_with_override_applies_both_weights(
        self, mock_sdxl, mock_vm, store, tmp_path
    ):
        """Multiple LoRAs with overrides each get their specified weight."""
        _, mock_pipe, _ = mock_sdxl
        svc = self._setup_storyboard_with_loras(
            store, tmp_path, "wt-6", ["lora-a", "lora-b"]
        )
        svc._vm = mock_vm

        svc.generate_single_scene(
            "wt-6", scene_idx=0,
            prompt_override=None, seed=1,
            lora_names=["lora-a", "lora-b"],
            lora_weights={"lora-a": 0.2, "lora-b": 0.8},
        )

        # set_adapters called twice: once by _load_loras (registry), once by override
        calls = mock_pipe.set_adapters.call_args_list
        # Last call is the override call
        last_call = calls[-1]
        weights_arg = last_call[1].get("adapter_weights") or last_call[0][1]
        assert set(weights_arg) == {0.2, 0.8}, f"Expected {{0.2, 0.8}}, got {weights_arg}"
