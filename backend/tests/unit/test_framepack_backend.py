"""FramePack backend unit tests (TDD — no real GPU needed).

All heavy imports (torch, diffusers, torchao, PIL) are patched at the
module boundary so these tests run in <1s with no VRAM.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, call, patch

import pytest

# framepack.py defers all heavy imports (torch/diffusers/PIL/torchao) to
# _import_heavy() which is called at inference time, so no module-level
# stubbing is needed — we patch specific names at test boundaries instead.
from backend.services.prompt.types import ComposedPrompt, LoRAConfig
from backend.services.video.backends.framepack import (
    FramePackBackend,
    _MAX_FRAMES,
    _NATIVE_HEIGHT,
    _NATIVE_WIDTH,
    _VRAM_GB,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _prompt(**kwargs) -> ComposedPrompt:
    defaults = dict(
        positive="rob_char, man on a beach at sunset",
        negative="blurry, low quality",
        cfg_scale=6.0,
        steps=25,
        model="framepack",
        nsfw=False,
        base_checkpoint="",
        lora_configs=[],
        init_image_path="",
    )
    defaults.update(kwargs)
    return ComposedPrompt(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# Interface / identity
# ══════════════════════════════════════════════════════════════════════════════

class TestFramePackInterface:
    def setup_method(self):
        self.backend = FramePackBackend()

    def test_name(self):
        assert self.backend.name() == "framepack"

    def test_vram_required_gb(self):
        assert self.backend.vram_required_gb() == _VRAM_GB
        assert _VRAM_GB >= 8.0  # fp8 realistic minimum

    def test_supports_style_cinematic(self):
        assert self.backend.supports_style("cinematic") is True

    def test_supports_style_music_video(self):
        assert self.backend.supports_style("music_video") is True

    def test_supports_style_unsupported(self):
        assert self.backend.supports_style("oil_painting") is False

    def test_estimated_cost_zero(self):
        assert self.backend.estimated_cost_per_scene() == 0.0

    def test_estimated_time_per_scene_returns_float(self):
        t = self.backend.estimated_time_per_scene()
        assert isinstance(t, float)
        assert t > 0.0

    def test_initial_current_model_empty(self):
        assert self.backend._current_model == ""


# ══════════════════════════════════════════════════════════════════════════════
# _snap_frames — 4k+1 constraint
# ══════════════════════════════════════════════════════════════════════════════

class TestSnapFrames:
    def test_minimum_clamps_to_nine(self):
        assert FramePackBackend._snap_frames(1) == 9

    def test_already_valid_nine(self):
        assert FramePackBackend._snap_frames(9) == 9

    def test_already_valid_thirteen(self):
        assert FramePackBackend._snap_frames(13) == 13

    def test_ten_snaps_down_to_nine(self):
        # (10-1)//4 = 2  →  2*4+1 = 9
        assert FramePackBackend._snap_frames(10) == 9

    def test_seventy_two_snaps_to_sixty_nine(self):
        # 3 sec @ 24fps = 72 raw  →  (71//4)*4+1 = 17*4+1 = 69
        assert FramePackBackend._snap_frames(72) == 69

    def test_caps_at_max_frames(self):
        assert FramePackBackend._snap_frames(9999) == _MAX_FRAMES

    def test_max_frames_is_itself_valid(self):
        assert (_MAX_FRAMES - 1) % 4 == 0  # i.e. MAX = 4k+1


# ══════════════════════════════════════════════════════════════════════════════
# is_available
# ══════════════════════════════════════════════════════════════════════════════

class TestIsAvailable:
    def test_available_when_hf_cache_hit(self):
        backend = FramePackBackend()
        with patch("backend.services.video.backends.framepack._hf_cached") as mock_cached:
            mock_cached.return_value = "/cache/some/path"
            assert backend.is_available() is True

    def test_unavailable_when_hf_cache_miss(self):
        backend = FramePackBackend()
        with patch("backend.services.video.backends.framepack._hf_cached") as mock_cached:
            mock_cached.return_value = None
            assert backend.is_available() is False

    def test_unavailable_on_exception(self):
        backend = FramePackBackend()
        with patch("backend.services.video.backends.framepack._hf_cached") as mock_cached:
            mock_cached.side_effect = Exception("network error")
            assert backend.is_available() is False


# ══════════════════════════════════════════════════════════════════════════════
# generate_clip — delegates to _run_pipeline, builds VideoClip
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateClip:
    def setup_method(self):
        self.backend = FramePackBackend()

    def _patch_run(self, tmp_path):
        clip_file = str(tmp_path / "clip_abc12345.mp4")
        clip_file_obj = Path(clip_file)
        clip_file_obj.touch()
        self.backend._run_pipeline = MagicMock(return_value=clip_file)
        return clip_file

    def test_returns_video_clip(self, tmp_path):
        from backend.services.video.types import VideoClip
        self._patch_run(tmp_path)
        clip = self.backend.generate_clip(_prompt(), 3.0, (1280, 720))
        assert isinstance(clip, VideoClip)

    def test_scene_index_is_minus_one(self, tmp_path):
        self._patch_run(tmp_path)
        clip = self.backend.generate_clip(_prompt(), 3.0, (1280, 720))
        assert clip.scene_index == -1

    def test_backend_used_is_framepack(self, tmp_path):
        self._patch_run(tmp_path)
        clip = self.backend.generate_clip(_prompt(), 3.0, (1280, 720))
        assert clip.backend_used == "framepack"

    def test_duration_matches_snapped_frames_at_24fps(self, tmp_path):
        # 3 sec × 24 fps = 72 raw → snapped 69 → 69/24 = 2.875 sec
        self._patch_run(tmp_path)
        clip = self.backend.generate_clip(_prompt(), 3.0, (1280, 720), fps=24)
        expected = FramePackBackend._snap_frames(max(9, int(3.0 * 24))) / 24
        assert abs(clip.duration_sec - expected) < 0.001

    def test_native_resolution_in_clip(self, tmp_path):
        self._patch_run(tmp_path)
        clip = self.backend.generate_clip(_prompt(), 3.0, (1280, 720))
        assert clip.width == _NATIVE_WIDTH
        assert clip.height == _NATIVE_HEIGHT

    def test_passes_fps_to_run_pipeline(self, tmp_path):
        self._patch_run(tmp_path)
        self.backend.generate_clip(_prompt(), 3.0, (1280, 720), fps=30)
        args = self.backend._run_pipeline.call_args
        # _run_pipeline(prompt, duration_sec, fps, seed, init_image_path)
        assert args[0][2] == 30 or args[1].get("fps") == 30  # positional or keyword

    def test_passes_seed_to_run_pipeline(self, tmp_path):
        self._patch_run(tmp_path)
        self.backend.generate_clip(_prompt(), 3.0, (1280, 720), seed=42)
        args = self.backend._run_pipeline.call_args
        assert 42 in args[0] or args[1].get("seed") == 42


# ══════════════════════════════════════════════════════════════════════════════
# generate_batch
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateBatch:
    def test_returns_list_of_clips(self, tmp_path):
        backend = FramePackBackend()
        from backend.services.video.types import VideoClip
        clip_file = str(tmp_path / "clip.mp4")
        Path(clip_file).touch()
        backend._run_pipeline = MagicMock(return_value=clip_file)

        prompts = [_prompt(), _prompt()]
        clips = backend.generate_batch(prompts, [3.0, 2.0], (1280, 720))
        assert len(clips) == 2
        assert all(isinstance(c, VideoClip) for c in clips)

    def test_calls_run_pipeline_per_prompt(self, tmp_path):
        backend = FramePackBackend()
        clip_file = str(tmp_path / "clip.mp4")
        Path(clip_file).touch()
        backend._run_pipeline = MagicMock(return_value=clip_file)

        prompts = [_prompt(positive="scene A"), _prompt(positive="scene B")]
        backend.generate_batch(prompts, [3.0, 2.0], (1280, 720))
        assert backend._run_pipeline.call_count == 2


# ══════════════════════════════════════════════════════════════════════════════
# kill()
# ══════════════════════════════════════════════════════════════════════════════

class TestKill:
    def test_kill_resets_current_model(self):
        backend = FramePackBackend()
        backend._current_model = "some/repo"
        with patch("backend.services.video.backends.framepack._vram_manager") as mock_vm:
            backend.kill()
        assert backend._current_model == ""

    def test_kill_delegates_to_vram_manager(self):
        backend = FramePackBackend()
        with patch("backend.services.video.backends.framepack._vram_manager") as mock_vm:
            backend.kill()
            mock_vm.kill.assert_called_once()


# ══════════════════════════════════════════════════════════════════════════════
# _run_pipeline — logic tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRunPipeline:
    def _make_pipeline_mock(self):
        """Return a mock that mimics pipeline(image=...) → output.frames[0]."""
        frames = [MagicMock()]  # list of PIL Images
        output = MagicMock()
        output.frames = [frames]
        pipeline = MagicMock(return_value=output)
        return pipeline

    def _setup(self, tmp_path):
        """Return backend with mocked vram_manager and _load_pipeline."""
        backend = FramePackBackend()
        backend._current_model = backend._transformer_repo  # pretend loaded
        self._pipeline_mock = self._make_pipeline_mock()

        self._mock_vm = MagicMock()
        self._mock_vm._current_pipeline = self._pipeline_mock

        self._patcher_vm = patch(
            "backend.services.video.backends.framepack._vram_manager",
            self._mock_vm,
        )
        self._patcher_vm.start()

        # Patch deferred imports inside _run_pipeline
        self._patcher_export = patch(
            "backend.services.video.backends.framepack.FramePackBackend._export_frames"
        )
        self._mock_export = self._patcher_export.start()

        return backend

    def teardown_method(self):
        patch.stopall()

    def test_output_path_in_clips_dir(self, tmp_path):
        backend = self._setup(tmp_path)
        with patch("backend.services.video.backends.framepack.Path") as mock_path_cls:
            # Let the real Path work for most cases, just intercept mkdir
            real_path = Path
            mock_path_cls.side_effect = real_path
            result = backend._run_pipeline(_prompt(), 3.0, 24, -1, "")
        assert "clip_" in result or result.endswith(".mp4")

    def test_loads_pipeline_when_model_changes(self, tmp_path):
        backend = FramePackBackend()
        backend._current_model = ""  # not yet loaded
        backend._load_pipeline = MagicMock(return_value=self._make_pipeline_mock())
        mock_vm = MagicMock()
        mock_vm._current_pipeline = self._make_pipeline_mock()
        with patch("backend.services.video.backends.framepack._vram_manager", mock_vm):
            with patch("backend.services.video.backends.framepack.FramePackBackend._export_frames"):
                with patch("backend.services.video.backends.framepack.FramePackBackend._open_image", return_value=MagicMock()):
                    backend._run_pipeline(_prompt(), 3.0, 24, -1, "")
        backend._load_pipeline.assert_called_once()

    def test_skips_load_when_model_already_loaded(self, tmp_path):
        backend = self._setup(tmp_path)
        backend._load_pipeline = MagicMock()
        with patch("backend.services.video.backends.framepack.FramePackBackend._open_image", return_value=MagicMock()):
            backend._run_pipeline(_prompt(), 3.0, 24, -1, "")
        backend._load_pipeline.assert_not_called()

    def test_black_frame_when_no_init_image(self, tmp_path):
        backend = self._setup(tmp_path)
        with patch("backend.services.video.backends.framepack.FramePackBackend._open_image") as mock_open:
            mock_open.return_value = None  # simulates missing path
            with patch("backend.services.video.backends.framepack.Image") as mock_img:
                backend._run_pipeline(_prompt(init_image_path=""), 3.0, 24, -1, "")
                mock_img.new.assert_called_once_with("RGB", (_NATIVE_WIDTH, _NATIVE_HEIGHT))

    def test_init_image_opened_from_path(self, tmp_path):
        backend = self._setup(tmp_path)
        fake_img_path = str(tmp_path / "keyframe.png")
        Path(fake_img_path).write_bytes(b"fake")
        with patch("backend.services.video.backends.framepack.FramePackBackend._open_image") as mock_open:
            mock_open.return_value = MagicMock()
            backend._run_pipeline(_prompt(init_image_path=fake_img_path), 3.0, 24, -1, fake_img_path)
            mock_open.assert_called_once_with(fake_img_path)

    def test_lora_configs_ignored_with_warning(self, tmp_path, caplog):
        import logging
        backend = self._setup(tmp_path)
        lora = LoRAConfig(name="crossfadeclub", trigger_token="crossfadeclub_style", weight=0.7, lora_type="style")
        p = _prompt(lora_configs=[lora])
        with patch("backend.services.video.backends.framepack.FramePackBackend._open_image", return_value=MagicMock()):
            with caplog.at_level(logging.WARNING, logger="beat_studio.video.framepack"):
                backend._run_pipeline(p, 3.0, 24, -1, "")
        assert any("LoRA" in r.message and "ignored" in r.message for r in caplog.records)

    def test_negative_seed_uses_default(self, tmp_path):
        backend = self._setup(tmp_path)
        with patch("backend.services.video.backends.framepack.FramePackBackend._open_image", return_value=MagicMock()):
            backend._run_pipeline(_prompt(), 3.0, 24, seed=-1, init_image_path="")
        # pipeline should have been called — just verify no exception raised
        assert self._pipeline_mock.call_count == 1

    def test_num_frames_passed_to_pipeline(self, tmp_path):
        backend = self._setup(tmp_path)
        with patch("backend.services.video.backends.framepack.FramePackBackend._open_image", return_value=MagicMock()):
            backend._run_pipeline(_prompt(), 3.0, 24, -1, "")
        call_kwargs = self._pipeline_mock.call_args[1]
        assert "num_frames" in call_kwargs
        # 3 sec × 24 fps = 72 raw → snapped to 69
        assert call_kwargs["num_frames"] == FramePackBackend._snap_frames(max(9, int(3.0 * 24)))


# ══════════════════════════════════════════════════════════════════════════════
# _load_pipeline — fp8 and offload
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadPipeline:
    def test_fp8_quantization_applied_when_torchao_available(self):
        backend = FramePackBackend()
        mock_transformer = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.vae = MagicMock()

        with patch("backend.services.video.backends.framepack.HunyuanVideoFramepackTransformer3DModel") as MockT:
            MockT.from_pretrained.return_value = mock_transformer
            with patch("backend.services.video.backends.framepack.HunyuanVideoFramepackPipeline") as MockP:
                MockP.from_pretrained.return_value = mock_pipe
                with patch("backend.services.video.backends.framepack.quantize_") as mock_q:
                    with patch("backend.services.video.backends.framepack.float8_weight_only") as mock_f8:
                        backend._load_pipeline()
                        mock_q.assert_called_once_with(mock_transformer, mock_f8.return_value)

    def test_bf16_fallback_when_torchao_unavailable(self):
        backend = FramePackBackend()
        mock_transformer = MagicMock()
        mock_pipe = MagicMock()
        mock_pipe.vae = MagicMock()

        with patch("backend.services.video.backends.framepack.HunyuanVideoFramepackTransformer3DModel") as MockT:
            MockT.from_pretrained.return_value = mock_transformer
            with patch("backend.services.video.backends.framepack.HunyuanVideoFramepackPipeline") as MockP:
                MockP.from_pretrained.return_value = mock_pipe
                # Simulate torchao not available
                with patch("backend.services.video.backends.framepack.quantize_", side_effect=ImportError):
                    pipe = backend._load_pipeline()  # should not raise
        assert pipe is mock_pipe

    def test_cpu_offload_enabled(self):
        backend = FramePackBackend()
        mock_pipe = MagicMock()
        mock_pipe.vae = MagicMock()

        with patch("backend.services.video.backends.framepack.HunyuanVideoFramepackTransformer3DModel") as MockT:
            MockT.from_pretrained.return_value = MagicMock()
            with patch("backend.services.video.backends.framepack.HunyuanVideoFramepackPipeline") as MockP:
                MockP.from_pretrained.return_value = mock_pipe
                with patch("backend.services.video.backends.framepack.quantize_"):
                    with patch("backend.services.video.backends.framepack.float8_weight_only"):
                        backend._load_pipeline()
        mock_pipe.enable_model_cpu_offload.assert_called_once()

    def test_vae_tiling_enabled(self):
        backend = FramePackBackend()
        mock_pipe = MagicMock()
        mock_pipe.vae = MagicMock()

        with patch("backend.services.video.backends.framepack.HunyuanVideoFramepackTransformer3DModel") as MockT:
            MockT.from_pretrained.return_value = MagicMock()
            with patch("backend.services.video.backends.framepack.HunyuanVideoFramepackPipeline") as MockP:
                MockP.from_pretrained.return_value = mock_pipe
                with patch("backend.services.video.backends.framepack.quantize_"):
                    with patch("backend.services.video.backends.framepack.float8_weight_only"):
                        backend._load_pipeline()
        mock_pipe.vae.enable_tiling.assert_called_once()
