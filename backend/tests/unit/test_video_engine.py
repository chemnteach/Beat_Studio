"""Tests for Phase 6 Video Generation Engine (backends, ModelRouter, CostEstimator)."""
from __future__ import annotations

from typing import List, Tuple
from unittest.mock import MagicMock, patch
import pytest

from backend.services.prompt.types import ComposedPrompt
from backend.services.video.types import ExecutionPlan, VideoClip


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_composed_prompt():
    return ComposedPrompt(
        positive="cinematic scene, high quality",
        negative="low quality, blurry",
        cfg_scale=7.5,
        steps=25,
        model="animatediff",
        nsfw=False,
    )


def _make_video_clip(idx=0):
    return VideoClip(
        file_path=f"/tmp/clip_{idx}.mp4",
        duration_sec=4.0,
        width=576,
        height=1024,
        fps=24,
        scene_index=idx,
    )


# ── VideoBackend abstract interface ──────────────────────────────────────────

class TestVideoBackendInterface:
    """Verify all concrete backends implement the abstract interface."""

    def _check_backend(self, backend_cls):
        from backend.services.video.backends.base import VideoBackend
        assert issubclass(backend_cls, VideoBackend)
        # Must have required methods
        for method in ("name", "vram_required_gb", "supports_style",
                       "is_available", "estimated_time_per_scene",
                       "estimated_cost_per_scene", "generate_clip",
                       "generate_batch", "kill"):
            assert hasattr(backend_cls, method), f"Missing method: {method}"

    def test_animatediff_implements_interface(self):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        self._check_backend(AnimateDiffBackend)

    def test_wan26_local_implements_interface(self):
        from backend.services.video.backends.wan26_local import WAN26LocalBackend
        self._check_backend(WAN26LocalBackend)

    def test_wan26_cloud_implements_interface(self):
        from backend.services.video.backends.wan26_cloud import WAN26CloudBackend
        self._check_backend(WAN26CloudBackend)

    def test_skyreels_implements_interface(self):
        from backend.services.video.backends.skyreels import SkyReelsBackend
        self._check_backend(SkyReelsBackend)

    def test_svd_implements_interface(self):
        from backend.services.video.backends.svd import SVDBackend
        self._check_backend(SVDBackend)

    def test_sdxl_controlnet_implements_interface(self):
        from backend.services.video.backends.sdxl_controlnet import SDXLControlNetBackend
        self._check_backend(SDXLControlNetBackend)

    def test_cogvideox_implements_interface(self):
        from backend.services.video.backends.cogvideox import CogVideoXBackend
        self._check_backend(CogVideoXBackend)

    def test_mochi_implements_interface(self):
        from backend.services.video.backends.mochi import MochiBackend
        self._check_backend(MochiBackend)

    def test_ltx_video_implements_interface(self):
        from backend.services.video.backends.ltx_video import LTXVideoBackend
        self._check_backend(LTXVideoBackend)


# ── AnimateDiff backend ──────────────────────────────────────────────────────

class TestAnimateDiffBackend:
    def test_name(self):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        b = AnimateDiffBackend()
        assert b.name() == "animatediff"

    def test_vram_required(self):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        b = AnimateDiffBackend()
        assert b.vram_required_gb() == pytest.approx(5.6)

    def test_supports_cinematic(self):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        b = AnimateDiffBackend()
        assert b.supports_style("cinematic")

    def test_supports_anime(self):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        assert AnimateDiffBackend().supports_style("anime")

    def test_does_not_support_photorealistic(self):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        # Photorealistic is better handled by WAN 2.6
        b = AnimateDiffBackend()
        assert not b.supports_style("photorealistic")

    def test_local_cost_is_zero(self):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        assert AnimateDiffBackend().estimated_cost_per_scene() == 0.0

    def test_is_available_no_model(self, tmp_path):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        b = AnimateDiffBackend(model_path=str(tmp_path / "nonexistent.pt"))
        assert not b.is_available()

    def test_generate_clip_calls_internal(self):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        b = AnimateDiffBackend()
        prompt = _make_composed_prompt()
        with patch.object(b, "_run_pipeline", return_value="/tmp/out.mp4"):
            clip = b.generate_clip(prompt, duration_sec=4.0, resolution=(576, 1024))
        assert isinstance(clip, VideoClip)

    def test_generate_batch_returns_list(self):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        b = AnimateDiffBackend()
        prompts = [_make_composed_prompt(), _make_composed_prompt()]
        with patch.object(b, "_run_pipeline", return_value="/tmp/out.mp4"):
            clips = b.generate_batch(prompts, [4.0, 4.0], (576, 1024))
        assert len(clips) == 2

    def test_kill_clears_pipeline(self):
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        b = AnimateDiffBackend()
        b._pipeline = MagicMock()
        b.kill()
        assert b._pipeline is None


# ── WAN26 Local backend ───────────────────────────────────────────────────────

class TestWAN26LocalBackend:
    def test_name(self):
        from backend.services.video.backends.wan26_local import WAN26LocalBackend
        assert WAN26LocalBackend().name() == "wan26_local"

    def test_vram_required(self):
        from backend.services.video.backends.wan26_local import WAN26LocalBackend
        assert WAN26LocalBackend().vram_required_gb() >= 12.0

    def test_supports_photorealistic(self):
        from backend.services.video.backends.wan26_local import WAN26LocalBackend
        assert WAN26LocalBackend().supports_style("photorealistic")

    def test_local_cost_zero(self):
        from backend.services.video.backends.wan26_local import WAN26LocalBackend
        assert WAN26LocalBackend().estimated_cost_per_scene() == 0.0


# ── WAN26 Cloud backend ───────────────────────────────────────────────────────

class TestWAN26CloudBackend:
    def test_name(self):
        from backend.services.video.backends.wan26_cloud import WAN26CloudBackend
        assert WAN26CloudBackend().name() == "wan26_cloud"

    def test_vram_zero_cloud(self):
        from backend.services.video.backends.wan26_cloud import WAN26CloudBackend
        assert WAN26CloudBackend().vram_required_gb() == 0.0

    def test_cloud_has_cost(self):
        from backend.services.video.backends.wan26_cloud import WAN26CloudBackend
        assert WAN26CloudBackend().estimated_cost_per_scene() > 0.0

    def test_is_available_no_endpoint(self, monkeypatch):
        from backend.services.video.backends.wan26_cloud import WAN26CloudBackend
        monkeypatch.delenv("RUNPOD_WAN26_ENDPOINT", raising=False)
        assert not WAN26CloudBackend().is_available()

    def test_is_available_with_endpoint(self, monkeypatch):
        from backend.services.video.backends.wan26_cloud import WAN26CloudBackend
        monkeypatch.setenv("RUNPOD_WAN26_ENDPOINT", "https://api.runpod.io/test")
        assert WAN26CloudBackend().is_available()


# ── SkyReels backend ──────────────────────────────────────────────────────────

class TestSkyReelsBackend:
    def test_name(self):
        from backend.services.video.backends.skyreels import SkyReelsBackend
        assert SkyReelsBackend().name() == "skyreels"

    def test_vram_zero(self):
        from backend.services.video.backends.skyreels import SkyReelsBackend
        assert SkyReelsBackend().vram_required_gb() == 0.0

    def test_is_available_no_endpoint(self, monkeypatch):
        from backend.services.video.backends.skyreels import SkyReelsBackend
        monkeypatch.delenv("RUNPOD_SKYREELS_ENDPOINT", raising=False)
        assert not SkyReelsBackend().is_available()


# ── SVD backend ───────────────────────────────────────────────────────────────

class TestSVDBackend:
    def test_name(self):
        from backend.services.video.backends.svd import SVDBackend
        assert SVDBackend().name() == "svd"

    def test_vram_required(self):
        from backend.services.video.backends.svd import SVDBackend
        assert SVDBackend().vram_required_gb() == pytest.approx(7.5)

    def test_local_cost_zero(self):
        from backend.services.video.backends.svd import SVDBackend
        assert SVDBackend().estimated_cost_per_scene() == 0.0


# ── SDXL+ControlNet backend ───────────────────────────────────────────────────

class TestSDXLControlNetBackend:
    def test_name(self):
        from backend.services.video.backends.sdxl_controlnet import SDXLControlNetBackend
        assert SDXLControlNetBackend().name() == "sdxl_controlnet"

    def test_supports_rotoscope(self):
        from backend.services.video.backends.sdxl_controlnet import SDXLControlNetBackend
        assert SDXLControlNetBackend().supports_style("rotoscope")

    def test_local_cost_zero(self):
        from backend.services.video.backends.sdxl_controlnet import SDXLControlNetBackend
        assert SDXLControlNetBackend().estimated_cost_per_scene() == 0.0


# ── Stub backends ─────────────────────────────────────────────────────────────

class TestStubBackends:
    def test_cogvideox_is_stub(self):
        from backend.services.video.backends.cogvideox import CogVideoXBackend
        b = CogVideoXBackend()
        assert b.name() == "cogvideox"
        assert not b.is_available()  # stub always returns False

    def test_mochi_is_stub(self):
        from backend.services.video.backends.mochi import MochiBackend
        b = MochiBackend()
        assert b.name() == "mochi"
        assert not b.is_available()

    def test_ltx_video_is_stub(self):
        from backend.services.video.backends.ltx_video import LTXVideoBackend
        b = LTXVideoBackend()
        assert b.name() == "ltx_video"
        assert not b.is_available()


# ── ModelRouter ───────────────────────────────────────────────────────────────

class TestModelRouter:
    def _make_router_with_mock_backends(self):
        from backend.services.video.model_router import ModelRouter
        router = ModelRouter()

        animatediff_mock = MagicMock()
        animatediff_mock.name.return_value = "animatediff"
        animatediff_mock.is_available.return_value = True
        animatediff_mock.vram_required_gb.return_value = 5.6
        animatediff_mock.supports_style.return_value = True
        animatediff_mock.estimated_cost_per_scene.return_value = 0.0
        animatediff_mock.estimated_time_per_scene.return_value = 30.0

        wan_cloud_mock = MagicMock()
        wan_cloud_mock.name.return_value = "wan26_cloud"
        wan_cloud_mock.is_available.return_value = True
        wan_cloud_mock.vram_required_gb.return_value = 0.0
        wan_cloud_mock.supports_style.return_value = True
        wan_cloud_mock.estimated_cost_per_scene.return_value = 0.05
        wan_cloud_mock.estimated_time_per_scene.return_value = 120.0

        router._backends = [animatediff_mock, wan_cloud_mock]
        return router, animatediff_mock, wan_cloud_mock

    def test_select_backend_returns_backend(self):
        from backend.services.video.model_router import ModelRouter
        router, _, _ = self._make_router_with_mock_backends()
        backend = router.select_backend("cinematic", quality="high")
        assert backend is not None

    def test_select_backend_local_preferred(self):
        from backend.services.video.model_router import ModelRouter
        router, animatediff_mock, _ = self._make_router_with_mock_backends()
        backend = router.select_backend("cinematic", quality="high", local_preferred=True)
        # Should prefer the free local backend
        assert backend.estimated_cost_per_scene() == 0.0

    def test_select_backend_cloud_when_no_local(self):
        from backend.services.video.model_router import ModelRouter
        router, animatediff_mock, wan_cloud_mock = self._make_router_with_mock_backends()
        animatediff_mock.is_available.return_value = False
        backend = router.select_backend("cinematic", quality="high", local_preferred=True)
        assert backend.name() == "wan26_cloud"

    def test_select_backend_no_available_raises(self):
        from backend.services.video.model_router import ModelRouter, NoBackendAvailableError
        router, animatediff_mock, wan_cloud_mock = self._make_router_with_mock_backends()
        animatediff_mock.is_available.return_value = False
        wan_cloud_mock.is_available.return_value = False
        with pytest.raises(NoBackendAvailableError):
            router.select_backend("cinematic", quality="high")

    def test_get_execution_plan_returns_plan(self):
        from backend.services.video.model_router import ModelRouter
        from backend.services.prompt.types import ScenePrompt
        router, _, _ = self._make_router_with_mock_backends()
        scenes = [
            ScenePrompt(
                scene_index=i, start_sec=i*4.0, end_sec=(i+1)*4.0,
                duration_sec=4.0, is_hero=(i == 1), energy_level=0.7,
                positive="cinematic scene", negative="low quality",
                style="cinematic", lora_names=[], transition_hint="cut",
                cfg_scale=7.5, steps=25,
            )
            for i in range(3)
        ]
        plan = router.get_execution_plan(scenes, style="cinematic", quality="high")
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.scenes) == 3

    def test_execution_plan_has_cost_estimate(self):
        from backend.services.video.model_router import ModelRouter
        from backend.services.prompt.types import ScenePrompt
        router, _, _ = self._make_router_with_mock_backends()
        scenes = [
            ScenePrompt(
                scene_index=0, start_sec=0, end_sec=4, duration_sec=4,
                is_hero=False, energy_level=0.5,
                positive="test", negative="bad", style="cinematic",
                lora_names=[], transition_hint="cut", cfg_scale=7.5, steps=20,
            )
        ]
        plan = router.get_execution_plan(scenes, "cinematic", "high")
        assert plan.total_cost_usd >= 0.0
        assert plan.total_time_sec >= 0.0


# ── CostEstimator ─────────────────────────────────────────────────────────────

class TestCostEstimator:
    def test_local_backend_cost_zero(self):
        from backend.services.video.cost_estimator import CostEstimator
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        est = CostEstimator()
        cost = est.estimate_scene_cost(AnimateDiffBackend(), resolution=(576, 1024))
        assert cost == 0.0

    def test_cloud_backend_has_positive_cost(self):
        from backend.services.video.cost_estimator import CostEstimator
        from backend.services.video.backends.wan26_cloud import WAN26CloudBackend
        est = CostEstimator()
        cost = est.estimate_scene_cost(WAN26CloudBackend(), resolution=(1920, 1080))
        assert cost > 0.0

    def test_estimate_total_sums_scenes(self):
        from backend.services.video.cost_estimator import CostEstimator
        from backend.services.video.backends.animatediff import AnimateDiffBackend
        est = CostEstimator()
        backend = AnimateDiffBackend()
        total = est.estimate_total(backend, num_scenes=10, resolution=(576, 1024))
        assert total == 0.0  # local = free

    def test_estimate_total_cloud_scales_with_scenes(self):
        from backend.services.video.cost_estimator import CostEstimator
        from backend.services.video.backends.wan26_cloud import WAN26CloudBackend
        est = CostEstimator()
        backend = WAN26CloudBackend()
        cost_5 = est.estimate_total(backend, num_scenes=5, resolution=(1920, 1080))
        cost_10 = est.estimate_total(backend, num_scenes=10, resolution=(1920, 1080))
        assert cost_10 > cost_5
