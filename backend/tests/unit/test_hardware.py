"""Tests for HardwareDetector."""
from unittest.mock import MagicMock, patch

import pytest

from backend.services.shared.hardware_detector import (
    ExecutionRecommendation,
    HardwareProfile,
)


class TestHardwareProfile:
    def test_detect_returns_profile(self):
        profile = HardwareProfile.detect()
        assert isinstance(profile, HardwareProfile)

    def test_profile_has_required_fields(self):
        profile = HardwareProfile.detect()
        assert isinstance(profile.cuda_available, bool)
        assert isinstance(profile.cpu_cores, int)
        assert profile.cpu_cores > 0
        assert isinstance(profile.ram_gb, float)
        assert profile.ram_gb > 0

    def test_can_run_local_animatediff_with_enough_vram(self):
        profile = HardwareProfile(
            gpu_name="RTX 4090",
            vram_total_gb=24.0,
            vram_available_gb=20.0,
            cuda_available=True,
            cuda_version="12.1",
            cpu_cores=16,
            ram_gb=64.0,
        )
        assert profile.can_run_local("animatediff_lightning") is True

    def test_cannot_run_wan26_with_insufficient_vram(self):
        profile = HardwareProfile(
            gpu_name="GTX 1060",
            vram_total_gb=6.0,
            vram_available_gb=5.0,
            cuda_available=True,
            cuda_version="11.8",
            cpu_cores=8,
            ram_gb=16.0,
        )
        assert profile.can_run_local("wan26_local") is False

    def test_unknown_model_returns_false(self):
        profile = HardwareProfile(
            gpu_name="RTX 4090",
            vram_total_gb=24.0,
            vram_available_gb=20.0,
            cuda_available=True,
            cuda_version="12.1",
            cpu_cores=16,
            ram_gb=64.0,
        )
        assert profile.can_run_local("unknown_model_xyz") is False

    def test_no_cuda_cannot_run_any_model(self):
        profile = HardwareProfile(
            gpu_name="None",
            vram_total_gb=0.0,
            vram_available_gb=0.0,
            cuda_available=False,
            cuda_version="",
            cpu_cores=8,
            ram_gb=16.0,
        )
        assert profile.can_run_local("animatediff_lightning") is False


class TestExecutionRecommendation:
    def test_recommend_local_when_possible(self):
        profile = HardwareProfile(
            gpu_name="RTX 4090",
            vram_total_gb=24.0,
            vram_available_gb=20.0,
            cuda_available=True,
            cuda_version="12.1",
            cpu_cores=16,
            ram_gb=64.0,
        )
        rec = profile.recommend_execution("animatediff_lightning")
        assert rec.backend == "animatediff_lightning"
        assert rec.use_cloud is False
        assert rec.estimated_cost_usd == 0.0

    def test_recommend_cloud_when_local_insufficient(self):
        profile = HardwareProfile(
            gpu_name="GTX 1060",
            vram_total_gb=6.0,
            vram_available_gb=5.0,
            cuda_available=True,
            cuda_version="11.8",
            cpu_cores=8,
            ram_gb=16.0,
        )
        rec = profile.recommend_execution("wan26_local")
        assert rec.use_cloud is True

    def test_recommendation_has_time_estimate(self):
        profile = HardwareProfile(
            gpu_name="RTX 4090",
            vram_total_gb=24.0,
            vram_available_gb=20.0,
            cuda_available=True,
            cuda_version="12.1",
            cpu_cores=16,
            ram_gb=64.0,
        )
        rec = profile.recommend_execution("animatediff_lightning")
        assert rec.estimated_seconds_per_scene > 0
