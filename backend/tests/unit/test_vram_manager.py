"""Tests for VRAMManager — kill-and-revive pattern."""
from unittest.mock import MagicMock, patch, call
import gc

import pytest

from backend.services.shared.vram_manager import VRAMManager, VRAMStatus


class TestVRAMStatus:
    def test_get_status_no_cuda(self):
        with patch("backend.services.shared.vram_manager.CUDA_AVAILABLE", False):
            mgr = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)
            status = mgr.get_status()
        assert isinstance(status, VRAMStatus)
        assert status.cuda_available is False
        assert status.total_gb == 0.0
        assert status.free_gb == 0.0

    def test_baseline_threshold_stored(self):
        mgr = VRAMManager(budget_gb=12.0, baseline_threshold_gb=2.0)
        assert mgr.baseline_threshold_gb == 2.0

    def test_budget_stored(self):
        mgr = VRAMManager(budget_gb=24.0, baseline_threshold_gb=1.5)
        assert mgr.budget_gb == 24.0


class TestKillOperation:
    def test_kill_deletes_pipeline_reference(self):
        mgr = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)
        mock_pipe = MagicMock()
        mock_pipe.unload_lora_weights = MagicMock()
        mgr._current_pipeline = mock_pipe
        mgr._current_model_name = "animatediff"

        with patch("backend.services.shared.vram_manager.CUDA_AVAILABLE", False):
            mgr.kill()

        assert mgr._current_pipeline is None
        assert mgr._current_model_name is None

    def test_kill_calls_unload_lora_weights_if_present(self):
        mgr = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)
        mock_pipe = MagicMock(spec=["unload_lora_weights"])
        mgr._current_pipeline = mock_pipe

        with patch("backend.services.shared.vram_manager.CUDA_AVAILABLE", False):
            mgr.kill()

        mock_pipe.unload_lora_weights.assert_called_once()

    def test_kill_handles_no_lora_method(self):
        mgr = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)
        mock_pipe = MagicMock(spec=[])  # no unload_lora_weights
        mgr._current_pipeline = mock_pipe

        with patch("backend.services.shared.vram_manager.CUDA_AVAILABLE", False):
            mgr.kill()  # must not raise

        assert mgr._current_pipeline is None

    def test_kill_when_nothing_loaded_is_safe(self):
        mgr = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)
        with patch("backend.services.shared.vram_manager.CUDA_AVAILABLE", False):
            mgr.kill()  # must not raise
        assert mgr._current_pipeline is None


class TestCurrentModel:
    def test_current_model_initially_none(self):
        mgr = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)
        assert mgr.current_model_name is None

    def test_set_current_pipeline(self):
        mgr = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)
        mock_pipe = MagicMock()
        mgr.set_pipeline(mock_pipe, "animatediff")
        assert mgr.current_model_name == "animatediff"
        assert mgr._current_pipeline is mock_pipe

    def test_set_pipeline_kills_previous(self):
        mgr = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)
        mock_old = MagicMock(spec=["unload_lora_weights"])
        mock_new = MagicMock()

        mgr.set_pipeline(mock_old, "animatediff")
        with patch("backend.services.shared.vram_manager.CUDA_AVAILABLE", False):
            mgr.set_pipeline(mock_new, "svd")

        mock_old.unload_lora_weights.assert_called_once()
        assert mgr.current_model_name == "svd"
