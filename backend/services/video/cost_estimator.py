"""CostEstimator — calculates video generation cost and time estimates."""
from __future__ import annotations

import logging
from typing import Tuple

from backend.services.video.backends.base import VideoBackend

logger = logging.getLogger("beat_studio.video.cost_estimator")

# Resolution scaling factor (relative to 1080p baseline)
_BASELINE_RES = (1920, 1080)
_BASELINE_PIXELS = _BASELINE_RES[0] * _BASELINE_RES[1]


class CostEstimator:
    """Estimates cost and generation time for video scenes.

    Local backends are always free (cost_usd=0.0).
    Cloud backends scale by pixel count relative to 1080p baseline.

    Usage::

        est = CostEstimator()
        cost = est.estimate_scene_cost(backend, resolution=(1920, 1080))
        total = est.estimate_total(backend, num_scenes=24, resolution=(1920, 1080))
    """

    def estimate_scene_cost(
        self,
        backend: VideoBackend,
        resolution: Tuple[int, int] = (1920, 1080),
    ) -> float:
        """Estimate cost in USD for generating one scene at the given resolution.

        Args:
            backend: The VideoBackend that will generate the scene.
            resolution: (width, height) in pixels.

        Returns:
            Estimated cost in USD. 0.0 for local backends.
        """
        base_cost = backend.estimated_cost_per_scene()
        if base_cost == 0.0:
            return 0.0

        # Scale cost by pixel count relative to 1080p
        pixels = resolution[0] * resolution[1]
        scale = pixels / _BASELINE_PIXELS
        return round(base_cost * scale, 4)

    def estimate_scene_time(
        self,
        backend: VideoBackend,
        resolution: Tuple[int, int] = (1920, 1080),
    ) -> float:
        """Estimate generation time in seconds for one scene.

        Args:
            backend: The VideoBackend.
            resolution: (width, height) in pixels.

        Returns:
            Estimated seconds per scene.
        """
        return backend.estimated_time_per_scene(resolution)

    def estimate_total(
        self,
        backend: VideoBackend,
        num_scenes: int,
        resolution: Tuple[int, int] = (1920, 1080),
    ) -> float:
        """Estimate total cost in USD for all scenes.

        Args:
            backend: The VideoBackend.
            num_scenes: Number of scenes to generate.
            resolution: (width, height) in pixels.

        Returns:
            Total estimated cost in USD.
        """
        return self.estimate_scene_cost(backend, resolution) * num_scenes

    def estimate_total_time(
        self,
        backend: VideoBackend,
        num_scenes: int,
        resolution: Tuple[int, int] = (1920, 1080),
    ) -> float:
        """Estimate total generation time in seconds for all scenes."""
        return self.estimate_scene_time(backend, resolution) * num_scenes
