"""ModelRouter — selects the best video backend for a given task."""
from __future__ import annotations

import logging
from typing import List, Optional

from backend.services.prompt.types import ScenePrompt
from backend.services.video.backends.animatediff import AnimateDiffBackend
from backend.services.video.backends.base import VideoBackend
from backend.services.video.backends.cogvideox import CogVideoXBackend
from backend.services.video.backends.ltx_video import LTXVideoBackend
from backend.services.video.backends.mochi import MochiBackend
from backend.services.video.backends.sdxl_controlnet import SDXLControlNetBackend
from backend.services.video.backends.skyreels import SkyReelsBackend
from backend.services.video.backends.svd import SVDBackend
from backend.services.video.backends.wan26_cloud import WAN26CloudBackend
from backend.services.video.backends.wan26_local import WAN26LocalBackend
from backend.services.video.cost_estimator import CostEstimator
from backend.services.video.types import ExecutionPlan, ScenePlan

logger = logging.getLogger("beat_studio.video.model_router")


class NoBackendAvailableError(RuntimeError):
    """Raised when no backend can handle the requested style/quality."""


class ModelRouter:
    """Selects the best video backend for a given task.

    Selection logic:
    1. Filter backends that support the requested style
    2. Filter by availability (models downloaded, GPU sufficient, API keys set)
    3. If local_preferred and local options exist:
       - Rank by quality/capability, select highest that fits
    4. If no local option available:
       - Fall back to cloud backends
    5. Raise NoBackendAvailableError if nothing works

    Usage::

        router = ModelRouter()
        backend = router.select_backend("cinematic", quality="high")
        plan = router.get_execution_plan(scenes, "cinematic", "high")
    """

    def __init__(self):
        self._backends: List[VideoBackend] = self._discover_backends()
        self._cost_estimator = CostEstimator()

    # ── public ────────────────────────────────────────────────────────────────

    def select_backend(
        self,
        style: str,
        quality: str = "high",
        local_preferred: bool = True,
    ) -> VideoBackend:
        """Select the best available backend for the given style and quality.

        Args:
            style: Animation style name (e.g. "cinematic", "anime").
            quality: "draft" | "high" | "cinematic"
            local_preferred: If True, prefer local backends over cloud.

        Returns:
            The selected VideoBackend.

        Raises:
            NoBackendAvailableError: If no backend can handle the request.
        """
        compatible = [b for b in self._backends if b.supports_style(style)]
        available = [b for b in compatible if b.is_available()]

        if not available:
            raise NoBackendAvailableError(
                f"No backend available for style='{style}', quality='{quality}'. "
                f"Compatible but unavailable: {[b.name() for b in compatible]}"
            )

        if local_preferred:
            local = [b for b in available if b.vram_required_gb() > 0]
            if local:
                # Prefer higher VRAM backends for better quality
                return sorted(local, key=lambda b: -b.vram_required_gb())[0]

        # Fall back to cloud or any available
        return sorted(available, key=lambda b: b.estimated_cost_per_scene())[0]

    def get_execution_plan(
        self,
        scenes: List[ScenePrompt],
        style: str,
        quality: str = "high",
        local_preferred: bool = True,
        resolution: tuple = (1920, 1080),
    ) -> ExecutionPlan:
        """Create a complete execution plan for generating all scenes.

        Args:
            scenes: List of ScenePrompts to generate.
            style: Animation style name.
            quality: "draft" | "high" | "cinematic"
            local_preferred: Prefer local backends.
            resolution: Target output resolution.

        Returns:
            ExecutionPlan with per-scene details and totals.
        """
        try:
            backend = self.select_backend(style, quality, local_preferred)
        except NoBackendAvailableError:
            # If local fails, try cloud
            try:
                backend = self.select_backend(style, quality, local_preferred=False)
            except NoBackendAvailableError:
                raise

        scene_plans: List[ScenePlan] = []
        for scene in scenes:
            cost = self._cost_estimator.estimate_scene_cost(backend, resolution)
            time_est = backend.estimated_time_per_scene(resolution)
            scene_plans.append(ScenePlan(
                scene_index=scene.scene_index,
                backend_name=backend.name(),
                estimated_time_sec=time_est,
                estimated_cost_usd=cost,
                resolution=resolution,
            ))

        total_cost = sum(p.estimated_cost_usd for p in scene_plans)
        total_time = sum(p.estimated_time_sec for p in scene_plans)

        return ExecutionPlan(
            scenes=scene_plans,
            total_time_sec=total_time,
            total_cost_usd=total_cost,
            primary_backend=backend.name(),
        )

    def list_available_backends(self) -> List[str]:
        """Return names of all currently available backends."""
        return [b.name() for b in self._backends if b.is_available()]

    # ── internal ──────────────────────────────────────────────────────────────

    def _discover_backends(self) -> List[VideoBackend]:
        """Instantiate all known backends in priority order."""
        return [
            WAN26LocalBackend(),        # Best quality local
            SVDBackend(),               # Image-to-video local
            SDXLControlNetBackend(),    # Rotoscope local
            AnimateDiffBackend(),       # Fast animated local
            WAN26CloudBackend(),        # Best quality cloud
            SkyReelsBackend(),          # Seamless stitching cloud
            CogVideoXBackend(),         # Stub
            MochiBackend(),             # Stub
            LTXVideoBackend(),          # Stub
        ]
