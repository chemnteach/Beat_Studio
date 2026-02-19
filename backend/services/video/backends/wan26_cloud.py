"""WAN 2.6 Cloud (RunPod) backend — high-quality cloud video generation."""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

from backend.services.prompt.types import ComposedPrompt
from backend.services.video.backends.base import VideoBackend
from backend.services.video.types import VideoClip

logger = logging.getLogger("beat_studio.video.wan26_cloud")

_SUPPORTED_STYLES = {
    "photorealistic", "cinematic", "abstract", "watercolor", "oil_painting",
    "impressionist", "lofi", "anime", "cel_animation", "synthwave",
}
_COST_PER_SCENE_USD = 0.05  # ~$0.05 per scene at 1080p on RunPod A100
_ENV_KEY = "RUNPOD_WAN26_ENDPOINT"


class WAN26CloudBackend(VideoBackend):
    """WAN 2.6 cloud backend running on RunPod.

    Best for: Photorealistic 1080p video, no local VRAM constraint.
    Cost: ~$0.05 per scene (RunPod GPU time).
    Requires: RUNPOD_WAN26_ENDPOINT environment variable.
    """

    def __init__(self, endpoint_url: Optional[str] = None):
        self._endpoint_url = endpoint_url or os.getenv(_ENV_KEY, "")
        self._pipeline = None

    def name(self) -> str:
        return "wan26_cloud"

    def vram_required_gb(self) -> float:
        return 0.0  # cloud — no local VRAM needed

    def supports_style(self, style: str) -> bool:
        return style in _SUPPORTED_STYLES

    def is_available(self) -> bool:
        return bool(self._endpoint_url or os.getenv(_ENV_KEY))

    def estimated_time_per_scene(self, resolution: Tuple[int, int] = (1920, 1080)) -> float:
        return 90.0  # ~90s per 1080p scene on cloud A100

    def estimated_cost_per_scene(self) -> float:
        return _COST_PER_SCENE_USD

    def generate_clip(
        self, prompt: ComposedPrompt, duration_sec: float,
        resolution: Tuple[int, int], fps: int = 24, seed: int = -1,
    ) -> VideoClip:
        t0 = time.time()
        out_path = self._call_api(prompt, duration_sec, resolution, fps, seed)
        return VideoClip(
            file_path=out_path, duration_sec=duration_sec,
            width=resolution[0], height=resolution[1], fps=fps,
            scene_index=-1, backend_used=self.name(),
            generation_time_sec=time.time() - t0,
            cost_usd=_COST_PER_SCENE_USD,
        )

    def generate_batch(
        self, prompts: List[ComposedPrompt], durations: List[float],
        resolution: Tuple[int, int], fps: int = 24,
    ) -> List[VideoClip]:
        return [self.generate_clip(p, d, resolution, fps) for p, d in zip(prompts, durations)]

    def kill(self) -> None:
        self._pipeline = None  # no GPU resources held

    def _call_api(
        self, prompt: ComposedPrompt, duration_sec: float,
        resolution: Tuple[int, int], fps: int, seed: int,
    ) -> str:
        """Call the RunPod WAN 2.6 endpoint. Override in tests."""
        raise NotImplementedError("RunPod WAN 2.6 API call not implemented.")
