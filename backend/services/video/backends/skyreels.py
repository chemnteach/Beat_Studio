"""SkyReels V2 DF (RunPod) backend — seamless scene stitching via diffusion forcing."""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

from backend.services.prompt.types import ComposedPrompt
from backend.services.video.backends.base import VideoBackend
from backend.services.video.types import VideoClip

logger = logging.getLogger("beat_studio.video.skyreels")

_SUPPORTED_STYLES = {"photorealistic", "cinematic"}
_COST_PER_SCENE_USD = 0.04
_ENV_KEY = "RUNPOD_SKYREELS_ENDPOINT"


class SkyReelsBackend(VideoBackend):
    """SkyReels V2 Diffusion Forcing cloud backend.

    Primary use: Post-processing step to stitch WAN 2.6 scenes seamlessly.
    Also usable for direct generation of photorealistic/cinematic content.
    Requires: RUNPOD_SKYREELS_ENDPOINT environment variable.
    """

    def __init__(self, endpoint_url: Optional[str] = None):
        self._endpoint_url = endpoint_url or os.getenv(_ENV_KEY, "")

    def name(self) -> str:
        return "skyreels"

    def vram_required_gb(self) -> float:
        return 0.0

    def supports_style(self, style: str) -> bool:
        return style in _SUPPORTED_STYLES

    def is_available(self) -> bool:
        return bool(self._endpoint_url or os.getenv(_ENV_KEY))

    def estimated_time_per_scene(self, resolution: Tuple[int, int] = (1920, 1080)) -> float:
        return 60.0

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
        pass  # cloud backend, no local resources

    def _call_api(
        self, prompt: ComposedPrompt, duration_sec: float,
        resolution: Tuple[int, int], fps: int, seed: int,
    ) -> str:
        raise NotImplementedError("SkyReels API call not implemented.")
