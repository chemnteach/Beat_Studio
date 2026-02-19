"""WAN 2.6 Local backend — high-quality local video generation."""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

from backend.services.prompt.types import ComposedPrompt
from backend.services.video.backends.base import VideoBackend
from backend.services.video.types import VideoClip

logger = logging.getLogger("beat_studio.video.wan26_local")

_SUPPORTED_STYLES = {
    "photorealistic", "cinematic", "abstract", "lofi", "watercolor",
    "oil_painting", "impressionist",
}
_DEFAULT_MODEL_PATH = "backend/models/wan26/wan2.6_14B.safetensors"
_VRAM_GB = 12.0


class WAN26LocalBackend(VideoBackend):
    """WAN 2.6 local GPU backend.

    Best for: Photorealistic and high-quality animated video locally.
    VRAM: 12GB+ recommended (14B model).
    Cost: Free (local GPU)
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = Path(model_path or _DEFAULT_MODEL_PATH)
        self._pipeline = None

    def name(self) -> str:
        return "wan26_local"

    def vram_required_gb(self) -> float:
        return _VRAM_GB

    def supports_style(self, style: str) -> bool:
        return style in _SUPPORTED_STYLES

    def is_available(self) -> bool:
        return self._model_path.exists()

    def estimated_time_per_scene(self, resolution: Tuple[int, int] = (1280, 720)) -> float:
        return 120.0  # ~2 min per scene at 720p on RTX 3080

    def estimated_cost_per_scene(self) -> float:
        return 0.0

    def generate_clip(
        self, prompt: ComposedPrompt, duration_sec: float,
        resolution: Tuple[int, int], fps: int = 24, seed: int = -1,
    ) -> VideoClip:
        t0 = time.time()
        out_path = self._run_pipeline(prompt, duration_sec, resolution, fps, seed)
        return VideoClip(
            file_path=out_path, duration_sec=duration_sec,
            width=resolution[0], height=resolution[1], fps=fps,
            scene_index=-1, backend_used=self.name(),
            generation_time_sec=time.time() - t0, cost_usd=0.0,
        )

    def generate_batch(
        self, prompts: List[ComposedPrompt], durations: List[float],
        resolution: Tuple[int, int], fps: int = 24,
    ) -> List[VideoClip]:
        return [self.generate_clip(p, d, resolution, fps) for p, d in zip(prompts, durations)]

    def kill(self) -> None:
        del self._pipeline
        self._pipeline = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _run_pipeline(
        self, prompt: ComposedPrompt, duration_sec: float,
        resolution: Tuple[int, int], fps: int, seed: int,
    ) -> str:
        raise NotImplementedError("WAN 2.6 local pipeline not loaded.")
