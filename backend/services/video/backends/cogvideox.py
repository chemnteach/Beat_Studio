"""CogVideoX backend stub — evaluate during development."""
from __future__ import annotations

from typing import List, Tuple

from backend.services.prompt.types import ComposedPrompt
from backend.services.video.backends.base import VideoBackend
from backend.services.video.types import VideoClip


class CogVideoXBackend(VideoBackend):
    """CogVideoX backend stub.

    Status: Evaluate during development. BeatCanvas had a stub but never integrated.
    VRAM: 14-16GB required.
    """

    def name(self) -> str:
        return "cogvideox"

    def vram_required_gb(self) -> float:
        return 14.0

    def supports_style(self, style: str) -> bool:
        return style in {"cinematic", "photorealistic"}

    def is_available(self) -> bool:
        return False  # stub — not yet implemented

    def estimated_time_per_scene(self, resolution: Tuple[int, int] = (1920, 1080)) -> float:
        return 180.0

    def estimated_cost_per_scene(self) -> float:
        return 0.0

    def generate_clip(self, prompt, duration_sec, resolution, fps=24, seed=-1) -> VideoClip:
        raise NotImplementedError("CogVideoX is a stub — not yet integrated.")

    def generate_batch(self, prompts, durations, resolution, fps=24) -> List[VideoClip]:
        raise NotImplementedError("CogVideoX is a stub — not yet integrated.")

    def kill(self) -> None:
        pass
