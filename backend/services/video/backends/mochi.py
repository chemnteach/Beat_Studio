"""Mochi backend stub — evaluate during development."""
from __future__ import annotations

from typing import List, Tuple

from backend.services.prompt.types import ComposedPrompt
from backend.services.video.backends.base import VideoBackend
from backend.services.video.types import VideoClip


class MochiBackend(VideoBackend):
    """Mochi backend stub. Not yet integrated — evaluate during development."""

    def name(self) -> str:
        return "mochi"

    def vram_required_gb(self) -> float:
        return 20.0

    def supports_style(self, style: str) -> bool:
        return style in {"cinematic", "photorealistic"}

    def is_available(self) -> bool:
        return False  # stub

    def estimated_time_per_scene(self, resolution: Tuple[int, int] = (1920, 1080)) -> float:
        return 240.0

    def estimated_cost_per_scene(self) -> float:
        return 0.0

    def generate_clip(self, prompt, duration_sec, resolution, fps=24, seed=-1) -> VideoClip:
        raise NotImplementedError("Mochi is a stub — not yet integrated.")

    def generate_batch(self, prompts, durations, resolution, fps=24) -> List[VideoClip]:
        raise NotImplementedError("Mochi is a stub — not yet integrated.")

    def kill(self) -> None:
        pass
