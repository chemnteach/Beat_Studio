"""Abstract VideoBackend interface — all video generation backends implement this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from backend.services.prompt.types import ComposedPrompt
from backend.services.video.types import VideoClip


class VideoBackend(ABC):
    """Abstract base for all video generation backends.

    Each backend wraps one video model or cloud service and exposes a
    uniform interface for the ModelRouter to select and invoke.

    Backends are responsible for:
    - Declaring their VRAM requirements and style support
    - Checking their own availability (models downloaded, GPU sufficient, API key set)
    - Generating video clips from ComposedPrompts
    - Releasing GPU memory via kill()
    """

    @abstractmethod
    def name(self) -> str:
        """Short identifier for this backend (e.g. "animatediff")."""

    @abstractmethod
    def vram_required_gb(self) -> float:
        """Minimum VRAM required to run this backend locally. 0.0 for cloud."""

    @abstractmethod
    def supports_style(self, style: str) -> bool:
        """Return True if this backend can produce the requested animation style."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can run right now.

        For local backends: model file exists AND sufficient VRAM.
        For cloud backends: endpoint URL is configured.
        """

    @abstractmethod
    def estimated_time_per_scene(self, resolution: Tuple[int, int] = (1920, 1080)) -> float:
        """Estimated seconds to generate one scene at the given resolution."""

    @abstractmethod
    def estimated_cost_per_scene(self) -> float:
        """Estimated USD cost per scene. 0.0 for local backends."""

    @abstractmethod
    def generate_clip(
        self,
        prompt: ComposedPrompt,
        duration_sec: float,
        resolution: Tuple[int, int],
        fps: int = 24,
        seed: int = -1,
    ) -> VideoClip:
        """Generate a single video clip from a prompt.

        Args:
            prompt: Assembled ComposedPrompt with positive/negative/cfg/steps.
            duration_sec: Target clip duration in seconds.
            resolution: (width, height) in pixels.
            fps: Frames per second.
            seed: Random seed (-1 for random).

        Returns:
            VideoClip pointing to the generated file.
        """

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[ComposedPrompt],
        durations: List[float],
        resolution: Tuple[int, int],
        fps: int = 24,
    ) -> List[VideoClip]:
        """Generate multiple clips (may be parallelized by the backend).

        Args:
            prompts: List of ComposedPrompts, one per scene.
            durations: Duration for each clip in seconds.
            resolution: (width, height) for all clips.
            fps: Frames per second.

        Returns:
            List of VideoClip objects, one per prompt.
        """

    @abstractmethod
    def kill(self) -> None:
        """Release all GPU resources held by this backend.

        Implementations must:
        1. Unload LoRA weights (if any)
        2. Delete the pipeline reference
        3. Collect garbage
        4. Clear CUDA cache
        """
