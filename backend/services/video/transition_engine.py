"""TransitionEngine — selects and applies smooth transitions between video clips."""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Optional

from backend.services.video.beat_sync import SyncedScenePlan
from backend.services.video.types import VideoClip

logger = logging.getLogger("beat_studio.video.transition_engine")


class TransitionType(enum.Enum):
    """Available transition types, ordered from simplest to highest quality."""
    CROSSFADE = "crossfade"
    MORPH = "morph"
    MOTION_BLEND = "motion_blend"
    SKYREELS_STITCH = "skyreels_stitch"
    TEMPORAL_INTERP = "temporal_interp"


@dataclass
class TransitionConfig:
    """Configuration for a single transition between two clips.

    Attributes:
        transition_type: The chosen :class:`TransitionType`.
        duration_sec: Duration of the transition overlap in seconds.
        notes: Optional description.
    """
    transition_type: TransitionType
    duration_sec: float = 0.75
    notes: str = ""


class TransitionEngine:
    """Creates smooth transitions between generated video clips.

    Selection logic:
    - SkyReels backend → prefer SKYREELS_STITCH for cinematic tier
    - Hero scene boundary → use MORPH for professional/cinematic
    - All other cases → CROSSFADE (safe fallback)

    GPU-heavy operations (blending) are isolated in ``_blend_frames()``
    so unit tests can mock them.

    Usage::

        engine = TransitionEngine()
        config = engine.select_transition(scene_a, scene_b, backend, "professional")
        result_clip = engine.apply_transition(clip_a, clip_b, config)
    """

    # Default transition duration by tier (seconds)
    _DEFAULT_DURATIONS = {
        "basic": 0.5,
        "standard": 0.75,
        "professional": 0.75,
        "cinematic": 1.0,
    }

    def select_transition(
        self,
        scene_a: SyncedScenePlan,
        scene_b: SyncedScenePlan,
        backend: object,
        quality_tier: str = "standard",
    ) -> TransitionConfig:
        """Select the best transition type for a scene boundary.

        Args:
            scene_a: The outgoing scene.
            scene_b: The incoming scene.
            backend: The video backend in use (checked for SkyReels).
            quality_tier: ``"basic"`` | ``"standard"`` | ``"professional"`` |
                ``"cinematic"``.

        Returns:
            :class:`TransitionConfig` with the selected type and duration.
        """
        duration = self._DEFAULT_DURATIONS.get(quality_tier, 0.75)
        backend_name = getattr(backend, "name", lambda: "")()

        # SkyReels Diffusion Forcing stitching — highest quality
        if backend_name == "skyreels" and quality_tier in ("professional", "cinematic"):
            return TransitionConfig(
                transition_type=TransitionType.SKYREELS_STITCH,
                duration_sec=duration,
                notes="skyreels diffusion forcing",
            )

        # Hero scene boundary → morph for visual punch
        if scene_b.is_hero and quality_tier in ("professional", "cinematic"):
            return TransitionConfig(
                transition_type=TransitionType.MORPH,
                duration_sec=duration,
                notes="hero scene boundary",
            )

        # Default: crossfade (always available, no GPU required)
        return TransitionConfig(
            transition_type=TransitionType.CROSSFADE,
            duration_sec=duration,
        )

    def apply_transition(
        self,
        clip_a: VideoClip,
        clip_b: VideoClip,
        config: TransitionConfig,
    ) -> VideoClip:
        """Apply a transition, returning a blended boundary clip.

        The actual blending is performed by ``_blend_frames()``.

        Args:
            clip_a: Outgoing video clip.
            clip_b: Incoming video clip.
            config: Transition configuration from :meth:`select_transition`.

        Returns:
            A new :class:`VideoClip` representing the blended output
            (covers the transition region of both clips).
        """
        logger.debug(
            "Applying %s transition (%.2fs) between clips %d and %d",
            config.transition_type.value,
            config.duration_sec,
            clip_a.scene_index,
            clip_b.scene_index,
        )
        blended_path = self._blend_frames(clip_a, clip_b, config)
        return VideoClip(
            file_path=blended_path,
            duration_sec=config.duration_sec,
            width=clip_a.width,
            height=clip_a.height,
            fps=clip_a.fps,
            scene_index=clip_a.scene_index,
        )

    # ── Internal — mockable for unit tests ────────────────────────────────────

    def _blend_frames(
        self,
        clip_a: VideoClip,
        clip_b: VideoClip,
        config: TransitionConfig,
    ) -> str:
        """Blend the boundary frames of clip_a and clip_b.

        Production: uses MoviePy/FFmpeg/optical flow depending on transition
        type.  Mock in tests.

        Returns:
            Path to the blended output video file.
        """
        raise NotImplementedError(
            "_blend_frames requires FFmpeg/MoviePy. Mock in tests."
        )
