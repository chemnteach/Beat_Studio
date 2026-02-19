"""VideoAssembler — assembles generated clips into a continuous final video."""
from __future__ import annotations

import logging
from typing import List, Tuple

from backend.services.video.transition_engine import TransitionConfig
from backend.services.video.types import VideoClip

logger = logging.getLogger("beat_studio.video.assembler")


class VideoAssembler:
    """Assembles video clips into a continuous final video.

    Critical principle: **No Ken Burns effects. No static images.**
    Every frame of the output must be generated or interpolated.

    All heavy FFmpeg operations are isolated in ``_run_ffmpeg_concat()``
    so unit tests can mock without real files or binaries.

    Usage::

        assembler = VideoAssembler()
        output = assembler.assemble(
            clips=clips,
            transitions=transitions,
            audio_path="/path/to/audio.wav",
            output_path="/path/to/output.mp4",
        )
    """

    # Hard rule — must never be changed
    ALLOW_KEN_BURNS: bool = False

    def assemble(
        self,
        clips: List[VideoClip],
        transitions: List[TransitionConfig],
        audio_path: str,
        output_path: str,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 24,
    ) -> str:
        """Assemble clips into a final video with audio.

        Assembly pipeline:
        1. Validate clips and transitions (len(transitions) == len(clips) - 1)
        2. Upscale clips to target resolution if needed
        3. Apply transitions between consecutive clips
        4. Mux with audio track
        5. Encode with FFmpeg

        Args:
            clips: Ordered list of :class:`VideoClip` objects.
            transitions: Transition configs between clips.
                Must have exactly ``len(clips) - 1`` entries.
            audio_path: Path to the audio file to mux in.
            output_path: Destination path for the final video.
            resolution: Target output resolution ``(width, height)``.
            fps: Target frames per second.

        Returns:
            Path to the assembled video file (same as ``output_path``).

        Raises:
            ValueError: If ``len(transitions) != len(clips) - 1``.
        """
        expected_transitions = max(0, len(clips) - 1)
        if len(transitions) != expected_transitions:
            raise ValueError(
                f"Expected {expected_transitions} transitions for {len(clips)} clips, "
                f"got {len(transitions)}."
            )

        logger.info(
            "Assembling %d clips → %s (resolution=%s, fps=%d)",
            len(clips),
            output_path,
            resolution,
            fps,
        )

        result = self._run_ffmpeg_concat(
            clips=clips,
            transitions=transitions,
            audio_path=audio_path,
            output_path=output_path,
            resolution=resolution,
            fps=fps,
        )
        return result

    # ── Internal — mockable for unit tests ────────────────────────────────────

    def _run_ffmpeg_concat(
        self,
        clips: List[VideoClip],
        transitions: List[TransitionConfig],
        audio_path: str,
        output_path: str,
        resolution: Tuple[int, int],
        fps: int,
    ) -> str:
        """Run FFmpeg concat + audio mux.

        Production: builds FFmpeg filter graph for crossfades / transitions,
        applies audio track, encodes to H.264.  Mock in tests.

        Returns:
            Path to the assembled video.
        """
        raise NotImplementedError(
            "_run_ffmpeg_concat requires FFmpeg. Mock in tests."
        )
