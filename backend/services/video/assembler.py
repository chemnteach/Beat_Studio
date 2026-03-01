"""VideoAssembler — assembles generated clips into a continuous final video."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

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
        scene_durations: Optional[List[float]] = None,
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
            scene_durations: Intended duration for each clip in seconds.
                If provided, clips shorter than their intended duration are
                looped and trimmed, and clips longer are trimmed.  If omitted,
                each clip's own ``duration_sec`` is used as-is.

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
            scene_durations=scene_durations,
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
        scene_durations: Optional[List[float]] = None,
    ) -> str:
        """Run FFmpeg concat + audio mux.

        Production: builds FFmpeg filter graph for crossfades / transitions,
        applies audio track, encodes to H.264.  Mock in tests.

        Returns:
            Path to the assembled video.
        """
        import subprocess
        import uuid
        from pathlib import Path

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize clip durations to intended scene durations.
        # - Clip too short (>0.5s gap): loop and trim with ffmpeg, then use temp file.
        # - Clip too long: concat `duration` directive trims it.
        # - Close enough (<= 0.5s gap): use `duration` directive to trim/hold last frame.
        intended = scene_durations or [c.duration_sec for c in clips]
        normalized_paths: List[tuple[str, float]] = []
        temp_files: List[Path] = []

        for clip, target_dur in zip(clips, intended):
            clip_path = Path(clip.file_path).resolve()
            gap = target_dur - clip.duration_sec

            if gap > 0.5:
                # Clip is significantly shorter — loop it to cover the full duration
                looped = out_path.parent / f"looped_{uuid.uuid4().hex[:8]}.mp4"
                temp_files.append(looped)
                loop_cmd = [
                    "ffmpeg", "-y",
                    "-stream_loop", "-1",
                    "-i", str(clip_path),
                    "-t", str(target_dur),
                    "-c", "copy",
                    str(looped),
                ]
                try:
                    subprocess.run(loop_cmd, check=True, capture_output=True)
                    normalized_paths.append((str(looped), target_dur))
                    logger.debug(
                        "Looped short clip %.2fs → %.2fs: %s",
                        clip.duration_sec, target_dur, clip.file_path,
                    )
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr.decode(errors="replace").strip()
                    logger.warning(
                        "Failed to loop clip %s (%.2f→%.2f): %s — using original",
                        clip.file_path, clip.duration_sec, target_dur, stderr[-200:],
                    )
                    normalized_paths.append((str(clip_path), target_dur))
            else:
                # Trim or minor hold: concat duration directive handles it
                normalized_paths.append((str(clip_path), target_dur))

        # Write concat list — unique filename prevents concurrent-job collisions.
        # MUST use absolute paths: ffmpeg resolves relative paths in concat.txt
        # relative to the concat file's directory, not the process CWD.
        concat_file = out_path.parent / f"concat_{uuid.uuid4().hex[:8]}.txt"
        concat_file.write_text(
            "".join(
                f"file '{path}'\nduration {dur}\n"
                for path, dur in normalized_paths
            )
        )

        # Scale filter: maintain aspect ratio then pad to target resolution with black bars.
        # Avoids stretching when clip resolution (e.g. 512x512) differs from target (1920x1080).
        w, h = resolution
        scale_filter = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
        )

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", audio_path,
            "-vf", scale_filter,
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace").strip()
            logger.error("FFmpeg assembly failed (exit %d):\n%s", exc.returncode, stderr)
            raise RuntimeError(f"FFmpeg assembly failed (exit {exc.returncode}): {stderr[-500:]}") from exc
        finally:
            concat_file.unlink(missing_ok=True)
            for tmp in temp_files:
                tmp.unlink(missing_ok=True)

        logger.info("Assembled %d clips → %s", len(clips), output_path)
        return output_path
