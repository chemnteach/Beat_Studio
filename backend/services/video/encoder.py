"""VideoEncoder — FFmpeg-based final encoding with platform presets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger("beat_studio.video.encoder")


@dataclass
class EncoderPreset:
    """Quality preset for H.264 encoding.

    Attributes:
        crf: Constant Rate Factor (lower = higher quality, larger file).
        preset: FFmpeg ``-preset`` value (ultrafast … veryslow).
    """
    crf: int
    preset: str


@dataclass
class PlatformPreset:
    """Platform-specific output specification.

    Attributes:
        resolution: ``(width, height)`` in pixels.
        video_bitrate: Target video bitrate string (e.g. ``"8M"``).
        audio_bitrate: Target audio bitrate string (e.g. ``"320k"``).
    """
    resolution: Tuple[int, int]
    video_bitrate: str
    audio_bitrate: str


class VideoEncoder:
    """FFmpeg-based final encoding with quality and platform presets.

    Quality presets:
        draft     → CRF 23, fast
        standard  → CRF 18, medium
        broadcast → CRF 15, slow

    Platform presets:
        tiktok/reels/shorts → 1080×1920, 5M video, 192k audio
        youtube/standard    → 1920×1080, 8M video, 320k audio

    Heavy FFmpeg calls are in ``_run_ffmpeg()`` for easy mocking in tests.

    Usage::

        enc = VideoEncoder()
        output = enc.encode(
            input_path="/tmp/assembled.mp4",
            output_path="/tmp/final.mp4",
            quality="standard",
            platform="youtube",
        )
    """

    _QUALITY_PRESETS = {
        "draft":     EncoderPreset(crf=23, preset="fast"),
        "standard":  EncoderPreset(crf=18, preset="medium"),
        "broadcast": EncoderPreset(crf=15, preset="slow"),
    }

    _PLATFORM_PRESETS = {
        "tiktok":   PlatformPreset(resolution=(1080, 1920), video_bitrate="5M",  audio_bitrate="192k"),
        "reels":    PlatformPreset(resolution=(1080, 1920), video_bitrate="5M",  audio_bitrate="192k"),
        "shorts":   PlatformPreset(resolution=(1080, 1920), video_bitrate="5M",  audio_bitrate="192k"),
        "youtube":  PlatformPreset(resolution=(1920, 1080), video_bitrate="8M",  audio_bitrate="320k"),
        "standard": PlatformPreset(resolution=(1920, 1080), video_bitrate="8M",  audio_bitrate="320k"),
    }

    def get_preset(self, quality: str) -> EncoderPreset:
        """Return the :class:`EncoderPreset` for ``quality``.

        Raises:
            KeyError: If ``quality`` is not recognised.
        """
        if quality not in self._QUALITY_PRESETS:
            raise KeyError(
                f"Unknown quality preset: '{quality}'. "
                f"Valid: {list(self._QUALITY_PRESETS)}"
            )
        return self._QUALITY_PRESETS[quality]

    def get_platform(self, platform: str) -> PlatformPreset:
        """Return the :class:`PlatformPreset` for ``platform``.

        Raises:
            KeyError: If ``platform`` is not recognised.
        """
        if platform not in self._PLATFORM_PRESETS:
            raise KeyError(
                f"Unknown platform: '{platform}'. "
                f"Valid: {list(self._PLATFORM_PRESETS)}"
            )
        return self._PLATFORM_PRESETS[platform]

    def encode(
        self,
        input_path: str,
        output_path: str,
        quality: str = "standard",
        platform: str = "youtube",
    ) -> str:
        """Encode a video file with the given quality and platform preset.

        Args:
            input_path: Path to the assembled input video.
            output_path: Destination path for the encoded output.
            quality: ``"draft"`` | ``"standard"`` | ``"broadcast"``.
            platform: ``"tiktok"`` | ``"reels"`` | ``"shorts"`` |
                ``"youtube"`` | ``"standard"``.

        Returns:
            Path to the encoded video file (same as ``output_path``).

        Raises:
            KeyError: If ``quality`` or ``platform`` is not recognised.
        """
        enc_preset = self.get_preset(quality)    # raises KeyError for invalid
        plat_preset = self.get_platform(platform)  # raises KeyError for invalid

        logger.info(
            "Encoding: %s → %s  quality=%s platform=%s",
            input_path,
            output_path,
            quality,
            platform,
        )

        return self._run_ffmpeg(
            input_path=input_path,
            output_path=output_path,
            enc_preset=enc_preset,
            plat_preset=plat_preset,
        )

    # ── Internal — mockable for unit tests ────────────────────────────────────

    def _run_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        enc_preset: EncoderPreset,
        plat_preset: PlatformPreset,
    ) -> str:
        """Run FFmpeg encoding.

        Production: builds FFmpeg command with CRF, preset, scale, bitrate
        constraints, and executes via subprocess.  Mock in tests.

        Returns:
            Path to the encoded file.
        """
        raise NotImplementedError(
            "_run_ffmpeg requires FFmpeg binary. Mock in tests."
        )
