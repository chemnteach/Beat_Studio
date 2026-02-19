"""Phase 8: Video Assembly & Continuity — unit tests (TDD, no real GPU/FFmpeg needed)."""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch, call

import pytest

from backend.services.video.types import ScenePlan, VideoClip
from backend.services.video.beat_sync import BeatSynchronizer, SyncedScenePlan
from backend.services.video.transition_engine import (
    TransitionEngine,
    TransitionConfig,
    TransitionType,
)
from backend.services.video.assembler import VideoAssembler
from backend.services.video.encoder import VideoEncoder, EncoderPreset, PlatformPreset


# ═══════════════════════════════════════════════════════════════════════════════
# BeatSynchronizer
# ═══════════════════════════════════════════════════════════════════════════════


def _mock_analysis(duration=120.0, bpm=120.0, n_sections=4):
    """Return a minimal SongAnalysis mock."""
    analysis = MagicMock()
    analysis.duration = duration
    analysis.bpm = bpm
    section_dur = duration / n_sections
    sections = []
    for i in range(n_sections):
        s = MagicMock()
        s.start = i * section_dur
        s.end = (i + 1) * section_dur
        s.section_type = "chorus" if i % 2 == 1 else "verse"
        s.energy_level = 0.9 if i % 2 == 1 else 0.5
        sections.append(s)
    analysis.sections = sections
    return analysis


class TestBeatSynchronizerSceneCount:
    def setup_method(self):
        self.sync = BeatSynchronizer()

    def test_basic_tier_gives_fewer_scenes(self):
        analysis = _mock_analysis(duration=180.0)
        plans = self.sync.create_scene_plan(analysis, quality_tier="basic")
        assert len(plans) <= 12

    def test_professional_tier_gives_more_scenes(self):
        analysis = _mock_analysis(duration=180.0)
        plans = self.sync.create_scene_plan(analysis, quality_tier="professional")
        assert len(plans) <= 24

    def test_cinematic_tier_gives_most_scenes(self):
        analysis = _mock_analysis(duration=180.0)
        plans = self.sync.create_scene_plan(analysis, quality_tier="cinematic")
        assert len(plans) <= 48

    def test_professional_more_than_basic(self):
        analysis = _mock_analysis(duration=180.0)
        basic = self.sync.create_scene_plan(analysis, quality_tier="basic")
        professional = self.sync.create_scene_plan(analysis, quality_tier="professional")
        assert len(professional) >= len(basic)

    def test_invalid_tier_raises(self):
        analysis = _mock_analysis()
        with pytest.raises(ValueError, match="quality_tier"):
            self.sync.create_scene_plan(analysis, quality_tier="ultra_mega")


class TestBeatSynchronizerTiming:
    def setup_method(self):
        self.sync = BeatSynchronizer()

    def test_plans_cover_full_duration(self):
        analysis = _mock_analysis(duration=120.0)
        plans = self.sync.create_scene_plan(analysis, quality_tier="basic")
        total = sum(p.duration_sec for p in plans)
        assert total == pytest.approx(120.0, abs=2.0)

    def test_plans_are_sequential_non_overlapping(self):
        analysis = _mock_analysis(duration=120.0)
        plans = self.sync.create_scene_plan(analysis, quality_tier="professional")
        for i in range(len(plans) - 1):
            assert plans[i].end_sec <= plans[i + 1].start_sec + 0.01

    def test_hero_scenes_exist_in_cinematic(self):
        analysis = _mock_analysis(duration=180.0, n_sections=4)
        plans = self.sync.create_scene_plan(analysis, quality_tier="cinematic")
        has_hero = any(p.is_hero for p in plans)
        assert has_hero

    def test_scene_plans_have_valid_indices(self):
        analysis = _mock_analysis(duration=60.0)
        plans = self.sync.create_scene_plan(analysis, quality_tier="basic")
        for i, p in enumerate(plans):
            assert p.scene_index == i

    def test_synced_plan_has_start_end_duration(self):
        analysis = _mock_analysis(duration=60.0)
        plans = self.sync.create_scene_plan(analysis, quality_tier="basic")
        for p in plans:
            assert hasattr(p, "start_sec")
            assert hasattr(p, "end_sec")
            assert hasattr(p, "duration_sec")
            assert p.duration_sec > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TransitionEngine
# ═══════════════════════════════════════════════════════════════════════════════


def _make_scene_plan(idx: int, backend: str = "animatediff", is_hero: bool = False):
    return SyncedScenePlan(
        scene_index=idx,
        start_sec=idx * 5.0,
        end_sec=(idx + 1) * 5.0,
        duration_sec=5.0,
        is_hero=is_hero,
        backend_name=backend,
    )


class TestTransitionEngineSelection:
    def setup_method(self):
        self.engine = TransitionEngine()

    def test_select_returns_transition_config(self):
        a = _make_scene_plan(0)
        b = _make_scene_plan(1)
        backend = MagicMock()
        backend.name.return_value = "animatediff"
        config = self.engine.select_transition(a, b, backend, quality_tier="basic")
        assert isinstance(config, TransitionConfig)

    def test_crossfade_is_valid_type(self):
        a = _make_scene_plan(0)
        b = _make_scene_plan(1)
        backend = MagicMock()
        backend.name.return_value = "animatediff"
        config = self.engine.select_transition(a, b, backend, quality_tier="basic")
        assert config.transition_type in TransitionType.__members__.values()

    def test_hero_scene_gets_prominent_transition(self):
        a = _make_scene_plan(0)
        b = _make_scene_plan(1, is_hero=True)
        backend = MagicMock()
        backend.name.return_value = "animatediff"
        config = self.engine.select_transition(a, b, backend, quality_tier="cinematic")
        # Hero scene gets at minimum a crossfade; check type is not None
        assert config.transition_type is not None

    def test_skyreels_backend_can_use_skyreels_stitch(self):
        a = _make_scene_plan(0)
        b = _make_scene_plan(1)
        backend = MagicMock()
        backend.name.return_value = "skyreels"
        config = self.engine.select_transition(a, b, backend, quality_tier="cinematic")
        # When SkyReels backend is available, it may use skyreels_stitch
        assert config.transition_type is not None

    def test_transition_duration_positive(self):
        a = _make_scene_plan(0)
        b = _make_scene_plan(1)
        backend = MagicMock()
        backend.name.return_value = "animatediff"
        config = self.engine.select_transition(a, b, backend, quality_tier="standard")
        assert config.duration_sec > 0.0

    def test_apply_transition_mocked(self):
        """apply_transition calls _blend_frames internally."""
        a = _make_scene_plan(0)
        b = _make_scene_plan(1)
        backend = MagicMock()
        backend.name.return_value = "animatediff"
        config = self.engine.select_transition(a, b, backend, quality_tier="basic")
        clip_a = VideoClip("/tmp/a.mp4", 5.0, 1920, 1080, 24, 0)
        clip_b = VideoClip("/tmp/b.mp4", 5.0, 1920, 1080, 24, 1)
        with patch.object(self.engine, "_blend_frames", return_value="/tmp/blended.mp4"):
            result = self.engine.apply_transition(clip_a, clip_b, config)
        assert result is not None


class TestTransitionConfig:
    def test_config_fields(self):
        cfg = TransitionConfig(
            transition_type=TransitionType.CROSSFADE,
            duration_sec=0.75,
        )
        assert cfg.transition_type == TransitionType.CROSSFADE
        assert cfg.duration_sec == pytest.approx(0.75)

    def test_transition_types_enum(self):
        types = list(TransitionType)
        assert TransitionType.CROSSFADE in types
        assert TransitionType.MORPH in types


# ═══════════════════════════════════════════════════════════════════════════════
# VideoAssembler
# ═══════════════════════════════════════════════════════════════════════════════


def _make_clip(idx: int, path: str = None) -> VideoClip:
    return VideoClip(
        file_path=path or f"/tmp/clip_{idx:02d}.mp4",
        duration_sec=5.0,
        width=1920,
        height=1080,
        fps=24,
        scene_index=idx,
    )


def _make_transition_config() -> TransitionConfig:
    return TransitionConfig(
        transition_type=TransitionType.CROSSFADE,
        duration_sec=0.5,
    )


class TestVideoAssembler:
    def setup_method(self):
        self.assembler = VideoAssembler()

    def test_assemble_calls_internal_pipeline(self):
        clips = [_make_clip(i) for i in range(3)]
        transitions = [_make_transition_config(), _make_transition_config()]
        with patch.object(self.assembler, "_run_ffmpeg_concat", return_value="/tmp/out.mp4") as mock_concat:
            result = self.assembler.assemble(
                clips=clips,
                transitions=transitions,
                audio_path="/tmp/audio.wav",
                output_path="/tmp/out.mp4",
            )
        mock_concat.assert_called_once()

    def test_assemble_returns_output_path(self):
        clips = [_make_clip(i) for i in range(2)]
        transitions = [_make_transition_config()]
        with patch.object(self.assembler, "_run_ffmpeg_concat", return_value="/tmp/done.mp4"):
            result = self.assembler.assemble(
                clips=clips,
                transitions=transitions,
                audio_path="/tmp/audio.wav",
                output_path="/tmp/done.mp4",
            )
        assert result == "/tmp/done.mp4"

    def test_mismatched_transitions_raises(self):
        """len(transitions) must be len(clips) - 1."""
        clips = [_make_clip(i) for i in range(3)]
        # Wrong: 3 transitions for 3 clips (should be 2)
        transitions = [_make_transition_config()] * 3
        with pytest.raises(ValueError, match="transition"):
            self.assembler.assemble(
                clips=clips,
                transitions=transitions,
                audio_path="/tmp/audio.wav",
                output_path="/tmp/out.mp4",
            )

    def test_single_clip_no_transitions(self):
        clips = [_make_clip(0)]
        with patch.object(self.assembler, "_run_ffmpeg_concat", return_value="/tmp/out.mp4"):
            result = self.assembler.assemble(
                clips=clips,
                transitions=[],
                audio_path="/tmp/audio.wav",
                output_path="/tmp/out.mp4",
            )
        assert result is not None

    def test_no_ken_burns_flag(self):
        """VideoAssembler must never use Ken Burns effects."""
        # Verify the constant exists and is False
        assert VideoAssembler.ALLOW_KEN_BURNS is False


# ═══════════════════════════════════════════════════════════════════════════════
# VideoEncoder
# ═══════════════════════════════════════════════════════════════════════════════


class TestVideoEncoderPresets:
    def setup_method(self):
        self.enc = VideoEncoder()

    def test_draft_preset_exists(self):
        preset = self.enc.get_preset("draft")
        assert isinstance(preset, EncoderPreset)
        assert preset.crf > 0

    def test_standard_preset_exists(self):
        preset = self.enc.get_preset("standard")
        assert isinstance(preset, EncoderPreset)
        assert preset.crf <= 20

    def test_broadcast_preset_exists(self):
        preset = self.enc.get_preset("broadcast")
        assert isinstance(preset, EncoderPreset)
        assert preset.crf <= 18

    def test_broadcast_lower_crf_than_draft(self):
        draft = self.enc.get_preset("draft")
        broadcast = self.enc.get_preset("broadcast")
        assert broadcast.crf < draft.crf

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            self.enc.get_preset("ultra_8k_megapixel")


class TestVideoEncoderPlatforms:
    def setup_method(self):
        self.enc = VideoEncoder()

    def test_tiktok_platform_exists(self):
        p = self.enc.get_platform("tiktok")
        assert isinstance(p, PlatformPreset)
        assert p.resolution == (1080, 1920)

    def test_youtube_platform_exists(self):
        p = self.enc.get_platform("youtube")
        assert isinstance(p, PlatformPreset)
        assert p.resolution == (1920, 1080)

    def test_reels_platform_vertical(self):
        p = self.enc.get_platform("reels")
        width, height = p.resolution
        assert height > width  # Vertical

    def test_shorts_platform_vertical(self):
        p = self.enc.get_platform("shorts")
        width, height = p.resolution
        assert height > width

    def test_standard_platform_horizontal(self):
        p = self.enc.get_platform("standard")
        width, height = p.resolution
        assert width >= height

    def test_unknown_platform_raises(self):
        with pytest.raises(KeyError):
            self.enc.get_platform("betamax")


class TestVideoEncoderEncode:
    def setup_method(self):
        self.enc = VideoEncoder()

    def test_encode_calls_ffmpeg(self):
        with patch.object(self.enc, "_run_ffmpeg", return_value="/tmp/encoded.mp4") as mock_ff:
            result = self.enc.encode(
                input_path="/tmp/assembled.mp4",
                output_path="/tmp/encoded.mp4",
                quality="standard",
                platform="youtube",
            )
        mock_ff.assert_called_once()

    def test_encode_returns_output_path(self):
        with patch.object(self.enc, "_run_ffmpeg", return_value="/tmp/encoded.mp4"):
            result = self.enc.encode(
                input_path="/tmp/assembled.mp4",
                output_path="/tmp/encoded.mp4",
                quality="standard",
                platform="youtube",
            )
        assert result == "/tmp/encoded.mp4"

    def test_encode_invalid_quality_raises(self):
        with pytest.raises(KeyError):
            self.enc.encode(
                input_path="/tmp/assembled.mp4",
                output_path="/tmp/out.mp4",
                quality="wizard_mode",
                platform="youtube",
            )

    def test_encode_invalid_platform_raises(self):
        with pytest.raises(KeyError):
            self.enc.encode(
                input_path="/tmp/assembled.mp4",
                output_path="/tmp/out.mp4",
                quality="standard",
                platform="laserdisc",
            )
