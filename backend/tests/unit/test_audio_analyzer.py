"""Tests for the unified AudioAnalyzer (Phase 2.1)."""
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from backend.services.audio.types import SongAnalysis, SceneTiming, SectionInfo


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_section(section_type="verse", start=0.0, end=30.0, energy=0.5):
    return SectionInfo(
        section_type=section_type,
        start_sec=start,
        end_sec=end,
        duration_sec=end - start,
        energy_level=energy,
        spectral_centroid=2000.0,
        tempo_stability=0.8,
        vocal_density="medium",
        vocal_intensity=0.5,
        lyrical_content="",
        emotional_tone="hopeful",
        lyrical_function="narrative",
        themes=["love"],
    )


def _make_analysis(bpm=120.0, key="Cmaj", duration=180.0, sections=None):
    return SongAnalysis(
        artist="Test Artist",
        title="Test Song",
        file_path="/tmp/test.wav",
        bpm=bpm,
        key=key,
        camelot="8B",
        duration_sec=duration,
        sample_rate=44100,
        energy_level=0.5,
        first_downbeat_sec=0.1,
        sections=sections or [],
        beat_times=[0.5 * i for i in range(int(duration * 2))],
        transcript="test lyrics here",
        mood_summary="upbeat",
        genres=["pop"],
        primary_genre="pop",
    )


# ── AudioAnalyzer class tests ─────────────────────────────────────────────────

class TestAudioAnalyzerInit:
    def test_instantiation_default(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer()
        assert az is not None

    def test_instantiation_with_custom_config(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer(sample_rate=22050, whisper_model="tiny")
        assert az.sample_rate == 22050
        assert az.whisper_model == "tiny"


class TestAnalyzeDepthBasic:
    """basic depth: BPM, key, duration, energy (no I/O for heavy steps)."""

    def test_basic_depth_returns_song_analysis(self, tmp_dir):
        from backend.services.audio.analyzer import AudioAnalyzer
        dummy_wav = tmp_dir / "song.wav"
        dummy_wav.write_bytes(b"\x00" * 100)

        az = AudioAnalyzer()
        fake_y = np.zeros(44100)
        fake_sr = 44100

        with patch("backend.services.audio.analyzer.librosa.load",
                   return_value=(fake_y, fake_sr)), \
             patch("backend.services.audio.analyzer.librosa.to_mono",
                   return_value=fake_y), \
             patch("backend.services.audio.analyzer._analyze_signal_basic",
                   return_value={"bpm": 128.0, "key": "Gmaj", "camelot": "9B",
                                 "duration_sec": 1.0, "energy_level": 0.6,
                                 "first_downbeat_sec": 0.0}):
            result = az.analyze(str(dummy_wav), artist="A", title="T", depth="basic")

        assert isinstance(result, SongAnalysis)
        assert result.bpm == 128.0
        assert result.key == "Gmaj"
        assert result.camelot == "9B"
        assert result.artist == "A"
        assert result.title == "T"

    def test_basic_depth_skips_transcription(self, tmp_dir):
        from backend.services.audio.analyzer import AudioAnalyzer
        dummy_wav = tmp_dir / "song.wav"
        dummy_wav.write_bytes(b"\x00" * 100)
        az = AudioAnalyzer()
        fake_y = np.zeros(44100)

        with patch("backend.services.audio.analyzer.librosa.load",
                   return_value=(fake_y, 44100)), \
             patch("backend.services.audio.analyzer.librosa.to_mono",
                   return_value=fake_y), \
             patch("backend.services.audio.analyzer._analyze_signal_basic",
                   return_value={"bpm": 120.0, "key": "Cmaj", "camelot": "8B",
                                 "duration_sec": 1.0, "energy_level": 0.5,
                                 "first_downbeat_sec": 0.0}), \
             patch("backend.services.audio.analyzer.whisper") as mock_whisper:
            az.analyze(str(dummy_wav), artist="A", title="T", depth="basic")
            mock_whisper.load_model.assert_not_called()

    def test_basic_depth_skips_sections(self, tmp_dir):
        from backend.services.audio.analyzer import AudioAnalyzer
        dummy_wav = tmp_dir / "song.wav"
        dummy_wav.write_bytes(b"\x00" * 100)
        az = AudioAnalyzer()
        fake_y = np.zeros(44100)

        with patch("backend.services.audio.analyzer.librosa.load",
                   return_value=(fake_y, 44100)), \
             patch("backend.services.audio.analyzer.librosa.to_mono",
                   return_value=fake_y), \
             patch("backend.services.audio.analyzer._analyze_signal_basic",
                   return_value={"bpm": 120.0, "key": "Cmaj", "camelot": "8B",
                                 "duration_sec": 1.0, "energy_level": 0.5,
                                 "first_downbeat_sec": 0.0}), \
             patch("backend.services.audio.analyzer.detect_sections") as mock_det:
            az.analyze(str(dummy_wav), artist="A", title="T", depth="basic")
            mock_det.assert_not_called()


class TestAnalyzeDepthStandard:
    def test_standard_depth_includes_sections(self, tmp_dir):
        from backend.services.audio.analyzer import AudioAnalyzer
        dummy_wav = tmp_dir / "song.wav"
        dummy_wav.write_bytes(b"\x00" * 100)
        az = AudioAnalyzer()
        fake_y = np.zeros(44100 * 3)

        fake_sections = [_make_section("verse", 0.0, 30.0)]

        with patch("backend.services.audio.analyzer.librosa.load",
                   return_value=(fake_y, 44100)), \
             patch("backend.services.audio.analyzer.librosa.to_mono",
                   return_value=fake_y), \
             patch("backend.services.audio.analyzer.librosa.beat.beat_track",
                   return_value=(np.array([120.0]), np.array([0, 10, 20]))), \
             patch("backend.services.audio.analyzer.librosa.frames_to_time",
                   return_value=np.array([0.0, 0.5, 1.0])), \
             patch("backend.services.audio.analyzer._analyze_signal_basic",
                   return_value={"bpm": 120.0, "key": "Cmaj", "camelot": "8B",
                                 "duration_sec": 3.0, "energy_level": 0.5,
                                 "first_downbeat_sec": 0.0}), \
             patch("backend.services.audio.analyzer.detect_sections",
                   return_value=[(0.0, 30.0)]), \
             patch("backend.services.audio.analyzer._build_sections",
                   return_value=fake_sections):
            result = az.analyze(str(dummy_wav), artist="A", title="T", depth="standard")

        assert len(result.sections) == 1
        assert result.sections[0].section_type == "verse"

    def test_standard_depth_skips_llm(self, tmp_dir):
        from backend.services.audio.analyzer import AudioAnalyzer
        dummy_wav = tmp_dir / "song.wav"
        dummy_wav.write_bytes(b"\x00" * 100)
        az = AudioAnalyzer()
        fake_y = np.zeros(44100)

        with patch("backend.services.audio.analyzer.librosa.load",
                   return_value=(fake_y, 44100)), \
             patch("backend.services.audio.analyzer.librosa.to_mono",
                   return_value=fake_y), \
             patch("backend.services.audio.analyzer.librosa.beat.beat_track",
                   return_value=(np.array([120.0]), np.array([0, 10]))), \
             patch("backend.services.audio.analyzer.librosa.frames_to_time",
                   return_value=np.array([0.0, 0.5])), \
             patch("backend.services.audio.analyzer._analyze_signal_basic",
                   return_value={"bpm": 120.0, "key": "Cmaj", "camelot": "8B",
                                 "duration_sec": 1.0, "energy_level": 0.5,
                                 "first_downbeat_sec": 0.0}), \
             patch("backend.services.audio.analyzer.detect_sections",
                   return_value=[]), \
             patch("backend.services.audio.analyzer._build_sections",
                   return_value=[]), \
             patch("backend.services.audio.analyzer.analyze_song_semantics") as mock_sem:
            az.analyze(str(dummy_wav), artist="A", title="T", depth="standard")
            mock_sem.assert_not_called()


class TestGetSceneTimings:
    def test_returns_scene_timings_list(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer()
        sections = [
            _make_section("intro", 0, 20, energy=0.3),
            _make_section("verse", 20, 60, energy=0.5),
            _make_section("chorus", 60, 90, energy=0.9),
            _make_section("outro", 90, 120, energy=0.2),
        ]
        analysis = _make_analysis(duration=120.0, sections=sections)
        timings = az.get_scene_timings(analysis)
        assert len(timings) > 0
        assert all(isinstance(t, SceneTiming) for t in timings)

    def test_hero_scenes_are_high_energy(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer()
        sections = [
            _make_section("verse", 0, 30, energy=0.4),
            _make_section("chorus", 30, 60, energy=0.95),
            _make_section("verse", 60, 90, energy=0.5),
            _make_section("outro", 90, 120, energy=0.2),
        ]
        analysis = _make_analysis(duration=120.0, sections=sections)
        timings = az.get_scene_timings(analysis)
        hero_scenes = [t for t in timings if t.is_hero]
        non_hero = [t for t in timings if not t.is_hero]
        if hero_scenes and non_hero:
            assert min(t.energy_level for t in hero_scenes) >= \
                   min(t.energy_level for t in non_hero) - 0.01

    def test_scene_min_duration_respected(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer()
        sections = [_make_section("verse", 0, 60, energy=0.5)]
        analysis = _make_analysis(duration=60.0, sections=sections)
        timings = az.get_scene_timings(analysis, min_duration=2.5)
        assert all(t.duration_sec >= 2.5 for t in timings)

    def test_scene_max_duration_respected(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer()
        # One very long section that should be split
        sections = [_make_section("verse", 0, 120, energy=0.5)]
        analysis = _make_analysis(duration=120.0, sections=sections)
        analysis.beat_times = [0.5 * i for i in range(240)]
        timings = az.get_scene_timings(analysis, max_duration=8.0)
        assert all(t.duration_sec <= 8.0 + 0.01 for t in timings)

    def test_no_sections_falls_back_to_uniform(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer()
        analysis = _make_analysis(duration=60.0, sections=[])
        timings = az.get_scene_timings(analysis)
        assert len(timings) > 0
        total = sum(t.duration_sec for t in timings)
        assert abs(total - 60.0) < 1.0

    def test_scene_index_sequential(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer()
        sections = [
            _make_section("verse", 0, 30, energy=0.5),
            _make_section("chorus", 30, 60, energy=0.8),
        ]
        analysis = _make_analysis(duration=60.0, sections=sections)
        timings = az.get_scene_timings(analysis)
        for i, t in enumerate(timings):
            assert t.scene_index == i


class TestGetMashupMetadata:
    def test_returns_dict_with_required_keys(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer()
        analysis = _make_analysis()
        meta = az.get_mashup_metadata(analysis)
        for key in ("bpm", "key", "camelot", "duration_sec", "energy_level",
                    "mood_summary", "genres", "primary_genre"):
            assert key in meta

    def test_bpm_value_matches_analysis(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer()
        analysis = _make_analysis(bpm=140.0)
        meta = az.get_mashup_metadata(analysis)
        assert meta["bpm"] == 140.0

    def test_sections_preserved(self):
        from backend.services.audio.analyzer import AudioAnalyzer
        az = AudioAnalyzer()
        secs = [_make_section("chorus", 0, 30, energy=0.9)]
        analysis = _make_analysis(sections=secs)
        meta = az.get_mashup_metadata(analysis)
        assert "sections" in meta
        assert len(meta["sections"]) == 1


# ── _analyze_signal_basic unit tests ─────────────────────────────────────────

class TestAnalyzeSignalBasic:
    def test_returns_dict_with_expected_keys(self):
        from backend.services.audio.analyzer import _analyze_signal_basic
        fake_y = np.zeros(44100)
        with patch("backend.services.audio.analyzer.librosa.beat.beat_track",
                   return_value=(np.array([120.0]), np.array([0, 22, 44]))), \
             patch("backend.services.audio.analyzer.librosa.frames_to_time",
                   return_value=np.array([0.0, 0.5, 1.0])), \
             patch("backend.services.audio.analyzer.librosa.feature.chroma_cqt",
                   return_value=np.ones((12, 100))), \
             patch("backend.services.audio.analyzer.librosa.feature.rms",
                   return_value=np.array([[0.5] * 100])), \
             patch("backend.services.audio.analyzer.librosa.get_duration",
                   return_value=3.0), \
             patch("backend.services.audio.analyzer.estimate_key",
                   return_value="Cmaj"), \
             patch("backend.services.audio.analyzer.key_to_camelot",
                   return_value="8B"):
            result = _analyze_signal_basic(fake_y, 44100)
        assert "bpm" in result
        assert "key" in result
        assert "camelot" in result
        assert "energy_level" in result
        assert "duration_sec" in result
        assert "first_downbeat_sec" in result

    def test_bpm_from_beat_track(self):
        from backend.services.audio.analyzer import _analyze_signal_basic
        fake_y = np.zeros(44100)
        with patch("backend.services.audio.analyzer.librosa.beat.beat_track",
                   return_value=(np.array([100.0]), np.array([0, 22]))), \
             patch("backend.services.audio.analyzer.librosa.frames_to_time",
                   return_value=np.array([0.0, 0.5])), \
             patch("backend.services.audio.analyzer.librosa.feature.chroma_cqt",
                   return_value=np.ones((12, 100))), \
             patch("backend.services.audio.analyzer.librosa.feature.rms",
                   return_value=np.array([[0.3] * 100])), \
             patch("backend.services.audio.analyzer.librosa.get_duration",
                   return_value=1.0), \
             patch("backend.services.audio.analyzer.estimate_key",
                   return_value="Amin"), \
             patch("backend.services.audio.analyzer.key_to_camelot",
                   return_value="8A"):
            result = _analyze_signal_basic(fake_y, 44100)
        assert result["bpm"] == pytest.approx(100.0, abs=1.0)
