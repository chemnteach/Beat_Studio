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


# ── _align_lyrics_to_sections unit tests ─────────────────────────────────────

class TestAlignLyricsToSections:
    def _make_sec(self, stype, start, end):
        return _make_section(stype, start, end)

    def test_assigns_segment_to_containing_section(self):
        from backend.services.audio.analyzer import _align_lyrics_to_sections
        secs = [self._make_sec("intro", 0.0, 30.0),
                self._make_sec("verse", 30.0, 90.0),
                self._make_sec("outro", 90.0, 120.0)]
        timings = [{"start": 31.0, "end": 35.0, "text": "island girl"}]
        result = _align_lyrics_to_sections(secs, timings)
        assert result[0].lyrical_content == ""
        assert result[1].lyrical_content == "island girl"
        assert result[2].lyrical_content == ""

    def test_multiple_segments_concatenated(self):
        from backend.services.audio.analyzer import _align_lyrics_to_sections
        secs = [self._make_sec("verse", 0.0, 60.0),
                self._make_sec("outro", 60.0, 120.0)]
        timings = [
            {"start": 5.0, "end": 8.0, "text": "hello world"},
            {"start": 12.0, "end": 15.0, "text": "foo bar"},
        ]
        result = _align_lyrics_to_sections(secs, timings)
        assert "hello world" in result[0].lyrical_content
        assert "foo bar" in result[0].lyrical_content

    def test_segment_past_last_boundary_appended_to_last(self):
        from backend.services.audio.analyzer import _align_lyrics_to_sections
        secs = [self._make_sec("verse", 0.0, 30.0)]
        timings = [{"start": 35.0, "end": 40.0, "text": "overflow text"}]
        result = _align_lyrics_to_sections(secs, timings)
        assert "overflow text" in result[0].lyrical_content

    def test_empty_timings_leaves_sections_unchanged(self):
        from backend.services.audio.analyzer import _align_lyrics_to_sections
        secs = [self._make_sec("verse", 0.0, 30.0)]
        result = _align_lyrics_to_sections(secs, [])
        assert result[0].lyrical_content == ""

    def test_blank_text_segments_skipped(self):
        from backend.services.audio.analyzer import _align_lyrics_to_sections
        secs = [self._make_sec("verse", 0.0, 60.0)]
        timings = [{"start": 5.0, "end": 8.0, "text": "  "},
                   {"start": 10.0, "end": 12.0, "text": ""}]
        result = _align_lyrics_to_sections(secs, timings)
        assert result[0].lyrical_content == ""

    def test_uses_segment_midpoint_not_start(self):
        from backend.services.audio.analyzer import _align_lyrics_to_sections
        # Segment starts at 28s (in section 0, end=30) but midpoint at 31s (section 1)
        secs = [self._make_sec("intro", 0.0, 30.0),
                self._make_sec("verse", 30.0, 90.0)]
        timings = [{"start": 28.0, "end": 34.0, "text": "mid crosses boundary"}]
        result = _align_lyrics_to_sections(secs, timings)
        assert result[1].lyrical_content == "mid crosses boundary"
        assert result[0].lyrical_content == ""


# ── relabel_by_lyrical_repetition unit tests ─────────────────────────────────

class TestRelabelByLyricalRepetition:
    def _sec(self, stype, start, end, lyrics=""):
        s = _make_section(stype, start, end)
        s.lyrical_content = lyrics
        return s

    def test_repeated_lyrics_promote_to_chorus(self):
        from backend.services.audio.analysis import relabel_by_lyrical_repetition
        chorus_lyrics = "island girl you came and you gave without taking"
        secs = [
            self._sec("intro", 0, 20),
            self._sec("verse", 20, 60, "she was sitting there by the side of the road"),
            self._sec("verse", 60, 100, chorus_lyrics),
            self._sec("verse", 100, 140, chorus_lyrics),
            self._sec("outro", 140, 160),
        ]
        result = relabel_by_lyrical_repetition(secs)
        assert result[0].section_type == "intro"
        assert result[4].section_type == "outro"
        # Both sections with identical lyrics become chorus
        assert result[2].section_type == "chorus"
        assert result[3].section_type == "chorus"

    def test_unique_lyrics_not_promoted(self):
        from backend.services.audio.analysis import relabel_by_lyrical_repetition
        secs = [
            self._sec("intro", 0, 20),
            self._sec("verse", 20, 60, "one flew over the cuckoo nest"),
            self._sec("verse", 60, 100, "something completely different here entirely"),
            self._sec("outro", 100, 120),
        ]
        result = relabel_by_lyrical_repetition(secs)
        # Low similarity — types unchanged
        assert result[1].section_type == "verse"
        assert result[2].section_type == "verse"

    def test_no_lyrics_returns_unchanged(self):
        from backend.services.audio.analysis import relabel_by_lyrical_repetition
        secs = [
            self._sec("intro", 0, 20),
            self._sec("verse", 20, 60, ""),
            self._sec("chorus", 60, 100, ""),
            self._sec("outro", 100, 120),
        ]
        result = relabel_by_lyrical_repetition(secs)
        assert result[1].section_type == "verse"
        assert result[2].section_type == "chorus"

    def test_intro_and_outro_never_relabeled(self):
        from backend.services.audio.analysis import relabel_by_lyrical_repetition
        repeated = "island girl island girl island girl"
        secs = [
            self._sec("intro", 0, 20, repeated),
            self._sec("verse", 20, 60, repeated),
            self._sec("outro", 60, 80, repeated),
        ]
        result = relabel_by_lyrical_repetition(secs)
        assert result[0].section_type == "intro"
        assert result[-1].section_type == "outro"

    def test_two_sections_returns_unchanged(self):
        from backend.services.audio.analysis import relabel_by_lyrical_repetition
        secs = [self._sec("intro", 0, 30, "same text"),
                self._sec("outro", 30, 60, "same text")]
        result = relabel_by_lyrical_repetition(secs)
        assert result[0].section_type == "intro"
        assert result[1].section_type == "outro"


# ── _adaptive_k unit tests ────────────────────────────────────────────────────

class TestAdaptiveK:
    """_adaptive_k must return a value in [min_k, max_k] and degrade gracefully."""

    def _flat_chroma(self, n_frames: int = 200) -> "np.ndarray":
        """Uniform chroma — no structural content, should return min_k."""
        return np.ones((12, n_frames), dtype=np.float32) / 12

    def _striped_chroma(self, n_frames: int = 400, n_stripes: int = 8) -> "np.ndarray":
        """Alternating chroma blocks simulate distinct sections."""
        chroma = np.zeros((12, n_frames), dtype=np.float32)
        stripe_len = n_frames // n_stripes
        for i in range(n_stripes):
            note = i % 12
            chroma[note, i * stripe_len:(i + 1) * stripe_len] = 1.0
        return chroma

    def test_returns_int_in_valid_range(self):
        from backend.services.audio.analysis import _adaptive_k
        chroma = self._flat_chroma(300)
        fake_y = np.zeros(44100 * 5, dtype=np.float32)
        with patch("backend.services.audio.analysis.librosa.beat.beat_track",
                   return_value=(np.array([120.0]), np.arange(0, 300, 2))), \
             patch("backend.services.audio.analysis.librosa.frames_to_time",
                   return_value=np.linspace(0, 60, 150)), \
             patch("backend.services.audio.analysis.librosa.get_duration",
                   return_value=60.0), \
             patch("backend.services.audio.analysis.librosa.util.sync",
                   return_value=chroma[:, :150]), \
             patch("backend.services.audio.analysis.librosa.segment.recurrence_matrix",
                   return_value=np.eye(150, dtype=np.float32)), \
             patch("backend.services.audio.analysis.librosa.util.peak_pick",
                   return_value=np.array([20, 50, 80, 110, 130])):
            k = _adaptive_k(chroma, fake_y, 44100)
        assert isinstance(k, int)
        assert 4 <= k <= 14

    def test_clamps_to_min_k(self):
        from backend.services.audio.analysis import _adaptive_k
        chroma = self._flat_chroma(50)
        fake_y = np.zeros(44100, dtype=np.float32)
        with patch("backend.services.audio.analysis.librosa.beat.beat_track",
                   return_value=(np.array([120.0]), np.arange(0, 50, 2))), \
             patch("backend.services.audio.analysis.librosa.frames_to_time",
                   return_value=np.linspace(0, 30, 25)), \
             patch("backend.services.audio.analysis.librosa.get_duration",
                   return_value=30.0), \
             patch("backend.services.audio.analysis.librosa.util.sync",
                   return_value=chroma[:, :25]), \
             patch("backend.services.audio.analysis.librosa.segment.recurrence_matrix",
                   return_value=np.eye(25, dtype=np.float32)), \
             patch("backend.services.audio.analysis.librosa.util.peak_pick",
                   return_value=np.array([])):  # no peaks
            k = _adaptive_k(chroma, fake_y, 44100, min_k=4)
        assert k == 4  # min_k when no peaks detected → 0+2=2, clamped to 4

    def test_clamps_to_max_k(self):
        from backend.services.audio.analysis import _adaptive_k
        chroma = self._flat_chroma(400)
        fake_y = np.zeros(44100 * 10, dtype=np.float32)
        # 20 peaks all in the middle of the song → should clamp to max_k=14
        mid_times = np.linspace(20, 180, 400)  # 400 frames, all in 20s-180s range
        with patch("backend.services.audio.analysis.librosa.beat.beat_track",
                   return_value=(np.array([120.0]), np.arange(0, 400, 2))), \
             patch("backend.services.audio.analysis.librosa.frames_to_time",
                   return_value=mid_times), \
             patch("backend.services.audio.analysis.librosa.get_duration",
                   return_value=200.0), \
             patch("backend.services.audio.analysis.librosa.util.sync",
                   return_value=chroma[:, :200]), \
             patch("backend.services.audio.analysis.librosa.segment.recurrence_matrix",
                   return_value=np.eye(200, dtype=np.float32)), \
             patch("backend.services.audio.analysis.librosa.util.peak_pick",
                   return_value=np.arange(20)):  # 20 peaks → k=22, clamped to 14
            k = _adaptive_k(chroma, fake_y, 44100, max_k=14)
        assert k == 14

    def test_falls_back_on_exception(self):
        from backend.services.audio.analysis import _adaptive_k
        chroma = self._flat_chroma(100)
        fake_y = np.zeros(44100, dtype=np.float32)
        with patch("backend.services.audio.analysis.librosa.beat.beat_track",
                   side_effect=RuntimeError("beat track failed")):
            k = _adaptive_k(chroma, fake_y, 44100)
        assert k == 9  # default fallback

    def test_very_short_audio_returns_min_k(self):
        from backend.services.audio.analysis import _adaptive_k
        chroma = self._flat_chroma(5)
        fake_y = np.zeros(44100, dtype=np.float32)
        with patch("backend.services.audio.analysis.librosa.beat.beat_track",
                   return_value=(np.array([120.0]), np.arange(0, 5))), \
             patch("backend.services.audio.analysis.librosa.frames_to_time",
                   return_value=np.linspace(0, 5, 5)), \
             patch("backend.services.audio.analysis.librosa.get_duration",
                   return_value=5.0), \
             patch("backend.services.audio.analysis.librosa.util.sync",
                   return_value=chroma[:, :5]):
            k = _adaptive_k(chroma, fake_y, 44100, min_k=4)
        assert k == 4  # n_frames < 8 → return min_k immediately


# ── pre_chorus in post_process_sections unit tests ───────────────────────────

class TestPreChorusLabeling:
    """pre_chorus fires in step 4 of post_process_sections."""

    def _sec(self, stype, start, end, energy):
        s = _make_section(stype, start, end, energy)
        return s

    def test_verse_before_chorus_becomes_pre_chorus(self):
        from backend.services.audio.analysis import post_process_sections
        # verse (energy=0.75) → chorus (energy=0.90): 0.75 >= 0.6*0.90=0.54 AND 0.75 < 0.90
        secs = [
            self._sec("intro",  0,  20, 0.30),
            self._sec("verse",  20, 40, 0.75),
            self._sec("chorus", 40, 80, 0.90),
            self._sec("outro",  80, 100, 0.20),
        ]
        result = post_process_sections(secs, 100.0)
        types = [s.section_type for s in result]
        assert "pre_chorus" in types
        pre_idx = types.index("pre_chorus")
        assert result[pre_idx + 1].section_type == "chorus"

    def test_low_energy_verse_before_chorus_stays_verse(self):
        from backend.services.audio.analysis import post_process_sections
        # verse (energy=0.40) → chorus (energy=0.90): 0.40 < 0.6*0.90=0.54 → no pre_chorus
        secs = [
            self._sec("intro",  0,  20, 0.30),
            self._sec("verse",  20, 40, 0.40),
            self._sec("chorus", 40, 80, 0.90),
            self._sec("outro",  80, 100, 0.20),
        ]
        result = post_process_sections(secs, 100.0)
        types = [s.section_type for s in result]
        assert "pre_chorus" not in types

    def test_pre_chorus_not_assigned_when_same_energy_as_chorus(self):
        from backend.services.audio.analysis import post_process_sections
        # If verse energy == chorus energy, condition `curr < nxt` fails → no pre_chorus
        secs = [
            self._sec("intro",  0,  20, 0.30),
            self._sec("verse",  20, 40, 0.90),
            self._sec("chorus", 40, 80, 0.90),
            self._sec("outro",  80, 100, 0.20),
        ]
        result = post_process_sections(secs, 100.0)
        types = [s.section_type for s in result]
        assert "pre_chorus" not in types

    def test_intro_and_outro_never_become_pre_chorus(self):
        from backend.services.audio.analysis import post_process_sections
        secs = [
            self._sec("intro",  0,  20, 0.85),
            self._sec("chorus", 20, 60, 0.90),
            self._sec("outro",  60, 80, 0.20),
        ]
        result = post_process_sections(secs, 80.0)
        assert result[0].section_type == "intro"
        assert result[-1].section_type == "outro"


# ── relabel_by_hook_phrase unit tests ─────────────────────────────────────────

class TestRelabelByHookPhrase:
    """relabel_by_hook_phrase promotes sections with the repeated hook to chorus."""

    def _sec(self, stype, start, end, energy, lyrics=""):
        s = _make_section(stype, start, end, energy)
        from dataclasses import replace
        return SectionInfo(
            section_type=stype,
            start_sec=start,
            end_sec=end,
            duration_sec=end - start,
            energy_level=energy,
            spectral_centroid=2000.0,
            tempo_stability=0.8,
            vocal_density="medium",
            vocal_intensity=energy,
            lyrical_content=lyrics,
            emotional_tone="neutral",
            lyrical_function="narrative",
            themes=[],
        )

    def test_hook_sections_become_chorus(self):
        from backend.services.audio.analysis import relabel_by_hook_phrase
        hook = "island girl"
        secs = [
            self._sec("intro",  0,  20, 0.30, ""),
            self._sec("verse",  20, 50, 0.60, "walking down the street alone"),
            self._sec("verse",  50, 80, 0.85, f"she is an {hook} dancing in the night"),
            self._sec("verse",  80, 110, 0.85, f"remember {hook} always on my mind"),
            self._sec("outro",  110, 130, 0.20, ""),
        ]
        result = relabel_by_hook_phrase(secs)
        types = [s.section_type for s in result]
        # sections 2 and 3 (inner idx 1 and 2) contain "island girl" → chorus
        assert types[2] == "chorus"
        assert types[3] == "chorus"

    def test_non_hook_sections_unchanged(self):
        from backend.services.audio.analysis import relabel_by_hook_phrase
        secs = [
            self._sec("intro",  0,  20, 0.30, ""),
            self._sec("verse",  20, 50, 0.60, "totally unique verse lyrics here"),
            self._sec("chorus", 50, 80, 0.85, "island girl dancing in the night"),
            self._sec("verse",  80, 110, 0.70, "another verse with different words"),
            self._sec("outro",  110, 130, 0.20, ""),
        ]
        result = relabel_by_hook_phrase(secs)
        # Only the sections with island girl get chorus; verse at idx 1 stays verse
        assert result[1].section_type == "verse"

    def test_no_repeated_phrases_returns_unchanged(self):
        from backend.services.audio.analysis import relabel_by_hook_phrase
        secs = [
            self._sec("intro",  0,  20, 0.30, ""),
            self._sec("verse",  20, 50, 0.60, "unique words no repetition here"),
            self._sec("chorus", 50, 80, 0.85, "completely different content there"),
            self._sec("outro",  80, 100, 0.20, ""),
        ]
        result = relabel_by_hook_phrase(secs)
        types = [s.section_type for s in result]
        assert types == ["intro", "verse", "chorus", "outro"]

    def test_intro_and_outro_protected(self):
        from backend.services.audio.analysis import relabel_by_hook_phrase
        hook = "island girl"
        secs = [
            self._sec("intro",  0,  20, 0.30, f"{hook} welcome to the show"),
            self._sec("verse",  20, 50, 0.85, f"she is {hook} dancing all night long"),
            self._sec("verse",  50, 80, 0.85, f"remember {hook} always on my mind"),
            self._sec("outro",  80, 100, 0.20, f"{hook} goodbye see you soon"),
        ]
        result = relabel_by_hook_phrase(secs)
        assert result[0].section_type == "intro"
        assert result[-1].section_type == "outro"

    def test_bridge_detected_between_two_choruses(self):
        from backend.services.audio.analysis import relabel_by_hook_phrase
        hook = "island girl"
        secs = [
            self._sec("intro",   0,  20, 0.30, ""),
            self._sec("verse",  20,  50, 0.85, f"she is {hook} dancing all night"),
            self._sec("verse",  50,  80, 0.50, "something completely different here"),
            self._sec("verse",  80, 110, 0.85, f"remember {hook} always on my mind"),
            self._sec("outro", 110, 130, 0.20, ""),
        ]
        result = relabel_by_hook_phrase(secs)
        types = [s.section_type for s in result]
        # sections 1 and 3 → chorus (hook); section 2 between them, lower energy → bridge
        assert types[1] == "chorus"
        assert types[3] == "chorus"
        assert types[2] == "bridge"

    def test_pre_chorus_assigned_before_hook_chorus(self):
        from backend.services.audio.analysis import relabel_by_hook_phrase
        hook = "island girl"
        secs = [
            self._sec("intro",   0,  20, 0.30, ""),
            self._sec("verse",  20,  50, 0.60, "walking down the street alone at night"),
            self._sec("bridge", 50,  80, 0.75, "building up the tension now rising"),
            self._sec("verse",  80, 110, 0.85, f"she is {hook} dancing all night long"),
            self._sec("verse", 110, 140, 0.85, f"remember {hook} always on my mind"),
            self._sec("outro", 140, 160, 0.20, ""),
        ]
        result = relabel_by_hook_phrase(secs)
        types = [s.section_type for s in result]
        # Inner sections: idx 0=verse, 1=bridge, 2=hook-chorus, 3=hook-chorus
        # bridge (idx 1) precedes hook-chorus (idx 2) and isn't verse → pre_chorus
        assert types[3] == "chorus"  # first hook section
        assert types[2] == "pre_chorus"  # bridge before it → pre_chorus

    def test_empty_lyrics_skips_gracefully(self):
        from backend.services.audio.analysis import relabel_by_hook_phrase
        secs = [
            self._sec("intro",  0,  20, 0.30, ""),
            self._sec("verse",  20, 50, 0.60, ""),
            self._sec("chorus", 50, 80, 0.85, ""),
            self._sec("outro",  80, 100, 0.20, ""),
        ]
        result = relabel_by_hook_phrase(secs)
        # No lyrics → no change
        types = [s.section_type for s in result]
        assert types == ["intro", "verse", "chorus", "outro"]
