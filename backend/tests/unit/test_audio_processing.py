"""Tests for audio processing utilities (Phase 2.3)."""
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest


class TestSeparateStems:
    def test_returns_dict_with_stem_keys(self, tmp_dir):
        from backend.services.audio.processing import separate_stems
        dummy = tmp_dir / "song.wav"
        dummy.write_bytes(b"\x00" * 100)
        mock_result = {
            "vocals": np.zeros(1000),
            "drums": np.zeros(1000),
            "bass": np.zeros(1000),
            "other": np.zeros(1000),
        }
        with patch("backend.services.audio.processing._run_demucs",
                   return_value=mock_result):
            result = separate_stems(str(dummy))
        for key in ("vocals", "drums", "bass", "other"):
            assert key in result

    def test_returns_numpy_arrays(self, tmp_dir):
        from backend.services.audio.processing import separate_stems
        dummy = tmp_dir / "song.wav"
        dummy.write_bytes(b"\x00" * 100)
        mock_result = {k: np.zeros(100) for k in ("vocals", "drums", "bass", "other")}
        with patch("backend.services.audio.processing._run_demucs",
                   return_value=mock_result):
            result = separate_stems(str(dummy))
        for key in ("vocals", "drums", "bass", "other"):
            assert isinstance(result[key], np.ndarray)


class TestTimeStretch:
    def test_returns_numpy_array(self):
        from backend.services.audio.processing import time_stretch
        y = np.ones(44100, dtype=np.float32)
        result = time_stretch(y, rate=1.0)
        assert isinstance(result, np.ndarray)

    def test_identity_ratio_unchanged_length(self):
        from backend.services.audio.processing import time_stretch
        y = np.ones(44100, dtype=np.float32)
        result = time_stretch(y, rate=1.0)
        assert len(result) == len(y)

    def test_stretch_2x_doubles_length(self):
        from backend.services.audio.processing import time_stretch
        y = np.ones(44100, dtype=np.float32)
        with patch("backend.services.audio.processing._pyrubberband_stretch",
                   side_effect=lambda y, sr, rate: np.ones(int(len(y) / rate))):
            result = time_stretch(y, rate=0.5, sr=44100)
        assert len(result) == pytest.approx(44100 * 2, abs=100)


class TestAlignToGrid:
    def test_returns_float(self):
        from backend.services.audio.processing import align_to_downbeat
        result = align_to_downbeat(0.1, beat_times=[0.0, 0.5, 1.0])
        assert isinstance(result, float)

    def test_snaps_to_nearest_beat(self):
        from backend.services.audio.processing import align_to_downbeat
        beat_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        # Downbeat at 0.1 — should snap to 0.0
        result = align_to_downbeat(0.1, beat_times=beat_times)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_empty_beat_times_returns_original(self):
        from backend.services.audio.processing import align_to_downbeat
        result = align_to_downbeat(0.75, beat_times=[])
        assert result == pytest.approx(0.75, abs=0.001)


class TestNormalizeLufs:
    def test_returns_audio_segment(self):
        from backend.services.audio.processing import normalize_lufs
        from pydub import AudioSegment
        seg = AudioSegment.silent(duration=1000)
        result = normalize_lufs(seg, target_lufs=-14)
        assert isinstance(result, AudioSegment)

    def test_silent_segment_not_raised(self):
        from backend.services.audio.processing import normalize_lufs
        from pydub import AudioSegment
        seg = AudioSegment.silent(duration=500)
        result = normalize_lufs(seg, target_lufs=-14)
        assert result is not None


class TestPitchShift:
    def test_zero_semitones_returns_same_length(self):
        from backend.services.audio.processing import pitch_shift
        y = np.ones(44100, dtype=np.float32)
        result = pitch_shift(y, sr=44100, semitones=0)
        assert len(result) == len(y)

    def test_returns_numpy_array(self):
        from backend.services.audio.processing import pitch_shift
        y = np.ones(44100, dtype=np.float32)
        result = pitch_shift(y, sr=44100, semitones=0)
        assert isinstance(result, np.ndarray)


class TestCalculateSemitoneShift:
    def test_same_key_returns_zero(self):
        from backend.services.audio.processing import calculate_semitone_shift
        assert calculate_semitone_shift("Cmaj", "Cmaj") == 0

    def test_known_shift_up(self):
        from backend.services.audio.processing import calculate_semitone_shift
        # C to G = 7 semitones, but chromatic circle may normalise to -5
        shift = calculate_semitone_shift("Cmaj", "Gmaj")
        assert abs(shift) <= 6

    def test_shift_within_range(self):
        from backend.services.audio.processing import calculate_semitone_shift
        keys = ["Cmaj", "Gmaj", "Dmaj", "Amaj", "Emaj", "Bmaj",
                "Fmaj", "Amin", "Emin", "Bmin"]
        for src in keys:
            for tgt in keys:
                shift = calculate_semitone_shift(src, tgt)
                assert -6 <= shift <= 6, f"shift {src}→{tgt} = {shift} out of range"


class TestExportAudio:
    def test_creates_file(self, tmp_dir):
        from backend.services.audio.processing import export_audio
        from pydub import AudioSegment
        seg = AudioSegment.silent(duration=500)
        out_path = str(tmp_dir / "out.mp3")
        export_audio(seg, out_path, fmt="mp3")
        import os
        assert os.path.exists(out_path)
