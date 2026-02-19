"""Tests for the mashup analyst agent (Phase 2)."""
from unittest.mock import MagicMock, patch
import pytest


class TestProfileAudio:
    def _make_fake_analysis(self):
        return {
            "bpm": 120.0,
            "key": "Cmaj",
            "camelot": "8B",
            "duration_sec": 180.0,
            "energy_level": 0.6,
            "first_downbeat_sec": 0.1,
            "transcript": "test lyrics",
            "word_timings": [],
            "has_vocals": True,
            "mood_summary": "upbeat",
            "genres": ["pop"],
            "primary_genre": "pop",
            "irony_score": 0,
            "valence": 7,
            "emotional_arc": "hopeful",
            "sections": [],
        }

    def test_returns_dict_with_status(self, tmp_dir):
        from backend.services.mashup.analyst import profile_audio
        wav = tmp_dir / "song.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch("backend.services.mashup.analyst.AudioAnalyzer") as mock_az_cls:
            mock_az = MagicMock()
            mock_analysis = MagicMock()
            mock_analysis.bpm = 120.0
            mock_analysis.key = "Cmaj"
            mock_analysis.camelot = "8B"
            mock_analysis.duration_sec = 180.0
            mock_analysis.energy_level = 0.6
            mock_analysis.first_downbeat_sec = 0.1
            mock_analysis.transcript = "lyrics"
            mock_analysis.word_timings = []
            mock_analysis.has_vocals = True
            mock_analysis.mood_summary = "upbeat"
            mock_analysis.genres = ["pop"]
            mock_analysis.primary_genre = "pop"
            mock_analysis.irony_score = 0
            mock_analysis.valence = 7
            mock_analysis.emotional_arc = "hopeful"
            mock_analysis.sections = []
            mock_az.analyze.return_value = mock_analysis
            mock_az.get_mashup_metadata.return_value = self._make_fake_analysis()
            mock_az_cls.return_value = mock_az

            with patch("backend.services.mashup.analyst.MashupLibrary"):
                result = profile_audio(
                    file_path=str(wav),
                    song_id="test_song",
                    artist="Test",
                    title="Song",
                )

        assert result["status"] == "success"
        assert result["song_id"] == "test_song"

    def test_returns_metadata_in_result(self, tmp_dir):
        from backend.services.mashup.analyst import profile_audio
        wav = tmp_dir / "song.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch("backend.services.mashup.analyst.AudioAnalyzer") as mock_az_cls:
            mock_az = MagicMock()
            mock_analysis = MagicMock()
            mock_analysis.bpm = 128.0
            mock_analysis.sections = []
            mock_az.analyze.return_value = mock_analysis
            mock_az.get_mashup_metadata.return_value = {"bpm": 128.0, "key": "Gmaj"}
            mock_az_cls.return_value = mock_az

            with patch("backend.services.mashup.analyst.MashupLibrary"):
                result = profile_audio(str(wav), "test_id", "Artist", "Song")

        assert "metadata" in result

    def test_failed_analysis_raises(self, tmp_dir):
        from backend.services.mashup.analyst import profile_audio, AnalysisError
        wav = tmp_dir / "song.wav"
        wav.write_bytes(b"RIFF" + b"\x00" * 100)

        with patch("backend.services.mashup.analyst.AudioAnalyzer") as mock_az_cls:
            mock_az = MagicMock()
            mock_az.analyze.side_effect = RuntimeError("librosa failed")
            mock_az_cls.return_value = mock_az

            with pytest.raises(AnalysisError):
                profile_audio(str(wav), "fail_id", "Artist", "Song")
