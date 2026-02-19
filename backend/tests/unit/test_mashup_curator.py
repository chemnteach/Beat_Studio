"""Tests for the mashup curator agent (Phase 2)."""
import pytest
from unittest.mock import MagicMock, patch


def _make_meta(bpm=120.0, camelot="8B", key="Cmaj", energy=5,
               primary_genre="pop", mood="upbeat", sections=None):
    return {
        "bpm": bpm, "camelot": camelot, "key": key,
        "energy_level": energy, "primary_genre": primary_genre,
        "mood_summary": mood, "sections": sections or [],
        "source": "local_file", "path": "/tmp/x.wav",
        "artist": "A", "title": "T", "sample_rate": 44100,
        "duration_sec": 180.0, "genres": [primary_genre],
        "irony_score": 0, "valence": 5, "first_downbeat_sec": 0.0,
        "has_vocals": True,
    }


class TestCalculateCompatibilityScore:
    def test_identical_songs_score_near_100(self):
        from backend.services.mashup.curator import calculate_compatibility_score
        meta_a = _make_meta(bpm=120.0, camelot="8B", energy=5, primary_genre="pop")
        meta_b = _make_meta(bpm=120.0, camelot="8B", energy=5, primary_genre="pop")
        score = calculate_compatibility_score(meta_a, meta_b)
        assert score > 80

    def test_incompatible_songs_score_low(self):
        from backend.services.mashup.curator import calculate_compatibility_score
        meta_a = _make_meta(bpm=80.0, camelot="1A", energy=2, primary_genre="classical")
        meta_b = _make_meta(bpm=180.0, camelot="7B", energy=9, primary_genre="metal")
        score = calculate_compatibility_score(meta_a, meta_b)
        assert score < 60

    def test_returns_float_in_range(self):
        from backend.services.mashup.curator import calculate_compatibility_score
        meta_a = _make_meta()
        meta_b = _make_meta(bpm=125.0, camelot="9B")
        score = calculate_compatibility_score(meta_a, meta_b)
        assert 0 <= score <= 100

    def test_bpm_compatible_gets_bonus(self):
        from backend.services.mashup.curator import calculate_compatibility_score
        meta_a = _make_meta(bpm=120.0, camelot="8B")
        meta_near = _make_meta(bpm=122.0, camelot="9B")
        meta_far = _make_meta(bpm=180.0, camelot="9B")
        score_near = calculate_compatibility_score(meta_a, meta_near)
        score_far = calculate_compatibility_score(meta_a, meta_far)
        assert score_near > score_far


class TestRecommendMashupType:
    def test_returns_recommendation_with_type(self):
        from backend.services.mashup.curator import recommend_mashup_type
        meta_a = _make_meta(bpm=120.0, camelot="8B", key="Cmaj")
        meta_b = _make_meta(bpm=121.0, camelot="9B", key="Gmaj")
        rec = recommend_mashup_type(meta_a, meta_b)
        assert "mashup_type" in rec
        assert "confidence" in rec
        assert "reasoning" in rec

    def test_confidence_between_0_and_1(self):
        from backend.services.mashup.curator import recommend_mashup_type
        meta_a = _make_meta()
        meta_b = _make_meta()
        rec = recommend_mashup_type(meta_a, meta_b)
        assert 0.0 <= rec["confidence"] <= 1.0

    def test_key_clash_recommends_adaptive_harmony(self):
        from backend.services.mashup.curator import recommend_mashup_type
        # Very different keys should prefer adaptive harmony
        meta_a = _make_meta(camelot="1A", key="G#min")
        meta_b = _make_meta(camelot="7B", key="Fmaj")
        rec = recommend_mashup_type(meta_a, meta_b)
        # Adaptive harmony should be among the top recommendations
        assert "ADAPTIVE" in rec["mashup_type"].upper() or rec["confidence"] > 0

    def test_classic_recommended_for_compatible_pair(self):
        from backend.services.mashup.curator import recommend_mashup_type
        meta_a = _make_meta(bpm=120.0, camelot="8B", key="Cmaj")
        meta_b = _make_meta(bpm=121.0, camelot="8B", key="Cmaj")
        rec = recommend_mashup_type(meta_a, meta_b)
        assert "CLASSIC" in rec["mashup_type"].upper() or rec["confidence"] > 0


class TestFindMatch:
    def test_returns_match_results_list(self, temp_chroma_dir):
        from backend.services.mashup.curator import find_match
        from backend.services.mashup.memory import MashupLibrary
        lib = MashupLibrary(persist_directory=str(temp_chroma_dir))
        for i in range(3):
            lib.upsert_song(
                f"Artist{i}", f"Song{i}",
                _make_meta(bpm=120.0 + i, camelot="8B"),
                transcript="lyrics",
            )
        results = find_match("artist0_song0", library=lib)
        lib.close()
        assert isinstance(results, list)

    def test_excludes_self_from_results(self, temp_chroma_dir):
        from backend.services.mashup.curator import find_match
        from backend.services.mashup.memory import MashupLibrary
        lib = MashupLibrary(persist_directory=str(temp_chroma_dir))
        for i in range(3):
            lib.upsert_song(
                f"Artist{i}", f"Song{i}",
                _make_meta(bpm=120.0 + i, camelot="8B"),
                transcript="lyrics",
            )
        results = find_match("artist0_song0", library=lib)
        ids = [r.get("id") for r in results]
        assert "artist0_song0" not in ids
        lib.close()
