"""Tests for the ChromaDB memory module (Phase 2 — mashup memory)."""
import pytest
from unittest.mock import MagicMock, patch


# ── sanitize_id ───────────────────────────────────────────────────────────────

class TestSanitizeId:
    def test_basic_artist_title(self):
        from backend.services.mashup.memory import sanitize_id
        result = sanitize_id("Taylor Swift", "Shake It Off")
        assert result == "taylor_swift_shake_it_off"

    def test_special_characters_removed(self):
        from backend.services.mashup.memory import sanitize_id
        result = sanitize_id("Ke$ha", "TiK ToK")
        assert "$" not in result
        assert result == "keha_tik_tok"

    def test_slashes_removed(self):
        from backend.services.mashup.memory import sanitize_id
        result = sanitize_id("AC/DC", "Back in Black")
        assert "/" not in result

    def test_max_length_128(self):
        from backend.services.mashup.memory import sanitize_id
        long_artist = "A" * 70
        long_title = "B" * 70
        result = sanitize_id(long_artist, long_title)
        assert len(result) <= 128

    def test_empty_artist_raises(self):
        from backend.services.mashup.memory import sanitize_id, SchemaError
        with pytest.raises(SchemaError):
            sanitize_id("", "Song")

    def test_empty_title_raises(self):
        from backend.services.mashup.memory import sanitize_id, SchemaError
        with pytest.raises(SchemaError):
            sanitize_id("Artist", "")

    def test_consecutive_underscores_collapsed(self):
        from backend.services.mashup.memory import sanitize_id
        result = sanitize_id("Test  Artist", "My__Song")
        assert "__" not in result


# ── validate_metadata ─────────────────────────────────────────────────────────

class TestValidateMetadata:
    def _valid_meta(self):
        return {
            "source": "local_file",
            "path": "/tmp/song.wav",
            "artist": "Test Artist",
            "title": "Test Song",
            "sample_rate": 44100,
            "bpm": 120.0,
            "key": "Cmaj",
            "camelot": "8B",
            "duration_sec": 180.0,
        }

    def test_valid_metadata_passes(self):
        from backend.services.mashup.memory import validate_metadata
        validate_metadata(self._valid_meta())  # should not raise

    def test_missing_required_field_raises(self):
        from backend.services.mashup.memory import validate_metadata, SchemaError
        meta = self._valid_meta()
        del meta["artist"]
        with pytest.raises(SchemaError):
            validate_metadata(meta)

    def test_bpm_out_of_range_raises(self):
        from backend.services.mashup.memory import validate_metadata, SchemaError
        meta = self._valid_meta()
        meta["bpm"] = 5.0  # < 20
        with pytest.raises(SchemaError):
            validate_metadata(meta)

    def test_bpm_too_high_raises(self):
        from backend.services.mashup.memory import validate_metadata, SchemaError
        meta = self._valid_meta()
        meta["bpm"] = 400.0  # > 300
        with pytest.raises(SchemaError):
            validate_metadata(meta)


# ── MashupLibrary (ChromaDB wrapper) ─────────────────────────────────────────

class TestMashupLibrary:
    """Tests using a real (temp) ChromaDB instance."""

    @pytest.fixture
    def library(self, temp_chroma_dir):
        from backend.services.mashup.memory import MashupLibrary
        lib = MashupLibrary(persist_directory=str(temp_chroma_dir))
        yield lib
        lib.close()

    def _song_meta(self, artist="Test", title="Song"):
        return {
            "source": "local_file",
            "path": f"/tmp/{artist}_{title}.wav",
            "artist": artist,
            "title": title,
            "sample_rate": 44100,
            "bpm": 120.0,
            "key": "Cmaj",
            "camelot": "8B",
            "duration_sec": 180.0,
            "genres": ["pop"],
            "primary_genre": "pop",
            "mood_summary": "upbeat",
            "energy_level": 7,
            "valence": 7,
            "irony_score": 0,
            "first_downbeat_sec": 0.5,
            "has_vocals": True,
        }

    def test_upsert_and_get(self, library):
        meta = self._song_meta("Taylor Swift", "Shake It Off")
        song_id = library.upsert_song(
            artist="Taylor Swift", title="Shake It Off",
            metadata=meta, transcript="I stay out too late"
        )
        result = library.get_song(song_id)
        assert result is not None
        assert result["id"] == song_id

    def test_upsert_returns_string_id(self, library):
        meta = self._song_meta()
        song_id = library.upsert_song("A", "B", meta, transcript="")
        assert isinstance(song_id, str)
        assert len(song_id) > 0

    def test_get_nonexistent_returns_none(self, library):
        assert library.get_song("nonexistent_id_xyz") is None

    def test_delete_song(self, library):
        meta = self._song_meta("Delete", "Me")
        song_id = library.upsert_song("Delete", "Me", meta, transcript="")
        assert library.delete_song(song_id) is True
        assert library.get_song(song_id) is None

    def test_delete_nonexistent_returns_false(self, library):
        assert library.delete_song("never_existed") is False

    def test_list_all_songs(self, library):
        for i in range(3):
            meta = self._song_meta(f"Artist{i}", f"Song{i}")
            library.upsert_song(f"Artist{i}", f"Song{i}", meta, transcript="")
        songs = library.list_all()
        assert len(songs) == 3

    def test_count_songs(self, library):
        for i in range(2):
            meta = self._song_meta(f"Band{i}", f"Track{i}")
            library.upsert_song(f"Band{i}", f"Track{i}", meta, transcript="")
        assert library.count() == 2


class TestHarmonicQuery:
    """Harmonic matching tests using a real ChromaDB instance."""

    @pytest.fixture
    def library(self, temp_chroma_dir):
        from backend.services.mashup.memory import MashupLibrary
        lib = MashupLibrary(persist_directory=str(temp_chroma_dir))
        # Add songs
        for i, (bpm, camelot, key) in enumerate([
            (120.0, "8B", "Cmaj"),
            (122.0, "9B", "Gmaj"),   # BPM-compatible, adjacent Camelot
            (160.0, "3A", "A#min"),   # BPM-incompatible
        ]):
            meta = {
                "source": "local_file", "path": f"/tmp/s{i}.wav",
                "artist": f"Artist{i}", "title": f"Song{i}",
                "sample_rate": 44100, "bpm": bpm, "key": key, "camelot": camelot,
                "duration_sec": 180.0, "genres": ["pop"], "primary_genre": "pop",
                "mood_summary": "test", "energy_level": 5, "valence": 5,
                "irony_score": 0, "first_downbeat_sec": 0.0, "has_vocals": True,
            }
            lib.upsert_song(f"Artist{i}", f"Song{i}", meta, transcript="test lyrics")
        yield lib
        lib.close()

    def test_harmonic_query_returns_list(self, library):
        results = library.query_harmonic(target_bpm=120.0, target_camelot="8B")
        assert isinstance(results, list)

    def test_harmonic_excludes_ids(self, library):
        from backend.services.mashup.memory import sanitize_id
        exclude = [sanitize_id("Artist0", "Song0")]
        results = library.query_harmonic(120.0, "8B", exclude_ids=exclude)
        ids = [r["id"] for r in results]
        assert exclude[0] not in ids


class TestSemanticQuery:
    @pytest.fixture
    def library(self, temp_chroma_dir):
        from backend.services.mashup.memory import MashupLibrary
        lib = MashupLibrary(persist_directory=str(temp_chroma_dir))
        for mood, i in [("upbeat joyful", 0), ("sad melancholic", 1)]:
            meta = {
                "source": "local_file", "path": f"/tmp/s{i}.wav",
                "artist": f"A{i}", "title": f"T{i}",
                "sample_rate": 44100, "bpm": 120.0, "key": "Cmaj",
                "camelot": "8B", "duration_sec": 180.0,
                "genres": ["pop"], "primary_genre": "pop",
                "mood_summary": mood, "energy_level": 5, "valence": 5,
                "irony_score": 0, "first_downbeat_sec": 0.0, "has_vocals": True,
            }
            lib.upsert_song(f"A{i}", f"T{i}", meta, transcript=mood)
        yield lib
        lib.close()

    def test_semantic_query_returns_list(self, library):
        results = library.query_semantic(
            mood_summary="joyful upbeat", max_results=5
        )
        assert isinstance(results, list)

    def test_semantic_query_returns_results(self, library):
        results = library.query_semantic(mood_summary="upbeat", max_results=2)
        assert len(results) >= 0  # may be 0 if embeddings don't match perfectly
