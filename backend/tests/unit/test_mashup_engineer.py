"""Tests for the mashup engineer (all 8 mashup types) — Phase 2."""
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest


# ── shared test helpers ───────────────────────────────────────────────────────

def _make_meta(bpm=120.0, key="Cmaj", camelot="8B", first_downbeat=0.1,
               sections=None, has_vocals=True):
    return {
        "bpm": bpm, "key": key, "camelot": camelot,
        "first_downbeat_sec": first_downbeat,
        "sections": sections or [],
        "has_vocals": has_vocals,
        "source": "local_file", "path": "/tmp/x.wav",
        "artist": "A", "title": "T", "sample_rate": 44100,
        "duration_sec": 180.0, "genres": ["pop"], "primary_genre": "pop",
        "irony_score": 0, "valence": 5, "energy_level": 5,
        "mood_summary": "upbeat", "emotional_arc": "hopeful",
    }


def _section(stype="verse", start=0.0, end=30.0, energy=0.5, func="narrative",
             themes=None, tone="neutral"):
    return {
        "section_type": stype, "start_sec": start, "end_sec": end,
        "duration_sec": end - start, "energy_level": energy,
        "lyrical_function": func, "themes": themes or ["love"],
        "emotional_tone": tone, "vocal_density": "medium",
    }


FAKE_AUDIO = np.zeros(44100, dtype=np.float32)


# ── Classic mashup ────────────────────────────────────────────────────────────

class TestCreateClassicMashup:
    def test_creates_output_file(self, tmp_dir):
        from backend.services.mashup.engineer import create_classic_mashup

        out = str(tmp_dir / "classic.mp3")
        meta_vocal = _make_meta()
        meta_inst = _make_meta(bpm=121.0)

        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[
                       (FAKE_AUDIO, meta_vocal),
                       (FAKE_AUDIO, meta_inst),
                   ]), \
             patch("backend.services.mashup.engineer.separate_stems",
                   return_value={k: FAKE_AUDIO for k in ("vocals", "drums", "bass", "other")}), \
             patch("backend.services.mashup.engineer.time_stretch",
                   return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.align_tracks",
                   return_value=(FAKE_AUDIO, FAKE_AUDIO)), \
             patch("backend.services.mashup.engineer.mix_and_export",
                   return_value=out):
            result = create_classic_mashup(
                vocal_song_id="a", inst_song_id="b", output_path=out
            )

        assert result == out

    def test_incompatible_bpm_still_processes(self, tmp_dir):
        from backend.services.mashup.engineer import create_classic_mashup
        out = str(tmp_dir / "stretch.mp3")
        meta_vocal = _make_meta(bpm=90.0)
        meta_inst = _make_meta(bpm=120.0)
        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[(FAKE_AUDIO, meta_vocal), (FAKE_AUDIO, meta_inst)]), \
             patch("backend.services.mashup.engineer.separate_stems",
                   return_value={k: FAKE_AUDIO for k in ("vocals", "drums", "bass", "other")}), \
             patch("backend.services.mashup.engineer.time_stretch", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.align_tracks",
                   return_value=(FAKE_AUDIO, FAKE_AUDIO)), \
             patch("backend.services.mashup.engineer.mix_and_export", return_value=out):
            result = create_classic_mashup("a", "b", out)
        assert result == out


# ── Stem swap mashup ──────────────────────────────────────────────────────────

class TestCreateStemSwapMashup:
    def test_combines_stems_from_multiple_songs(self, tmp_dir):
        from backend.services.mashup.engineer import create_stem_swap_mashup
        out = str(tmp_dir / "stem_swap.mp3")

        config = {
            "vocals": "song_a",
            "drums": "song_b",
            "bass": "song_a",
            "other": "song_b",
        }
        with patch("backend.services.mashup.engineer._load_song_audio",
                   return_value=(FAKE_AUDIO, _make_meta())), \
             patch("backend.services.mashup.engineer.separate_stems",
                   return_value={k: FAKE_AUDIO for k in ("vocals", "drums", "bass", "other")}), \
             patch("backend.services.mashup.engineer.time_stretch", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.combine_stems", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.mix_and_export", return_value=out):
            result = create_stem_swap_mashup(config, out)
        assert result == out


# ── Energy matched mashup ─────────────────────────────────────────────────────

class TestCreateEnergyMatchedMashup:
    def test_requires_sections(self, tmp_dir):
        from backend.services.mashup.engineer import create_energy_matched_mashup, EngineerError
        out = str(tmp_dir / "energy.mp3")
        # No sections → should raise
        with patch("backend.services.mashup.engineer._load_song_audio",
                   return_value=(FAKE_AUDIO, _make_meta(sections=[]))):
            with pytest.raises(EngineerError):
                create_energy_matched_mashup("a", "b", out)

    def test_creates_output_with_sections(self, tmp_dir):
        from backend.services.mashup.engineer import create_energy_matched_mashup
        out = str(tmp_dir / "energy.mp3")
        secs_a = [_section("chorus", 0, 30, energy=0.9)]
        secs_b = [_section("verse", 0, 30, energy=0.5)]
        meta_a = _make_meta(sections=secs_a)
        meta_b = _make_meta(sections=secs_b)

        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[(FAKE_AUDIO, meta_a), (FAKE_AUDIO, meta_b)]), \
             patch("backend.services.mashup.engineer.time_stretch", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.mix_and_export", return_value=out):
            result = create_energy_matched_mashup("a", "b", out)
        assert result == out


# ── Adaptive harmony mashup ───────────────────────────────────────────────────

class TestCreateAdaptiveHarmonyMashup:
    def test_applies_pitch_shift_when_keys_differ(self, tmp_dir):
        from backend.services.mashup.engineer import create_adaptive_harmony_mashup
        out = str(tmp_dir / "harmony.mp3")
        meta_vocal = _make_meta(key="Cmaj", bpm=120.0)
        meta_inst = _make_meta(key="Gmaj", bpm=120.0)

        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[(FAKE_AUDIO, meta_vocal), (FAKE_AUDIO, meta_inst)]), \
             patch("backend.services.mashup.engineer.separate_stems",
                   return_value={k: FAKE_AUDIO for k in ("vocals", "drums", "bass", "other")}), \
             patch("backend.services.mashup.engineer.pitch_shift",
                   return_value=FAKE_AUDIO) as mock_ps, \
             patch("backend.services.mashup.engineer.time_stretch", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.align_tracks",
                   return_value=(FAKE_AUDIO, FAKE_AUDIO)), \
             patch("backend.services.mashup.engineer.mix_and_export", return_value=out):
            create_adaptive_harmony_mashup("a", "b", out)
            # pitch_shift should be called (keys differ: C to G)
            mock_ps.assert_called()

    def test_skips_pitch_shift_when_same_key(self, tmp_dir):
        from backend.services.mashup.engineer import create_adaptive_harmony_mashup
        out = str(tmp_dir / "harmony_same.mp3")
        meta_vocal = _make_meta(key="Cmaj", bpm=120.0)
        meta_inst = _make_meta(key="Cmaj", bpm=120.0)

        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[(FAKE_AUDIO, meta_vocal), (FAKE_AUDIO, meta_inst)]), \
             patch("backend.services.mashup.engineer.separate_stems",
                   return_value={k: FAKE_AUDIO for k in ("vocals", "drums", "bass", "other")}), \
             patch("backend.services.mashup.engineer.pitch_shift",
                   return_value=FAKE_AUDIO) as mock_ps, \
             patch("backend.services.mashup.engineer.time_stretch", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.align_tracks",
                   return_value=(FAKE_AUDIO, FAKE_AUDIO)), \
             patch("backend.services.mashup.engineer.mix_and_export", return_value=out):
            create_adaptive_harmony_mashup("a", "b", out)
            mock_ps.assert_not_called()


# ── Theme fusion mashup ───────────────────────────────────────────────────────

class TestCreateThemeFusionMashup:
    def test_raises_if_no_matching_sections(self, tmp_dir):
        from backend.services.mashup.engineer import create_theme_fusion_mashup, EngineerError
        out = str(tmp_dir / "theme.mp3")
        secs = [_section("verse", 0, 30, themes=["sadness"])]
        meta = _make_meta(sections=secs)

        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[(FAKE_AUDIO, meta), (FAKE_AUDIO, meta)]):
            with pytest.raises(EngineerError):
                create_theme_fusion_mashup("a", "b", out, theme="love_theme_xyz")

    def test_creates_output_with_matching_sections(self, tmp_dir):
        from backend.services.mashup.engineer import create_theme_fusion_mashup
        out = str(tmp_dir / "theme.mp3")
        secs = [_section("chorus", 0, 30, themes=["love"])]
        meta = _make_meta(sections=secs)

        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[(FAKE_AUDIO, meta), (FAKE_AUDIO, meta)]), \
             patch("backend.services.mashup.engineer.time_stretch", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.mix_and_export", return_value=out):
            result = create_theme_fusion_mashup("a", "b", out, theme="love")
        assert result == out


# ── Semantic aligned mashup ───────────────────────────────────────────────────

class TestCreateSemanticAlignedMashup:
    def test_creates_output(self, tmp_dir):
        from backend.services.mashup.engineer import create_semantic_aligned_mashup
        out = str(tmp_dir / "semantic.mp3")
        secs_a = [_section("verse", 0, 30, func="question")]
        secs_b = [_section("chorus", 0, 30, func="answer")]
        meta_a = _make_meta(sections=secs_a)
        meta_b = _make_meta(sections=secs_b)

        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[(FAKE_AUDIO, meta_a), (FAKE_AUDIO, meta_b)]), \
             patch("backend.services.mashup.engineer.time_stretch", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.mix_and_export", return_value=out):
            result = create_semantic_aligned_mashup("a", "b", out)
        assert result == out

    def test_warns_if_no_pairs(self, tmp_dir):
        from backend.services.mashup.engineer import create_semantic_aligned_mashup
        import logging
        out = str(tmp_dir / "semantic2.mp3")
        secs_a = [_section("verse", 0, 30, func="narrative")]
        secs_b = [_section("verse", 0, 30, func="narrative")]
        meta_a = _make_meta(sections=secs_a)
        meta_b = _make_meta(sections=secs_b)

        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[(FAKE_AUDIO, meta_a), (FAKE_AUDIO, meta_b)]), \
             patch("backend.services.mashup.engineer.time_stretch", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.mix_and_export", return_value=out):
            # Should complete even with no semantic pairs (falls back)
            result = create_semantic_aligned_mashup("a", "b", out)
        assert isinstance(result, str)


# ── Role-aware mashup ─────────────────────────────────────────────────────────

class TestCreateRoleAwareMashup:
    def test_creates_output(self, tmp_dir):
        from backend.services.mashup.engineer import create_role_aware_mashup
        out = str(tmp_dir / "role.mp3")
        secs = [_section("verse", 0, 30, energy=0.5)]
        meta = _make_meta(sections=secs)

        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[(FAKE_AUDIO, meta), (FAKE_AUDIO, meta)]), \
             patch("backend.services.mashup.engineer.separate_stems",
                   return_value={k: FAKE_AUDIO for k in ("vocals", "drums", "bass", "other")}), \
             patch("backend.services.mashup.engineer.time_stretch", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.mix_and_export", return_value=out):
            result = create_role_aware_mashup("a", "b", out)
        assert result == out


# ── Conversational mashup ─────────────────────────────────────────────────────

class TestCreateConversationalMashup:
    def test_creates_output(self, tmp_dir):
        from backend.services.mashup.engineer import create_conversational_mashup
        out = str(tmp_dir / "conv.mp3")
        secs_a = [_section("verse", 0, 10), _section("chorus", 10, 20)]
        secs_b = [_section("verse", 0, 10), _section("chorus", 10, 20)]
        meta_a = _make_meta(sections=secs_a)
        meta_b = _make_meta(sections=secs_b)

        with patch("backend.services.mashup.engineer._load_song_audio",
                   side_effect=[(FAKE_AUDIO, meta_a), (FAKE_AUDIO, meta_b)]), \
             patch("backend.services.mashup.engineer.separate_stems",
                   return_value={k: FAKE_AUDIO for k in ("vocals", "drums", "bass", "other")}), \
             patch("backend.services.mashup.engineer.time_stretch", return_value=FAKE_AUDIO), \
             patch("backend.services.mashup.engineer.mix_and_export", return_value=out):
            result = create_conversational_mashup("a", "b", out)
        assert result == out


# ── _load_song_audio ──────────────────────────────────────────────────────────

class TestLoadSongAudio:
    def test_raises_if_song_not_found(self):
        from backend.services.mashup.engineer import _load_song_audio, EngineerError, SongNotFoundError
        with patch("backend.services.mashup.engineer.MashupLibrary") as mock_lib_cls:
            mock_lib = MagicMock()
            mock_lib.get_song.return_value = None
            mock_lib_cls.return_value = mock_lib
            with pytest.raises((EngineerError, SongNotFoundError)):
                _load_song_audio("nonexistent_id")
