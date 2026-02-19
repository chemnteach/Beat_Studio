"""Mashup engineer — all 8 mashup types, ported from AI_Mixer mixer/agents/engineer.py.

Implements:
  1. Classic           — vocal A + instrumental B
  2. Stem Swap         — mix stems from 3+ songs
  3. Energy Matched    — dynamic section selection by energy
  4. Adaptive Harmony  — auto-fix key clashes via pitch-shifting
  5. Theme Fusion      — filter sections by lyrical theme
  6. Semantic Aligned  — question→answer, narrative→reflection pairing
  7. Role Aware        — lead / harmony / call / response / texture
  8. Conversational    — songs talk to each other with silence gaps
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.services.audio.processing import (
    align_tracks,
    calculate_semitone_shift,
    combine_stems,
    mix_and_export,
    pitch_shift,
    separate_stems,
    time_stretch,
)
from backend.services.mashup.memory import MashupLibrary

logger = logging.getLogger("beat_studio.mashup.engineer")

# ── exceptions ────────────────────────────────────────────────────────────────


class EngineerError(Exception):
    """Base engineer exception."""


class SongNotFoundError(EngineerError):
    """Song not found in library."""


class MashupConfigError(EngineerError):
    """Invalid mashup configuration."""


# ── internal helpers ──────────────────────────────────────────────────────────

def _load_song_audio(
    song_id: str,
    library: Optional[MashupLibrary] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load audio array and metadata for a song from the library."""
    if library is None:
        library = MashupLibrary()
    record = library.get_song(song_id)
    if record is None:
        raise SongNotFoundError(f"Song not found in library: {song_id}")

    meta = record["metadata"]
    audio_path = meta.get("path", "")
    if not audio_path or not Path(audio_path).exists():
        raise EngineerError(f"Audio file not found: {audio_path}")

    import librosa
    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    return y, meta


def _calculate_stretch_ratio(source_bpm: float, target_bpm: float) -> float:
    """Calculate time-stretch ratio to align source BPM to target BPM."""
    if target_bpm <= 0 or source_bpm <= 0:
        return 1.0
    ratio = target_bpm / source_bpm
    # Cap ratio (pyrubberband works best within 0.7-1.3)
    return max(0.7, min(1.3, ratio))


def _extract_section_audio(
    y: np.ndarray,
    section: Dict[str, Any],
    sr: int = 44100,
) -> np.ndarray:
    """Extract audio for a single section."""
    start = int(section["start_sec"] * sr)
    end = int(section["end_sec"] * sr)
    return y[start:end]


def _concat_numpy(*arrays: np.ndarray) -> np.ndarray:
    """Concatenate numpy arrays, handling empty arrays gracefully."""
    non_empty = [a for a in arrays if len(a) > 0]
    if not non_empty:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(non_empty)


# ── 1. Classic mashup ─────────────────────────────────────────────────────────

def create_classic_mashup(
    vocal_song_id: str,
    inst_song_id: str,
    output_path: str,
    quality_preset: str = "high",
    output_format: str = "mp3",
    vocal_attenuation_db: float = -2.0,
    library: Optional[MashupLibrary] = None,
) -> str:
    """Classic mashup: vocals from song A + instrumental from song B.

    Steps:
    1. Load both songs.
    2. Separate stems (Demucs).
    3. Time-stretch vocals to match instrumental BPM.
    4. Align by first downbeat.
    5. Mix and export.
    """
    logger.info("Creating CLASSIC mashup: %s + %s", vocal_song_id, inst_song_id)

    vocal_y, vocal_meta = _load_song_audio(vocal_song_id, library)
    inst_y, inst_meta = _load_song_audio(inst_song_id, library)

    vocal_bpm = vocal_meta.get("bpm", 120.0) or 120.0
    inst_bpm = inst_meta.get("bpm", 120.0) or 120.0

    # Separate stems
    vocal_stems = separate_stems(vocal_meta["path"])
    inst_stems = separate_stems(inst_meta["path"])

    vocals = vocal_stems["vocals"]
    instrumental = _concat_numpy(
        inst_stems["drums"], inst_stems["bass"], inst_stems["other"]
    )

    # Time-stretch vocals to match instrumental BPM
    stretch_ratio = _calculate_stretch_ratio(vocal_bpm, inst_bpm)
    if abs(stretch_ratio - 1.0) > 0.01:
        logger.info("Stretching vocals %.3f×", stretch_ratio)
        vocals = time_stretch(vocals, rate=stretch_ratio)

    # Align by downbeat
    vocal_db = vocal_meta.get("first_downbeat_sec", 0.0) or 0.0
    inst_db = inst_meta.get("first_downbeat_sec", 0.0) or 0.0
    vocals, instrumental = align_tracks(vocals, instrumental, vocal_db, inst_db, stretch_ratio)

    return mix_and_export(vocals, instrumental, output_path,
                          output_format=output_format,
                          vocal_attenuation_db=vocal_attenuation_db)


# ── 2. Stem swap mashup ───────────────────────────────────────────────────────

def create_stem_swap_mashup(
    stem_config: Dict[str, str],   # {"vocals": song_a_id, "drums": song_b_id, ...}
    output_path: str,
    quality_preset: str = "high",
    output_format: str = "mp3",
    library: Optional[MashupLibrary] = None,
) -> str:
    """Stem swap: mix stems from 3+ songs.

    stem_config maps stem names ("vocals", "drums", "bass", "other") to song IDs.
    """
    logger.info("Creating STEM_SWAP mashup")

    stem_names = ("vocals", "drums", "bass", "other")
    for stem in stem_names:
        if stem not in stem_config:
            stem_config[stem] = list(stem_config.values())[0]

    # Load all unique songs
    unique_ids = set(stem_config.values())
    loaded: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
    for song_id in unique_ids:
        loaded[song_id] = _load_song_audio(song_id, library)

    # Determine target BPM (median of all)
    bpms = [m.get("bpm", 120.0) or 120.0 for _, m in loaded.values()]
    target_bpm = float(np.median(bpms))

    # Separate and time-stretch all songs
    separated: Dict[str, Dict[str, np.ndarray]] = {}
    for song_id, (y, meta) in loaded.items():
        stems = separate_stems(meta["path"])
        src_bpm = meta.get("bpm", 120.0) or 120.0
        ratio = _calculate_stretch_ratio(src_bpm, target_bpm)
        if abs(ratio - 1.0) > 0.01:
            stems = {k: time_stretch(v, rate=ratio) for k, v in stems.items()}
        separated[song_id] = stems

    # Assemble from config
    chosen_stems: Dict[str, np.ndarray] = {}
    for stem_name in stem_names:
        song_id = stem_config[stem_name]
        chosen_stems[stem_name] = separated[song_id][stem_name]

    combined = combine_stems(chosen_stems)

    from pydub import AudioSegment
    from backend.services.audio.processing import numpy_to_audiosegment, export_audio
    seg = numpy_to_audiosegment(combined, 44100)
    export_audio(seg, output_path, fmt=output_format)
    return output_path


# ── 3. Energy matched mashup ──────────────────────────────────────────────────

def create_energy_matched_mashup(
    song_a_id: str,
    song_b_id: str,
    output_path: str,
    quality_preset: str = "high",
    output_format: str = "mp3",
    library: Optional[MashupLibrary] = None,
) -> str:
    """Energy matched: dynamically select high-energy sections from either song.

    Requires section-level metadata.
    """
    logger.info("Creating ENERGY_MATCHED mashup: %s + %s", song_a_id, song_b_id)

    y_a, meta_a = _load_song_audio(song_a_id, library)
    y_b, meta_b = _load_song_audio(song_b_id, library)

    sections_a = meta_a.get("sections", [])
    sections_b = meta_b.get("sections", [])

    if not sections_a or not sections_b:
        raise EngineerError(
            "Energy matched mashup requires section-level metadata for both songs. "
            "Run profile_audio(depth='full') first."
        )

    bpm_a = meta_a.get("bpm", 120.0) or 120.0
    bpm_b = meta_b.get("bpm", 120.0) or 120.0
    target_bpm = (bpm_a + bpm_b) / 2.0

    # Pool all sections with source labelled
    all_sections = [(s, "a") for s in sections_a] + [(s, "b") for s in sections_b]
    # Sort by energy descending
    all_sections.sort(key=lambda x: x[0].get("energy_level", 0), reverse=True)

    # Build sequence by alternating sources
    sequence = []
    used_a = used_b = 0
    for sec, source in all_sections:
        if len(sequence) >= max(len(sections_a), len(sections_b)):
            break
        sequence.append((sec, source))

    # Extract, time-stretch, concatenate
    chunks: List[np.ndarray] = []
    for sec, source in sequence:
        y_src = y_a if source == "a" else y_b
        meta_src = meta_a if source == "a" else meta_b
        bpm_src = meta_src.get("bpm", 120.0) or 120.0
        chunk = _extract_section_audio(y_src, sec)
        ratio = _calculate_stretch_ratio(bpm_src, target_bpm)
        if abs(ratio - 1.0) > 0.01:
            chunk = time_stretch(chunk, rate=ratio)
        chunks.append(chunk)

    combined = _concat_numpy(*chunks) if chunks else np.zeros(44100)
    return mix_and_export(combined, np.zeros(len(combined)), output_path,
                          output_format=output_format)


# ── 4. Adaptive harmony mashup ────────────────────────────────────────────────

def create_adaptive_harmony_mashup(
    vocal_song_id: str,
    inst_song_id: str,
    output_path: str,
    quality_preset: str = "high",
    output_format: str = "mp3",
    library: Optional[MashupLibrary] = None,
) -> str:
    """Adaptive harmony: classic mashup with automatic pitch correction.

    Pitch-shifts the instrumental to match the vocal key.
    """
    logger.info("Creating ADAPTIVE_HARMONY mashup: %s + %s", vocal_song_id, inst_song_id)

    vocal_y, vocal_meta = _load_song_audio(vocal_song_id, library)
    inst_y, inst_meta = _load_song_audio(inst_song_id, library)

    vocal_key = vocal_meta.get("key", "Cmaj") or "Cmaj"
    inst_key = inst_meta.get("key", "Cmaj") or "Cmaj"
    semitones = calculate_semitone_shift(inst_key, vocal_key)
    logger.info("Key shift: %s → %s = %+d semitones", inst_key, vocal_key, semitones)

    vocal_stems = separate_stems(vocal_meta["path"])
    inst_stems = separate_stems(inst_meta["path"])

    vocals = vocal_stems["vocals"]
    instrumental = _concat_numpy(
        inst_stems["drums"], inst_stems["bass"], inst_stems["other"]
    )

    # Pitch-shift instrumental if needed
    if semitones != 0:
        instrumental = pitch_shift(instrumental, sr=44100, semitones=semitones)

    # Time-stretch
    bpm_v = vocal_meta.get("bpm", 120.0) or 120.0
    bpm_i = inst_meta.get("bpm", 120.0) or 120.0
    ratio = _calculate_stretch_ratio(bpm_v, bpm_i)
    if abs(ratio - 1.0) > 0.01:
        vocals = time_stretch(vocals, rate=ratio)

    # Align
    vocal_db = vocal_meta.get("first_downbeat_sec", 0.0) or 0.0
    inst_db = inst_meta.get("first_downbeat_sec", 0.0) or 0.0
    vocals, instrumental = align_tracks(vocals, instrumental, vocal_db, inst_db, ratio)

    return mix_and_export(vocals, instrumental, output_path, output_format=output_format)


# ── 5. Theme fusion mashup ────────────────────────────────────────────────────

def create_theme_fusion_mashup(
    song_a_id: str,
    song_b_id: str,
    output_path: str,
    theme: str = "love",
    quality_preset: str = "high",
    output_format: str = "mp3",
    library: Optional[MashupLibrary] = None,
) -> str:
    """Theme fusion: filter sections by lyrical theme, combine matching sections."""
    logger.info("Creating THEME_FUSION mashup: %s + %s, theme='%s'",
                song_a_id, song_b_id, theme)

    y_a, meta_a = _load_song_audio(song_a_id, library)
    y_b, meta_b = _load_song_audio(song_b_id, library)

    def _matching(sections: List[Dict[str, Any]], theme: str) -> List[Dict[str, Any]]:
        t_lower = theme.lower()
        return [
            s for s in sections
            if any(t_lower in th.lower() for th in s.get("themes", []))
        ]

    matching_a = _matching(meta_a.get("sections", []), theme)
    matching_b = _matching(meta_b.get("sections", []), theme)

    if not matching_a and not matching_b:
        raise EngineerError(
            f"No sections found matching theme '{theme}' in either song. "
            "Try a different theme keyword."
        )

    # Sort by energy for coherent flow
    all_matching = [(s, "a", y_a, meta_a) for s in matching_a] + \
                   [(s, "b", y_b, meta_b) for s in matching_b]
    all_matching.sort(key=lambda x: x[0].get("energy_level", 0))

    target_bpm = (
        (meta_a.get("bpm") or 120.0) + (meta_b.get("bpm") or 120.0)
    ) / 2.0

    chunks: List[np.ndarray] = []
    for sec, source, y_src, meta_src in all_matching:
        chunk = _extract_section_audio(y_src, sec)
        bpm_src = meta_src.get("bpm", 120.0) or 120.0
        ratio = _calculate_stretch_ratio(bpm_src, target_bpm)
        if abs(ratio - 1.0) > 0.01:
            chunk = time_stretch(chunk, rate=ratio)
        chunks.append(chunk)

    combined = _concat_numpy(*chunks) if chunks else np.zeros(44100)
    return mix_and_export(combined, np.zeros(len(combined)), output_path,
                          output_format=output_format)


# ── 6. Semantic aligned mashup ────────────────────────────────────────────────

#: Lyrical function pairs for conversational structure
_SEMANTIC_PAIRS = [
    ("question", "answer"),
    ("narrative", "reflection"),
    ("hook", "hook"),
    ("call", "response"),
]


def create_semantic_aligned_mashup(
    song_a_id: str,
    song_b_id: str,
    output_path: str,
    quality_preset: str = "high",
    output_format: str = "mp3",
    library: Optional[MashupLibrary] = None,
) -> str:
    """Semantic aligned: pair sections by lyrical function (question→answer, etc.)."""
    logger.info("Creating SEMANTIC_ALIGNED mashup: %s + %s", song_a_id, song_b_id)

    y_a, meta_a = _load_song_audio(song_a_id, library)
    y_b, meta_b = _load_song_audio(song_b_id, library)

    sections_a = {s.get("lyrical_function", "narrative"): s
                  for s in meta_a.get("sections", [])}
    sections_b = {s.get("lyrical_function", "narrative"): s
                  for s in meta_b.get("sections", [])}

    # Build paired sequence
    pairs: List[Tuple[Dict, str, Dict, str]] = []
    for func_a, func_b in _SEMANTIC_PAIRS:
        if func_a in sections_a and func_b in sections_b:
            pairs.append((sections_a[func_a], "a", sections_b[func_b], "b"))
        elif func_b in sections_a and func_a in sections_b:
            pairs.append((sections_a[func_b], "a", sections_b[func_a], "b"))

    if not pairs:
        logger.warning("No semantic pairs found — falling back to section-order alternation")
        secs_a = meta_a.get("sections", [])
        secs_b = meta_b.get("sections", [])
        for sa, sb in zip(secs_a, secs_b):
            pairs.append((sa, "a", sb, "b"))

    target_bpm = (
        (meta_a.get("bpm") or 120.0) + (meta_b.get("bpm") or 120.0)
    ) / 2.0

    chunks: List[np.ndarray] = []
    for sec_a, src_a, sec_b, src_b in pairs:
        for sec, src, y_src, meta_src in [
            (sec_a, src_a, y_a, meta_a),
            (sec_b, src_b, y_b, meta_b),
        ]:
            chunk = _extract_section_audio(y_src, sec)
            bpm_src = meta_src.get("bpm", 120.0) or 120.0
            ratio = _calculate_stretch_ratio(bpm_src, target_bpm)
            if abs(ratio - 1.0) > 0.01:
                chunk = time_stretch(chunk, rate=ratio)
            chunks.append(chunk)

    combined = _concat_numpy(*chunks) if chunks else np.zeros(44100)
    return mix_and_export(combined, np.zeros(len(combined)), output_path,
                          output_format=output_format)


# ── 7. Role aware mashup ──────────────────────────────────────────────────────

def create_role_aware_mashup(
    song_a_id: str,
    song_b_id: str,
    output_path: str,
    quality_preset: str = "high",
    output_format: str = "mp3",
    library: Optional[MashupLibrary] = None,
) -> str:
    """Role aware: vocals shift between lead/harmony/call/response based on energy."""
    logger.info("Creating ROLE_AWARE mashup: %s + %s", song_a_id, song_b_id)

    y_a, meta_a = _load_song_audio(song_a_id, library)
    y_b, meta_b = _load_song_audio(song_b_id, library)

    stems_a = separate_stems(meta_a["path"])
    stems_b = separate_stems(meta_b["path"])

    sections_a = meta_a.get("sections", [])
    sections_b = meta_b.get("sections", [])

    bpm_a = meta_a.get("bpm", 120.0) or 120.0
    bpm_b = meta_b.get("bpm", 120.0) or 120.0
    target_bpm = (bpm_a + bpm_b) / 2.0

    # Time-stretch stems to target BPM
    ratio_a = _calculate_stretch_ratio(bpm_a, target_bpm)
    ratio_b = _calculate_stretch_ratio(bpm_b, target_bpm)

    for k in stems_a:
        if abs(ratio_a - 1.0) > 0.01:
            stems_a[k] = time_stretch(stems_a[k], rate=ratio_a)
    for k in stems_b:
        if abs(ratio_b - 1.0) > 0.01:
            stems_b[k] = time_stretch(stems_b[k], rate=ratio_b)

    # Alternate lead/harmony roles by energy
    # High energy section → lead; low energy → harmony/texture
    chunks: List[np.ndarray] = []
    max_sections = max(len(sections_a), len(sections_b), 1)

    for i in range(max_sections):
        sec_a = sections_a[i] if i < len(sections_a) else None
        sec_b = sections_b[i] if i < len(sections_b) else None

        energy_a = sec_a.get("energy_level", 0.5) if sec_a else 0.5
        energy_b = sec_b.get("energy_level", 0.5) if sec_b else 0.5

        if energy_a >= energy_b:
            lead_stems = stems_a
            harmony_stems = stems_b
            lead_sec = sec_a
        else:
            lead_stems = stems_b
            harmony_stems = stems_a
            lead_sec = sec_b

        lead_chunk = _concat_numpy(
            lead_stems.get("vocals", np.zeros(0)),
            lead_stems.get("drums", np.zeros(0)),
            lead_stems.get("bass", np.zeros(0)),
        )

        # Add harmony vocals at reduced volume
        harmony_vocals = harmony_stems.get("vocals", np.zeros(0))
        if len(harmony_vocals) > 0 and len(lead_chunk) > 0:
            min_len = min(len(lead_chunk), len(harmony_vocals))
            lead_chunk[:min_len] += harmony_vocals[:min_len] * 0.4

        if lead_sec and len(lead_chunk) > 0:
            sr = 44100
            start = int(lead_sec["start_sec"] * sr)
            end = int(lead_sec["end_sec"] * sr)
            if start < len(lead_chunk):
                chunks.append(lead_chunk[start:min(end, len(lead_chunk))])
        elif len(lead_chunk) > 0:
            chunks.append(lead_chunk[:44100])  # 1 second fallback

    combined = _concat_numpy(*chunks) if chunks else np.zeros(44100)
    return mix_and_export(combined, np.zeros(len(combined)), output_path,
                          output_format=output_format)


# ── 8. Conversational mashup ──────────────────────────────────────────────────

def create_conversational_mashup(
    song_a_id: str,
    song_b_id: str,
    output_path: str,
    silence_gap_sec: float = 0.5,
    quality_preset: str = "high",
    output_format: str = "mp3",
    library: Optional[MashupLibrary] = None,
) -> str:
    """Conversational: songs take turns speaking (A section → gap → B section → gap → …)."""
    logger.info("Creating CONVERSATIONAL mashup: %s + %s", song_a_id, song_b_id)

    y_a, meta_a = _load_song_audio(song_a_id, library)
    y_b, meta_b = _load_song_audio(song_b_id, library)

    stems_a = separate_stems(meta_a["path"])
    stems_b = separate_stems(meta_b["path"])

    sections_a = meta_a.get("sections", [])
    sections_b = meta_b.get("sections", [])

    bpm_a = meta_a.get("bpm", 120.0) or 120.0
    bpm_b = meta_b.get("bpm", 120.0) or 120.0
    target_bpm = (bpm_a + bpm_b) / 2.0

    ratio_a = _calculate_stretch_ratio(bpm_a, target_bpm)
    ratio_b = _calculate_stretch_ratio(bpm_b, target_bpm)

    for k in stems_a:
        if abs(ratio_a - 1.0) > 0.01:
            stems_a[k] = time_stretch(stems_a[k], rate=ratio_a)
    for k in stems_b:
        if abs(ratio_b - 1.0) > 0.01:
            stems_b[k] = time_stretch(stems_b[k], rate=ratio_b)

    sr = 44100
    silence = np.zeros(int(silence_gap_sec * sr), dtype=np.float32)
    chunks: List[np.ndarray] = []

    max_rounds = max(len(sections_a), len(sections_b), 1)
    for i in range(max_rounds):
        # Song A speaks
        if i < len(sections_a):
            sec = sections_a[i]
            vocals_a = stems_a.get("vocals", np.zeros(0))
            start = int(sec["start_sec"] * sr)
            end = int(sec["end_sec"] * sr)
            if start < len(vocals_a):
                chunks.append(vocals_a[start:min(end, len(vocals_a))])
                chunks.append(silence.copy())

        # Song B responds
        if i < len(sections_b):
            sec = sections_b[i]
            vocals_b = stems_b.get("vocals", np.zeros(0))
            start = int(sec["start_sec"] * sr)
            end = int(sec["end_sec"] * sr)
            if start < len(vocals_b):
                chunks.append(vocals_b[start:min(end, len(vocals_b))])
                chunks.append(silence.copy())

    combined = _concat_numpy(*chunks) if chunks else np.zeros(44100)
    return mix_and_export(combined, np.zeros(len(combined)), output_path,
                          output_format=output_format)
