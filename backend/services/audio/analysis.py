"""Audio signal analysis — ported from AI_Mixer mixer/audio/analysis.py.

Section detection, energy analysis, key estimation, Camelot wheel mapping.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("beat_studio.audio.analysis")

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore[assignment]


def detect_sections(
    y: np.ndarray,
    sr: int,
    n_segments: int = 8,
) -> List[Tuple[float, float]]:
    """Detect section boundaries using spectral agglomerative clustering.

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    try:
        duration = librosa.get_duration(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        k = min(8, len(chroma[0]) // 10)
        boundaries = librosa.segment.agglomerative(chroma, k)
        boundary_times = librosa.frames_to_time(boundaries, sr=sr)

        raw = []
        for i in range(len(boundary_times) - 1):
            raw.append((float(boundary_times[i]), float(boundary_times[i + 1])))

        # Extend last section to cover full duration (librosa may leave a tail)
        if raw and raw[-1][1] < duration - 0.5:
            raw[-1] = (raw[-1][0], float(duration))

        # Merge sections shorter than MIN_SECTION_SEC into their longer neighbor.
        # Process inner sections first, then check first/last.
        MIN_SEC = 10.0
        changed = True
        while changed and len(raw) > 2:
            changed = False
            for i in range(1, len(raw) - 1):   # skip first and last
                if raw[i][1] - raw[i][0] < MIN_SEC:
                    prev_dur = raw[i - 1][1] - raw[i - 1][0]
                    next_dur = raw[i + 1][1] - raw[i + 1][0]
                    if next_dur >= prev_dur:
                        raw = raw[:i] + [(raw[i][0], raw[i + 1][1])] + raw[i + 2:]
                    else:
                        raw = raw[:i - 1] + [(raw[i - 1][0], raw[i][1])] + raw[i + 1:]
                    changed = True
                    break

        # Absorb a tiny first or last section into its neighbor
        if len(raw) > 1 and raw[0][1] - raw[0][0] < MIN_SEC:
            raw = [(raw[0][0], raw[1][1])] + raw[2:]
        if len(raw) > 1 and raw[-1][1] - raw[-1][0] < MIN_SEC:
            raw = raw[:-2] + [(raw[-2][0], raw[-1][1])]

        logger.info("Detected %d sections (after merge)", len(raw))
        return raw

    except Exception as exc:
        logger.error("Section detection failed: %s", exc)
        duration = librosa.get_duration(y=y, sr=sr)
        n_chunks = max(1, int(duration / 8))
        chunk_dur = duration / n_chunks
        return [(i * chunk_dur, min((i + 1) * chunk_dur, duration))
                for i in range(n_chunks)]


# Pre-defined inner-section patterns (between intro and outro) for common counts.
# Keyed by number of inner sections.
_INNER_PATTERNS: dict = {
    1: ["verse"],
    2: ["verse", "chorus"],
    3: ["verse", "chorus", "verse"],
    4: ["verse", "chorus", "verse", "chorus"],
    5: ["verse", "chorus", "verse", "chorus", "bridge"],
    6: ["verse", "chorus", "verse", "chorus", "bridge", "verse"],
    7: ["verse", "chorus", "verse", "chorus", "bridge", "verse", "chorus"],
}


def classify_section_type(
    section_idx: int,
    total_sections: int,
    energy_level: float,
    spectral_centroid: float,
) -> str:
    """Assign section type by position.

    First section is always intro, last is always outro.
    Inner sections use a pre-defined pattern scaled to the detected count.
    """
    if total_sections == 1:
        return "verse"
    if section_idx == 0:
        return "intro"
    if section_idx == total_sections - 1:
        return "outro"

    n_inner = total_sections - 2
    inner_idx = section_idx - 1

    if n_inner in _INNER_PATTERNS:
        return _INNER_PATTERNS[n_inner][inner_idx]

    # > 7 inner sections: alternate verse/chorus
    return "verse" if inner_idx % 2 == 0 else "chorus"


def analyze_section_energy(
    y: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> dict:
    """Return energy_level, spectral_centroid, tempo_stability for a section."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    section_audio = y[start_sample:end_sample]

    if len(section_audio) == 0:
        return {"energy_level": 0.0, "spectral_centroid": 0.0, "tempo_stability": 0.0}

    try:
        rms = librosa.feature.rms(y=section_audio)[0]
        energy_level = float(np.mean(rms))

        centroid = librosa.feature.spectral_centroid(y=section_audio, sr=sr)[0]
        spectral_centroid = float(np.mean(centroid))

        try:
            tempo, beats = librosa.beat.beat_track(y=section_audio, sr=sr)
            if len(beats) > 1:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                beat_intervals = np.diff(beat_times)
                tempo_stability = float(np.exp(-np.var(beat_intervals) * 10))
            else:
                tempo_stability = 0.0
        except Exception:
            tempo_stability = 0.5

        return {
            "energy_level": energy_level,
            "spectral_centroid": spectral_centroid,
            "tempo_stability": tempo_stability,
        }
    except Exception as exc:
        logger.error("Energy analysis failed: %s", exc)
        return {"energy_level": 0.0, "spectral_centroid": 0.0, "tempo_stability": 0.0}


def _merge_two_sections(a: "SectionInfo", b: "SectionInfo") -> "SectionInfo":  # type: ignore[name-defined]
    """Merge two adjacent sections into one, keeping the longer one's type."""
    from backend.services.audio.types import SectionInfo
    total = a.duration_sec + b.duration_sec
    energy = (a.energy_level * a.duration_sec + b.energy_level * b.duration_sec) / total
    centroid = (a.spectral_centroid * a.duration_sec + b.spectral_centroid * b.duration_sec) / total
    longer = a if a.duration_sec >= b.duration_sec else b
    vocal = "dense" if energy > 0.65 else ("medium" if energy > 0.30 else "sparse")
    return SectionInfo(
        section_type=longer.section_type,
        start_sec=min(a.start_sec, b.start_sec),
        end_sec=max(a.end_sec, b.end_sec),
        duration_sec=total,
        energy_level=energy,
        spectral_centroid=centroid,
        tempo_stability=(a.tempo_stability + b.tempo_stability) / 2,
        vocal_density=vocal,
        vocal_intensity=energy,
        lyrical_content="",
        emotional_tone="neutral",
        lyrical_function="narrative",
        themes=[],
    )


def _retype_section(sec: "SectionInfo", new_type: str) -> "SectionInfo":  # type: ignore[name-defined]
    """Return a copy of sec with section_type replaced."""
    from backend.services.audio.types import SectionInfo
    return SectionInfo(
        section_type=new_type,
        start_sec=sec.start_sec,
        end_sec=sec.end_sec,
        duration_sec=sec.duration_sec,
        energy_level=sec.energy_level,
        spectral_centroid=sec.spectral_centroid,
        tempo_stability=sec.tempo_stability,
        vocal_density=sec.vocal_density,
        vocal_intensity=sec.vocal_intensity,
        lyrical_content=sec.lyrical_content,
        emotional_tone=sec.emotional_tone,
        lyrical_function=sec.lyrical_function,
        themes=sec.themes,
    )


def post_process_sections(
    sections: List,
    total_duration: float,
    min_sec: float = 12.0,
) -> List:
    """Merge tiny/adjacent-same-type sections and re-label with alternating pattern.

    Steps:
      1. Merge sections shorter than min_sec into their longer neighbor.
      2. Merge consecutive sections of the same type.
      3. Re-label inner sections using energy rank + verse/chorus alternation.
    """
    if len(sections) <= 1:
        return sections

    # ── Step 1: Absorb tiny sections ──────────────────────────────────────────
    changed = True
    while changed and len(sections) > 1:
        changed = False
        for i, sec in enumerate(sections):
            if sec.duration_sec < min_sec:
                if i < len(sections) - 1:
                    sections = sections[:i] + [_merge_two_sections(sec, sections[i + 1])] + sections[i + 2:]
                else:
                    sections = sections[:i - 1] + [_merge_two_sections(sections[i - 1], sec)] + sections[i + 1:]
                changed = True
                break

    # ── Step 2: Merge adjacent same-type sections ──────────────────────────────
    changed = True
    while changed and len(sections) > 1:
        changed = False
        for i in range(len(sections) - 1):
            if sections[i].section_type == sections[i + 1].section_type:
                sections = sections[:i] + [_merge_two_sections(sections[i], sections[i + 1])] + sections[i + 2:]
                changed = True
                break

    if len(sections) <= 2:
        return sections

    # ── Step 3: Re-label inner sections ───────────────────────────────────────
    inner = sections[1:-1]
    if not inner:
        return sections

    energies = sorted(s.energy_level for s in inner)
    median_e = energies[len(energies) // 2]

    labeled: List = []
    for sec in inner:
        if sec.energy_level >= median_e:
            new_type = "chorus"
        elif sec.energy_level < median_e * 0.55:
            new_type = "bridge"
        else:
            new_type = "verse"
        labeled.append(_retype_section(sec, new_type))

    # Ensure first inner section is verse, not chorus
    if labeled and labeled[0].section_type == "chorus":
        labeled[0] = _retype_section(labeled[0], "verse")

    # Break up any run of 3+ same type by flipping the middle one
    for i in range(1, len(labeled) - 1):
        if (labeled[i - 1].section_type == labeled[i].section_type ==
                labeled[i + 1].section_type):
            flip = "verse" if labeled[i].section_type == "chorus" else "chorus"
            labeled[i] = _retype_section(labeled[i], flip)

    return [sections[0]] + labeled + [sections[-1]]


def estimate_key(chroma: np.ndarray) -> str:
    """Estimate musical key using Krumhansl-Schmuckler profile matching."""
    chroma_mean = np.mean(chroma, axis=1)
    total = np.sum(chroma_mean)
    if total == 0:
        return "Unknown"
    chroma_mean = chroma_mean / total

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_profile = major_profile / np.sum(major_profile)
    minor_profile = minor_profile / np.sum(minor_profile)

    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best_corr = -1.0
    best_key = "Unknown"

    for shift in range(12):
        rotated = np.roll(chroma_mean, shift)
        corr_maj = float(np.corrcoef(rotated, major_profile)[0, 1])
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = f"{pitch_classes[shift]}maj"
        corr_min = float(np.corrcoef(rotated, minor_profile)[0, 1])
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = f"{pitch_classes[shift]}min"

    return best_key


def key_to_camelot(key: str) -> str:
    """Convert musical key string to Camelot wheel notation."""
    camelot_map = {
        "Cmaj": "8B",  "Gmaj": "9B",  "Dmaj": "10B", "Amaj": "11B",
        "Emaj": "12B", "Bmaj": "1B",  "F#maj": "2B", "C#maj": "3B",
        "G#maj": "4B", "D#maj": "5B", "A#maj": "6B", "Fmaj": "7B",
        "Amin": "8A",  "Emin": "9A",  "Bmin": "10A", "F#min": "11A",
        "C#min": "12A","G#min": "1A", "D#min": "2A", "A#min": "3A",
        "Fmin": "4A",  "Cmin": "5A",  "Gmin": "6A",  "Dmin": "7A",
    }
    return camelot_map.get(key, "Unknown")
