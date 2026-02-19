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
        adjusted = max(4, min(16, int(duration / 20)))

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        boundaries = librosa.segment.agglomerative(chroma, adjusted)
        boundary_times = librosa.frames_to_time(boundaries, sr=sr)

        sections = []
        for i in range(len(boundary_times) - 1):
            sections.append((float(boundary_times[i]), float(boundary_times[i + 1])))
        logger.info("Detected %d sections", len(sections))
        return sections

    except Exception as exc:
        logger.error("Section detection failed: %s", exc)
        duration = librosa.get_duration(y=y, sr=sr)
        n_chunks = max(1, int(duration / 8))
        chunk_dur = duration / n_chunks
        return [(i * chunk_dur, min((i + 1) * chunk_dur, duration))
                for i in range(n_chunks)]


def classify_section_type(
    section_idx: int,
    total_sections: int,
    energy_level: float,
    spectral_centroid: float,
) -> str:
    """Classify section type from position and audio features."""
    if section_idx == 0:
        return "intro"
    if section_idx == total_sections - 1:
        return "outro"
    if energy_level > 0.6 and spectral_centroid > 2000:
        return "chorus"
    if energy_level < 0.3:
        return "bridge"
    return "verse"


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
