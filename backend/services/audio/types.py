"""Data types for the Beat_Studio audio engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SectionInfo:
    """Per-section audio metadata."""
    section_type: str          # "intro" | "verse" | "chorus" | "bridge" | "outro"
    start_sec: float
    end_sec: float
    duration_sec: float

    # Energy (from librosa)
    energy_level: float        # 0.0-1.0 (RMS energy)
    spectral_centroid: float   # Brightness in Hz
    tempo_stability: float     # Beat consistency 0-1

    # Vocal characteristics
    vocal_density: str         # "sparse" | "medium" | "dense"
    vocal_intensity: float     # 0.0-1.0
    lyrical_content: str       # Lyrics in this section

    # Semantic (LLM-derived)
    emotional_tone: str        # "hopeful" | "melancholic" | "defiant" | etc.
    lyrical_function: str      # "narrative" | "hook" | "question" | "answer" | "reflection"
    themes: List[str]          # ["love", "loss", "rebellion"]


@dataclass
class SceneTiming:
    """Video scene boundary with metadata for generation."""
    scene_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    is_hero: bool              # Top 25% energy — gets more VRAM / better model
    energy_level: float        # 0.0-1.0
    section_type: str          # "intro" | "verse" | "chorus" | "bridge" | "outro"
    beat_aligned: bool         # True if start_sec snapped to a beat boundary


@dataclass
class SongAnalysis:
    """Complete analysis result from AudioAnalyzer."""
    # Identity
    artist: str
    title: str
    file_path: str

    # Basic signal (depth="basic")
    bpm: float
    key: str               # e.g. "Cmaj", "Amin"
    camelot: str           # e.g. "8B", "5A"
    duration_sec: float
    sample_rate: int
    energy_level: float    # 0.0-1.0 overall energy
    first_downbeat_sec: float

    # Standard (depth="standard" or "full")
    sections: List[SectionInfo] = field(default_factory=list)
    beat_times: List[float] = field(default_factory=list)

    # Full (depth="full")
    transcript: str = ""
    word_timings: List[Dict[str, Any]] = field(default_factory=list)
    has_vocals: bool = True
    mood_summary: str = ""
    genres: List[str] = field(default_factory=list)
    primary_genre: str = ""
    irony_score: int = 0
    valence: int = 5
    emotional_arc: str = ""
