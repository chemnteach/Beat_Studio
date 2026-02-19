"""Unified AudioAnalyzer — merges AI_Mixer section detection with BeatCanvas
scene timing generation into a single service.

depth levels:
  "basic"    — BPM, key, duration, energy (fast, no heavy models)
  "standard" — + sections, beat_times, scene-ready output
  "full"     — + Whisper transcript, word timings, LLM semantic analysis
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("beat_studio.audio.analyzer")

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore[assignment]

try:
    import whisper as _whisper_module
    whisper = _whisper_module
except ImportError:  # pragma: no cover
    whisper = None  # type: ignore[assignment]

from backend.services.audio.types import SceneTiming, SectionInfo, SongAnalysis

# ── local analysis helpers (ported from AI_Mixer) ─────────────────────────────
from backend.services.audio.analysis import (
    analyze_section_energy,
    classify_section_type,
    detect_sections,
    estimate_key,
    key_to_camelot,
)

# ── optional LLM helpers ──────────────────────────────────────────────────────
try:
    from backend.services.llm.semantic import (
        analyze_section_semantics,
        analyze_song_semantics,
        generate_emotional_arc,
    )
    _HAS_LLM = True
except ImportError:  # pragma: no cover
    _HAS_LLM = False

    def analyze_song_semantics(*a, **kw):  # type: ignore[misc]
        return {"genres": [], "primary_genre": "unknown", "irony_score": 0,
                "mood_summary": "", "valence": 5}

    def analyze_section_semantics(*a, **kw):  # type: ignore[misc]
        return {"emotional_tone": "neutral", "lyrical_function": "narrative",
                "themes": []}

    def generate_emotional_arc(sections):  # type: ignore[misc]
        return ""


# ── public helpers (used by tests) ────────────────────────────────────────────

def _analyze_signal_basic(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Extract BPM, key, camelot, energy, duration, first downbeat."""
    tempo_arr, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo_arr[0]) if hasattr(tempo_arr, "__len__") else float(tempo_arr)

    beat_times = librosa.frames_to_time(beats, sr=sr)
    first_downbeat = float(beat_times[0]) if len(beat_times) > 0 else 0.0

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key = estimate_key(chroma)
    camelot = key_to_camelot(key)

    rms = librosa.feature.rms(y=y)[0]
    energy_level = float(np.mean(rms))

    duration = librosa.get_duration(y=y, sr=sr)

    return {
        "bpm": bpm,
        "key": key,
        "camelot": camelot,
        "duration_sec": duration,
        "energy_level": energy_level,
        "first_downbeat_sec": first_downbeat,
    }


def _build_sections(
    y: np.ndarray,
    sr: int,
    boundaries: List[Tuple[float, float]],
    transcript_data: Optional[Dict[str, Any]] = None,
) -> List[SectionInfo]:
    """Build SectionInfo list from boundary tuples + optional transcript data."""
    sections: List[SectionInfo] = []
    total = len(boundaries)
    transcript_data = transcript_data or {}

    # Pre-compute energy across all sections to normalise energy_level
    raw_energies: List[Dict[str, Any]] = []
    for start, end in boundaries:
        raw_energies.append(analyze_section_energy(y, sr, start, end))

    max_energy = max((e["energy_level"] for e in raw_energies), default=1.0) or 1.0

    for idx, (start, end) in enumerate(boundaries):
        energy_data = raw_energies[idx]
        energy_norm = energy_data["energy_level"] / max_energy

        section_type = classify_section_type(
            idx, total, energy_norm, energy_data["spectral_centroid"]
        )

        # Vocal density heuristic (crude but fast — LLM pass enriches if full)
        if energy_norm > 0.65:
            vocal_density = "dense"
        elif energy_norm > 0.30:
            vocal_density = "medium"
        else:
            vocal_density = "sparse"

        sections.append(SectionInfo(
            section_type=section_type,
            start_sec=start,
            end_sec=end,
            duration_sec=end - start,
            energy_level=energy_norm,
            spectral_centroid=energy_data["spectral_centroid"],
            tempo_stability=energy_data["tempo_stability"],
            vocal_density=vocal_density,
            vocal_intensity=energy_norm,
            lyrical_content="",          # enriched in full depth pass
            emotional_tone="neutral",    # enriched in full depth pass
            lyrical_function="narrative",# enriched in full depth pass
            themes=[],                   # enriched in full depth pass
        ))

    return sections


# ── main class ────────────────────────────────────────────────────────────────

class AudioAnalyzer:
    """Unified audio analysis serving both the mashup and video pipelines.

    Usage::

        az = AudioAnalyzer()
        analysis = az.analyze("song.wav", artist="X", title="Y", depth="full")
        scenes   = az.get_scene_timings(analysis)
        meta     = az.get_mashup_metadata(analysis)  # ChromaDB-ready dict
    """

    #: Hero scene threshold — top N% by energy are flagged is_hero=True
    HERO_ENERGY_PERCENTILE = 0.75

    #: Fallback scene duration when there are no sections at all
    FALLBACK_SCENE_SEC = 4.0

    def __init__(
        self,
        sample_rate: int = 44100,
        whisper_model: str = "base",
    ):
        self.sample_rate = sample_rate
        self.whisper_model = whisper_model

    # ── public: analyze ───────────────────────────────────────────────────────

    def analyze(
        self,
        audio_path: str,
        artist: str,
        title: str,
        depth: str = "full",
    ) -> SongAnalysis:
        """Analyze an audio file.

        Args:
            audio_path: Absolute path to WAV (or any librosa-compatible format)
            artist: Artist name
            title: Song title
            depth: "basic" | "standard" | "full"

        Returns:
            SongAnalysis populated according to depth.
        """
        logger.info("Analyzing '%s — %s' at depth=%s", artist, title, depth)

        # ── 1. Load audio ─────────────────────────────────────────────────────
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
        if y.ndim > 1:
            y = librosa.to_mono(y)

        # ── 2. Basic signal analysis (always) ─────────────────────────────────
        basic = _analyze_signal_basic(y, sr)

        if depth == "basic":
            return SongAnalysis(
                artist=artist,
                title=title,
                file_path=audio_path,
                bpm=basic["bpm"],
                key=basic["key"],
                camelot=basic["camelot"],
                duration_sec=basic["duration_sec"],
                sample_rate=sr,
                energy_level=basic["energy_level"],
                first_downbeat_sec=basic["first_downbeat_sec"],
            )

        # ── 3. Beat grid ───────────────────────────────────────────────────────
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = [float(t) for t in librosa.frames_to_time(beats, sr=sr)]

        # ── 4. Section detection ───────────────────────────────────────────────
        boundaries = detect_sections(y, sr)
        sections = _build_sections(y, sr, boundaries)

        if depth == "standard":
            return SongAnalysis(
                artist=artist,
                title=title,
                file_path=audio_path,
                bpm=basic["bpm"],
                key=basic["key"],
                camelot=basic["camelot"],
                duration_sec=basic["duration_sec"],
                sample_rate=sr,
                energy_level=basic["energy_level"],
                first_downbeat_sec=basic["first_downbeat_sec"],
                sections=sections,
                beat_times=beat_times,
            )

        # ── 5. Whisper transcription (full only) ───────────────────────────────
        transcript_data = self._transcribe(audio_path)

        # ── 6. Song-level LLM semantics ────────────────────────────────────────
        semantic = analyze_song_semantics(
            transcript_data["transcript"],
            basic["bpm"],
            basic["key"],
            basic["energy_level"],
        )

        # ── 7. Enrich sections with semantic data ─────────────────────────────
        for sec in sections:
            sem = analyze_section_semantics(sec.lyrical_content or "")
            sec.emotional_tone = sem.get("emotional_tone", "neutral")
            sec.lyrical_function = sem.get("lyrical_function", "narrative")
            sec.themes = sem.get("themes", [])

        emotional_arc = generate_emotional_arc(sections)

        return SongAnalysis(
            artist=artist,
            title=title,
            file_path=audio_path,
            bpm=basic["bpm"],
            key=basic["key"],
            camelot=basic["camelot"],
            duration_sec=basic["duration_sec"],
            sample_rate=sr,
            energy_level=basic["energy_level"],
            first_downbeat_sec=basic["first_downbeat_sec"],
            sections=sections,
            beat_times=beat_times,
            transcript=transcript_data["transcript"],
            word_timings=transcript_data.get("word_timings", []),
            has_vocals=transcript_data.get("has_vocals", True),
            mood_summary=semantic.get("mood_summary", ""),
            genres=semantic.get("genres", []),
            primary_genre=semantic.get("primary_genre", ""),
            irony_score=semantic.get("irony_score", 0),
            valence=semantic.get("valence", 5),
            emotional_arc=emotional_arc,
        )

    # ── public: get_scene_timings ─────────────────────────────────────────────

    def get_scene_timings(
        self,
        analysis: SongAnalysis,
        min_duration: float = 2.5,
        max_duration: float = 8.0,
    ) -> List[SceneTiming]:
        """Generate beat-aligned video scene boundaries from a SongAnalysis.

        Hero scenes (top 25% energy) are flagged is_hero=True so the video
        pipeline can allocate more resources to them.

        Args:
            analysis: Completed SongAnalysis (any depth, but better with sections)
            min_duration: Minimum scene length in seconds (default 2.5)
            max_duration: Maximum scene length in seconds (default 8.0)

        Returns:
            List of SceneTiming, sequential, covering the full song duration.
        """
        if not analysis.sections:
            return self._uniform_scenes(analysis.duration_sec, min_duration, max_duration)

        # Compute hero threshold from section energies
        energies = [s.energy_level for s in analysis.sections]
        hero_threshold = float(np.percentile(energies, self.HERO_ENERGY_PERCENTILE * 100))

        # Build raw scenes from sections, splitting long ones
        raw_scenes: List[Dict[str, Any]] = []
        for sec in analysis.sections:
            if sec.duration_sec <= max_duration:
                raw_scenes.append({
                    "start": sec.start_sec,
                    "end": sec.end_sec,
                    "energy": sec.energy_level,
                    "section_type": sec.section_type,
                })
            else:
                # Split long section at beat boundaries
                split = self._split_section(
                    sec, analysis.beat_times, max_duration, min_duration
                )
                raw_scenes.extend(split)

        # Merge scenes that are shorter than min_duration with the next
        merged = self._merge_short_scenes(raw_scenes, min_duration)

        # Build SceneTiming objects
        timings: List[SceneTiming] = []
        for idx, scene in enumerate(merged):
            dur = scene["end"] - scene["start"]
            timings.append(SceneTiming(
                scene_index=idx,
                start_sec=scene["start"],
                end_sec=scene["end"],
                duration_sec=dur,
                is_hero=scene["energy"] >= hero_threshold,
                energy_level=scene["energy"],
                section_type=scene.get("section_type", "verse"),
                beat_aligned=scene.get("beat_aligned", False),
            ))
        return timings

    # ── public: get_mashup_metadata ───────────────────────────────────────────

    def get_mashup_metadata(self, analysis: SongAnalysis) -> Dict[str, Any]:
        """Return a ChromaDB-compatible metadata dict from a SongAnalysis."""
        sections_as_dicts = []
        for sec in analysis.sections:
            sections_as_dicts.append({
                "section_type": sec.section_type,
                "start_sec": sec.start_sec,
                "end_sec": sec.end_sec,
                "duration_sec": sec.duration_sec,
                "energy_level": sec.energy_level,
                "spectral_centroid": sec.spectral_centroid,
                "tempo_stability": sec.tempo_stability,
                "vocal_density": sec.vocal_density,
                "vocal_intensity": sec.vocal_intensity,
                "lyrical_content": sec.lyrical_content,
                "emotional_tone": sec.emotional_tone,
                "lyrical_function": sec.lyrical_function,
                "themes": sec.themes,
            })

        return {
            "bpm": analysis.bpm,
            "key": analysis.key,
            "camelot": analysis.camelot,
            "duration_sec": analysis.duration_sec,
            "sample_rate": analysis.sample_rate,
            "energy_level": int(analysis.energy_level * 10),  # 0-10 scale for ChromaDB
            "first_downbeat_sec": analysis.first_downbeat_sec,
            "has_vocals": analysis.has_vocals,
            "mood_summary": analysis.mood_summary,
            "genres": analysis.genres,
            "primary_genre": analysis.primary_genre,
            "irony_score": analysis.irony_score,
            "valence": analysis.valence,
            "sections": sections_as_dicts,
            "emotional_arc": analysis.emotional_arc,
            "word_timings": analysis.word_timings,
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Run Whisper transcription. Returns transcript + word timings."""
        if whisper is None:  # pragma: no cover
            return {"transcript": "", "word_timings": [], "has_vocals": False}
        try:
            model = whisper.load_model(self.whisper_model)
            result = model.transcribe(audio_path, word_timestamps=True)
            transcript = result.get("text", "").strip()
            has_vocals = bool(transcript)
            return {
                "transcript": transcript,
                "word_timings": result.get("segments", []),
                "has_vocals": has_vocals,
            }
        except Exception as exc:
            logger.warning("Whisper transcription failed: %s", exc)
            return {"transcript": "", "word_timings": [], "has_vocals": False}

    def _uniform_scenes(
        self, duration: float, min_dur: float, max_dur: float
    ) -> List[SceneTiming]:
        """Fallback: divide into equal-length scenes when there are no sections."""
        scene_dur = max(min_dur, min(max_dur, self.FALLBACK_SCENE_SEC))
        n = max(1, int(duration / scene_dur))
        actual_dur = duration / n
        timings = []
        for i in range(n):
            start = i * actual_dur
            end = min((i + 1) * actual_dur, duration)
            timings.append(SceneTiming(
                scene_index=i,
                start_sec=start,
                end_sec=end,
                duration_sec=end - start,
                is_hero=False,
                energy_level=0.5,
                section_type="verse",
                beat_aligned=False,
            ))
        return timings

    def _split_section(
        self,
        sec: SectionInfo,
        beat_times: List[float],
        max_dur: float,
        min_dur: float,
    ) -> List[Dict[str, Any]]:
        """Split a long section at beat boundaries."""
        scenes = []
        cursor = sec.start_sec
        while cursor < sec.end_sec:
            target_end = min(cursor + max_dur, sec.end_sec)
            # Snap to nearest beat if available
            beat_aligned = False
            if beat_times:
                candidates = [b for b in beat_times if cursor < b <= target_end]
                if candidates:
                    target_end = candidates[-1]
                    beat_aligned = True
            dur = target_end - cursor
            if dur < min_dur and scenes:
                # Extend last scene instead of creating a tiny one
                scenes[-1]["end"] = target_end
                scenes[-1]["energy"] = sec.energy_level
            else:
                scenes.append({
                    "start": cursor,
                    "end": target_end,
                    "energy": sec.energy_level,
                    "section_type": sec.section_type,
                    "beat_aligned": beat_aligned,
                })
            cursor = target_end
        return scenes

    def _merge_short_scenes(
        self, scenes: List[Dict[str, Any]], min_dur: float
    ) -> List[Dict[str, Any]]:
        """Merge scenes shorter than min_dur into adjacent scenes."""
        if not scenes:
            return scenes
        merged = [dict(scenes[0])]
        for scene in scenes[1:]:
            dur = scene["end"] - scene["start"]
            if dur < min_dur:
                merged[-1]["end"] = scene["end"]
                # Use higher energy of the two
                merged[-1]["energy"] = max(merged[-1]["energy"], scene["energy"])
            else:
                merged.append(dict(scene))
        return merged
