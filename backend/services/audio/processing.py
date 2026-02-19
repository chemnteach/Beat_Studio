"""Audio processing — ported from AI_Mixer mixer/audio/processing.py.

Stem separation, time-stretching, alignment, mixing, pitch-shifting, export.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize

logger = logging.getLogger("beat_studio.audio.processing")


class ProcessingError(Exception):
    """Base exception for audio processing errors."""


# ── internal helpers ──────────────────────────────────────────────────────────

def _run_demucs(audio_path: str, model_name: str, device: str) -> Dict[str, np.ndarray]:
    """Run Demucs model and return stem dict. Separated so it can be mocked."""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    import torch
    import torchaudio

    model = get_model(model_name)
    model.to(device)

    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
    wav = wav.to(device)

    with torch.no_grad():
        stems = apply_model(model, wav[None], device=device)[0]

    result: Dict[str, np.ndarray] = {}
    for i, name in enumerate(model.sources):
        stem = stems[i]
        if stem.shape[0] > 1:
            stem = stem.mean(dim=0)
        result[name] = stem.cpu().numpy()
    return result


def _pyrubberband_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Wrap pyrubberband; separated for easy mocking."""
    import pyrubberband as pyrb
    return pyrb.time_stretch(y, sr, rate)


# ── public API ────────────────────────────────────────────────────────────────

def separate_stems(
    audio_path: str,
    model_name: str = "htdemucs",
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Separate audio into stems using Demucs.

    Returns:
        Dict with keys "vocals", "drums", "bass", "other" (numpy arrays).
    """
    try:
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Separating stems: model=%s device=%s", model_name, device)
        return _run_demucs(audio_path, model_name, device)
    except ImportError as exc:
        raise ProcessingError(f"Demucs not installed: {exc}") from exc
    except Exception as exc:
        raise ProcessingError(f"Stem separation failed: {exc}") from exc


def time_stretch(
    audio: np.ndarray,
    rate: float,
    sr: int = 44100,
    quality: str = "high",
) -> np.ndarray:
    """Time-stretch audio (pitch-preserving) via pyrubberband.

    Args:
        audio: Input audio array.
        rate: Stretch rate (>1 = faster, <1 = slower).
        sr: Sample rate.
        quality: "draft" | "high" | "broadcast"

    Returns:
        Stretched audio array.
    """
    if rate == 1.0:
        return audio

    if not (0.5 <= rate <= 2.0):
        raise ProcessingError(f"Stretch rate {rate:.2f} out of acceptable bounds (0.5-2.0)")

    try:
        return _pyrubberband_stretch(audio, sr, rate)
    except Exception as exc:
        raise ProcessingError(f"Time-stretching failed: {exc}") from exc


def align_to_downbeat(
    downbeat_sec: float,
    beat_times: List[float],
) -> float:
    """Snap a timestamp to the nearest beat boundary.

    Returns the original value if beat_times is empty.
    """
    if not beat_times:
        return downbeat_sec
    beats = np.array(beat_times)
    idx = int(np.argmin(np.abs(beats - downbeat_sec)))
    return float(beats[idx])


def align_tracks(
    vocals: np.ndarray,
    instrumental: np.ndarray,
    vocal_downbeat_sec: float,
    inst_downbeat_sec: float,
    stretch_ratio: float,
    sr: int = 44100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align vocal and instrumental by downbeat offset."""
    adjusted_vocal_db = vocal_downbeat_sec * stretch_ratio
    offset_sec = inst_downbeat_sec - adjusted_vocal_db
    offset_samples = int(offset_sec * sr)

    if offset_samples > 0:
        vocals_aligned = np.pad(vocals, (offset_samples, 0), mode="constant")
    elif offset_samples < 0:
        vocals_aligned = vocals[abs(offset_samples):]
    else:
        vocals_aligned = vocals

    min_len = min(len(instrumental), len(vocals_aligned))
    return vocals_aligned[:min_len], instrumental[:min_len]


def numpy_to_audiosegment(audio: np.ndarray, sr: int) -> AudioSegment:
    """Convert float numpy array to pydub AudioSegment."""
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    return AudioSegment(
        audio.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1,
    )


def normalize_lufs(segment: AudioSegment, target_lufs: int = -14) -> AudioSegment:
    """Normalise an AudioSegment.  Uses pydub normalize (headroom-based)."""
    # pyloudnorm would be more precise for true LUFS, but pydub normalize
    # is sufficient for mashup use-cases and avoids extra dependencies.
    try:
        return normalize(segment, headroom=abs(target_lufs) - 14 + 2.0)
    except Exception as exc:
        logger.warning("normalize_lufs failed: %s — returning original", exc)
        return segment


def mix_and_export(
    vocals: np.ndarray,
    instrumental: np.ndarray,
    output_path: str,
    sr: int = 44100,
    output_format: str = "mp3",
    vocal_attenuation_db: float = -2.0,
) -> str:
    """Mix vocal + instrumental and export to file."""
    vocals_seg = numpy_to_audiosegment(vocals, sr)
    inst_seg = numpy_to_audiosegment(instrumental, sr)

    vocals_seg = normalize(vocals_seg, headroom=2.0)
    inst_seg = normalize(inst_seg, headroom=2.0)
    vocals_seg = vocals_seg + vocal_attenuation_db

    mashup = inst_seg.overlay(vocals_seg)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    export_audio(mashup, output_path, fmt=output_format, sr=sr)
    return os.path.abspath(output_path)


def combine_stems(
    stem_dict: Dict[str, np.ndarray],
    exclude: Optional[List[str]] = None,
) -> np.ndarray:
    """Combine multiple stems into a single track."""
    exclude = exclude or []
    combined: Optional[np.ndarray] = None
    for name, audio in stem_dict.items():
        if name in exclude:
            continue
        if combined is None:
            combined = audio.copy()
        else:
            min_len = min(len(combined), len(audio))
            combined = combined[:min_len] + audio[:min_len]
    if combined is None:
        raise ProcessingError("No stems to combine")
    return combined


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Pitch-shift audio by semitones using librosa."""
    if semitones == 0:
        return audio
    try:
        import librosa
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=semitones)
    except Exception as exc:
        raise ProcessingError(f"Pitch-shifting failed: {exc}") from exc


def calculate_semitone_shift(source_key: str, target_key: str) -> int:
    """Calculate chromatic-circle shortest-path semitone shift between keys.

    Returns value in [-6, 6].
    """
    def _normalise(key: str) -> str:
        key = key.strip()
        key = key.replace(" major", "maj").replace(" minor", "min")
        key = key.replace("Major", "maj").replace("Minor", "min")
        if len(key) >= 2 and key[-1] == "M" and key[-2] != "m":
            key = key[:-1] + "maj"
        elif len(key) >= 2 and key[-1] == "m" and (
            len(key) == 2 or key[-2] in ("#", "b")
        ):
            key = key + "in"
        return key

    note_semitone = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    }

    def _root(key: str) -> str:
        return key[:2] if len(key) >= 2 and key[1] in ("#", "b") else key[0]

    try:
        src = _normalise(source_key)
        tgt = _normalise(target_key)
        shift = note_semitone[_root(tgt)] - note_semitone[_root(src)]
        if shift > 6:
            shift -= 12
        elif shift < -6:
            shift += 12
        return shift
    except KeyError as exc:
        raise ProcessingError(f"Invalid key: {exc}") from exc


def export_audio(
    segment: AudioSegment,
    output_path: str,
    fmt: str = "mp3",
    sr: int = 44100,
) -> None:
    """Export pydub AudioSegment to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "mp3":
        segment.export(output_path, format="mp3", bitrate="320k",
                       parameters=["-q:a", "0"])
    elif fmt == "wav":
        segment.export(output_path, format="wav",
                       parameters=["-ac", "2", "-ar", str(sr)])
    else:
        raise ProcessingError(f"Unsupported format: {fmt}")
