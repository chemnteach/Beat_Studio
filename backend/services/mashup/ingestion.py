"""Mashup ingestion agent — ported from AI_Mixer mixer/agents/ingestion.py.

Handles local files and YouTube URLs. Caches to WAV for analysis.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("beat_studio.mashup.ingestion")

# ── cache directory (overridable for tests) ───────────────────────────────────
_CACHE_DIR = "backend/data/library_cache/audio"


# ── exceptions ────────────────────────────────────────────────────────────────

class IngestionError(Exception):
    """Base ingestion exception."""


class InvalidInputError(IngestionError):
    """Invalid input (bad URL, missing file, etc.)."""


class DownloadError(IngestionError):
    """YouTube download failure."""


class ValidationError(IngestionError):
    """Audio file validation failure."""


# ── source detection ──────────────────────────────────────────────────────────

def detect_source_type(input_source: str) -> str:
    """Return "youtube" or "local_file" based on input_source string."""
    normalized = input_source.lower()
    if "youtube.com" in normalized or "youtu.be" in normalized:
        return "youtube"
    return "local_file"


# ── artist/title extraction ───────────────────────────────────────────────────

def extract_artist_title_from_filename(file_path: str) -> tuple[str, str]:
    """Extract artist and title from 'Artist - Title.ext' filename convention.

    Falls back to ('Unknown Artist', stem) if no dash separator found.
    """
    stem = Path(file_path).stem
    if " - " in stem:
        parts = stem.split(" - ", 1)
        return parts[0].strip(), parts[1].strip()
    return "Unknown Artist", stem.strip()


def extract_artist_title_from_youtube_title(video_title: str) -> tuple[str, str]:
    """Extract artist/title from a YouTube video title."""
    for sep in (" - ", " – ", " | "):
        if sep in video_title:
            parts = video_title.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    return "Unknown Artist", video_title.strip()


# ── cache helpers ─────────────────────────────────────────────────────────────

def check_cache(song_id: str) -> Optional[str]:
    """Return cached WAV path if it exists, else None."""
    cache_path = Path(_CACHE_DIR) / f"{song_id}.wav"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return str(cache_path)
    return None


# ── validation ────────────────────────────────────────────────────────────────

def validate_audio_file(file_path: str) -> None:
    """Raise IngestionError if file doesn't exist or is empty."""
    p = Path(file_path)
    if not p.exists():
        raise IngestionError(f"File not found: {file_path}")
    if p.stat().st_size == 0:
        raise IngestionError(f"File is empty: {file_path}")


# ── format conversion ─────────────────────────────────────────────────────────

def convert_to_standard_wav(input_path: str, output_path: str) -> str:
    """Convert any audio format to standard WAV using ffmpeg via pydub."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(44100).set_channels(2).set_sample_width(2)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        audio.export(output_path, format="wav")
        logger.info("Converted to WAV: %s", output_path)
        return output_path
    except Exception as exc:
        raise IngestionError(f"Format conversion failed: {exc}") from exc


# ── local file ingestion ──────────────────────────────────────────────────────

def ingest_local_file(file_path: str) -> Dict[str, Any]:
    """Ingest a local audio file. Returns IngestionResult dict."""
    validate_audio_file(file_path)

    artist, title = extract_artist_title_from_filename(file_path)
    song_id = _make_id(artist, title)

    # Check cache first
    cached_path = check_cache(song_id)
    if cached_path:
        logger.info("Cache hit: %s", song_id)
        return {
            "id": song_id,
            "path": cached_path,
            "cached": True,
            "source": "local_file",
            "metadata": None,
        }

    # Convert to standard WAV
    Path(_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(_CACHE_DIR) / f"{song_id}.wav")
    convert_to_standard_wav(file_path, out_path)

    logger.info("Ingested: %s → %s", song_id, out_path)
    return {
        "id": song_id,
        "path": out_path,
        "cached": False,
        "source": "local_file",
        "metadata": None,
    }


# ── YouTube ingestion ─────────────────────────────────────────────────────────

def _download_youtube(url: str) -> Dict[str, Any]:
    """Download YouTube audio via yt-dlp. Separated for mocking."""
    import yt_dlp
    Path(_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(Path(_CACHE_DIR) / "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_title = info.get("title", "Unknown")
    artist, title = extract_artist_title_from_youtube_title(video_title)
    song_id = _make_id(artist, title)

    # yt-dlp writes the file as <title>.wav
    expected = Path(_CACHE_DIR) / f"{video_title}.wav"
    final = Path(_CACHE_DIR) / f"{song_id}.wav"
    if expected.exists():
        expected.rename(final)

    return {
        "id": song_id,
        "path": str(final),
        "cached": False,
        "source": "youtube",
        "metadata": None,
    }


def ingest_youtube_url(url: str, max_retries: int = 3) -> Dict[str, Any]:
    """Download audio from YouTube URL and cache as WAV."""
    try:
        return _download_youtube(url)
    except Exception as exc:
        raise IngestionError(f"YouTube download failed for {url}: {exc}") from exc


# ── main entry point ──────────────────────────────────────────────────────────

def ingest_song(input_source: str) -> Dict[str, Any]:
    """Unified ingestion: auto-detects local file vs YouTube URL."""
    source_type = detect_source_type(input_source)
    if source_type == "youtube":
        return ingest_youtube_url(input_source)
    return ingest_local_file(input_source)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_id(artist: str, title: str) -> str:
    """Create a safe filesystem-friendly song ID."""
    combined = f"{artist}_{title}".lower()
    combined = combined.replace(" ", "_")
    combined = re.sub(r"[^a-z0-9_]", "", combined)
    combined = re.sub(r"_+", "_", combined).strip("_")
    return combined[:128] or "unknown_song"
