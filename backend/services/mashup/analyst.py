"""Mashup analyst agent — wraps AudioAnalyzer and stores result in ChromaDB.

Thin orchestration layer: calls AudioAnalyzer.analyze() then upserts
the result into MashupLibrary. Ported from AI_Mixer mixer/agents/analyst.py.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from backend.services.audio.analyzer import AudioAnalyzer
from backend.services.mashup.memory import MashupLibrary

logger = logging.getLogger("beat_studio.mashup.analyst")


class AnalysisError(Exception):
    """Analysis pipeline failure."""


def profile_audio(
    file_path: str,
    song_id: str,
    artist: str,
    title: str,
    depth: str = "full",
    library: Optional[MashupLibrary] = None,
) -> Dict[str, Any]:
    """Analyze audio, extract metadata, and store in ChromaDB.

    Args:
        file_path: Path to WAV file.
        song_id: Unique song identifier.
        artist: Artist name.
        title: Song title.
        depth: Analysis depth ("basic" | "standard" | "full").
        library: MashupLibrary instance (creates default if None).

    Returns:
        {"status": "success", "song_id": str, "metadata": dict}
    """
    logger.info("Profiling audio: %s — %s (depth=%s)", artist, title, depth)

    try:
        az = AudioAnalyzer()
        analysis = az.analyze(file_path, artist=artist, title=title, depth=depth)
        metadata = az.get_mashup_metadata(analysis)

        # Enrich with identity fields
        metadata["source"] = "local_file"
        metadata["path"] = file_path
        metadata["artist"] = artist
        metadata["title"] = title
        metadata["date_added"] = datetime.utcnow().isoformat()
        metadata["sample_rate"] = analysis.sample_rate

        # Store in ChromaDB
        transcript = analysis.transcript or ""
        if library is None:
            library = MashupLibrary()
        library.upsert_song(
            artist=artist,
            title=title,
            metadata=metadata,
            transcript=f"{transcript}\n\n[MOOD]: {metadata.get('mood_summary', '')}",
            force_id=song_id,
        )

        logger.info("Profile complete: %s", song_id)
        return {
            "status": "success",
            "song_id": song_id,
            "metadata": metadata,
        }

    except Exception as exc:
        logger.error("Profile failed for %s: %s", song_id, exc, exc_info=True)
        raise AnalysisError(f"Analysis failed for {song_id}: {exc}") from exc
