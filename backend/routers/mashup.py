"""Mashup router — ingest, library, matching, and mashup creation."""
from __future__ import annotations

import dataclasses
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel

from backend.services.mashup.curator import CuratorError, find_match
from backend.services.mashup.engineer import (
    EngineerError,
    SongNotFoundError,
    create_adaptive_harmony_mashup,
    create_classic_mashup,
    create_conversational_mashup,
    create_energy_matched_mashup,
    create_role_aware_mashup,
    create_semantic_aligned_mashup,
    create_stem_swap_mashup,
    create_theme_fusion_mashup,
)
from backend.services.mashup.ingestion import IngestionError, ingest_song
from backend.services.mashup.memory import MashupLibrary
from backend.services.shared.task_manager import TaskManager

logger = logging.getLogger("beat_studio.routers.mashup")
router = APIRouter()

# ── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_DIR  = Path(__file__).parent.parent
_MASHUPS_DIR  = _BACKEND_DIR / "data" / "mashups"

# ── Module-level singletons (lazy init) ───────────────────────────────────────
_task_manager: Optional[TaskManager] = None
_library: Optional[MashupLibrary] = None


def _get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(db_path=str(_BACKEND_DIR / "tasks.db"))
    return _task_manager


def _get_library() -> MashupLibrary:
    global _library
    if _library is None:
        _library = MashupLibrary(
            persist_directory=str(_BACKEND_DIR / "data" / "library_cache" / "chroma")
        )
    return _library


# ── Pydantic models ───────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    source: str              # file path or YouTube URL
    artist: str = ""         # optional override
    title: str = ""          # optional override


class MashupCreateRequest(BaseModel):
    song_a_id: str
    song_b_id: str
    mashup_type: str = "classic"   # one of the 8 types


# ── Background workers ────────────────────────────────────────────────────────

_MASHUP_TYPE_MAP = {
    "classic":          create_classic_mashup,
    "energy_match":     create_energy_matched_mashup,
    "adaptive_harmony": create_adaptive_harmony_mashup,
    "theme_fusion":     create_theme_fusion_mashup,
    "semantic_aligned": create_semantic_aligned_mashup,
    "role_aware":       create_role_aware_mashup,
    "conversational":   create_conversational_mashup,
}


def _run_ingest(source: str, task_id: str, artist_hint: str, title_hint: str) -> None:
    """Ingest a song and register it in the ChromaDB library.

    Steps:
      1. ingest_song() — download/convert to WAV, return song_id + path
      2. AudioAnalyzer.analyze() — BPM, key, mood, sections
      3. MashupLibrary.upsert_song() — store metadata for matching
    """
    from backend.services.audio.analyzer import AudioAnalyzer

    tm = _get_task_manager()
    try:
        tm.update_progress(task_id, stage="ingesting", percent=10.0,
                           message="Downloading / converting audio")
        result = ingest_song(source)
        song_id = result["id"]
        audio_path = result["path"]

        tm.update_progress(task_id, stage="analyzing", percent=40.0,
                           message="Analyzing audio (BPM, key, sections)")
        analyzer = AudioAnalyzer(sample_rate=44100, whisper_model="base")
        artist = artist_hint or result.get("artist", "Unknown")
        title  = title_hint  or result.get("title",  song_id)
        analysis = analyzer.analyze(audio_path, artist=artist, title=title, depth="standard")

        tm.update_progress(task_id, stage="indexing", percent=80.0,
                           message="Indexing in library")
        lib = _get_library()

        # Map SongAnalysis → library metadata dict
        metadata: Dict[str, Any] = {
            "source":            source,
            "path":              audio_path,
            "artist":            analysis.artist,
            "title":             analysis.title,
            "sample_rate":       analysis.sample_rate,
            "bpm":               analysis.bpm,
            "key":               analysis.key,
            "camelot":           analysis.camelot,
            "duration_sec":      analysis.duration_sec,
            "energy_level":      round(analysis.energy_level * 10, 1),  # 0-10 scale
            "first_downbeat_sec": analysis.first_downbeat_sec,
            "mood_summary":      analysis.mood_summary,
            "primary_genre":     analysis.primary_genre,
            "irony_score":       analysis.irony_score,
            "valence":           analysis.valence,
            "sections":          [dataclasses.asdict(s) for s in analysis.sections],
        }

        upserted_id = lib.upsert_song(
            artist=analysis.artist,
            title=analysis.title,
            metadata=metadata,
            transcript=analysis.transcript,
            force_id=song_id,
        )

        tm.complete_task(task_id, result={"song_id": upserted_id, "source": source,
                                          "cached": result.get("cached", False)})
        logger.info("Ingested and indexed: %s", upserted_id)

    except IngestionError as exc:
        logger.error("Ingestion failed for %r: %s", source, exc)
        tm.fail_task(task_id, error=f"Ingestion error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error ingesting %r: %s", source, exc)
        tm.fail_task(task_id, error=str(exc))


def _run_create_mashup(
    song_a_id: str,
    song_b_id: str,
    mashup_type: str,
    task_id: str,
) -> None:
    """Create a mashup and save to data/mashups/."""
    tm = _get_task_manager()
    try:
        _MASHUPS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(_MASHUPS_DIR / f"{task_id}.mp3")

        tm.update_progress(task_id, stage="creating", percent=20.0,
                           message=f"Creating {mashup_type} mashup")
        lib = _get_library()

        if mashup_type == "stem_swap":
            # Default stem config: vocals from A, drums/bass/other from B
            stem_config = {
                "vocals": song_a_id,
                "drums":  song_b_id,
                "bass":   song_b_id,
                "other":  song_b_id,
            }
            result_path = create_stem_swap_mashup(
                stem_config=stem_config,
                output_path=output_path,
                library=lib,
            )
        elif mashup_type in _MASHUP_TYPE_MAP:
            fn = _MASHUP_TYPE_MAP[mashup_type]
            result_path = fn(
                song_a_id=song_a_id,
                song_b_id=song_b_id,
                output_path=output_path,
                library=lib,
            )
        else:
            tm.fail_task(task_id, error=f"Unknown mashup type: {mashup_type!r}")
            return

        tm.complete_task(task_id, result={
            "output_path": result_path,
            "song_a_id":   song_a_id,
            "song_b_id":   song_b_id,
            "mashup_type": mashup_type,
        })
        logger.info("Mashup created: %s → %s", mashup_type, result_path)

    except SongNotFoundError as exc:
        tm.fail_task(task_id, error=f"Song not found: {exc}")
    except EngineerError as exc:
        tm.fail_task(task_id, error=f"Mashup error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected mashup error: %s", exc)
        tm.fail_task(task_id, error=str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_song_endpoint(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Queue ingestion of a song from a local file or YouTube URL.

    The song is downloaded/converted, analyzed (BPM/key/mood), and indexed in
    the ChromaDB library.  Poll ``GET /api/tasks/{task_id}`` for progress.
    """
    task_id = _get_task_manager().create_task(
        task_type="song_ingest",
        params={"source": request.source},
    )
    background_tasks.add_task(
        _run_ingest,
        source=request.source,
        task_id=task_id,
        artist_hint=request.artist,
        title_hint=request.title,
    )
    return {"task_id": task_id, "status": "queued", "source": request.source}


@router.post("/ingest/batch", status_code=status.HTTP_202_ACCEPTED)
async def ingest_batch(
    sources: List[str],
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Queue batch ingestion of multiple songs."""
    task_ids = []
    for source in sources:
        task_id = _get_task_manager().create_task(
            task_type="song_ingest",
            params={"source": source},
        )
        background_tasks.add_task(
            _run_ingest,
            source=source,
            task_id=task_id,
            artist_hint="",
            title_hint="",
        )
        task_ids.append(task_id)
    return {"task_ids": task_ids, "count": len(task_ids), "status": "queued"}


@router.get("/library")
async def list_library() -> Dict[str, Any]:
    """List all songs in the ChromaDB library."""
    lib = _get_library()
    songs = lib.list_all()
    return {"songs": songs, "total": lib.count()}


@router.get("/library/search")
async def search_library(q: str) -> Dict[str, Any]:
    """Search the library by semantic mood/vibe query."""
    lib = _get_library()
    results = lib.query_semantic(mood_summary=q, max_results=10)
    return {"results": results, "query": q}


@router.post("/match")
async def match_songs(
    song_id: str,
    criteria: str = "hybrid",
    top: int = 5,
) -> Dict[str, Any]:
    """Find compatible songs for mashup pairing.

    ``criteria`` accepts ``harmonic``, ``semantic``, or ``hybrid`` (default).
    Returns a ranked list with compatibility scores and recommended mashup type.
    """
    lib = _get_library()
    try:
        matches = find_match(
            target_song_id=song_id,
            criteria=criteria,
            max_results=top,
            library=lib,
        )
    except CuratorError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return {"song_id": song_id, "matches": matches, "criteria": criteria}


@router.post("/create", status_code=status.HTTP_202_ACCEPTED)
async def create_mashup(
    request: MashupCreateRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Queue mashup creation for two songs.

    ``mashup_type`` must be one of the 8 supported types.
    Poll ``GET /api/tasks/{task_id}`` for progress; result includes output path.
    """
    valid_types = set(_MASHUP_TYPE_MAP) | {"stem_swap"}
    if request.mashup_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown mashup_type {request.mashup_type!r}. "
                   f"Valid: {sorted(valid_types)}",
        )

    task_id = _get_task_manager().create_task(
        task_type="mashup_create",
        params={
            "song_a_id":   request.song_a_id,
            "song_b_id":   request.song_b_id,
            "mashup_type": request.mashup_type,
        },
    )
    background_tasks.add_task(
        _run_create_mashup,
        song_a_id=request.song_a_id,
        song_b_id=request.song_b_id,
        mashup_type=request.mashup_type,
        task_id=task_id,
    )
    return {"task_id": task_id, "status": "queued", "type": request.mashup_type}


@router.get("/types")
async def list_mashup_types() -> Dict[str, Any]:
    """List all 8 mashup types with descriptions."""
    types = [
        {"name": "classic",          "description": "Vocal from A + instrumental from B"},
        {"name": "stem_swap",        "description": "Mix stems from 3+ songs"},
        {"name": "energy_match",     "description": "Align high-energy sections"},
        {"name": "adaptive_harmony", "description": "Auto-fix key clashes via pitch-shifting"},
        {"name": "theme_fusion",     "description": "Filter sections by lyrical themes"},
        {"name": "semantic_aligned", "description": "Question→answer pairing"},
        {"name": "role_aware",       "description": "Vocals shift between lead/harmony/call/response"},
        {"name": "conversational",   "description": "Songs talk to each other like a dialogue"},
    ]
    return {"types": types}
