"""Audio router — upload, analyze, and retrieve song analysis."""
from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from backend.services.audio.analyzer import AudioAnalyzer
from backend.services.shared.task_manager import TaskManager

logger = logging.getLogger("beat_studio.routers.audio")
router = APIRouter()

# ── Paths ─────────────────────────────────────────────────────────────────────
# __file__ = backend/routers/audio.py  →  .parent.parent = backend/
_BACKEND_DIR  = Path(__file__).parent.parent
_UPLOADS_DIR  = _BACKEND_DIR / "data" / "uploads"
_ANALYSIS_DIR = _BACKEND_DIR / "data" / "analysis"

_ALLOWED_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}

# ── Module-level singletons (lazy init) ───────────────────────────────────────
_task_manager: Optional[TaskManager] = None
_analyzer: Optional[AudioAnalyzer] = None


def _get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(db_path=str(_BACKEND_DIR / "tasks.db"))
    return _task_manager


def _get_analyzer() -> AudioAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = AudioAnalyzer(sample_rate=44100, whisper_model="base")
    return _analyzer


# ── Pydantic models ───────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    audio_id: str
    depth: str = "standard"  # "basic" | "standard" | "full"
    artist: str = ""          # Optional; falls back to filename stem
    title: str = ""           # Optional; falls back to filename stem


class AnalysisResponse(BaseModel):
    audio_id: str
    status: str               # "queued" | "pending" | "running" | "complete" | "failed"
    task_id: Optional[str] = None
    analysis: Dict[str, Any] = {}


# ── Background worker ─────────────────────────────────────────────────────────


def _run_analysis(
    audio_id: str,
    task_id: str,
    audio_path: str,
    artist: str,
    title: str,
    depth: str,
) -> None:
    """Run AudioAnalyzer synchronously in a background thread.

    Called via FastAPI BackgroundTasks — Starlette runs sync functions in the
    thread pool, so librosa's CPU work doesn't block the event loop.
    """
    tm = _get_task_manager()
    try:
        tm.update_progress(task_id, stage="loading", percent=5.0, message="Loading audio")
        analyzer = _get_analyzer()

        tm.update_progress(task_id, stage="analyzing", percent=20.0,
                           message=f"Analyzing ({depth})")
        result = analyzer.analyze(audio_path, artist=artist, title=title, depth=depth)

        tm.update_progress(task_id, stage="saving", percent=90.0, message="Saving result")
        _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        (_ANALYSIS_DIR / f"{audio_id}.json").write_text(
            json.dumps(dataclasses.asdict(result))
        )

        tm.complete_task(task_id, result={"audio_id": audio_id})
        logger.info("Analysis complete for audio_id=%s", audio_id)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Analysis failed for audio_id=%s: %s", audio_id, exc)
        tm.fail_task(task_id, error=str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_audio(file: UploadFile = File(...)) -> Dict[str, str]:
    """Upload an audio file and return its assigned ID.

    Saves the file to ``data/uploads/{audio_id}{ext}`` and writes a sidecar
    ``{audio_id}.meta.json`` so downstream endpoints can locate the file by ID.
    The returned ``audio_id`` is a UUID — pass it to ``POST /analyze``.
    """
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {sorted(_ALLOWED_SUFFIXES)}"
            ),
        )

    audio_id = str(uuid.uuid4())
    _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    dest = _UPLOADS_DIR / f"{audio_id}{suffix}"
    content = await file.read()
    dest.write_bytes(content)

    # Sidecar: maps audio_id → original filename + absolute path
    meta = {"filename": filename, "suffix": suffix, "path": str(dest)}
    (_UPLOADS_DIR / f"{audio_id}.meta.json").write_text(json.dumps(meta))

    logger.info("Uploaded '%s' → audio_id=%s (%d bytes)", filename, audio_id, len(content))
    return {"audio_id": audio_id, "filename": filename, "status": "uploaded"}


@router.post("/analyze")
async def analyze_audio(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> AnalysisResponse:
    """Dispatch audio analysis as a background task.

    Validates that the file was previously uploaded (via ``POST /upload``),
    creates a persistent TaskManager entry, and queues the analysis work.
    Poll ``GET /api/tasks/{task_id}`` for progress or ``GET /analysis/{audio_id}``
    for the completed result.
    """
    meta_path = _UPLOADS_DIR / f"{request.audio_id}.meta.json"
    if not meta_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No uploaded file found for audio_id={request.audio_id!r}. "
                   "Call POST /upload first.",
        )

    meta = json.loads(meta_path.read_text())
    stem = Path(meta["filename"]).stem
    artist = request.artist or "Unknown"
    title  = request.title  or stem

    task_id = _get_task_manager().create_task(
        task_type="audio_analysis",
        params={"audio_id": request.audio_id, "depth": request.depth},
    )

    background_tasks.add_task(
        _run_analysis,
        audio_id=request.audio_id,
        task_id=task_id,
        audio_path=meta["path"],
        artist=artist,
        title=title,
        depth=request.depth,
    )

    return AnalysisResponse(
        audio_id=request.audio_id,
        status="queued",
        task_id=task_id,
    )


@router.get("/analysis/{audio_id}")
async def get_analysis(audio_id: str) -> AnalysisResponse:
    """Return the cached analysis for a previously analyzed audio file.

    Response ``status`` values:
    - ``"complete"`` — analysis JSON available in ``analysis`` field
    - ``"pending"``  — file uploaded but analysis not yet started/complete
    - 404            — ``audio_id`` not found at all
    """
    # Fast path: analysis JSON exists
    analysis_file = _ANALYSIS_DIR / f"{audio_id}.json"
    if analysis_file.exists():
        return AnalysisResponse(
            audio_id=audio_id,
            status="complete",
            analysis=json.loads(analysis_file.read_text()),
        )

    # File uploaded but analysis not done (or failed in background)
    if (_UPLOADS_DIR / f"{audio_id}.meta.json").exists():
        return AnalysisResponse(audio_id=audio_id, status="pending")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No upload or analysis found for audio_id={audio_id!r}.",
    )
