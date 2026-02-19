"""Video router — execution plan, generation, style management."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.services.video.beat_sync import BeatSynchronizer
from backend.services.video.cost_estimator import CostEstimator
from backend.services.video.model_router import ModelRouter, NoBackendAvailableError

logger = logging.getLogger("beat_studio.routers.video")
router = APIRouter()

_BACKEND_DIR = Path(__file__).parent.parent
_ANALYSIS_DIR = _BACKEND_DIR / "data" / "analysis"
_STYLES_YAML = _BACKEND_DIR / "config" / "animation_styles.yaml"

_beat_sync: Optional[BeatSynchronizer] = None
_model_router: Optional[ModelRouter] = None
_cost_estimator: Optional[CostEstimator] = None


def _get_beat_sync() -> BeatSynchronizer:
    global _beat_sync
    if _beat_sync is None:
        _beat_sync = BeatSynchronizer()
    return _beat_sync


def _get_model_router() -> ModelRouter:
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router


def _get_cost_estimator() -> CostEstimator:
    global _cost_estimator
    if _cost_estimator is None:
        _cost_estimator = CostEstimator()
    return _cost_estimator


def _load_analysis_for_sync(audio_id: str) -> SimpleNamespace:
    """Load cached SongAnalysis JSON and adapt to BeatSynchronizer duck-type.

    BeatSynchronizer expects:
      analysis.duration  (float)
      analysis.bpm       (float)
      analysis.sections  (list of objects with .start, .end, .section_type, .energy_level)
    SongAnalysis JSON stores duration_sec, start_sec, end_sec — so we remap.
    """
    path = _ANALYSIS_DIR / f"{audio_id}.json"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis found for audio_id {audio_id!r}. "
                   "Run POST /api/audio/analyze first.",
        )

    data = json.loads(path.read_text())

    sections = []
    for s in data.get("sections", []):
        sections.append(SimpleNamespace(
            start=s.get("start_sec", 0.0),
            end=s.get("end_sec", 0.0),
            section_type=s.get("section_type", "verse"),
            energy_level=s.get("energy_level", 0.5),
        ))

    return SimpleNamespace(
        duration=data.get("duration_sec", 0.0),
        bpm=data.get("bpm", 120.0),
        sections=sections,
    )


class PlanRequest(BaseModel):
    audio_id: str
    style: str = "cinematic"
    quality: str = "professional"
    local_preferred: bool = True
    resolution: List[int] = [1920, 1080]


class GenerateRequest(BaseModel):
    plan_id: str
    audio_id: str


class SceneEditRequest(BaseModel):
    video_id: str
    scene_index: int
    new_prompt: str


@router.post("/plan")
async def plan_video(request: PlanRequest) -> Dict[str, Any]:
    """Generate a video execution plan (backend selection, cost, time estimate).

    Loads the cached audio analysis, creates beat-aligned scenes, selects
    the best available backend, and returns cost and time estimates.
    """
    analysis = _load_analysis_for_sync(request.audio_id)
    resolution = tuple(request.resolution[:2]) if len(request.resolution) >= 2 else (1920, 1080)

    # Beat-aligned scene plan
    scenes = _get_beat_sync().create_scene_plan(analysis, quality_tier=request.quality)

    # Backend selection
    try:
        backend = _get_model_router().select_backend(
            style=request.style,
            quality=request.quality,
            local_preferred=request.local_preferred,
        )
        backend_name = backend.name()
        is_local = backend.vram_required_gb() > 0
    except NoBackendAvailableError as exc:
        logger.warning("No backend available: %s", exc)
        backend_name = "none"
        is_local = False

    # Cost and time estimates
    estimator = _get_cost_estimator()
    num_scenes = len(scenes)

    if backend_name != "none":
        try:
            total_cost = estimator.estimate_total(backend, num_scenes, resolution)
            total_time = estimator.estimate_total_time(backend, num_scenes, resolution)
        except Exception as exc:
            logger.warning("Cost estimation failed: %s", exc)
            total_cost = 0.0
            total_time = 0.0
    else:
        total_cost = 0.0
        total_time = 0.0

    plan_id = str(uuid.uuid4())

    return {
        "plan_id":            plan_id,
        "audio_id":           request.audio_id,
        "backend":            backend_name,
        "style":              request.style,
        "quality":            request.quality,
        "num_scenes":         num_scenes,
        "estimated_time_sec": round(total_time, 1),
        "estimated_cost_usd": round(total_cost, 4),
        "is_local":           is_local,
        "scenes": [
            {
                "scene_index": s.scene_index,
                "start_sec":   round(s.start_sec, 3),
                "end_sec":     round(s.end_sec, 3),
                "duration_sec": round(s.duration_sec, 3),
                "is_hero":     s.is_hero,
                "notes":       s.notes,
            }
            for s in scenes
        ],
    }


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_video(request: GenerateRequest) -> Dict[str, str]:
    """Start video generation as a background task."""
    return {"task_id": "stub", "status": "queued", "plan_id": request.plan_id}


@router.post("/scene/edit", status_code=status.HTTP_202_ACCEPTED)
async def edit_scene(request: SceneEditRequest) -> Dict[str, Any]:
    """Edit a single scene prompt and regenerate that scene."""
    return {
        "task_id":     "stub",
        "video_id":    request.video_id,
        "scene_index": request.scene_index,
        "status":      "queued",
    }


@router.get("/styles")
async def list_styles() -> Dict[str, Any]:
    """List all available animation styles from animation_styles.yaml."""
    try:
        raw = yaml.safe_load(_STYLES_YAML.read_text())
    except Exception as exc:
        logger.warning("Failed to load animation_styles.yaml: %s", exc)
        return {"styles": [], "total": 0}

    styles = []
    for name, data in raw.get("styles", {}).items():
        styles.append({
            "name":                name,
            "display_name":        data.get("display_name"),
            "category":            data.get("category"),
            "recommended_backend": data.get("recommended_backend"),
            "best_for":            data.get("best_for"),
            "guidance_scale":      data.get("guidance_scale"),
            "steps":               data.get("steps"),
        })

    return {"styles": styles, "total": len(styles)}


@router.get("/backends")
async def list_backends() -> Dict[str, Any]:
    """List available video generation backends."""
    available = _get_model_router().list_available_backends()
    return {"backends": available}


@router.get("/download/{video_id}")
async def download_video(video_id: str) -> Dict[str, str]:
    """Download a generated video by ID."""
    return {"video_id": video_id, "status": "stub", "url": ""}
