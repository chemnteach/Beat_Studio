"""Storyboard keyframe preview router.

Endpoints for generating, browsing, regenerating, and approving per-scene
SDXL keyframe images. The approved keyframes are passed as prompt overrides
into the video generation step.

Mounted at: /api/video/storyboard
"""
from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator

from backend.services.shared.task_manager import TaskManager
from backend.services.storyboard.service import StoryboardService
from backend.services.storyboard.types import SceneInput

logger = logging.getLogger("beat_studio.routers.storyboard")
router = APIRouter()

_BACKEND_DIR = Path(__file__).parent.parent
_STORYBOARD_BASE = _BACKEND_DIR.parent / "output" / "storyboard"

_service: Optional[StoryboardService] = None
_service_lock = threading.Lock()
_task_manager: Optional[TaskManager] = None
_task_manager_lock = threading.Lock()


def _get_service() -> StoryboardService:
    global _service
    if _service is not None:
        return _service
    with _service_lock:
        if _service is None:
            _service = StoryboardService()
    return _service


def _get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is not None:
        return _task_manager
    with _task_manager_lock:
        if _task_manager is None:
            _task_manager = TaskManager(db_path=str(_BACKEND_DIR / "tasks.db"))
    return _task_manager


# ── Pydantic request / response models ────────────────────────────────────────

class StoryboardSceneInput(BaseModel):
    """Input data for one scene in a storyboard generation request."""
    scene_idx: int
    storyboard_prompt: str        # Cinematic description (from /api/video/prompts storyboard field)
    positive_prompt: str          # Composed style prompt (style.prefix + scene description)


class StoryboardGenerateRequest(BaseModel):
    style: str                                  # e.g. "cinematic"
    lora_names: List[str] = []
    scenes: List[StoryboardSceneInput]

    @field_validator("scenes")
    @classmethod
    def scenes_must_not_be_empty(cls, v: List[StoryboardSceneInput]) -> List[StoryboardSceneInput]:
        if not v:
            raise ValueError("scenes must not be empty")
        return v


class StoryboardRegenerateRequest(BaseModel):
    seed: Optional[int] = None                  # None → random seed
    positive_prompt: Optional[str] = None       # None → reuse existing prompt
    lora_weights: Optional[Dict[str, float]] = None  # per-LoRA weight overrides keyed by registry name


class StoryboardApproveRequest(BaseModel):
    # scene_idx (as str key from JSON) → selected version number (1-indexed)
    selections: Dict[str, int]


class VersionEntryResponse(BaseModel):
    version: int
    url: str                                    # e.g. /api/video/storyboard/{id}/img/scene_0/v1.png
    seed: int
    timestamp: str
    lora_weights: Dict[str, float] = {}        # weights used when generating this version


class StoryboardSceneResponse(BaseModel):
    scene_idx: int
    storyboard_prompt: str
    positive_prompt: str
    approved_version: Optional[int]
    versions: List[VersionEntryResponse]


class StoryboardImagesResponse(BaseModel):
    storyboard_id: str
    status: str
    scenes: List[StoryboardSceneResponse]


class StoryboardApproveResponse(BaseModel):
    storyboard_id: str
    # scene_idx (str) → absolute path on disk — passed to GenerateRequest.user_overrides
    approved_paths: Dict[str, str]


# ── Background workers ─────────────────────────────────────────────────────────

def _run_generate_storyboard(
    task_id: str,
    storyboard_id: str,
    scenes: List[SceneInput],
    style: str,
    lora_names: List[str],
) -> None:
    """Background worker: generate SDXL keyframes for all scenes."""
    tm = _get_task_manager()
    svc = _get_service()
    try:
        svc.generate_all_scenes(storyboard_id, scenes, style, lora_names)
        tm.complete_task(task_id, {"storyboard_id": storyboard_id})
    except Exception as exc:
        logger.exception("Storyboard generation failed for %s: %s", storyboard_id, exc)
        tm.fail_task(task_id, str(exc))


def _run_regenerate_scene(
    task_id: str,
    storyboard_id: str,
    scene_idx: int,
    prompt_override: Optional[str],
    seed: Optional[int],
    lora_names: List[str],
    lora_weights: Optional[Dict[str, float]],
) -> None:
    """Background worker: regenerate a single scene keyframe."""
    tm = _get_task_manager()
    svc = _get_service()
    try:
        svc.generate_single_scene(
            storyboard_id,
            scene_idx,
            prompt_override=prompt_override,
            seed=seed,
            lora_names=lora_names,
            lora_weights=lora_weights,
        )
        tm.complete_task(task_id, {"storyboard_id": storyboard_id, "scene_idx": scene_idx})
    except Exception as exc:
        logger.exception(
            "Scene regeneration failed for %s/scene_%d: %s", storyboard_id, scene_idx, exc
        )
        tm.fail_task(task_id, str(exc))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_storyboard(
    request: StoryboardGenerateRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Start async SDXL keyframe generation for all scenes.

    Returns a task_id for progress tracking. Poll GET /{storyboard_id}/images
    when the task completes.
    """
    tm = _get_task_manager()
    storyboard_id = str(uuid.uuid4())
    task_id = tm.create_task("generate_storyboard", {"storyboard_id": storyboard_id})
    scenes = [
        SceneInput(
            scene_idx=s.scene_idx,
            storyboard_prompt=s.storyboard_prompt,
            positive_prompt=s.positive_prompt,
        )
        for s in request.scenes
    ]
    background_tasks.add_task(
        _run_generate_storyboard,
        task_id,
        storyboard_id,
        scenes,
        request.style,
        request.lora_names,
    )
    return {
        "task_id": task_id,
        "storyboard_id": storyboard_id,
        "status": "queued",
        "scene_count": len(request.scenes),
    }


@router.get("/{storyboard_id}/images")
async def get_storyboard_images(storyboard_id: str) -> StoryboardImagesResponse:
    """Return current state for all scenes: versions, URLs, approval status."""
    svc = _get_service()
    state = svc.get_state(storyboard_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Storyboard '{storyboard_id}' not found")

    scenes = []
    for scene in state.scenes:
        versions = [
            VersionEntryResponse(
                version=v.version,
                url=f"/api/video/storyboard/{storyboard_id}/img/scene_{scene.scene_idx}/{v.filename}",
                seed=v.seed,
                timestamp=v.timestamp,
                lora_weights=v.lora_weights,
            )
            for v in scene.versions
        ]
        scenes.append(StoryboardSceneResponse(
            scene_idx=scene.scene_idx,
            storyboard_prompt=scene.storyboard_prompt,
            positive_prompt=scene.positive_prompt,
            approved_version=scene.approved_version,
            versions=versions,
        ))

    return StoryboardImagesResponse(
        storyboard_id=storyboard_id,
        status=state.status,
        scenes=scenes,
    )


@router.post("/{storyboard_id}/scene/{scene_idx}/regenerate",
             status_code=status.HTTP_202_ACCEPTED)
async def regenerate_scene(
    storyboard_id: str,
    scene_idx: int,
    request: StoryboardRegenerateRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Regenerate a single scene keyframe, appending a new version."""
    svc = _get_service()
    tm = _get_task_manager()
    state = svc.get_state(storyboard_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Storyboard '{storyboard_id}' not found")

    task_id = tm.create_task("regenerate_scene", {
        "storyboard_id": storyboard_id,
        "scene_idx": scene_idx,
    })
    background_tasks.add_task(
        _run_regenerate_scene,
        task_id,
        storyboard_id,
        scene_idx,
        request.positive_prompt,
        request.seed,
        state.lora_names,
        request.lora_weights,
    )
    return {
        "task_id": task_id,
        "storyboard_id": storyboard_id,
        "scene_idx": scene_idx,
        "status": "queued",
    }


@router.post("/{storyboard_id}/approve")
async def approve_storyboard(
    storyboard_id: str,
    request: StoryboardApproveRequest,
) -> StoryboardApproveResponse:
    """Lock in version selections per scene. Returns approved image paths.

    The caller passes these paths as ``user_overrides`` keys in
    ``POST /api/video/generate`` to use approved keyframes as prompt anchors.
    """
    svc = _get_service()
    # JSON keys are always strings; convert to int before calling service
    selections = {int(k): v for k, v in request.selections.items()}
    try:
        approved_paths = svc.approve(storyboard_id, selections)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StoryboardApproveResponse(
        storyboard_id=storyboard_id,
        approved_paths={str(k): v for k, v in approved_paths.items()},
    )


@router.get("/{storyboard_id}/img/{scene_dir}/{filename}")
async def serve_keyframe_image(
    storyboard_id: str,
    scene_dir: str,
    filename: str,
) -> FileResponse:
    """Serve a keyframe PNG from disk.

    URL pattern: /api/video/storyboard/{id}/img/scene_0/v1.png
    """
    img_path = _STORYBOARD_BASE / storyboard_id / scene_dir / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    return FileResponse(str(img_path), media_type="image/png")
