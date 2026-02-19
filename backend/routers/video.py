"""Video router — execution plan, generation, style management."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


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
    """Generate a video execution plan (backend selection, cost, time estimate)."""
    return {
        "plan_id": "stub",
        "backend": "animatediff",
        "num_scenes": 12,
        "estimated_time_sec": 600,
        "estimated_cost_usd": 0.0,
        "is_local": True,
    }


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_video(request: GenerateRequest) -> Dict[str, str]:
    """Start video generation as a background task."""
    return {"task_id": "stub", "status": "queued", "plan_id": request.plan_id}


@router.post("/scene/edit", status_code=status.HTTP_202_ACCEPTED)
async def edit_scene(request: SceneEditRequest) -> Dict[str, Any]:
    """Edit a single scene prompt and regenerate that scene."""
    return {
        "task_id": "stub",
        "video_id": request.video_id,
        "scene_index": request.scene_index,
        "status": "queued",
    }


@router.get("/styles")
async def list_styles() -> Dict[str, Any]:
    """List all available animation styles."""
    return {"styles": [], "total": 0}


@router.get("/backends")
async def list_backends() -> Dict[str, Any]:
    """List available video generation backends with their status."""
    return {"backends": []}


@router.get("/download/{video_id}")
async def download_video(video_id: str) -> Dict[str, str]:
    """Download a generated video by ID."""
    return {"video_id": video_id, "status": "stub", "url": ""}
