"""LoRA router — registry, training, downloading, recommendations."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter()


class TrainRequest(BaseModel):
    dataset_path: str
    lora_type: str
    trigger_token: str
    output_name: str
    training_steps: int = 1500


class DownloadRequest(BaseModel):
    url: str
    name: str
    lora_type: str
    trigger_token: str
    weight: float = 0.8


class RegisterRequest(BaseModel):
    name: str
    lora_type: str
    trigger_token: str
    file_path: str
    weight: float = 0.8
    description: str = ""


class RecommendRequest(BaseModel):
    audio_id: str
    style: str


@router.get("/list")
async def list_loras(type_filter: str = None) -> Dict[str, Any]:
    """List all registered LoRAs, optionally filtered by type."""
    return {"loras": [], "total": 0, "type_filter": type_filter}


@router.post("/recommend")
async def recommend_loras(request: RecommendRequest) -> Dict[str, Any]:
    """Get LoRA recommendations for a video project."""
    return {
        "available": [],
        "downloadable": [],
        "trainable": [],
    }


@router.post("/train", status_code=status.HTTP_202_ACCEPTED)
async def train_lora(request: TrainRequest) -> Dict[str, str]:
    """Start LoRA training as a background task."""
    return {"task_id": "stub", "status": "queued", "output_name": request.output_name}


@router.post("/download", status_code=status.HTTP_202_ACCEPTED)
async def download_lora(request: DownloadRequest) -> Dict[str, str]:
    """Download a LoRA from HuggingFace or Civitai."""
    return {"task_id": "stub", "status": "queued", "name": request.name}


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_lora(request: RegisterRequest) -> Dict[str, str]:
    """Register an existing LoRA file in the registry."""
    return {"name": request.name, "status": "registered"}


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_lora(name: str) -> None:
    """Remove a LoRA from the registry."""
    return None
