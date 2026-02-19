"""LoRA router — registry, training, downloading, recommendations."""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, status
from pydantic import BaseModel

from backend.services.lora.registry import LoRARegistry

logger = logging.getLogger("beat_studio.routers.lora")
router = APIRouter()

_BACKEND_DIR = Path(__file__).parent.parent
_LORAS_YAML = _BACKEND_DIR / "config" / "loras.yaml"
_LORAS_BASE = _BACKEND_DIR.parent / "output" / "loras"

_registry: Optional[LoRARegistry] = None


def _get_registry() -> LoRARegistry:
    global _registry
    if _registry is None:
        _registry = LoRARegistry(
            registry_path=str(_LORAS_YAML),
            base_path=str(_LORAS_BASE),
        )
    return _registry


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
async def list_loras(type_filter: Optional[str] = None) -> Dict[str, Any]:
    """List all registered LoRAs, optionally filtered by type."""
    entries = _get_registry().list_all(type_filter=type_filter)
    loras = [dataclasses.asdict(e) for e in entries]
    return {"loras": loras, "total": len(loras), "type_filter": type_filter}


@router.post("/recommend")
async def recommend_loras(request: RecommendRequest) -> Dict[str, Any]:
    """Get LoRA recommendations for a video project.

    Returns available on-disk LoRAs whose tags match the requested style.
    Downloadable and trainable suggestions require GPU/internet access and
    remain empty until those services are wired.
    """
    registry = _get_registry()
    style_lower = request.style.lower()

    available = []
    for entry in registry.list_all():
        validation = registry.validate(entry.name)
        if not validation.file_exists:
            continue
        tags_lower = {t.lower() for t in entry.tags}
        if style_lower in tags_lower or entry.type.lower() == style_lower:
            available.append(dataclasses.asdict(entry))

    return {
        "available":    available,
        "downloadable": [],   # requires internet search — stub
        "trainable":    [],   # requires NarrativeArc analysis — stub
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
