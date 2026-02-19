"""System router — health, GPU status, model inventory."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
from fastapi import APIRouter, status
from pydantic import BaseModel

logger = logging.getLogger("beat_studio.routers.system")
router = APIRouter()

_BACKEND_DIR = Path(__file__).parent.parent
_CHECKPOINTS_YAML = _BACKEND_DIR / "config" / "checkpoints.yaml"
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


class InstallModelRequest(BaseModel):
    model_name: str


def _check_model_installed(model: dict) -> bool:
    """Return True if the model appears to be cached locally."""
    cache_subdir = model.get("cache_subdir")
    if cache_subdir:
        return (_HF_CACHE / cache_subdir).exists()
    # No cache_subdir means manually managed — assume present if source is local_only
    return model.get("source") == "local_only"


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@router.get("/gpu")
async def gpu_status() -> Dict[str, Any]:
    """Get GPU status and VRAM usage."""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            used = torch.cuda.memory_allocated(0) / 1e9
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "total_gb": round(total, 1),
                "used_gb": round(used, 2),
                "free_gb": round(total - used, 2),
            }
    except ImportError:
        pass
    return {"available": False, "name": "none", "total_gb": 0, "used_gb": 0, "free_gb": 0}


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    """List model inventory with installation status from checkpoints.yaml."""
    try:
        raw = yaml.safe_load(_CHECKPOINTS_YAML.read_text())
    except Exception as exc:
        logger.warning("Failed to load checkpoints.yaml: %s", exc)
        return {"models": [], "total": 0}

    models: List[Dict[str, Any]] = []
    for m in raw.get("models", []):
        installed = _check_model_installed(m)
        models.append({
            "name":          m.get("name"),
            "display_name":  m.get("display_name"),
            "purpose":       m.get("purpose"),
            "status":        m.get("status"),
            "size_gb":       m.get("size_gb"),
            "vram_gb":       m.get("vram_gb"),
            "installed":     installed,
            "source":        m.get("source"),
        })

    return {"models": models, "total": len(models)}


@router.post("/models/install", status_code=status.HTTP_202_ACCEPTED)
async def install_model(request: InstallModelRequest) -> Dict[str, str]:
    """Download and install a recommended model."""
    return {
        "task_id": "stub",
        "status": "queued",
        "model": request.model_name,
    }
