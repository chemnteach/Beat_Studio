"""System router — health, GPU status, model inventory."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter()


class InstallModelRequest(BaseModel):
    model_name: str


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
    """List installed model inventory."""
    return {"models": [], "total": 0}


@router.post("/models/install", status_code=status.HTTP_202_ACCEPTED)
async def install_model(request: InstallModelRequest) -> Dict[str, str]:
    """Download and install a recommended model."""
    return {
        "task_id": "stub",
        "status": "queued",
        "model": request.model_name,
    }
