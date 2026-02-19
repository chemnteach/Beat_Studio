"""Nova Fade router — character pipeline, drift testing, DJ video generation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import APIRouter, status
from pydantic import BaseModel

from backend.services.lora.registry import LoRARegistry

logger = logging.getLogger("beat_studio.routers.nova_fade")
router = APIRouter()

_BACKEND_DIR = Path(__file__).parent.parent
_CONSTITUTION_YAML = _BACKEND_DIR / "config" / "nova_fade_constitution.yaml"
_CANONICAL_DIR = _BACKEND_DIR.parent / "output" / "nova_fade" / "canonical"
_LORAS_YAML = _BACKEND_DIR / "config" / "loras.yaml"

_NOVA_IDENTITY_LORA = "novafade_id_v1"
_NOVA_STYLE_LORA = "crossfadeclub_style_v1"


def _get_lora_status(registry: LoRARegistry, name: str) -> str:
    """Return 'available', 'missing', or 'not_registered' for a LoRA."""
    entry = registry.get(name)
    if entry is None:
        return "not_registered"
    validation = registry.validate(name)
    return "available" if validation.file_exists else "missing"


class DJVideoRequest(BaseModel):
    mashup_id: str
    theme: str = "sponsor_neon"
    style: str = "3d_cartoon"
    output_platform: str = "youtube"


class DriftTestRequest(BaseModel):
    lora_path: str
    canonical_dir: str = "output/nova_fade/canonical"


class CanonicalGenRequest(BaseModel):
    num_images: int = 20
    expressions: list = []


@router.post("/generate-canonical", status_code=status.HTTP_202_ACCEPTED)
async def generate_canonical(request: CanonicalGenRequest) -> Dict[str, Any]:
    """Generate Nova Fade canonical reference images via SDXL."""
    return {
        "task_id": "stub",
        "status": "queued",
        "num_images": request.num_images,
    }


@router.post("/train-identity-lora", status_code=status.HTTP_202_ACCEPTED)
async def train_identity_lora() -> Dict[str, str]:
    """Train the novafade_id_v1 Identity LoRA from canonical images."""
    return {"task_id": "stub", "status": "queued", "lora": "novafade_id_v1"}


@router.post("/train-style-lora", status_code=status.HTTP_202_ACCEPTED)
async def train_style_lora() -> Dict[str, str]:
    """Train the crossfadeclub_style_v1 Style LoRA."""
    return {"task_id": "stub", "status": "queued", "lora": "crossfadeclub_style_v1"}


@router.post("/drift-test")
async def run_drift_test(request: DriftTestRequest) -> Dict[str, Any]:
    """Run CLIP-based drift detection on a Nova Fade LoRA checkpoint."""
    return {
        "task_id": "stub",
        "status": "queued",
        "lora_path": request.lora_path,
    }


@router.post("/dj-video", status_code=status.HTTP_202_ACCEPTED)
async def generate_dj_video(request: DJVideoRequest) -> Dict[str, str]:
    """Generate a Nova Fade DJ performance video for a mashup."""
    return {
        "task_id": "stub",
        "status": "queued",
        "mashup_id": request.mashup_id,
        "theme": request.theme,
    }


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get the Nova Fade character system status."""
    # Constitution version
    constitution_version: str = "unknown"
    try:
        data = yaml.safe_load(_CONSTITUTION_YAML.read_text())
        constitution_version = data.get("version", "unknown")
    except Exception as exc:
        logger.warning("Could not read nova_fade_constitution.yaml: %s", exc)

    # Canonical images on disk
    canonical_count = 0
    if _CANONICAL_DIR.exists():
        canonical_count = sum(
            1 for f in _CANONICAL_DIR.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )

    # LoRA registry status
    try:
        registry = LoRARegistry(
            registry_path=str(_LORAS_YAML),
            base_path=str(_BACKEND_DIR.parent / "output" / "loras"),
        )
        identity_lora_status = _get_lora_status(registry, _NOVA_IDENTITY_LORA)
        style_lora_status = _get_lora_status(registry, _NOVA_STYLE_LORA)
    except Exception as exc:
        logger.warning("Could not read LoRA registry: %s", exc)
        identity_lora_status = "unknown"
        style_lora_status = "unknown"

    return {
        "identity_lora":       identity_lora_status,
        "style_lora":          style_lora_status,
        "canonical_images":    canonical_count,
        "last_drift_test":     None,
        "constitution_version": constitution_version,
    }
