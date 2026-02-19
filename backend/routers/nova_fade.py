"""Nova Fade router — character pipeline, drift testing, DJ video generation."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter()


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
    return {
        "identity_lora": "missing",
        "style_lora": "missing",
        "canonical_images": 0,
        "last_drift_test": None,
        "constitution_version": "1.0",
    }
