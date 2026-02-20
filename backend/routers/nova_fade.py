"""Nova Fade router — character pipeline, drift testing, DJ video generation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, BackgroundTasks, status
from pydantic import BaseModel

from backend.services.lora.registry import LoRARegistry
from backend.services.lora.trainer import LoRATrainer
from backend.services.lora.types import LoRAEntry, LoRATrainingConfig
from backend.services.nova_fade.character import NovaFadeCharacter
from backend.services.shared.task_manager import TaskManager
from backend.services.shared.vram_manager import VRAMManager

logger = logging.getLogger("beat_studio.routers.nova_fade")
router = APIRouter()

_BACKEND_DIR = Path(__file__).parent.parent
_CONSTITUTION_YAML = _BACKEND_DIR / "config" / "nova_fade_constitution.yaml"
_CANONICAL_DIR = _BACKEND_DIR.parent / "output" / "nova_fade" / "canonical"
_LORAS_YAML = _BACKEND_DIR / "config" / "loras.yaml"
_LORA_BASE = _BACKEND_DIR.parent / "output" / "loras"

# Fixed: was "novafade_id_v1" — must match loras.yaml entry name
_NOVA_IDENTITY_LORA = "nova_fade_id_v1"
_NOVA_STYLE_LORA = "crossfadeclub_style_v1"

_task_manager: Optional[TaskManager] = None
_vram_manager: Optional[VRAMManager] = None


def _get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


def _get_vram_manager() -> VRAMManager:
    global _vram_manager
    if _vram_manager is None:
        _vram_manager = VRAMManager(budget_gb=12.0)
    return _vram_manager


def _get_lora_status(registry: LoRARegistry, name: str) -> str:
    """Return 'available', 'missing', or 'not_registered' for a LoRA."""
    entry = registry.get(name)
    if entry is None:
        return "not_registered"
    validation = registry.validate(name)
    return "available" if validation.file_exists else "missing"


# ── Pydantic models ────────────────────────────────────────────────────────────

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


class TrainLoRARequest(BaseModel):
    dataset_path: str = "output/nova_fade/canonical"
    training_steps: int = 1500
    rank: int = 16


# ── Background workers ─────────────────────────────────────────────────────────

def _run_generate_canonical(
    task_id: str,
    num_images: int,
    expressions: List[str],
    output_dir: Path,
) -> None:
    """Background worker: generate Nova Fade canonical reference images via SDXL."""
    tm = _get_task_manager()
    vm = _get_vram_manager()
    pipe = None
    try:
        tm.update_progress(task_id, "loading_model", 5.0, "Loading SDXL pipeline…")
        import torch
        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")
        vm.set_pipeline(pipe, "sdxl_canonical")

        char = NovaFadeCharacter()
        exprs = expressions if expressions else list(char.EXPRESSIONS)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = 0

        for i in range(num_images):
            expr = exprs[i % len(exprs)]
            positive, negative = char.get_canonical_header(expression=expr)
            seed = i * 137  # deterministic spread
            pct = 10.0 + (i / num_images) * 85.0
            tm.update_progress(
                task_id, "generating", pct, f"Image {i + 1}/{num_images} ({expr})"
            )

            image = pipe(
                prompt=positive,
                negative_prompt=negative,
                width=1024,
                height=1024,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]

            image.save(output_dir / f"{expr}_{seed:04d}.png")
            generated += 1

        tm.complete_task(task_id, {
            "canonical_images": generated,
            "output_dir": str(output_dir),
        })
    except Exception as exc:
        logger.exception("Canonical image generation failed: %s", exc)
        tm.fail_task(task_id, str(exc))
    finally:
        if pipe is not None:
            try:
                vm.kill()
            except Exception:
                pass


def _run_train_lora(
    task_id: str,
    config: LoRATrainingConfig,
    lora_name: str,
) -> None:
    """Background worker: train a LoRA from canonical images."""
    tm = _get_task_manager()
    vm = _get_vram_manager()
    try:
        vm.kill()  # Free VRAM before loading SDXL UNet for training
        tm.update_progress(task_id, "training", 10.0, f"Starting {lora_name} LoRA training…")

        result = LoRATrainer().train(config)

        if result.success:
            registry = LoRARegistry(
                registry_path=str(_LORAS_YAML),
                base_path=str(_LORA_BASE),
            )
            entry = registry.get(lora_name)
            if entry is not None:
                entry.status = "available"
                registry.register(entry)
            tm.complete_task(task_id, {
                "lora": lora_name,
                "lora_path": result.lora_path,
                "status": "available",
            })
        else:
            tm.fail_task(task_id, result.error or "Training returned failure")
    except Exception as exc:
        logger.exception("LoRA training worker failed: %s", exc)
        tm.fail_task(task_id, str(exc))


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/generate-canonical", status_code=status.HTTP_202_ACCEPTED)
async def generate_canonical(
    request: CanonicalGenRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Generate Nova Fade canonical reference images via SDXL."""
    task_id = _get_task_manager().create_task("generate_canonical", {
        "num_images": request.num_images,
        "expressions": request.expressions,
    })
    background_tasks.add_task(
        _run_generate_canonical,
        task_id,
        request.num_images,
        list(request.expressions),
        _CANONICAL_DIR,
    )
    return {
        "task_id": task_id,
        "status": "queued",
        "num_images": request.num_images,
    }


@router.post("/train-identity-lora", status_code=status.HTTP_202_ACCEPTED)
async def train_identity_lora(
    background_tasks: BackgroundTasks,
    request: Optional[TrainLoRARequest] = None,
) -> Dict[str, str]:
    """Train the nova_fade_id_v1 Identity LoRA from canonical images."""
    if request is None:
        request = TrainLoRARequest()
    config = LoRATrainingConfig(
        dataset_path=request.dataset_path,
        lora_type="identity",
        trigger_token="novafade_char",
        output_name=_NOVA_IDENTITY_LORA,
        training_steps=request.training_steps,
        rank=request.rank,
    )
    task_id = _get_task_manager().create_task("train_identity_lora", {
        "lora": _NOVA_IDENTITY_LORA,
        "dataset_path": request.dataset_path,
        "training_steps": request.training_steps,
    })
    background_tasks.add_task(_run_train_lora, task_id, config, _NOVA_IDENTITY_LORA)
    return {"task_id": task_id, "status": "queued", "lora": _NOVA_IDENTITY_LORA}


@router.post("/train-style-lora", status_code=status.HTTP_202_ACCEPTED)
async def train_style_lora(
    background_tasks: BackgroundTasks,
    request: Optional[TrainLoRARequest] = None,
) -> Dict[str, str]:
    """Train the crossfadeclub_style_v1 Style LoRA."""
    if request is None:
        request = TrainLoRARequest()
    config = LoRATrainingConfig(
        dataset_path=request.dataset_path,
        lora_type="style",
        trigger_token="crossfadeclub_style",
        output_name=_NOVA_STYLE_LORA,
        training_steps=request.training_steps,
        rank=request.rank,
    )
    task_id = _get_task_manager().create_task("train_style_lora", {
        "lora": _NOVA_STYLE_LORA,
        "dataset_path": request.dataset_path,
        "training_steps": request.training_steps,
    })
    background_tasks.add_task(_run_train_lora, task_id, config, _NOVA_STYLE_LORA)
    return {"task_id": task_id, "status": "queued", "lora": _NOVA_STYLE_LORA}


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
        "identity_lora":        identity_lora_status,
        "style_lora":           style_lora_status,
        "canonical_images":     canonical_count,
        "last_drift_test":      None,
        "constitution_version": constitution_version,
    }
