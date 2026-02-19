"""GPU and system hardware detection for Beat Studio.

Provides VRAM-based capability checks and local/cloud execution recommendations
for each video generation model.
"""
from __future__ import annotations

import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

import psutil


# ── VRAM requirements per model (GB) ─────────────────────────────────────────

_VRAM_REQUIREMENTS: dict[str, float] = {
    "animatediff_lightning": 5.6,
    "wan26_local": 12.0,
    "wan26_cloud": 0.0,      # cloud — no local VRAM needed
    "skyreels_cloud": 0.0,
    "cogvideox": 14.0,
    "svd": 7.5,
    "sdxl_controlnet": 8.0,
    "sdxl_lora_training": 10.0,
    "demucs_stems": 8.0,
    "mochi": 20.0,
    "ltx_video": 10.0,
}

# ── Estimated seconds per 4-second scene (local) ─────────────────────────────
_SECONDS_PER_SCENE: dict[str, float] = {
    "animatediff_lightning": 16.0,
    "wan26_local": 90.0,
    "wan26_cloud": 60.0,
    "skyreels_cloud": 45.0,
    "cogvideox": 120.0,
    "svd": 60.0,
    "sdxl_controlnet": 45.0,
}

# ── Cloud fallback mapping ────────────────────────────────────────────────────
_CLOUD_FALLBACK: dict[str, str] = {
    "wan26_local": "wan26_cloud",
    "cogvideox": "wan26_cloud",
    "sdxl_controlnet": "wan26_cloud",
    "svd": "wan26_cloud",
    "animatediff_lightning": "wan26_cloud",
}


@dataclass
class ExecutionRecommendation:
    backend: str
    use_cloud: bool
    estimated_seconds_per_scene: float
    estimated_cost_usd: float
    reason: str


@dataclass
class HardwareProfile:
    gpu_name: str
    vram_total_gb: float
    vram_available_gb: float
    cuda_available: bool
    cuda_version: str
    cpu_cores: int
    ram_gb: float

    # ── class method ─────────────────────────────────────────────────────────

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Detect current system hardware."""
        cuda_available = False
        cuda_version = ""
        gpu_name = "None"
        vram_total = 0.0
        vram_available = 0.0

        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda or ""
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_total = props.total_memory / (1024 ** 3)
                vram_available = (
                    props.total_memory - torch.cuda.memory_allocated(0)
                ) / (1024 ** 3)
        except ImportError:
            pass

        cpu_cores = multiprocessing.cpu_count()
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)

        return cls(
            gpu_name=gpu_name,
            vram_total_gb=round(vram_total, 2),
            vram_available_gb=round(vram_available, 2),
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            cpu_cores=cpu_cores,
            ram_gb=round(ram_gb, 2),
        )

    # ── capability checks ────────────────────────────────────────────────────

    def can_run_local(self, model: str) -> bool:
        """Return True if this GPU has enough VRAM for the given model."""
        if not self.cuda_available:
            return False
        required = _VRAM_REQUIREMENTS.get(model)
        if required is None:
            return False
        if required == 0.0:
            # Cloud model — doesn't need local VRAM
            return False
        return self.vram_total_gb >= required

    def recommend_execution(self, model: str) -> ExecutionRecommendation:
        """Return local or cloud recommendation for a given model."""
        if self.can_run_local(model):
            return ExecutionRecommendation(
                backend=model,
                use_cloud=False,
                estimated_seconds_per_scene=_SECONDS_PER_SCENE.get(model, 60.0),
                estimated_cost_usd=0.0,
                reason=f"Local GPU ({self.gpu_name}, {self.vram_total_gb:.1f} GB) is sufficient.",
            )

        cloud = _CLOUD_FALLBACK.get(model, "wan26_cloud")
        return ExecutionRecommendation(
            backend=cloud,
            use_cloud=True,
            estimated_seconds_per_scene=_SECONDS_PER_SCENE.get(cloud, 60.0),
            estimated_cost_usd=0.10,  # rough estimate per scene
            reason=(
                f"Local GPU insufficient for {model} "
                f"(requires {_VRAM_REQUIREMENTS.get(model, '?')} GB, "
                f"available {self.vram_total_gb:.1f} GB). "
                f"Routing to cloud: {cloud}."
            ),
        )
