"""LoRA Trainer — stub implementation for SDXL LoRA training.

The full training pipeline requires a GPU with ~10GB VRAM, ai-toolkit (or
kohya_ss), and a prepared dataset. This stub validates inputs and returns
an informative error when training infrastructure is unavailable.

When GPU + ai-toolkit are available, replace _run_training() with the
real training invocation.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from backend.services.lora.types import (
    DatasetResult, LoRATrainingConfig, TrainingResult,
)

logger = logging.getLogger("beat_studio.lora.trainer")

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class LoRATrainer:
    """Train SDXL LoRAs from captioned image datasets.

    Training defaults (from BeatCanvas character LoRA protocol):
    - 25+ images, 1500 steps, rank 16, adamw8bit, resolution 1024

    Usage::

        trainer = LoRATrainer()
        dataset = trainer.prepare_dataset("/path/to/images")
        if dataset.success:
            result = trainer.train(config)
    """

    def train(self, config: LoRATrainingConfig) -> TrainingResult:
        """Train a LoRA. Returns stub result when GPU is unavailable.

        Args:
            config: LoRATrainingConfig with all training parameters.

        Returns:
            TrainingResult. success=False if training infrastructure absent.
        """
        # Validate dataset exists
        dataset_path = Path(config.dataset_path)
        if not dataset_path.exists():
            return TrainingResult(
                success=False,
                lora_path="",
                error=f"Dataset path does not exist: {dataset_path}",
            )

        # Check for GPU and training toolkit
        try:
            available, reason = self._check_training_available()
        except Exception as exc:
            available, reason = False, str(exc)

        if not available:
            return TrainingResult(
                success=False,
                lora_path="",
                error=(
                    f"Training infrastructure unavailable: {reason}. "
                    "Requires GPU + ai-toolkit or kohya_ss."
                ),
            )

        return self._run_training(config)

    def prepare_dataset(
        self,
        images_dir: str,
        captions: Optional[Dict[str, str]] = None,
        auto_caption: bool = False,
    ) -> DatasetResult:
        """Validate and prepare a training dataset.

        Args:
            images_dir: Directory containing images.
            captions: Optional dict of filename → caption text.
            auto_caption: If True, auto-generate captions via BLIP-2 (requires GPU).

        Returns:
            DatasetResult with image count and any warnings.
        """
        dir_path = Path(images_dir)
        if not dir_path.exists():
            return DatasetResult(
                success=False,
                image_count=0,
                caption_count=0,
                dataset_path=images_dir,
                error=f"Directory does not exist: {images_dir}",
            )

        image_files = [
            f for f in dir_path.iterdir()
            if f.suffix.lower() in _SUPPORTED_IMAGE_EXTS
        ]
        image_count = len(image_files)

        if image_count == 0:
            return DatasetResult(
                success=False,
                image_count=0,
                caption_count=0,
                dataset_path=images_dir,
                error="No supported image files found in directory",
            )

        warnings: list[str] = []
        if image_count < 15:
            warnings.append(
                f"Only {image_count} images found. "
                "Recommend 20-30+ for good quality LoRA training."
            )

        # Count caption files (.txt sidecar or passed in dict)
        caption_files = [f for f in dir_path.iterdir() if f.suffix == ".txt"]
        caption_count = len(captions) if captions else len(caption_files)

        if caption_count == 0 and not auto_caption:
            warnings.append(
                "No captions provided. Training without captions may reduce quality. "
                "Pass captions dict or set auto_caption=True."
            )

        return DatasetResult(
            success=True,
            image_count=image_count,
            caption_count=caption_count,
            dataset_path=images_dir,
            warnings=warnings,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _check_training_available(self) -> tuple[bool, str]:
        """Check if GPU and training toolkit are available."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA not available"
        except ImportError:
            return False, "PyTorch not installed"

        # Check for ai-toolkit or kohya_ss
        for toolkit in ("ai_toolkit", "kohya_ss", "diffusers"):
            try:
                __import__(toolkit)
                return True, "ok"
            except ImportError:
                continue

        return False, "No LoRA training toolkit found (ai-toolkit or kohya_ss required)"

    def _run_training(self, config: LoRATrainingConfig) -> TrainingResult:
        """Run the actual LoRA training. Override or extend for real implementation."""
        logger.info(
            "Starting LoRA training: %s (%s steps, rank %d)",
            config.output_name, config.training_steps, config.rank,
        )
        # Placeholder — real implementation would invoke ai-toolkit CLI or Python API
        return TrainingResult(
            success=False,
            lora_path="",
            error="Real training not yet implemented. Override _run_training().",
        )
