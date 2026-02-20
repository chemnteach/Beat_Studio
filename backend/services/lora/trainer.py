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
        """Run SDXL LoRA training using diffusers + peft.

        Pipeline:
        1. Load full SDXL pipeline (gives VAE + both text encoders + UNet)
        2. Apply peft LoraConfig to UNet attention layers
        3. Pre-compute text embeddings for trigger token
        4. Noise-prediction training loop with DDPMScheduler + AdamW
        5. Save LoRA weights as safetensors
        """
        import gc
        import random
        from pathlib import Path

        logger.info(
            "Starting LoRA training: %s (%d steps, rank %d)",
            config.output_name, config.training_steps, config.rank,
        )

        try:
            import torch
            from diffusers import DDPMScheduler, StableDiffusionXLPipeline
            from peft import LoraConfig, get_peft_model
            from PIL import Image
            import torchvision.transforms as T
        except ImportError as exc:
            return TrainingResult(success=False, lora_path="", error=f"Missing package: {exc}")

        output_dir = Path("output/loras") / config.output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        safetensors_path = output_dir / f"{config.output_name}.safetensors"

        pipe = None
        try:
            # Load full SDXL — gives us VAE + both text encoders + UNet in one shot
            logger.info("Loading SDXL pipeline for LoRA training…")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to("cuda")

            # DDPMScheduler for noise-prediction training
            noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

            # Apply peft LoRA to UNet attention layers
            lora_cfg = LoraConfig(
                r=config.rank,
                lora_alpha=config.rank,
                target_modules=[
                    "to_k", "to_q", "to_v", "to_out.0",
                    "add_k_proj", "add_v_proj",
                ],
                lora_dropout=0.0,
                bias="none",
            )
            pipe.unet = get_peft_model(pipe.unet, lora_cfg)
            pipe.unet.train()

            # Freeze everything except the LoRA adapter weights
            pipe.vae.requires_grad_(False)
            pipe.text_encoder.requires_grad_(False)
            pipe.text_encoder_2.requires_grad_(False)

            # ── Dataset ───────────────────────────────────────────────────────
            dataset_path = Path(config.dataset_path)
            image_files = [
                f for f in dataset_path.iterdir()
                if f.suffix.lower() in _SUPPORTED_IMAGE_EXTS
            ]
            if not image_files:
                return TrainingResult(
                    success=False, lora_path="",
                    error=f"No images found in {dataset_path}",
                )

            transform = T.Compose([
                T.Resize(
                    (config.resolution, config.resolution),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.CenterCrop(config.resolution),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ])
            img_tensors = []
            for f in image_files:
                try:
                    img_tensors.append(transform(Image.open(f).convert("RGB")))
                except Exception as img_exc:
                    logger.warning("Skipped %s: %s", f.name, img_exc)

            if not img_tensors:
                return TrainingResult(
                    success=False, lora_path="",
                    error="No valid images after loading",
                )

            # ── Pre-compute trigger-token text embeddings (SDXL: 2 encoders) ─
            trigger = config.trigger_token
            with torch.no_grad():
                tok1 = pipe.tokenizer(
                    trigger, return_tensors="pt", padding="max_length",
                    max_length=pipe.tokenizer.model_max_length, truncation=True,
                )
                tok2 = pipe.tokenizer_2(
                    trigger, return_tensors="pt", padding="max_length",
                    max_length=pipe.tokenizer_2.model_max_length, truncation=True,
                )
                enc1 = pipe.text_encoder(
                    tok1.input_ids.to("cuda"), output_hidden_states=True,
                )
                enc2 = pipe.text_encoder_2(
                    tok2.input_ids.to("cuda"), output_hidden_states=True,
                )
                # SDXL concatenates both encoder hidden states along the channel dim
                encoder_hidden_states = torch.cat(
                    [enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1
                )  # (1, 77, 2048)
                pooled_embeds = enc2[0]  # (1, 1280)
                add_time_ids = torch.tensor(
                    [[
                        config.resolution, config.resolution,
                        0, 0,
                        config.resolution, config.resolution,
                    ]],
                    dtype=torch.float16,
                    device="cuda",
                )
                added_cond_kwargs = {
                    "text_embeds": pooled_embeds,
                    "time_ids": add_time_ids,
                }

            # ── Training loop ─────────────────────────────────────────────────
            optimizer = torch.optim.AdamW(
                [p for p in pipe.unet.parameters() if p.requires_grad],
                lr=config.learning_rate,
            )

            logger.info(
                "Training for %d steps on %d images",
                config.training_steps, len(img_tensors),
            )
            for step in range(config.training_steps):
                img_t = random.choice(img_tensors).unsqueeze(0).to("cuda", dtype=torch.float16)

                with torch.no_grad():
                    latents = (
                        pipe.vae.encode(img_t).latent_dist.sample()
                        * pipe.vae.config.scaling_factor
                    )

                noise = torch.randn_like(latents)
                ts = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (1,), device="cuda",
                ).long()
                noisy = noise_scheduler.add_noise(latents, noise, ts)

                pred = pipe.unet(
                    noisy, ts,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                ).sample

                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 100 == 0:
                    logger.info(
                        "Step %d/%d  loss=%.4f",
                        step, config.training_steps, loss.item(),
                    )

            # ── Save ─────────────────────────────────────────────────────────
            pipe.unet.save_pretrained(str(output_dir))

            try:
                from safetensors.torch import save_file
                lora_weights = {
                    k: v.cpu().contiguous()
                    for k, v in pipe.unet.state_dict().items()
                    if "lora" in k.lower()
                }
                if lora_weights:
                    save_file(lora_weights, str(safetensors_path))
                else:
                    safetensors_path = output_dir / "adapter_model.safetensors"
            except Exception as save_exc:
                logger.warning("safetensors save failed, using peft dir: %s", save_exc)
                safetensors_path = output_dir / "adapter_model.safetensors"

            logger.info("LoRA training complete: %s", safetensors_path)
            return TrainingResult(success=True, lora_path=str(safetensors_path))

        except Exception as exc:
            logger.exception("LoRA training failed: %s", exc)
            return TrainingResult(success=False, lora_path="", error=str(exc))
        finally:
            if pipe is not None:
                del pipe
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
