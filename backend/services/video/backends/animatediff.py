"""AnimateDiff-Lightning backend — fast local animated video generation."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

from backend.services.prompt.types import ComposedPrompt
from backend.services.shared.vram_manager import VRAMManager
from backend.services.video.backends.base import VideoBackend
from backend.services.video.types import VideoClip

logger = logging.getLogger("beat_studio.video.animatediff")

# Module-level VRAMManager — shared across all AnimateDiff instances in the process
_vram_manager = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)

# Styles this backend handles well
_SUPPORTED_STYLES = {
    "cinematic", "anime", "lofi", "abstract", "cel_animation", "watercolor",
    "ink_wash", "pencil_sketch", "motion_graphics", "collage_mixed_media",
    "pixel_art", "low_poly_3d", "isometric", "oil_painting", "comic_book",
    "pop_art", "ukiyo_e", "synthwave", "graffiti", "art_deco", "impressionist",
    "psychedelic",
}

_DEFAULT_MODEL_PATH = "backend/models/animatediff/mm_sd_v15_v2.ckpt"
_VRAM_GB = 5.6

# AnimateDiff (SD1.5-based) limits — exceed these and VRAM blows on a 12 GB card.
# Temporal attention is O(n_frames²): 720 frames @ 24fps → 15+ GB intermediate tensors.
_MAX_FRAMES = 16   # Lightning 4-step is optimised for exactly 16 frames
_NATIVE_WIDTH  = 512   # SD1.5 native resolution
_NATIVE_HEIGHT = 512


class AnimateDiffBackend(VideoBackend):
    """AnimateDiff-Lightning local backend.

    Best for: Fast animated clips, stylized content (16 frames per generation).
    VRAM: ~5.6GB
    Cost: Free (local GPU)

    Reference: BeatCanvas animatediff_generator.py + animatediff_pipeline.py
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = Path(model_path or _DEFAULT_MODEL_PATH)
        self._current_checkpoint: str = ""

    # ── VideoBackend interface ─────────────────────────────────────────────────

    def name(self) -> str:
        return "animatediff"

    def vram_required_gb(self) -> float:
        return _VRAM_GB

    def supports_style(self, style: str) -> bool:
        return style in _SUPPORTED_STYLES

    def is_available(self) -> bool:
        """Check if AnimateDiff adapter is available locally or in HF hub cache."""
        # Legacy: check explicit local .ckpt path
        if self._model_path.exists():
            return True
        try:
            from huggingface_hub import try_to_load_from_cache
            # Standard guoyww motion adapter
            if try_to_load_from_cache(
                "guoyww/animatediff-motion-adapter-v1-5-2",
                filename="diffusion_pytorch_model.safetensors",
            ) is not None:
                return True
            # ByteDance AnimateDiff-Lightning (diffusers-format motion adapter)
            if try_to_load_from_cache(
                "ByteDance/AnimateDiff-Lightning",
                filename="animatediff_lightning_4step_diffusers.safetensors",
            ) is not None:
                return True
        except Exception:
            pass
        return False

    def estimated_time_per_scene(self, resolution: Tuple[int, int] = (576, 1024)) -> float:
        # ~30 seconds on RTX 3080 for 16 frames at 576×1024
        return 30.0

    def estimated_cost_per_scene(self) -> float:
        return 0.0  # local = free

    def generate_clip(
        self,
        prompt: ComposedPrompt,
        duration_sec: float,
        resolution: Tuple[int, int],
        fps: int = 24,
        seed: int = -1,
    ) -> VideoClip:
        """Generate a video clip using AnimateDiff-Lightning."""
        t0 = time.time()
        out_path = self._run_pipeline(prompt, duration_sec, resolution, fps, seed)
        elapsed = time.time() - t0

        # Actual clip is _MAX_FRAMES long regardless of requested scene duration.
        # The assembler handles looping/stretching to fill the intended scene slot.
        actual_duration = min(max(8, int(duration_sec * fps)), _MAX_FRAMES) / fps
        return VideoClip(
            file_path=out_path,
            duration_sec=actual_duration,
            width=_NATIVE_WIDTH,
            height=_NATIVE_HEIGHT,
            fps=fps,
            scene_index=-1,
            backend_used=self.name(),
            generation_time_sec=elapsed,
            cost_usd=0.0,
        )

    def generate_batch(
        self,
        prompts: List[ComposedPrompt],
        durations: List[float],
        resolution: Tuple[int, int],
        fps: int = 24,
    ) -> List[VideoClip]:
        """Generate clips sequentially (AnimateDiff does not support true batching)."""
        return [
            self.generate_clip(p, d, resolution, fps)
            for p, d in zip(prompts, durations)
        ]

    def kill(self) -> None:
        """Release GPU resources via VRAMManager."""
        _vram_manager.kill()
        self._current_checkpoint = ""
        logger.debug("AnimateDiff pipeline killed")

    # ── internal (mockable in tests) ──────────────────────────────────────────

    @staticmethod
    def _load_motion_adapter(torch):
        """Load the best available AnimateDiff motion adapter.

        Prefers ByteDance AnimateDiff-Lightning (4-step, already cached) over
        the standard guoyww v1-5-2 adapter. Falls back to guoyww (will download).
        """
        from diffusers import MotionAdapter as _MotionAdapter
        from huggingface_hub import try_to_load_from_cache

        lightning_path = try_to_load_from_cache(
            "ByteDance/AnimateDiff-Lightning",
            filename="animatediff_lightning_4step_diffusers.safetensors",
        )
        if lightning_path:
            from safetensors.torch import load_file
            logger.info("Using ByteDance AnimateDiff-Lightning adapter")
            adapter = _MotionAdapter()
            adapter.load_state_dict(load_file(lightning_path))
            return adapter.to(dtype=torch.float16)

        logger.info("Using guoyww animatediff-motion-adapter-v1-5-2")
        return _MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype=torch.float16,
        )

    def _run_pipeline(
        self,
        prompt: ComposedPrompt,
        duration_sec: float,
        resolution: Tuple[int, int],
        fps: int,
        seed: int,
    ) -> str:
        """Run the AnimateDiff pipeline and return path to the generated clip."""
        import uuid

        import torch
        from diffusers import AnimateDiffPipeline, EulerDiscreteScheduler
        from diffusers.utils import export_to_video

        out_dir = Path("output/video/clips")
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_path = str(out_dir / f"clip_{uuid.uuid4().hex[:8]}.mp4")

        # Cap frames BEFORE touching VRAM — temporal attention is O(n_frames²).
        # 30s × 24fps = 720 frames → 15+ GB intermediates even at fp16. Hard cap at 16.
        num_frames = min(max(8, int(duration_sec * fps)), _MAX_FRAMES)

        checkpoint = prompt.base_checkpoint or "emilianJR/epiCRealism"

        if self._current_checkpoint != checkpoint:
            # ── Kill BEFORE loading — keeps old + new from coexisting in VRAM ──
            _vram_manager.kill()
            logger.info(
                "Loading AnimateDiff pipeline: checkpoint='%s', fp16, %dx%d, max %d frames",
                checkpoint, _NATIVE_WIDTH, _NATIVE_HEIGHT, _MAX_FRAMES,
            )
            adapter = self._load_motion_adapter(torch)
            pipe = AnimateDiffPipeline.from_pretrained(
                checkpoint,
                motion_adapter=adapter,
                torch_dtype=torch.float16,
            )
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                beta_schedule="linear",
                clip_sample=False,
                timestep_spacing="linspace",
                steps_offset=1,
            )
            pipe.enable_vae_slicing()
            pipe.enable_attention_slicing()
            pipe.to("cuda")
            # set_pipeline won't kill again because we already called kill() above
            _vram_manager.set_pipeline(pipe, f"animatediff/{checkpoint}")
            self._current_checkpoint = checkpoint

        pipeline = _vram_manager._current_pipeline  # noqa: SLF001

        # ── LoRA loading ──────────────────────────────────────────────────────
        # Always start clean so previous clip's LoRAs don't bleed into this one.
        lora_configs = getattr(prompt, "lora_configs", []) or []
        loras_loaded = False
        if lora_configs:
            try:
                adapter_names = []
                adapter_weights = []
                for lc in lora_configs:
                    lora_file = Path(lc.file_path) if lc.file_path else None
                    if not lora_file or not lora_file.exists():
                        logger.warning("LoRA file not found for '%s' (%s), skipping", lc.name, lc.file_path)
                        continue
                    adapter_name = lc.name.replace("-", "_")
                    pipeline.load_lora_weights(
                        str(lora_file.parent),
                        weight_name=lora_file.name,
                        adapter_name=adapter_name,
                    )
                    adapter_names.append(adapter_name)
                    adapter_weights.append(float(lc.weight))
                    logger.info("Loaded LoRA '%s' (weight=%.2f) from %s", lc.name, lc.weight, lora_file)

                if adapter_names:
                    pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
                    loras_loaded = True
                    logger.info("Active LoRA adapters: %s", adapter_names)
            except Exception as exc:
                logger.warning("LoRA loading failed (%s) — generating without LoRAs", exc)
                try:
                    pipeline.unload_lora_weights()
                except Exception:
                    pass

        # ── Inference ─────────────────────────────────────────────────────────
        effective_seed = seed if seed >= 0 else 42
        try:
            output = pipeline(
                prompt=prompt.positive,
                negative_prompt=prompt.negative,
                num_frames=num_frames,
                width=_NATIVE_WIDTH,
                height=_NATIVE_HEIGHT,
                guidance_scale=prompt.cfg_scale,
                num_inference_steps=prompt.steps,
                generator=torch.Generator("cuda").manual_seed(effective_seed),
            )
        finally:
            # ── LoRA unload — always clean up even if inference fails ─────────
            if loras_loaded:
                try:
                    pipeline.unload_lora_weights()
                    logger.debug("LoRA weights unloaded after clip")
                except Exception as exc:
                    logger.warning("LoRA unload failed: %s", exc)

        export_to_video(output.frames[0], clip_path, fps=fps)
        logger.info(
            "AnimateDiff clip: %s (%d frames @ %dfps = %.2fs)",
            clip_path, num_frames, fps, num_frames / fps,
        )
        return clip_path
