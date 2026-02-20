"""AnimateDiff-Lightning backend — fast local animated video generation."""
from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from backend.services.prompt.types import ComposedPrompt
from backend.services.video.backends.base import VideoBackend
from backend.services.video.types import VideoClip

logger = logging.getLogger("beat_studio.video.animatediff")

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


class AnimateDiffBackend(VideoBackend):
    """AnimateDiff-Lightning local backend.

    Best for: Fast animated clips, stylized content (16 frames per generation).
    VRAM: ~5.6GB
    Cost: Free (local GPU)

    Reference: BeatCanvas animatediff_generator.py + animatediff_pipeline.py
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = Path(model_path or _DEFAULT_MODEL_PATH)
        self._pipeline = None

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
        # Check HuggingFace hub cache for the HF-hosted adapter
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(
                "guoyww/animatediff-motion-adapter-v1-5-2",
                filename="diffusion_pytorch_model.safetensors",
            )
            return cached is not None
        except Exception:
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

        return VideoClip(
            file_path=out_path,
            duration_sec=duration_sec,
            width=resolution[0],
            height=resolution[1],
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
        """Release GPU resources."""
        if self._pipeline is not None:
            try:
                if hasattr(self._pipeline, "unload_lora_weights"):
                    self._pipeline.unload_lora_weights()
            except Exception:
                pass
            del self._pipeline
            self._pipeline = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass
        logger.debug("AnimateDiff pipeline killed")

    # ── internal (mockable in tests) ──────────────────────────────────────────

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
        from pathlib import Path

        import torch
        from diffusers import AnimateDiffPipeline, EulerDiscreteScheduler, MotionAdapter
        from diffusers.utils import export_to_video

        out_dir = Path("output/video/clips")
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_path = str(out_dir / f"clip_{uuid.uuid4().hex[:8]}.mp4")

        num_frames = max(8, int(duration_sec * fps))

        if self._pipeline is None:
            logger.info("Loading AnimateDiff pipeline…")
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2",
                torch_dtype=torch.float16,
            )
            pipe = AnimateDiffPipeline.from_pretrained(
                "emilianJR/epiCRealism",
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
            pipe.to("cuda")
            self._pipeline = pipe

        effective_seed = seed if seed >= 0 else 42
        output = self._pipeline(
            prompt=prompt.positive,
            negative_prompt=prompt.negative,
            num_frames=num_frames,
            guidance_scale=prompt.cfg_scale,
            num_inference_steps=prompt.steps,
            generator=torch.Generator("cuda").manual_seed(effective_seed),
        )

        export_to_video(output.frames[0], clip_path, fps=fps)
        logger.debug("AnimateDiff clip: %s (%d frames)", clip_path, num_frames)
        return clip_path
