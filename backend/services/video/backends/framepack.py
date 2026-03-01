"""FramePack image-to-video backend (lllyasviel/FramePack).

Uses the official diffusers HunyuanVideoFramepackPipeline with fp8 quantization
and model CPU offload to fit within 12 GB VRAM.

⚠  System RAM: this backend requires ~30 GB CPU RAM (full model resident when
   offloaded to CPU). On a 32 GB system, kill all non-essential processes before
   generation. If OOM, the fallback is the Wan2.1 1.3B I2V variant.

Input:  ComposedPrompt (prompt.init_image_path = storyboard keyframe PNG)
Output: VideoClip pointing to an MP4 in output/video/clips/

SDXL LoRAs are NOT compatible with HunyuanVideo architecture.  Apply them at
the storyboard stage (SDXL SDXL keyframe generation) — not here.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from backend.services.prompt.types import ComposedPrompt
from backend.services.shared.vram_manager import VRAMManager
from backend.services.video.backends.base import VideoBackend
from backend.services.video.types import VideoClip

logger = logging.getLogger("beat_studio.video.framepack")

# ── Module-level singletons ────────────────────────────────────────────────────
_vram_manager = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)

# ── Model repos ───────────────────────────────────────────────────────────────
# F1 is the newer variant: better motion, same size, faster sampling.
# Swap to FramePackI2V_HY for the original if F1 gives drift artefacts.
_TRANSFORMER_REPO = "lllyasviel/FramePack_F1_I2V_HY_20250503"
_BASE_REPO = "hunyuanvideo-community/HunyuanVideo"

# ── Generation constants ───────────────────────────────────────────────────────
_VRAM_GB = 10.0         # realistic peak: fp8 + cpu_offload on 12 GB card
_NATIVE_WIDTH = 848     # 16:9 at HunyuanVideo native scale (848×480)
_NATIVE_HEIGHT = 480
_MAX_FRAMES = 129       # 4×32+1; ~5.4 sec @ 24fps — hard cap for 12 GB
_DEFAULT_STEPS = 25     # 25 steps is the sweet spot for F1 quality/speed

_SUPPORTED_STYLES = {
    "cinematic", "music_video", "photorealistic", "documentary",
    "abstract", "minimal", "retro", "synthwave",
}

# ── Deferred heavy imports (re-exported so tests can patch at module level) ───
# These are populated on first _load_pipeline() call.
HunyuanVideoFramepackPipeline = None  # type: ignore[assignment]
HunyuanVideoFramepackTransformer3DModel = None  # type: ignore[assignment]
Image = None  # type: ignore[assignment]
quantize_ = None  # type: ignore[assignment]
float8_weight_only = None  # type: ignore[assignment]


def _hf_cached(repo_id: str, filename: str) -> Optional[str]:
    """Thin wrapper around huggingface_hub.try_to_load_from_cache (patchable)."""
    from huggingface_hub import try_to_load_from_cache
    return try_to_load_from_cache(repo_id, filename)


def _import_heavy() -> None:
    """Import torch/diffusers/torchao/PIL into module globals on first use."""
    global HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
    global Image, quantize_, float8_weight_only

    if HunyuanVideoFramepackPipeline is None:
        from diffusers import (  # type: ignore[assignment]
            HunyuanVideoFramepackPipeline as _P,
            HunyuanVideoFramepackTransformer3DModel as _T,
        )
        from PIL import Image as _Img  # type: ignore[assignment]
        HunyuanVideoFramepackPipeline = _P
        HunyuanVideoFramepackTransformer3DModel = _T
        Image = _Img

        try:
            from torchao.quantization import (  # type: ignore[assignment]
                float8_weight_only as _f8,
                quantize_ as _q,
            )
            quantize_ = _q
            float8_weight_only = _f8
        except ImportError:
            logger.warning(
                "torchao not available — FramePack will use bf16 "
                "(higher VRAM, may OOM on 12 GB; install torchao for fp8)"
            )


class FramePackBackend(VideoBackend):
    """Image-to-video backend using lllyasviel/FramePack (HunyuanVideo family).

    Best for: Animating storyboard keyframes into smooth video clips.
    VRAM:     ~8–10 GB (fp8 + cpu_offload)
    RAM:      ~30 GB system RAM required
    Cost:     Free (local GPU)

    The conditioning image comes from prompt.init_image_path — set this to the
    storyboard-approved SDXL keyframe path before calling generate_clip().
    If init_image_path is empty, a black frame is used (lower quality).
    """

    def __init__(self, transformer_repo: str = _TRANSFORMER_REPO):
        self._transformer_repo = transformer_repo
        self._current_model: str = ""  # tracks which repo is loaded in VRAM

    # ── VideoBackend interface ─────────────────────────────────────────────────

    def name(self) -> str:
        return "framepack"

    def vram_required_gb(self) -> float:
        return _VRAM_GB

    def supports_style(self, style: str) -> bool:
        return style.lower() in _SUPPORTED_STYLES

    def is_available(self) -> bool:
        """True if the transformer weights are in the HuggingFace hub cache."""
        try:
            result = _hf_cached(self._transformer_repo, "model.safetensors.index.json")
            return result is not None and result != ""
        except Exception:
            return False

    def estimated_time_per_scene(self, resolution: Tuple[int, int] = (848, 480)) -> float:
        # ~3 min for 69 frames @ 25 steps with fp8+cpu_offload on 12 GB
        return 180.0

    def estimated_cost_per_scene(self) -> float:
        return 0.0

    def generate_clip(
        self,
        prompt: ComposedPrompt,
        duration_sec: float,
        resolution: Tuple[int, int],
        fps: int = 24,
        seed: int = -1,
    ) -> VideoClip:
        """Animate a storyboard keyframe into a video clip."""
        t0 = time.time()
        init_image_path = getattr(prompt, "init_image_path", "") or ""
        out_path = self._run_pipeline(prompt, duration_sec, fps, seed, init_image_path)
        elapsed = time.time() - t0

        raw_frames = max(9, int(duration_sec * fps))
        num_frames = self._snap_frames(raw_frames)
        actual_duration = num_frames / fps

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
        return [
            self.generate_clip(p, d, resolution, fps)
            for p, d in zip(prompts, durations)
        ]

    def kill(self) -> None:
        """Release GPU resources — call before loading a different model."""
        _vram_manager.kill()
        self._current_model = ""
        logger.debug("FramePack pipeline killed")

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _snap_frames(raw: int) -> int:
        """Snap to FramePack's 4k+1 frame count constraint (min 9, max _MAX_FRAMES)."""
        raw = max(9, min(raw, _MAX_FRAMES))
        return ((raw - 1) // 4) * 4 + 1

    # ── Internal (split out for testability) ──────────────────────────────────

    def _open_image(self, path: str):  # → Optional[PIL.Image.Image]
        """Load image from path; return None if path is empty or missing."""
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            logger.warning("init_image_path '%s' not found — using black frame", path)
            return None
        _import_heavy()
        return Image.open(str(p)).convert("RGB")

    def _export_frames(self, frames, clip_path: str, fps: int) -> None:
        """Write frames list to MP4 (wrapper so tests can patch it)."""
        from diffusers.utils import export_to_video
        export_to_video(frames, clip_path, fps=fps)

    def _run_pipeline(
        self,
        prompt: ComposedPrompt,
        duration_sec: float,
        fps: int,
        seed: int,
        init_image_path: str,
    ) -> str:
        import torch

        _import_heavy()

        out_dir = Path("output/video/clips")
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_path = str(out_dir / f"clip_{uuid.uuid4().hex[:8]}.mp4")

        raw_frames = max(9, int(duration_sec * fps))
        num_frames = self._snap_frames(raw_frames)

        # ── Kill-before-load when switching models ──────────────────────────
        if self._current_model != self._transformer_repo:
            _vram_manager.kill()
            logger.info(
                "Loading FramePack pipeline: repo='%s', fp8=%s, %dx%d, max %d frames",
                self._transformer_repo,
                quantize_ is not None,
                _NATIVE_WIDTH, _NATIVE_HEIGHT, _MAX_FRAMES,
            )
            pipe = self._load_pipeline()
            _vram_manager.set_pipeline(pipe, f"framepack/{self._transformer_repo}")
            self._current_model = self._transformer_repo

        pipeline = _vram_manager._current_pipeline  # noqa: SLF001

        # ── Conditioning image ──────────────────────────────────────────────
        image = self._open_image(init_image_path)
        if image is None:
            if init_image_path:
                # Path given but file missing — already warned in _open_image
                pass
            else:
                logger.info("No init_image_path — using black frame (lower quality)")
            image = Image.new("RGB", (_NATIVE_WIDTH, _NATIVE_HEIGHT))

        # ── LoRA warning ────────────────────────────────────────────────────
        # Our LoRAs are SDXL-trained; HunyuanVideo has a different architecture.
        # Apply LoRAs at the storyboard stage (SDXL), not here.
        lora_configs = getattr(prompt, "lora_configs", []) or []
        if lora_configs:
            logger.warning(
                "FramePack backend: %d LoRA config(s) ignored — SDXL LoRAs are not "
                "compatible with HunyuanVideo architecture. Apply LoRAs at the "
                "storyboard (SDXL) stage instead.",
                len(lora_configs),
            )

        # ── Inference ───────────────────────────────────────────────────────
        effective_seed = seed if seed >= 0 else 42
        output = pipeline(
            image=image,
            prompt=prompt.positive,
            negative_prompt=prompt.negative,
            num_frames=num_frames,
            height=_NATIVE_HEIGHT,
            width=_NATIVE_WIDTH,
            num_inference_steps=getattr(prompt, "steps", _DEFAULT_STEPS),
            guidance_scale=getattr(prompt, "cfg_scale", 6.0),
            generator=torch.Generator("cuda").manual_seed(effective_seed),
        )

        self._export_frames(output.frames[0], clip_path, fps)
        logger.info(
            "FramePack clip: %s (%d frames @ %dfps = %.2fs)",
            clip_path, num_frames, fps, num_frames / fps,
        )
        return clip_path

    def _load_pipeline(self):
        """Load HunyuanVideoFramepackPipeline with fp8+cpu_offload.

        RAM warning: the full model (~30 GB) lives in CPU memory when offloaded.
        On a 32 GB system this leaves ~2 GB headroom — kill background processes
        before calling this.
        """
        import torch

        _import_heavy()

        logger.info("Loading FramePack transformer: %s (bfloat16)", self._transformer_repo)
        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
            self._transformer_repo,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        if quantize_ is not None:
            try:
                quantize_(transformer, float8_weight_only())
                logger.info("FramePack transformer quantized to fp8 (~25%% VRAM reduction)")
            except Exception as exc:
                logger.warning("fp8 quantization failed (%s) — continuing with bf16", exc)
        else:
            logger.warning(
                "Running FramePack in bf16 — peak VRAM ~12-14 GB (may OOM). "
                "Install torchao for fp8 support: pip install torchao"
            )

        pipe = HunyuanVideoFramepackPipeline.from_pretrained(
            _BASE_REPO,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        logger.info("FramePack pipeline ready (cpu_offload=True, vae_tiling=True)")
        return pipe
