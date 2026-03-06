"""Model loading and video generation for Beat Studio RunPod worker.

Each model is loaded on first use and cached in _loaded_models. Only one model
is kept in VRAM at a time — if a different model is requested the current one
is freed first.

All models write to the network volume HF cache (HF_HOME=/runpod-volume/hf_cache).
Pre-download the transformer weights with the setup script before deploying.

Model IDs (HuggingFace):
  framepack:       lllyasviel/FramePackI2V_HY  (+ hunyuanvideo-community/HunyuanVideo base)
  skyreels_v2_i2v: Skywork/SkyReels-V2-I2V-14B-720P
  skyreels_v2_df:  Skywork/SkyReels-V2-DF-14B-720P-Diffusers
  wan22_i2v:       Wan-AI/Wan2.2-I2V-14B-720P
  skyreels_v3_r2v: Skywork/SkyReels-V3-R2V-14B
"""
from __future__ import annotations

import gc
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

import imageio
import torch
from PIL import Image

logger = logging.getLogger("beat_studio.worker.models")

# ── Constants ────────────────────────────────────────────────────────────────

_MODELS_ROOT     = Path("/workspace/models")
_FRAMEPACK_TRANSFORMER = str(_MODELS_ROOT / "FramePackI2V_HY")
_FRAMEPACK_BASE        = "hunyuanvideo-community/HunyuanVideo"  # no local copy, keep HF
_FRAMEPACK_SIGLIP      = "lllyasviel/flux_redux_bfl"            # no local copy, keep HF
_SKYREELS_V2_I2V = str(_MODELS_ROOT / "SkyReels-V2-I2V-14B-720P")
_SKYREELS_V2_DF  = str(_MODELS_ROOT / "SkyReels-V2-DF-14B-720P-Diffusers")
_WAN22_I2V       = str(_MODELS_ROOT / "Wan2.2-I2V-A14B")        # fixed: was wrong ID
_SKYREELS_V3_R2V = str(_MODELS_ROOT / "SkyReels-V3-R2V-14B")
_SKYREELS_V3_SCRIPT = str(_MODELS_ROOT / "SkyReels-V3-R2V-14B" / "generate_video.py")

SUPPORTED_MODELS = {
    "framepack",
    "skyreels_v2_i2v",
    "skyreels_v2_df",
    "wan22_i2v",
    "skyreels_v3_r2v",
}

# Native fps per model
_MODEL_FPS = {
    "framepack":       30,
    "skyreels_v2_i2v": 24,
    "skyreels_v2_df":  24,
    "wan22_i2v":       24,
    "skyreels_v3_r2v": 24,
}

# ── Model cache ──────────────────────────────────────────────────────────────

_loaded_models: dict = {}   # {model_name: pipeline_object}
_current_model: str = ""    # which model is currently in VRAM


def _free_current() -> None:
    """Unload whichever model is in VRAM."""
    global _loaded_models, _current_model
    if _current_model and _current_model in _loaded_models:
        logger.info("Freeing model: %s", _current_model)
        pipe = _loaded_models.pop(_current_model)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        _current_model = ""


def _snap_framepack_frames(raw: int) -> int:
    """FramePack requires 4k+1 frame count (min 9)."""
    raw = max(9, min(raw, 257))
    return ((raw - 1) // 4) * 4 + 1


# ── Per-model loaders ────────────────────────────────────────────────────────

def _load_framepack():
    from diffusers import (
        HunyuanVideoFramepackPipeline,
        HunyuanVideoFramepackTransformer3DModel,
    )
    from transformers import SiglipImageProcessor, SiglipVisionModel

    logger.info("Loading FramePack transformer from %s", _FRAMEPACK_TRANSFORMER)
    transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
        _FRAMEPACK_TRANSFORMER, torch_dtype=torch.bfloat16,
    )
    feature_extractor = SiglipImageProcessor.from_pretrained(
        _FRAMEPACK_SIGLIP, subfolder="feature_extractor",
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        _FRAMEPACK_SIGLIP, subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        _FRAMEPACK_BASE,
        transformer=transformer,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def _load_skyreels_v2_i2v():
    # SkyReels-V2 I2V uses a Wan-based architecture.
    # Verify the exact diffusers class name from:
    #   https://huggingface.co/Skywork/SkyReels-V2-I2V-14B-720P
    # If WanImageToVideoPipeline is not available in your diffusers version,
    # check the model card for the correct import.
    from diffusers import WanImageToVideoPipeline

    logger.info("Loading SkyReels-V2 I2V from %s", _SKYREELS_V2_I2V)
    pipe = WanImageToVideoPipeline.from_pretrained(
        _SKYREELS_V2_I2V, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def _load_skyreels_v2_df():
    from diffusers import (
        AutoModel,
        SkyReelsV2DiffusionForcingPipeline,
        UniPCMultistepScheduler,
    )

    logger.info("Loading SkyReels-V2 DF from %s", _SKYREELS_V2_DF)
    vae = AutoModel.from_pretrained(
        _SKYREELS_V2_DF, subfolder="vae", torch_dtype=torch.float32,
    )
    pipe = SkyReelsV2DiffusionForcingPipeline.from_pretrained(
        _SKYREELS_V2_DF, vae=vae, torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=5.0,  # 5.0 for I2V mode
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def _load_wan22_i2v():
    # Wan 2.2 I2V — verify exact pipeline class at:
    #   https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-720P
    # WanImageToVideoPipeline is the expected class for Wan I2V diffusers models.
    from diffusers import WanImageToVideoPipeline

    logger.info("Loading Wan 2.2 I2V from %s", _WAN22_I2V)
    pipe = WanImageToVideoPipeline.from_pretrained(
        _WAN22_I2V, torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def _load_skyreels_v3_r2v() -> dict:
    """V3-R2V uses a custom inference script, not a standard diffusers pipeline.

    Returns a sentinel dict instead of a pipeline object. Actual generation
    is handled by _gen_skyreels_v3_r2v via subprocess.
    """
    script = Path(_SKYREELS_V3_SCRIPT)
    if not script.exists():
        raise FileNotFoundError(
            f"SkyReels-V3 inference script not found at {script}. "
            "Clone https://github.com/SkyworkAI/SkyReels-V3 into the model directory "
            "or copy generate_video.py there manually."
        )
    return {"type": "skyreels_v3_r2v", "script": str(script), "model_id": _SKYREELS_V3_R2V}


_LOADERS = {
    "framepack":       _load_framepack,
    "skyreels_v2_i2v": _load_skyreels_v2_i2v,
    "skyreels_v2_df":  _load_skyreels_v2_df,
    "wan22_i2v":       _load_wan22_i2v,
    "skyreels_v3_r2v": _load_skyreels_v3_r2v,
}


def _get_pipeline(model_name: str):
    global _loaded_models, _current_model
    if _current_model == model_name:
        return _loaded_models[model_name]
    _free_current()
    logger.info("Loading model: %s", model_name)
    pipe = _LOADERS[model_name]()
    _loaded_models[model_name] = pipe
    _current_model = model_name
    return pipe


# ── Per-model generation ─────────────────────────────────────────────────────

def _gen_framepack(
    pipe, image: Image.Image, prompt: str, duration_sec: float,
    resolution: tuple, seed: int, negative_prompt: str,
    ref_images: list[Image.Image],
) -> list[Image.Image]:
    h, w = resolution
    raw_frames = int(duration_sec * _MODEL_FPS["framepack"])
    num_frames = _snap_framepack_frames(raw_frames)

    generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None
    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        height=h,
        width=w,
        num_frames=num_frames,
        num_inference_steps=30,
        guidance_scale=9.0,
        sampling_type="inverted_anti_drifting",
        generator=generator,
    )
    return output.frames[0]


def _gen_skyreels_v2_i2v(
    pipe, image: Image.Image, prompt: str, duration_sec: float,
    resolution: tuple, seed: int, negative_prompt: str,
    ref_images: list[Image.Image],
) -> list[Image.Image]:
    h, w = resolution
    num_frames = int(duration_sec * _MODEL_FPS["skyreels_v2_i2v"])
    generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None
    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        height=h,
        width=w,
        num_frames=num_frames,
        num_inference_steps=30,
        guidance_scale=6.0,
        generator=generator,
    )
    return output.frames[0]


def _gen_skyreels_v2_df(
    pipe, image: Image.Image, prompt: str, duration_sec: float,
    resolution: tuple, seed: int, negative_prompt: str,
    ref_images: list[Image.Image],
) -> list[Image.Image]:
    h, w = resolution
    num_frames = int(duration_sec * _MODEL_FPS["skyreels_v2_df"])
    generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None
    # I2V mode: pass image as first_frame or image parameter (verify from model card)
    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        height=h,
        width=w,
        num_frames=num_frames,
        num_inference_steps=30,
        guidance_scale=6.0,
        generator=generator,
    )
    return output.frames[0]


def _gen_wan22_i2v(
    pipe, image: Image.Image, prompt: str, duration_sec: float,
    resolution: tuple, seed: int, negative_prompt: str,
    ref_images: list[Image.Image],
) -> list[Image.Image]:
    h, w = resolution
    num_frames = int(duration_sec * _MODEL_FPS["wan22_i2v"])
    generator = torch.Generator("cuda").manual_seed(seed) if seed >= 0 else None
    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        height=h,
        width=w,
        num_frames=num_frames,
        num_inference_steps=30,
        guidance_scale=6.0,
        generator=generator,
    )
    return output.frames[0]


def _gen_skyreels_v3_r2v(
    pipe_info: dict,
    image: Image.Image,
    prompt: str,
    duration_sec: float,
    resolution: tuple,
    seed: int,
    negative_prompt: str,
    ref_images: Optional[list] = None,
) -> bytes:
    """Run SkyReels-V3 R2V via its custom inference script in a subprocess."""
    import shutil
    import subprocess
    import tempfile

    work_dir = Path(tempfile.mkdtemp(prefix="skyreels_v3_"))
    try:
        # Save primary (keyframe) image
        primary_path = work_dir / "primary.png"
        image.save(primary_path)

        # Save reference images
        ref_paths = []
        for i, img in enumerate(ref_images or []):
            p = work_dir / f"ref_{i}.png"
            img.save(p)
            ref_paths.append(str(p))

        # Primary image is always first; V3 supports up to 4 total
        all_refs = [str(primary_path)] + ref_paths
        ref_imgs_arg = ",".join(all_refs[:4])

        output_path = work_dir / "output.mp4"
        height, width = resolution

        cmd = [
            "python3", pipe_info["script"],
            "--task_type", "reference_to_video",
            "--model_id", pipe_info["model_id"],
            "--ref_imgs", ref_imgs_arg,
            "--prompt", prompt,
            "--duration", str(int(duration_sec)),
            "--height", str(height),
            "--width", str(width),
            "--output_path", str(output_path),
            "--offload",
        ]
        if seed and seed != -1:
            cmd += ["--seed", str(seed)]

        logger.info("Running SkyReels-V3 R2V: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            raise RuntimeError(
                f"SkyReels-V3 script exited {result.returncode}:\n{result.stderr}"
            )

        if not output_path.exists():
            raise RuntimeError("SkyReels-V3 script completed but output.mp4 was not created.")

        return output_path.read_bytes()

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


_GENERATORS = {
    "framepack":       _gen_framepack,
    "skyreels_v2_i2v": _gen_skyreels_v2_i2v,
    "skyreels_v2_df":  _gen_skyreels_v2_df,
    "wan22_i2v":       _gen_wan22_i2v,
    "skyreels_v3_r2v": _gen_skyreels_v3_r2v,
}


# ── Public API ────────────────────────────────────────────────────────────────

def frames_to_mp4_bytes(frames: list[Image.Image], fps: int) -> bytes:
    """Encode PIL frames to MP4 bytes using imageio."""
    buf = io.BytesIO()
    with imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    return buf.getvalue()


def load_and_generate(
    model_name: str,
    image: Image.Image,
    prompt: str,
    duration_sec: float,
    resolution: tuple,          # (height, width)
    seed: int,
    negative_prompt: str,
    ref_images: Optional[list] = None,
) -> bytes:
    """Load model (cached), generate frames, encode to MP4 bytes."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name!r}. Must be one of {SUPPORTED_MODELS}")

    ref_pils: list[Image.Image] = ref_images or []
    fps = _MODEL_FPS[model_name]

    pipe = _get_pipeline(model_name)

    # V3 R2V uses subprocess-based inference, not a pipeline call
    if model_name == "skyreels_v3_r2v":
        return _gen_skyreels_v3_r2v(
            pipe, image, prompt, duration_sec, resolution,
            seed, negative_prompt, ref_pils,
        )

    gen_fn = _GENERATORS[model_name]

    logger.info(
        "Generating %s — prompt=%.60s… duration=%.1fs resolution=%s seed=%d",
        model_name, prompt, duration_sec, resolution, seed,
    )
    frames = gen_fn(pipe, image, prompt, duration_sec, resolution, seed, negative_prompt, ref_pils)
    logger.info("Generated %d frames with %s", len(frames), model_name)

    return frames_to_mp4_bytes(frames, fps)
