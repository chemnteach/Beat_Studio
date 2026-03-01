"""RunPod serverless handler for Beat Studio video generation.

Job input schema:
  model          str   — which model to use (see models.SUPPORTED_MODELS)
  image          str   — base64-encoded PNG (the storyboard keyframe)
  prompt         str   — motion/scene description
  duration_sec   float — target clip duration in seconds
  resolution     [int, int] — [height, width] in pixels
  seed           int | null — random seed (-1 for random)
  negative_prompt str  — optional negative prompt
  ref_images     list[str] — additional base64 PNGs (SkyReels-V3 R2V only)

Job output schema:
  video_b64      str   — base64-encoded MP4
  model          str   — model that was used
  duration_sec   float — actual clip duration (may differ from requested due to frame snapping)
  frames         int   — number of frames generated
"""
from __future__ import annotations

import base64
import io
import logging
import time
import traceback

import runpod
from PIL import Image

from models import SUPPORTED_MODELS, _MODEL_FPS, frames_to_mp4_bytes, load_and_generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("beat_studio.worker.handler")


def _decode_image(b64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def handler(job: dict) -> dict:
    """RunPod job handler."""
    inp = job.get("input", {})

    # ── Validate ──────────────────────────────────────────────────────────────
    model = inp.get("model", "")
    if not model:
        return {"error": "Missing required field: model"}
    if model not in SUPPORTED_MODELS:
        return {"error": f"Unknown model {model!r}. Supported: {sorted(SUPPORTED_MODELS)}"}

    image_b64 = inp.get("image", "")
    if not image_b64:
        return {"error": "Missing required field: image"}

    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "Missing required field: prompt"}

    duration_sec = float(inp.get("duration_sec", 4.0))
    resolution = tuple(inp.get("resolution", [720, 480]))  # (height, width)
    seed = int(inp.get("seed") or -1)
    negative_prompt = inp.get("negative_prompt", "blurry, low quality, distorted, deformed")
    ref_images_b64: list[str] = inp.get("ref_images", [])

    # ── Decode images ─────────────────────────────────────────────────────────
    try:
        image = _decode_image(image_b64)
    except Exception as exc:
        return {"error": f"Failed to decode image: {exc}"}

    ref_images = []
    for i, rb64 in enumerate(ref_images_b64):
        try:
            ref_images.append(_decode_image(rb64))
        except Exception as exc:
            return {"error": f"Failed to decode ref_images[{i}]: {exc}"}

    # ── Generate ──────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        mp4_bytes = load_and_generate(
            model_name=model,
            image=image,
            prompt=prompt,
            duration_sec=duration_sec,
            resolution=resolution,
            seed=seed,
            negative_prompt=negative_prompt,
            ref_images=ref_images,
        )
    except Exception:
        tb = traceback.format_exc()
        logger.error("Generation failed:\n%s", tb)
        return {"error": f"Generation failed: {tb}"}

    elapsed = time.time() - t0
    fps = _MODEL_FPS[model]
    # Estimate frames from bytes (approximate) — we count from duration
    estimated_frames = int(duration_sec * fps)
    actual_duration = estimated_frames / fps

    video_b64 = base64.b64encode(mp4_bytes).decode()

    logger.info(
        "Job complete: model=%s duration=%.1fs frames~%d elapsed=%.1fs size=%dKB",
        model, actual_duration, estimated_frames, elapsed, len(mp4_bytes) // 1024,
    )

    return {
        "video_b64": video_b64,
        "model": model,
        "duration_sec": actual_duration,
        "frames": estimated_frames,
        "elapsed_sec": round(elapsed, 1),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
