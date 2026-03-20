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
  video_url      str   — presigned Cloudflare R2 URL (1-hour TTL)
  model          str   — model that was used
  duration_sec   float — actual clip duration (may differ from requested due to frame snapping)
  frames         int   — number of frames generated
  elapsed_sec    float — generation time in seconds
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
import traceback
import uuid

import boto3
import botocore.config
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


def _upload_to_r2(mp4_bytes: bytes) -> str:
    """Upload MP4 bytes to Cloudflare R2 and return a presigned URL (1-hour TTL).

    Required env vars on the RunPod endpoint:
      R2_ACCOUNT_ID        — full endpoint URL: https://<account_id>.r2.cloudflarestorage.com
      R2_ACCESS_KEY_ID     — R2 API token access key
      R2_SECRET_ACCESS_KEY — R2 API token secret key
      R2_BUCKET_NAME       — bucket name (e.g. beat-studio-clips)
    """
    endpoint_url = os.environ.get("R2_ACCOUNT_ID", "")
    access_key   = os.environ.get("R2_ACCESS_KEY_ID", "")
    secret_key   = os.environ.get("R2_SECRET_ACCESS_KEY", "")
    bucket       = os.environ.get("R2_BUCKET_NAME", "")

    missing = [k for k, v in [
        ("R2_ACCOUNT_ID", endpoint_url), ("R2_ACCESS_KEY_ID", access_key),
        ("R2_SECRET_ACCESS_KEY", secret_key), ("R2_BUCKET_NAME", bucket),
    ] if not v]
    if missing:
        raise EnvironmentError(f"Missing R2 env vars: {', '.join(missing)}")

    # R2 requires SigV4 signing — must be set explicitly for boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=botocore.config.Config(signature_version="s3v4"),
    )

    key = f"clips/{uuid.uuid4().hex}.mp4"
    s3.put_object(Bucket=bucket, Key=key, Body=mp4_bytes, ContentType="video/mp4")

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=3600,  # 1 hour
    )
    logger.info("Uploaded clip to R2: key=%s size=%dKB", key, len(mp4_bytes) // 1024)
    return url


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

    # Upload to R2 and return URL — avoids ~10MB RunPod payload limit
    try:
        video_url = _upload_to_r2(mp4_bytes)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("R2 upload failed:\n%s", tb)
        # Truncate to keep payload small — full tb is in worker logs
        return {"error": f"R2 upload failed: {type(exc).__name__}: {exc}"[:500]}

    logger.info(
        "Job complete: model=%s duration=%.1fs frames~%d elapsed=%.1fs size=%dKB",
        model, actual_duration, estimated_frames, elapsed, len(mp4_bytes) // 1024,
    )

    return {
        "video_url": video_url,
        "model": model,
        "duration_sec": actual_duration,
        "frames": estimated_frames,
        "elapsed_sec": round(elapsed, 1),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
