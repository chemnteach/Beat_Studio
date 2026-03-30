"""RunPod serverless handler for Beat Studio video generation.

Job input schema (video generation):
  model          str   — which model to use (see models.SUPPORTED_MODELS)
  image          str   — base64-encoded PNG (the storyboard keyframe)
  prompt         str   — motion/scene description
  duration_sec   float — target clip duration in seconds
  resolution     [int, int] — [height, width] in pixels
  seed           int | null — random seed (-1 for random)
  negative_prompt str  — optional negative prompt
  ref_images     list[str] — additional base64 PNGs (SkyReels-V3 R2V only)

Job input schema (maintenance):
  maintenance_op str   — "download_models" | "list_hf_files" | "list_models_dir" | "delete_dirs"
  dirs           list  — (delete_dirs only) directory names under /runpod-volume/models/ to remove

Job output schema (video generation):
  video_url      str   — presigned Cloudflare R2 URL (1-hour TTL)
  model          str   — model that was used
  duration_sec   float — actual clip duration (may differ from requested due to frame snapping)
  frames         int   — number of frames generated
  elapsed_sec    float — generation time in seconds

Job output schema (maintenance):
  status         str   — "ok" or "error"
  downloaded     list  — files successfully downloaded with sizes
  deleted        list  — directories removed
  models_dir     dict  — {dirname: "X.X GB"} for all entries in /runpod-volume/models/
  error          str   — present only on failure

NOTE: Execution timeout must be set to 3600s in the RunPod endpoint dashboard:
  Serverless → your endpoint → Edit → Max Execution Time → 3600
"""
from __future__ import annotations

import base64
import io
import logging
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path

import boto3
import botocore.config
import runpod
import torch
from PIL import Image

from models import SUPPORTED_MODELS, _MODEL_FPS, load_and_generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("beat_studio.worker.handler")

# TF32 — faster matmul/convolutions on Ampere+ GPUs with negligible precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logger.info("TF32 enabled: matmul=%s cudnn=%s",
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32)

_MIN_MODEL_SIZE = 10 * 1024 ** 3  # 10 GiB — FP8 14B models are ~13-14 GB


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _scan_models_dir() -> dict:
    """Return {dirname: 'X.X GB'} for all entries in /runpod-volume/models/."""
    models_root = Path("/runpod-volume/models")
    result = {}
    if not models_root.exists():
        return result
    for entry in sorted(models_root.iterdir()):
        if entry.is_dir():
            size_bytes = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
            result[entry.name] = f"{size_bytes / (1024 ** 3):.1f} GB"
        elif entry.is_file():
            size_bytes = entry.stat().st_size
            result[entry.name] = f"{size_bytes / (1024 ** 3):.1f} GB"
    return result


# ── Maintenance handler ───────────────────────────────────────────────────────

def _maintenance_download_models() -> dict:
    """Download FP8 SkyReels V3 weights and clean up old models.

    Progress logged to stdout (visible in RunPod worker logs).
    Returns a result dict — regular function, not a generator.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        return {"status": "error", "error": f"huggingface_hub not available: {exc}"}

    # Disable hf_transfer — known to cause issues with large files on some platforms
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    HF_REPO      = "Kijai/WanVideo_comfy_fp8_scaled"
    HF_SUBFOLDER = "SkyReelsV3"

    # Note: first filename uses underscore (Wan21_SkyReelsV3); others use hyphen
    downloads = [
        {
            "filename": "Wan21_SkyReelsV3-R2V_fp8_scaled_mixed.safetensors",
            "local_dir": "/runpod-volume/models/SkyReelsV3-R2V-FP8",
        },
        {
            "filename": "Wan21-SkyReelsV3-V2V_fp8_scaled_mixed.safetensors",
            "local_dir": "/runpod-volume/models/SkyReelsV3-V2V-FP8",
        },
        {
            "filename": "Wan21-SkyReelsV3-V2V_shot_fp8_scaled_mixed.safetensors",
            "local_dir": "/runpod-volume/models/SkyReelsV3-V2V-FP8",
        },
    ]

    to_delete = [
        "/runpod-volume/models/Wan2.2-I2V-A14B",
        "/runpod-volume/models/FramePackI2V_HY",
        "/runpod-volume/models/SkyReels-V3-R2V-14B",
    ]

    downloaded = []

    for spec in downloads:
        filename  = spec["filename"]
        local_dir = spec["local_dir"]

        print(f"[MAINT] Downloading {filename} → {local_dir} ...", flush=True)
        logger.info("Starting download: %s → %s", filename, local_dir)

        try:
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            actual_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                subfolder=HF_SUBFOLDER,
                local_dir=local_dir,
            )
            dest_path = Path(actual_path)
            print(f"[MAINT] Downloaded to: {dest_path}", flush=True)
            logger.info("Downloaded to: %s", dest_path)
        except Exception as exc:
            logger.error("Download failed for %s: %s", filename, traceback.format_exc())
            return {"status": "error", "error": f"Download failed for {filename}: {exc}", "downloaded": downloaded}

        if not dest_path.exists():
            msg = f"Verification failed: {dest_path} does not exist after download"
            logger.error(msg)
            return {"status": "error", "error": msg, "downloaded": downloaded}

        size_bytes = dest_path.stat().st_size
        size_gb    = size_bytes / (1024 ** 3)
        if size_bytes < _MIN_MODEL_SIZE:
            msg = f"Verification failed: {filename} is {size_gb:.1f} GB, expected >10 GB"
            logger.error(msg)
            return {"status": "error", "error": msg, "downloaded": downloaded}

        downloaded.append({"file": str(dest_path), "size_gb": round(size_gb, 1)})
        print(f"[MAINT] Verified: {filename} ({size_gb:.1f} GB)", flush=True)
        logger.info("Verified: %s (%.1f GB)", filename, size_gb)

    # Delete old model directories
    deleted = []
    for path_str in to_delete:
        p = Path(path_str)
        if p.exists():
            print(f"[MAINT] Deleting {path_str} ...", flush=True)
            logger.info("Deleting %s", path_str)
            try:
                shutil.rmtree(p)
                deleted.append(path_str)
                print(f"[MAINT] Deleted {path_str}", flush=True)
            except Exception as exc:
                logger.warning("Failed to delete %s: %s", path_str, exc)
        else:
            logger.info("Skip delete (not found): %s", path_str)

    models_dir = _scan_models_dir()
    return {
        "status":     "ok",
        "downloaded": downloaded,
        "deleted":    deleted,
        "models_dir": models_dir,
    }


def _maintenance_list_hf_files() -> dict:
    """List all files in Kijai/WanVideo_comfy_fp8_scaled under SkyReelsV3/."""
    from huggingface_hub import list_repo_files

    repo_id  = "Kijai/WanVideo_comfy_fp8_scaled"
    prefix   = "SkyReelsV3/"

    logger.info("Listing repo files: %s (prefix=%s)", repo_id, prefix)
    try:
        all_files    = list(list_repo_files(repo_id))
        matched      = [f for f in all_files if f.startswith(prefix)]
        logger.info("Found %d files under %s", len(matched), prefix)
        return {
            "status": "ok",
            "repo":   repo_id,
            "prefix": prefix,
            "files":  matched,
            "count":  len(matched),
        }
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("list_hf_files failed: %s", tb)
        return {"status": "error", "error": str(exc)}


# ── Video generation (extracted from handler for clarity) ─────────────────────

def _run_video_generation(inp: dict) -> dict:
    """Validate, decode, generate, and upload one video clip. Returns result dict."""
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

    duration_sec    = float(inp.get("duration_sec", 4.0))
    resolution      = tuple(inp.get("resolution", [720, 480]))  # (height, width)
    seed            = int(inp.get("seed") or -1)
    negative_prompt = inp.get("negative_prompt", "blurry, low quality, distorted, deformed")
    ref_images_b64: list[str] = inp.get("ref_images", [])

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

    elapsed           = time.time() - t0
    fps               = _MODEL_FPS[model]
    estimated_frames  = int(duration_sec * fps)
    actual_duration   = estimated_frames / fps

    try:
        video_url = _upload_to_r2(mp4_bytes)
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("R2 upload failed:\n%s", tb)
        return {"error": f"R2 upload failed: {type(exc).__name__}: {exc}"[:500]}

    logger.info(
        "Job complete: model=%s duration=%.1fs frames~%d elapsed=%.1fs size=%dKB",
        model, actual_duration, estimated_frames, elapsed, len(mp4_bytes) // 1024,
    )

    return {
        "video_url":    video_url,
        "model":        model,
        "duration_sec": actual_duration,
        "frames":       estimated_frames,
        "elapsed_sec":  round(elapsed, 1),
    }


# ── Maintenance: list models dir ──────────────────────────────────────────────

def _maintenance_list_models_dir() -> dict:
    """Return a snapshot of /runpod-volume/models/ with per-entry sizes."""
    models_dir = _scan_models_dir()
    return {"status": "ok", "models_dir": models_dir}


def _maintenance_delete_dirs(dirs: list[str]) -> dict:
    """Delete specified directories under /runpod-volume/models/.

    Input: list of directory names (not full paths) to delete.
    Only deletes entries that exist under /runpod-volume/models/ — no
    absolute paths accepted, preventing accidental deletion elsewhere.
    """
    models_root = Path("/runpod-volume/models")
    deleted = []
    skipped = []
    errors = []

    for name in dirs:
        # Safety: reject anything with path separators
        if "/" in name or "\\" in name or name in (".", ".."):
            errors.append(f"Rejected unsafe name: {name!r}")
            continue
        p = models_root / name
        if not p.exists():
            skipped.append(name)
            continue
        try:
            shutil.rmtree(p)
            deleted.append(name)
            logger.info("Deleted: %s", p)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.error("Failed to delete %s: %s", p, exc)

    return {
        "status": "ok" if not errors else "partial",
        "deleted": deleted,
        "skipped": skipped,
        "errors":  errors,
        "models_dir": _scan_models_dir(),
    }


# ── Main handler ──────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """RunPod job handler — regular function, returns one result dict."""
    try:
        inp      = job.get("input", {})
        maint_op = inp.get("maintenance_op", "")

        if maint_op == "download_models":
            return _maintenance_download_models()

        if maint_op == "list_hf_files":
            return _maintenance_list_hf_files()

        if maint_op == "list_models_dir":
            return _maintenance_list_models_dir()

        if maint_op == "delete_dirs":
            dirs = inp.get("dirs", [])
            if not isinstance(dirs, list) or not dirs:
                return {"error": "delete_dirs requires a non-empty 'dirs' list of directory names"}
            return _maintenance_delete_dirs(dirs)

        if maint_op:
            return {"error": f"Unknown maintenance_op: {maint_op!r}. Supported: download_models, list_hf_files, list_models_dir, delete_dirs"}

        return _run_video_generation(inp)

    except Exception as exc:
        logger.error("Unhandled handler exception: %s", traceback.format_exc())
        return {"error": f"Unhandled exception: {exc}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
