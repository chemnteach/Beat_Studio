#!/usr/bin/env python3
"""Three-way model comparison for Beat Studio video generation.

Sends a single image + prompt to three RunPod models (skyreels_v3_r2v,
wan22_i2v, framepack) and saves the resulting videos side by side.

Usage:
    python scripts/compare_models.py \
        --image path/to/keyframe.png \
        --prompt "gentle ocean waves rolling onto golden shore"

    # Subset of models:
    python scripts/compare_models.py \
        --image path/to/keyframe.png \
        --prompt "a person walking through a field" \
        --models wan22_i2v framepack

    # Custom output directory, duration, resolution:
    python scripts/compare_models.py \
        --image path/to/keyframe.png \
        --prompt "cinematic drone shot" \
        --output-dir results/ \
        --duration 4.0 \
        --resolution 720 480 \
        --seed 42

Required env vars (or in backend/.env):
    RUNPOD_API_KEY
    RUNPOD_ENDPOINT_ID
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

# ── Constants ────────────────────────────────────────────────────────────────

ALL_MODELS = ["skyreels_v3_r2v", "wan22_i2v", "framepack"]

DEFAULT_DURATION = 5.0
DEFAULT_RESOLUTION = [720, 480]  # [height, width] — matches handler expectation
DEFAULT_SEED = -1
DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed, ugly"
POLL_INTERVAL = 5   # seconds
TIMEOUT = 2700      # 45 min — covers cold model load + generation


# ── RunPod helpers ───────────────────────────────────────────────────────────

def _api_key() -> str:
    key = os.getenv("RUNPOD_API_KEY", "")
    if not key:
        sys.exit("ERROR: RUNPOD_API_KEY not set. Add it to backend/.env or export it.")
    return key


def _endpoint_id() -> str:
    eid = os.getenv("RUNPOD_ENDPOINT_ID", "")
    if not eid:
        sys.exit("ERROR: RUNPOD_ENDPOINT_ID not set. Add it to backend/.env or export it.")
    return eid


def _headers() -> dict:
    return {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}


def _base_url() -> str:
    return f"https://api.runpod.ai/v2/{_endpoint_id()}"


def encode_image(path: Path) -> str:
    """Base64-encode an image file."""
    if not path.exists():
        sys.exit(f"ERROR: Image not found: {path}")
    return base64.b64encode(path.read_bytes()).decode()


def submit_job(
    model: str,
    image_b64: str,
    prompt: str,
    duration_sec: float,
    resolution: list[int],
    seed: int,
    negative_prompt: str,
) -> str:
    """Submit a generation job to RunPod. Returns the job ID."""
    payload = {
        "input": {
            "model": model,
            "image": image_b64,
            "prompt": prompt,
            "duration_sec": duration_sec,
            "resolution": resolution,      # [height, width]
            "seed": seed,
            "negative_prompt": negative_prompt,
            "ref_images": [],
        }
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{_base_url()}/run", json=payload, headers=_headers())
        resp.raise_for_status()
        return resp.json()["id"]


def poll_job(job_id: str) -> dict:
    """Poll RunPod until job completes or fails. Returns the output dict."""
    deadline = time.time() + TIMEOUT
    with httpx.Client(timeout=30) as client:
        while time.time() < deadline:
            resp = client.get(f"{_base_url()}/status/{job_id}", headers=_headers())
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")

            if status == "COMPLETED":
                if "output" not in data:
                    raise RuntimeError(
                        f"Job {job_id} completed but response has no 'output' key "
                        f"(likely endpoint execution timeout). Full response: {data}"
                    )
                output = data["output"]
                if "error" in output:
                    raise RuntimeError(f"Job {job_id} handler error: {output['error']}")
                return output
            if status == "FAILED":
                raise RuntimeError(f"Job {job_id} failed: {data.get('error', data)}")

            print(f"    status: {status}", end="\r", flush=True)
            time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Job {job_id} timed out after {TIMEOUT}s")


def save_video(output: dict, out_path: Path) -> int:
    """Fetch video from output (URL or base64) and write to disk. Returns size in bytes."""
    if "video_url" in output:
        resp = httpx.get(output["video_url"], timeout=120)
        resp.raise_for_status()
        video_bytes = resp.content
    else:
        video_bytes = base64.b64decode(output["video_b64"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(video_bytes)
    return len(video_bytes)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Three-way model comparison for Beat Studio video generation",
    )
    parser.add_argument(
        "--image", type=Path, required=True,
        help="Path to the input/reference image (PNG or JPG)",
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt describing the desired motion/scene",
    )
    parser.add_argument(
        "--models", nargs="+", default=ALL_MODELS,
        choices=ALL_MODELS, metavar="MODEL",
        help=f"Models to compare (default: all three: {', '.join(ALL_MODELS)})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("comparison_output"),
        help="Directory to save output videos (default: comparison_output/)",
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION,
        help=f"Clip duration in seconds (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=DEFAULT_RESOLUTION,
        metavar=("HEIGHT", "WIDTH"),
        help=f"Resolution as HEIGHT WIDTH (default: {DEFAULT_RESOLUTION[0]} {DEFAULT_RESOLUTION[1]})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed (-1 for random, default: -1)",
    )
    parser.add_argument(
        "--negative-prompt", type=str, default=DEFAULT_NEGATIVE,
        help="Negative prompt (default: standard quality negatives)",
    )
    args = parser.parse_args()

    # Validate image exists
    if not args.image.exists():
        sys.exit(f"ERROR: Image not found: {args.image}")

    # Validate env vars early
    _api_key()
    _endpoint_id()

    models = args.models
    output_dir: Path = args.output_dir

    print()
    print("=" * 60)
    print("  Beat Studio — Three-Way Model Comparison")
    print("=" * 60)
    print(f"  Endpoint:    {_endpoint_id()}")
    print(f"  Image:       {args.image}")
    print(f"  Prompt:      {args.prompt[:70]}{'...' if len(args.prompt) > 70 else ''}")
    print(f"  Models:      {', '.join(models)}")
    print(f"  Duration:    {args.duration}s")
    print(f"  Resolution:  {args.resolution[0]}x{args.resolution[1]} (HxW)")
    print(f"  Seed:        {args.seed}")
    print(f"  Output dir:  {output_dir.resolve()}")
    print("=" * 60)
    print()

    # Encode image once
    print("Encoding input image...", flush=True)
    image_b64 = encode_image(args.image)
    print(f"  Image encoded ({len(image_b64) // 1024} KB base64)")
    print()

    # Run each model sequentially (worker handles one model at a time)
    results: list[dict] = []

    for i, model in enumerate(models, 1):
        # Friendly name for output file
        short_name = {
            "skyreels_v3_r2v": "skyreels",
            "wan22_i2v": "wan22",
            "framepack": "framepack",
        }.get(model, model)

        out_path = output_dir / f"comparison_{short_name}.mp4"

        print(f"[{i}/{len(models)}] {model}")
        print(f"  Output: {out_path}")

        t0 = time.time()
        try:
            # Submit
            print("  Submitting job...", end="", flush=True)
            job_id = submit_job(
                model=model,
                image_b64=image_b64,
                prompt=args.prompt,
                duration_sec=args.duration,
                resolution=args.resolution,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
            )
            print(f" job_id={job_id}")

            # Poll
            print("  Waiting for completion...", flush=True)
            output = poll_job(job_id)
            elapsed = time.time() - t0

            # Save
            size_bytes = save_video(output, out_path)
            size_kb = size_bytes // 1024

            # Collect metadata from worker response
            worker_model = output.get("model", model)
            worker_duration = output.get("duration_sec", args.duration)
            worker_frames = output.get("frames", "?")
            worker_elapsed = output.get("elapsed_sec", "?")

            results.append({
                "model": model,
                "status": "ok",
                "total_elapsed_sec": round(elapsed, 1),
                "worker_elapsed_sec": worker_elapsed,
                "size_kb": size_kb,
                "frames": worker_frames,
                "duration_sec": worker_duration,
                "output_path": str(out_path),
            })

            print(f"  Done: {elapsed:.0f}s total, {size_kb} KB, {worker_frames} frames")
            print()

        except Exception as exc:
            elapsed = time.time() - t0
            results.append({
                "model": model,
                "status": f"FAILED: {exc}",
                "total_elapsed_sec": round(elapsed, 1),
                "worker_elapsed_sec": None,
                "size_kb": 0,
                "frames": 0,
                "duration_sec": 0,
                "output_path": None,
            })
            print(f"  FAILED after {elapsed:.0f}s: {exc}")
            print()

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<22} {'Total Time':>10} {'GPU Time':>10} {'Size':>8} {'Frames':>7}  Status")
    print("  " + "-" * 66)

    for r in results:
        status = "OK" if r["status"] == "ok" else r["status"][:25]
        gpu_time = f"{r['worker_elapsed_sec']}s" if r["worker_elapsed_sec"] else "—"
        print(
            f"  {r['model']:<22} "
            f"{r['total_elapsed_sec']:>8.0f}s "
            f"{gpu_time:>10} "
            f"{r['size_kb']:>6}KB "
            f"{r['frames']:>7}  "
            f"{status}"
        )

    ok_count = sum(1 for r in results if r["status"] == "ok")
    print()
    print(f"  {ok_count}/{len(models)} models completed successfully.")

    if ok_count > 0:
        print(f"\n  Output videos saved to: {output_dir.resolve()}/")
        print("  Files:")
        for r in results:
            if r["status"] == "ok":
                print(f"    - {r['output_path']}")

    print()


if __name__ == "__main__":
    main()
