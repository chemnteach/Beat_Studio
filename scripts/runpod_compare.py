#!/usr/bin/env python3
"""Beat Studio RunPod model comparison script.

Sends 3 storyboard scenes through all 5 models and saves the results to
output/comparison/{scene}/{model}.mp4.

Usage:
    python scripts/runpod_compare.py --storyboard-zip path/to/storyboard.zip

    # Or point at individual scene PNGs:
    python scripts/runpod_compare.py \
        --scene03 path/to/scene_03.png \
        --scene11 path/to/scene_11.png \
        --scene17 path/to/scene_17.png

Required env vars (or in backend/.env):
    RUNPOD_API_KEY
    RUNPOD_ENDPOINT_ID
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
import zipfile
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = [
    "framepack",
    "skyreels_v2_i2v",
    "skyreels_v2_df",
    "wan22_i2v",
    "skyreels_v3_r2v",
]

SCENES = {
    "scene_03": {
        "label": "Beach sunset wide shot",
        "prompt": (
            "gentle ocean waves rolling onto golden shore, warm sunset light slowly "
            "shifting across wet sand, palm tree shadow drifting"
        ),
    },
    "scene_11": {
        "label": "Character close-up",
        "prompt": (
            "man peacefully resting on warm sand, subtle breathing movement, "
            "gentle light shifting across face, soft ambient breeze"
        ),
    },
    "scene_17": {
        "label": "Beach party crowd",
        "prompt": (
            "lively beach party at golden hour, people swaying and dancing, "
            "string lights gently swinging, warm festive energy"
        ),
    },
}

DURATION_SEC = 5.0
RESOLUTION   = [720, 480]   # [height, width]
SEED         = 42
NEGATIVE     = "blurry, low quality, distorted, deformed, ugly"
POLL_EVERY   = 5   # seconds
TIMEOUT      = 600 # seconds

# ── RunPod client ─────────────────────────────────────────────────────────────

def _api_key() -> str:
    key = os.getenv("RUNPOD_API_KEY", "")
    if not key:
        sys.exit("RUNPOD_API_KEY not set. Add it to backend/.env or export it.")
    return key

def _endpoint_id() -> str:
    eid = os.getenv("RUNPOD_ENDPOINT_ID", "")
    if not eid:
        sys.exit("RUNPOD_ENDPOINT_ID not set. Add it to backend/.env or export it.")
    return eid

def _headers() -> dict:
    return {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}

def _base_url() -> str:
    return f"https://api.runpod.ai/v2/{_endpoint_id()}"


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _submit(model: str, scene_name: str, scene_info: dict, image_b64: str) -> str:
    payload = {
        "input": {
            "model": model,
            "image": image_b64,
            "prompt": scene_info["prompt"],
            "duration_sec": DURATION_SEC,
            "resolution": RESOLUTION,
            "seed": SEED,
            "negative_prompt": NEGATIVE,
            "ref_images": [],
        }
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{_base_url()}/run", json=payload, headers=_headers())
        resp.raise_for_status()
        return resp.json()["id"]


def _poll(job_id: str) -> dict:
    deadline = time.time() + TIMEOUT
    with httpx.Client(timeout=30) as client:
        while time.time() < deadline:
            resp = client.get(f"{_base_url()}/status/{job_id}", headers=_headers())
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "")
            if status == "COMPLETED":
                return data["output"]
            if status == "FAILED":
                raise RuntimeError(f"Job {job_id} failed: {data.get('error', data)}")
            print(f"    [{status}]", end="\r", flush=True)
            time.sleep(POLL_EVERY)
    raise TimeoutError(f"Job {job_id} timed out after {TIMEOUT}s")


def _save_result(output: dict, out_path: Path) -> None:
    video_bytes = base64.b64decode(output["video_b64"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(video_bytes)


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_from_zip(zip_path: Path) -> dict[str, Path]:
    """Extract scene_03, scene_11, scene_17 from a storyboard ZIP."""
    tmp_dir = Path("/tmp/beat_studio_compare")
    tmp_dir.mkdir(exist_ok=True)
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            stem = Path(name).stem  # e.g. "scene_03"
            if stem in ("scene_03", "scene_11", "scene_17"):
                target = tmp_dir / Path(name).name
                target.write_bytes(zf.read(name))
                out[stem] = target
    missing = set(SCENES) - set(out)
    if missing:
        sys.exit(f"ZIP is missing scenes: {missing}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Beat Studio RunPod model comparison")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--storyboard-zip", type=Path, help="Storyboard ZIP file")
    group.add_argument("--scene03", type=Path, help="scene_03.png path")
    parser.add_argument("--scene11", type=Path, help="scene_11.png path")
    parser.add_argument("--scene17", type=Path, help="scene_17.png path")
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        help="Which models to test (default: all 5)",
    )
    args = parser.parse_args()

    # Resolve scene image paths
    if args.storyboard_zip:
        scene_paths = _load_from_zip(args.storyboard_zip)
    else:
        if not (args.scene11 and args.scene17):
            parser.error("--scene11 and --scene17 required when using --scene03")
        scene_paths = {
            "scene_03": args.scene03,
            "scene_11": args.scene11,
            "scene_17": args.scene17,
        }
        for name, p in scene_paths.items():
            if not p.exists():
                sys.exit(f"File not found: {p} ({name})")

    models_to_test = args.models
    out_root = Path("output/comparison")

    total = len(SCENES) * len(models_to_test)
    done = 0
    results: list[dict] = []

    print(f"\nBeat Studio RunPod Comparison")
    print(f"Endpoint: {_endpoint_id()}")
    print(f"Models:   {models_to_test}")
    print(f"Scenes:   {list(SCENES)}")
    print(f"Total:    {total} clips\n")

    for scene_name, scene_info in SCENES.items():
        image_path = scene_paths[scene_name]
        image_b64 = _encode_image(image_path)
        print(f"Scene: {scene_name} — {scene_info['label']}")

        for model in models_to_test:
            done += 1
            out_path = out_root / scene_name / f"{model}.mp4"

            if out_path.exists():
                print(f"  [{done}/{total}] {model:20s} SKIP (already exists)")
                continue

            print(f"  [{done}/{total}] {model:20s} submitting…", end="", flush=True)
            t0 = time.time()

            try:
                job_id = _submit(model, scene_name, scene_info, image_b64)
                output = _poll(job_id)
                _save_result(output, out_path)
                elapsed = time.time() - t0
                size_kb = out_path.stat().st_size // 1024
                results.append({
                    "scene": scene_name, "model": model,
                    "elapsed_sec": round(elapsed, 1), "size_kb": size_kb,
                    "status": "ok",
                })
                print(f"  done in {elapsed:.0f}s ({size_kb} KB) → {out_path}")
            except Exception as exc:
                elapsed = time.time() - t0
                results.append({
                    "scene": scene_name, "model": model,
                    "elapsed_sec": round(elapsed, 1), "size_kb": 0,
                    "status": f"FAILED: {exc}",
                })
                print(f"  FAILED after {elapsed:.0f}s: {exc}")

        print()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Scene':<12} {'Model':<22} {'Time':>6} {'Size':>8}  Status")
    print("-" * 60)
    for r in results:
        status = "OK" if r["status"] == "ok" else r["status"][:20]
        print(
            f"{r['scene']:<12} {r['model']:<22} {r['elapsed_sec']:>5.0f}s "
            f"{r['size_kb']:>6}KB  {status}"
        )

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{ok}/{total} clips generated successfully.")
    print(f"\nOutputs saved to: {out_root.resolve()}")

    if ok == total:
        print("\nAll clips generated. Review them and pick a winner.")
        print("Then delete the losing model dirs from the RunPod network volume.")


if __name__ == "__main__":
    main()
