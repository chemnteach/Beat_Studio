"""Smoke test for the Beat Studio RunPod serverless endpoint.

Sends a minimal job and polls until complete or failed.
"""
import base64
import io
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

# ── Load env ─────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

import os
RUNPOD_API_KEY   = os.environ["RUNPOD_API_KEY"]
RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

# ── Build test image ──────────────────────────────────────────────────────────

img = Image.new("RGB", (480, 720), color=(0, 0, 0))
buf = io.BytesIO()
img.save(buf, format="PNG")
image_b64 = base64.b64encode(buf.getvalue()).decode()

# ── Payload ───────────────────────────────────────────────────────────────────

payload = {
    "input": {
        "model":           "skyreels_v3_r2v",
        "image":           image_b64,
        "prompt":          "gentle ocean waves, cinematic",
        "duration_sec":    3.0,
        "resolution":      [720, 480],
        "seed":            42,
        "negative_prompt": "blurry, low quality",
        "ref_images":      [],
    }
}

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type":  "application/json",
}

BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

# ── Submit ────────────────────────────────────────────────────────────────────

print(f"Submitting job to endpoint {RUNPOD_ENDPOINT_ID}...")
resp = requests.post(f"{BASE_URL}/run", json=payload, headers=HEADERS, timeout=30)
resp.raise_for_status()
job = resp.json()
job_id = job["id"]
print(f"Job submitted: {job_id}")
print("Polling (cold start may take 1-3 minutes)...")

# ── Poll ──────────────────────────────────────────────────────────────────────

TIMEOUT    = 600
POLL_EVERY = 5
start      = time.time()

while True:
    elapsed = time.time() - start
    if elapsed > TIMEOUT:
        print(f"\nTIMEOUT after {elapsed:.0f}s")
        sys.exit(1)

    time.sleep(POLL_EVERY)
    status_resp = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=15)
    status_resp.raise_for_status()
    data   = status_resp.json()
    status = data.get("status", "UNKNOWN")
    print(f"  [{elapsed:5.0f}s] {status}", flush=True)

    if status == "COMPLETED":
        output = data.get("output", {})
        video_b64 = output.get("video_b64")
        if not video_b64:
            print("ERROR: COMPLETED but no video_b64 in output")
            print("Output:", output)
            sys.exit(1)

        out_path = Path(__file__).parent / "smoke_test_output.mp4"
        out_path.write_bytes(base64.b64decode(video_b64))
        size_kb = out_path.stat().st_size / 1024
        print(f"\nSUCCESS in {elapsed:.1f}s")
        print(f"Saved to: {out_path}")
        print(f"File size: {size_kb:.1f} KB")
        sys.exit(0)

    elif status == "FAILED":
        error = data.get("error") or data.get("output", {}).get("error", "unknown error")
        print(f"\nFAILED: {error}")
        sys.exit(1)
