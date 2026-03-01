"""RunPod cloud video backend — routes generation to the Beat Studio RunPod worker.

Implements VideoBackend so ModelRouter can select it alongside local backends.
The worker runs one of 5 models; which model is used is controlled by the
RUNPOD_MODEL env var or the `model_name` constructor arg.

Required env vars:
  RUNPOD_API_KEY       — your RunPod API key
  RUNPOD_ENDPOINT_ID   — the serverless endpoint ID
  RUNPOD_MODEL         — default model name (e.g. "skyreels_v2_df")
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from backend.services.prompt.types import ComposedPrompt
from backend.services.video.backends.base import VideoBackend
from backend.services.video.types import VideoClip

logger = logging.getLogger("beat_studio.video.runpod_client")

_ENV_API_KEY    = "RUNPOD_API_KEY"
_ENV_ENDPOINT   = "RUNPOD_ENDPOINT_ID"
_ENV_MODEL      = "RUNPOD_MODEL"

_SUPPORTED_STYLES = {
    "photorealistic", "cinematic", "watercolor", "oil_painting",
    "impressionist", "anime", "abstract", "lofi", "synthwave",
    "cel_animation",
}
_COST_PER_SCENE_USD = 0.08   # ~$0.08 per 5s scene on A100 80GB
_POLL_INTERVAL_SEC  = 5
_DEFAULT_TIMEOUT    = 600    # 10 min hard limit


class RunPodBackend(VideoBackend):
    """Cloud video generation via the Beat Studio RunPod serverless worker.

    Sends storyboard keyframe + prompt to RunPod, polls for the MP4 result,
    writes it to a temp file, and returns a VideoClip.

    The conditioning image comes from ``prompt.init_image_path`` — set this to
    the approved storyboard PNG before calling generate_clip().
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint_id: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_sec: int = _DEFAULT_TIMEOUT,
    ):
        self._api_key     = api_key     or os.getenv(_ENV_API_KEY, "")
        self._endpoint_id = endpoint_id or os.getenv(_ENV_ENDPOINT, "")
        self._model_name  = model_name  or os.getenv(_ENV_MODEL, "skyreels_v2_df")
        self._timeout_sec = timeout_sec

    # ── VideoBackend interface ────────────────────────────────────────────────

    def name(self) -> str:
        return f"runpod_{self._model_name}"

    def vram_required_gb(self) -> float:
        return 0.0  # cloud — no local VRAM

    def supports_style(self, style: str) -> bool:
        return style in _SUPPORTED_STYLES

    def is_available(self) -> bool:
        return bool(self._api_key and self._endpoint_id)

    def estimated_time_per_scene(self, resolution: Tuple[int, int] = (1920, 1080)) -> float:
        return 120.0  # ~2 min per 5s scene on A100 80GB

    def estimated_cost_per_scene(self) -> float:
        return _COST_PER_SCENE_USD

    def generate_clip(
        self,
        prompt: ComposedPrompt,
        duration_sec: float,
        resolution: Tuple[int, int],
        fps: int = 24,
        seed: int = -1,
    ) -> VideoClip:
        t0 = time.time()
        out_path = self._submit_and_poll(prompt, duration_sec, resolution, seed)
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
            cost_usd=_COST_PER_SCENE_USD,
        )

    def generate_batch(
        self,
        prompts: List[ComposedPrompt],
        durations: List[float],
        resolution: Tuple[int, int],
        fps: int = 24,
    ) -> List[VideoClip]:
        # Sequential — RunPod worker is single-threaded per endpoint
        return [
            self.generate_clip(p, d, resolution, fps)
            for p, d in zip(prompts, durations)
        ]

    def kill(self) -> None:
        pass  # no local GPU resources

    # ── Internal ──────────────────────────────────────────────────────────────

    @property
    def _base_url(self) -> str:
        return f"https://api.runpod.ai/v2/{self._endpoint_id}"

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        prompt: ComposedPrompt,
        duration_sec: float,
        resolution: Tuple[int, int],
        seed: int,
    ) -> dict:
        """Build the RunPod job input payload."""
        # Encode the storyboard keyframe
        init_path = getattr(prompt, "init_image_path", "") or ""
        if init_path and Path(init_path).exists():
            image_b64 = base64.b64encode(Path(init_path).read_bytes()).decode()
        else:
            logger.warning("init_image_path missing or not found (%r) — sending blank", init_path)
            image_b64 = _blank_png_b64()

        return {
            "input": {
                "model": self._model_name,
                "image": image_b64,
                "prompt": prompt.positive,
                "duration_sec": duration_sec,
                "resolution": list(resolution),  # [height, width]
                "seed": seed,
                "negative_prompt": prompt.negative or "blurry, low quality, distorted, deformed",
                "ref_images": [],
            }
        }

    def _submit_and_poll(
        self,
        prompt: ComposedPrompt,
        duration_sec: float,
        resolution: Tuple[int, int],
        seed: int,
    ) -> str:
        """Submit job, poll until done, write MP4 to temp file, return path."""
        payload = self._build_payload(prompt, duration_sec, resolution, seed)

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self._base_url}/run",
                json=payload,
                headers=self._headers,
            )
            resp.raise_for_status()
            job_id = resp.json()["id"]

        logger.info("RunPod job submitted: %s (model=%s)", job_id, self._model_name)
        result = self._poll(job_id)

        # Decode and write MP4
        video_bytes = base64.b64decode(result["video_b64"])
        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp4", prefix="rp_", delete=False,
        )
        tmp.write(video_bytes)
        tmp.close()
        logger.info("RunPod clip saved: %s (%d KB)", tmp.name, len(video_bytes) // 1024)
        return tmp.name

    def _poll(self, job_id: str) -> dict:
        """Poll RunPod status endpoint until COMPLETED or FAILED."""
        deadline = time.time() + self._timeout_sec
        with httpx.Client(timeout=30) as client:
            while time.time() < deadline:
                resp = client.get(
                    f"{self._base_url}/status/{job_id}",
                    headers=self._headers,
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status", "")

                if status == "COMPLETED":
                    return data["output"]
                if status == "FAILED":
                    raise RuntimeError(
                        f"RunPod job {job_id} failed: {data.get('error', data)}"
                    )

                logger.debug("RunPod job %s status: %s", job_id, status)
                time.sleep(_POLL_INTERVAL_SEC)

        raise TimeoutError(
            f"RunPod job {job_id} did not complete within {self._timeout_sec}s"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blank_png_b64() -> str:
    """Return a base64-encoded 1×1 black PNG (fallback when no init image)."""
    import io
    from PIL import Image
    img = Image.new("RGB", (480, 720), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
