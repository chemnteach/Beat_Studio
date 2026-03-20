"""Unit tests for RunPodBackend — payload construction and polling logic.

Does NOT hit the real RunPod API. All HTTP calls are mocked.
"""
from __future__ import annotations

import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from backend.services.prompt.types import ComposedPrompt
from backend.services.video.backends.runpod_client import (
    MAX_CLIP_SEC,
    RunPodBackend,
    _blank_png_b64,
    _encode_ref_images,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_backend(**kwargs) -> RunPodBackend:
    return RunPodBackend(
        api_key="test-key",
        endpoint_id="test-ep",
        model_name=kwargs.get("model_name", "skyreels_v3_r2v"),
        timeout_sec=kwargs.get("timeout_sec", 30),
    )


def _make_prompt(init_path: str = "", ref_paths: list = None) -> ComposedPrompt:
    p = ComposedPrompt(
        positive="gentle ocean waves, cinematic",
        negative="blurry, low quality",
        cfg_scale=7.0,
        steps=30,
        model="runpod_skyreels_v3_r2v",
        nsfw=False,
        base_checkpoint="",
    )
    p.init_image_path = init_path
    p.ref_image_paths = ref_paths or []
    return p


# ── _build_payload ─────────────────────────────────────────────────────────────

class TestBuildPayload:
    def test_resolution_sent_as_height_width(self):
        """Worker expects [height, width]; router passes (width, height). Must swap."""
        backend = _make_backend()
        prompt = _make_prompt()
        payload = backend._build_payload(prompt, duration_sec=3.0, resolution=(1920, 1080), seed=42)
        resolution = payload["input"]["resolution"]
        # router passes (width=1920, height=1080); worker needs [height, width]
        assert resolution == [1080, 1920], f"Expected [1080, 1920], got {resolution}"

    def test_resolution_portrait(self):
        """480×720 portrait: router passes (width=480, height=720) → worker gets [720, 480]."""
        backend = _make_backend()
        prompt = _make_prompt()
        payload = backend._build_payload(prompt, duration_sec=3.0, resolution=(480, 720), seed=0)
        assert payload["input"]["resolution"] == [720, 480]

    def test_model_name_in_payload(self):
        backend = _make_backend(model_name="skyreels_v3_r2v")
        prompt = _make_prompt()
        payload = backend._build_payload(prompt, duration_sec=5.0, resolution=(1280, 720), seed=1)
        assert payload["input"]["model"] == "skyreels_v3_r2v"

    def test_prompt_fields(self):
        backend = _make_backend()
        prompt = _make_prompt()
        payload = backend._build_payload(prompt, duration_sec=4.0, resolution=(1280, 720), seed=7)
        inp = payload["input"]
        assert inp["prompt"] == "gentle ocean waves, cinematic"
        assert inp["negative_prompt"] == "blurry, low quality"
        assert inp["duration_sec"] == 4.0
        assert inp["seed"] == 7

    def test_blank_image_fallback_when_no_init_path(self, tmp_path):
        """Missing init_image_path → sends blank PNG, logs warning."""
        backend = _make_backend()
        prompt = _make_prompt(init_path="")
        payload = backend._build_payload(prompt, duration_sec=3.0, resolution=(1280, 720), seed=0)
        # Should have a non-empty base64 image (the blank fallback)
        assert payload["input"]["image"]

    def test_init_image_encoded_when_exists(self, tmp_path):
        img = Image.new("RGB", (480, 720), color=(255, 0, 0))
        img_path = tmp_path / "frame.png"
        img.save(img_path)
        backend = _make_backend()
        prompt = _make_prompt(init_path=str(img_path))
        payload = backend._build_payload(prompt, duration_sec=3.0, resolution=(480, 720), seed=0)
        decoded = base64.b64decode(payload["input"]["image"])
        loaded = Image.open(io.BytesIO(decoded))
        assert loaded.size == (480, 720)

    def test_ref_images_empty_by_default(self):
        backend = _make_backend()
        prompt = _make_prompt()
        payload = backend._build_payload(prompt, duration_sec=3.0, resolution=(1280, 720), seed=0)
        assert payload["input"]["ref_images"] == []


# ── _poll ──────────────────────────────────────────────────────────────────────

class TestPoll:
    def _mock_response(self, status: str, output: dict = None, error: str = None):
        resp = MagicMock()
        data = {"status": status}
        if output:
            data["output"] = output
        if error:
            data["error"] = error
        resp.json.return_value = data
        resp.raise_for_status = MagicMock()
        return resp

    def test_poll_returns_output_on_completed(self):
        backend = _make_backend(timeout_sec=30)
        completed_resp = self._mock_response("COMPLETED", output={"video_url": "https://r2.example.com/clips/abc.mp4"})
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = completed_resp

        with patch("backend.services.video.backends.runpod_client.httpx.Client", return_value=mock_client):
            result = backend._poll("job-123")

        assert result == {"video_url": "https://r2.example.com/clips/abc.mp4"}

    def test_poll_raises_when_output_key_missing(self):
        """COMPLETED with no 'output' key → payload too large — must raise RuntimeError."""
        backend = _make_backend(timeout_sec=30)
        # Simulate RunPod silently dropping oversized payload
        resp = MagicMock()
        resp.json.return_value = {"status": "COMPLETED"}  # no 'output' key
        resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = resp

        with patch("backend.services.video.backends.runpod_client.httpx.Client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="no 'output' key"):
                backend._poll("job-big")

    def test_poll_raises_when_handler_returns_error(self):
        """COMPLETED with output={'error': '...'} → handler error — must raise RuntimeError."""
        backend = _make_backend(timeout_sec=30)
        completed_resp = self._mock_response("COMPLETED", output={"error": "CUDA OOM"})
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = completed_resp

        with patch("backend.services.video.backends.runpod_client.httpx.Client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="CUDA OOM"):
                backend._poll("job-err")

    def test_poll_raises_on_failed(self):
        backend = _make_backend(timeout_sec=30)
        failed_resp = self._mock_response("FAILED", error="OOM during model load")
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = failed_resp

        with patch("backend.services.video.backends.runpod_client.httpx.Client", return_value=mock_client):
            with pytest.raises(RuntimeError, match="OOM during model load"):
                backend._poll("job-456")

    def test_poll_raises_timeout_when_deadline_exceeded(self):
        """If the deadline passes while still IN_PROGRESS, raises TimeoutError."""
        backend = _make_backend(timeout_sec=0)  # immediate deadline
        in_progress_resp = self._mock_response("IN_PROGRESS")
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = in_progress_resp

        with patch("backend.services.video.backends.runpod_client.httpx.Client", return_value=mock_client):
            with pytest.raises(TimeoutError):
                backend._poll("job-789")


# ── Default timeout ────────────────────────────────────────────────────────────

def test_default_timeout_is_2700():
    """Cold start + 14B model load + generation requires at least 45 min poll window."""
    from backend.services.video.backends.runpod_client import _DEFAULT_TIMEOUT
    assert _DEFAULT_TIMEOUT == 2700, (
        f"_DEFAULT_TIMEOUT is {_DEFAULT_TIMEOUT}s — must be >= 2700s to survive cold-start jobs"
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def test_blank_png_b64_is_valid_image():
    b64 = _blank_png_b64()
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert img.size == (480, 720)
    assert img.mode == "RGB"


def test_encode_ref_images_skips_missing(tmp_path):
    img = Image.new("RGB", (100, 100))
    real = tmp_path / "real.png"
    img.save(real)
    result = _encode_ref_images([str(real), "/nonexistent/path.png"])
    assert len(result) == 1


def test_encode_ref_images_respects_max(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (50, 50)).save(p)
        paths.append(str(p))
    result = _encode_ref_images(paths, max_refs=2)
    assert len(result) == 2


# ── is_available ──────────────────────────────────────────────────────────────

def test_is_available_true_when_creds_set():
    b = RunPodBackend(api_key="key", endpoint_id="ep", model_name="skyreels_v3_r2v")
    assert b.is_available() is True


def test_is_available_false_when_no_creds():
    b = RunPodBackend(api_key="", endpoint_id="", model_name="skyreels_v3_r2v")
    assert b.is_available() is False


def test_max_clip_sec_is_8():
    assert MAX_CLIP_SEC == 8.0


def test_generate_clip_clamps_duration_over_8(tmp_path):
    """generate_clip must clamp duration_sec to MAX_CLIP_SEC and log a warning."""
    backend = RunPodBackend(api_key="k", endpoint_id="ep", model_name="skyreels_v3_r2v")
    mp4_bytes = b"\x00" * 100

    with patch.object(backend, "_submit_and_poll", return_value=str(tmp_path / "out.mp4")) as mock_poll, \
         patch("builtins.open", create=True), \
         patch("tempfile.NamedTemporaryFile") as mock_tmp:
        mock_tmp.return_value.__enter__ = lambda s: s
        mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
        mock_tmp.return_value.name = str(tmp_path / "out.mp4")

        # _submit_and_poll returns a path; patch generate_clip at the poll level
        with patch.object(backend, "_submit_and_poll", return_value=str(tmp_path / "out.mp4")) as mock_poll:
            prompt = ComposedPrompt(
                positive="a scene", negative="", cfg_scale=7.0, steps=20,
                model="skyreels_v3_r2v", nsfw=False,
            )
            clip = backend.generate_clip(prompt, duration_sec=12.0, resolution=(720, 480))

    # _submit_and_poll should have been called with clamped duration
    call_args = mock_poll.call_args
    assert call_args.args[1] == MAX_CLIP_SEC  # duration_sec clamped to 8.0


def test_generate_clip_passes_duration_under_8(tmp_path):
    """generate_clip must pass duration_sec unchanged when it is within the limit."""
    backend = RunPodBackend(api_key="k", endpoint_id="ep", model_name="skyreels_v3_r2v")
    with patch.object(backend, "_submit_and_poll", return_value=str(tmp_path / "out.mp4")) as mock_poll:
        prompt = ComposedPrompt(
            positive="a scene", negative="", cfg_scale=7.0, steps=20,
            model="skyreels_v3_r2v", nsfw=False,
        )
        clip = backend.generate_clip(prompt, duration_sec=5.0, resolution=(720, 480))

    call_args = mock_poll.call_args
    assert call_args.args[1] == 5.0
