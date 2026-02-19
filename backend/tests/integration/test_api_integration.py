"""Phase 11: Integration tests for Beat Studio FastAPI backend.

Uses FastAPI TestClient to exercise every router end-to-end with real
HTTP requests through the ASGI stack.  No real GPU, file system, or
external services are needed — all heavy services return stub data.

Coverage targets:
  - All 7 routers (audio, mashup, video, lora, nova_fade, tasks, system)
  - HTTP status codes and response shape
  - Full-pipeline chain: audio → mashup → video plan
  - WebSocket task progress endpoint
"""
from __future__ import annotations

import io
import struct

import pytest
from fastapi.testclient import TestClient

from backend.main import app


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _minimal_wav() -> io.BytesIO:
    """Return a BytesIO containing a valid minimal WAV file (0.5 s silence)."""
    sample_rate = 44100
    num_samples = sample_rate // 2
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    chunk_size = 36 + data_size

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", chunk_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))           # PCM
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(b"\x00" * data_size)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Synchronous TestClient wrapping the Beat Studio FastAPI app."""
    with TestClient(app) as c:
        yield c


# ═════════════════════════════════════════════════════════════════════════════
# System router  /api/system
# ═════════════════════════════════════════════════════════════════════════════


class TestSystemRouter:
    def test_health_check_returns_ok(self, client: TestClient):
        resp = client.get("/api/system/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_health_check_has_version(self, client: TestClient):
        body = client.get("/api/system/health").json()
        assert body["version"] == "1.0.0"

    def test_gpu_status_returns_valid_shape(self, client: TestClient):
        resp = client.get("/api/system/gpu")
        assert resp.status_code == 200
        body = resp.json()
        assert "available" in body
        assert "name" in body
        assert "total_gb" in body
        assert "used_gb" in body
        assert "free_gb" in body

    def test_list_models_returns_list(self, client: TestClient):
        resp = client.get("/api/system/models")
        assert resp.status_code == 200
        body = resp.json()
        assert "models" in body
        assert isinstance(body["models"], list)

    def test_install_model_accepted(self, client: TestClient):
        resp = client.post(
            "/api/system/models/install",
            json={"model_name": "animatediff_lightning"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert "status" in body

    def test_install_model_returns_model_name(self, client: TestClient):
        resp = client.post(
            "/api/system/models/install",
            json={"model_name": "whisper_base"},
        )
        assert resp.json()["model"] == "whisper_base"


# ═════════════════════════════════════════════════════════════════════════════
# Audio router  /api/audio
# ═════════════════════════════════════════════════════════════════════════════


class TestAudioRouter:
    def test_upload_audio_created(self, client: TestClient):
        fake_mp3 = io.BytesIO(b"ID3\x03\x00" + b"\x00" * 100)
        resp = client.post(
            "/api/audio/upload",
            files={"file": ("test_song.mp3", fake_mp3, "audio/mpeg")},
        )
        assert resp.status_code == 201

    def test_upload_returns_audio_id(self, client: TestClient):
        fake_wav = io.BytesIO(b"RIFF" + b"\x00" * 100)
        body = client.post(
            "/api/audio/upload",
            files={"file": ("track.wav", fake_wav, "audio/wav")},
        ).json()
        assert "audio_id" in body
        assert "status" in body

    def test_analyze_audio_queued(self, client: TestClient):
        # Upload first, then analyze with the real audio_id
        upload = client.post(
            "/api/audio/upload",
            files={"file": ("song.wav", _minimal_wav(), "audio/wav")},
        )
        assert upload.status_code == 201
        audio_id = upload.json()["audio_id"]

        resp = client.post(
            "/api/audio/analyze",
            json={"audio_id": audio_id, "depth": "standard"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["audio_id"] == audio_id
        assert "status" in body

    def test_analyze_accepts_all_depths(self, client: TestClient):
        for depth in ("basic", "standard", "full"):
            upload = client.post(
                "/api/audio/upload",
                files={"file": ("song.wav", _minimal_wav(), "audio/wav")},
            )
            audio_id = upload.json()["audio_id"]
            resp = client.post(
                "/api/audio/analyze",
                json={"audio_id": audio_id, "depth": depth},
            )
            assert resp.status_code == 200, f"depth={depth} failed"

    def test_get_analysis_returns_id(self, client: TestClient):
        # Upload so the audio_id is known; analysis will be "pending" (not yet run)
        upload = client.post(
            "/api/audio/upload",
            files={"file": ("song.wav", _minimal_wav(), "audio/wav")},
        )
        audio_id = upload.json()["audio_id"]

        resp = client.get(f"/api/audio/analysis/{audio_id}")
        assert resp.status_code == 200
        assert resp.json()["audio_id"] == audio_id

    def test_get_analysis_has_status(self, client: TestClient):
        upload = client.post(
            "/api/audio/upload",
            files={"file": ("song.wav", _minimal_wav(), "audio/wav")},
        )
        audio_id = upload.json()["audio_id"]

        body = client.get(f"/api/audio/analysis/{audio_id}").json()
        assert "status" in body

    def test_get_analysis_unknown_id_is_404(self, client: TestClient):
        resp = client.get("/api/audio/analysis/no-such-id-xyz")
        assert resp.status_code == 404


# ═════════════════════════════════════════════════════════════════════════════
# Mashup router  /api/mashup
# ═════════════════════════════════════════════════════════════════════════════


class TestMashupRouter:
    def test_ingest_song_accepted(self, client: TestClient):
        resp = client.post(
            "/api/mashup/ingest",
            json={"source": "/tmp/track.mp3"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert "status" in body

    def test_ingest_batch_returns_count(self, client: TestClient):
        sources = ["/tmp/a.mp3", "/tmp/b.mp3", "/tmp/c.mp3"]
        body = client.post("/api/mashup/ingest/batch", json=sources).json()
        assert body["count"] == 3

    def test_list_library_empty_default(self, client: TestClient):
        body = client.get("/api/mashup/library").json()
        assert "songs" in body
        assert "total" in body
        assert isinstance(body["songs"], list)

    def test_search_library_echoes_query(self, client: TestClient):
        body = client.get("/api/mashup/library/search", params={"q": "upbeat pop"}).json()
        assert body["query"] == "upbeat pop"
        assert "results" in body

    def test_match_unknown_song_is_404(self, client: TestClient):
        # Song not in library → 404 (real match requires an ingested song)
        resp = client.post(
            "/api/mashup/match",
            params={"song_id": "not_in_library", "criteria": "hybrid", "top": 5},
        )
        assert resp.status_code == 404

    def test_create_mashup_accepted(self, client: TestClient):
        resp = client.post(
            "/api/mashup/create",
            json={"song_a_id": "song-001", "song_b_id": "song-002", "mashup_type": "classic"},
        )
        assert resp.status_code == 202
        assert "task_id" in resp.json()

    def test_mashup_types_returns_eight(self, client: TestClient):
        body = client.get("/api/mashup/types").json()
        assert "types" in body
        assert len(body["types"]) == 8

    def test_mashup_types_have_name_and_description(self, client: TestClient):
        types = client.get("/api/mashup/types").json()["types"]
        for t in types:
            assert "name" in t
            assert "description" in t

    def test_all_eight_mashup_type_names_present(self, client: TestClient):
        names = {t["name"] for t in client.get("/api/mashup/types").json()["types"]}
        expected = {
            "classic", "stem_swap", "energy_match", "adaptive_harmony",
            "theme_fusion", "semantic_aligned", "role_aware", "conversational",
        }
        assert names == expected


# ═════════════════════════════════════════════════════════════════════════════
# Video router  /api/video
# ═════════════════════════════════════════════════════════════════════════════


class TestVideoRouter:
    @staticmethod
    def _real_analysis_id(client: TestClient) -> str:
        """Upload a minimal WAV, run basic analysis, return audio_id.

        The audio analysis BackgroundTask runs synchronously in TestClient,
        so the JSON cache is written before this method returns.
        """
        upload_resp = client.post(
            "/api/audio/upload",
            files={"file": ("song.wav", _minimal_wav(), "audio/wav")},
        )
        assert upload_resp.status_code == 201
        audio_id = upload_resp.json()["audio_id"]
        analyze_resp = client.post(
            "/api/audio/analyze",
            json={"audio_id": audio_id, "depth": "basic"},
        )
        assert analyze_resp.status_code == 200
        return audio_id

    def test_plan_video_returns_plan_id(self, client: TestClient):
        audio_id = self._real_analysis_id(client)
        resp = client.post(
            "/api/video/plan",
            json={"audio_id": audio_id, "style": "cinematic", "quality": "professional"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "plan_id" in body
        assert "backend" in body

    def test_plan_has_cost_and_time(self, client: TestClient):
        audio_id = self._real_analysis_id(client)
        body = client.post(
            "/api/video/plan",
            json={"audio_id": audio_id},
        ).json()
        assert "estimated_cost_usd" in body
        assert "estimated_time_sec" in body
        assert "is_local" in body

    def test_generate_video_accepted(self, client: TestClient):
        resp = client.post(
            "/api/video/generate",
            json={"plan_id": "plan-001", "audio_id": "song-001"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert "status" in body

    def test_edit_scene_accepted(self, client: TestClient):
        resp = client.post(
            "/api/video/scene/edit",
            json={"video_id": "vid-001", "scene_index": 3, "new_prompt": "neon city at night"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert body["scene_index"] == 3

    def test_list_styles_returns_dict(self, client: TestClient):
        body = client.get("/api/video/styles").json()
        assert "styles" in body
        assert "total" in body

    def test_list_backends_returns_list_key(self, client: TestClient):
        body = client.get("/api/video/backends").json()
        assert "backends" in body

    def test_download_video_returns_id(self, client: TestClient):
        body = client.get("/api/video/download/vid-xyz").json()
        assert body["video_id"] == "vid-xyz"


# ═════════════════════════════════════════════════════════════════════════════
# LoRA router  /api/lora
# ═════════════════════════════════════════════════════════════════════════════


class TestLoRARouter:
    def test_list_loras_returns_loras_key(self, client: TestClient):
        body = client.get("/api/lora/list").json()
        assert "loras" in body
        assert isinstance(body["loras"], list)

    def test_recommend_loras_returns_available_key(self, client: TestClient):
        resp = client.post(
            "/api/lora/recommend",
            json={"audio_id": "song-001", "style": "cinematic"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Response has available/downloadable/trainable (not "recommendations")
        assert "available" in body
        assert "downloadable" in body
        assert "trainable" in body

    def test_train_lora_accepted(self, client: TestClient):
        resp = client.post(
            "/api/lora/train",
            json={
                "dataset_path": "/tmp/dataset",
                "lora_type": "style",
                "trigger_token": "mystyle",
                "output_name": "my-style-lora",
                "training_steps": 1500,
            },
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert body["output_name"] == "my-style-lora"

    def test_download_lora_accepted(self, client: TestClient):
        resp = client.post(
            "/api/lora/download",
            json={
                "url": "https://huggingface.co/example/lora",
                "name": "test-lora",
                "lora_type": "style",
                "trigger_token": "testlora",
            },
        )
        assert resp.status_code == 202
        assert "task_id" in resp.json()

    def test_register_lora_created(self, client: TestClient):
        resp = client.post(
            "/api/lora/register",
            json={
                "name": "local-lora",
                "lora_type": "scene",
                "trigger_token": "locallora",
                "file_path": "/models/loras/local.safetensors",
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["name"] == "local-lora"
        assert body["status"] == "registered"

    def test_delete_lora_no_content(self, client: TestClient):
        resp = client.delete("/api/lora/old-lora")
        assert resp.status_code == 204


# ═════════════════════════════════════════════════════════════════════════════
# Nova Fade router  /api/nova-fade
# ═════════════════════════════════════════════════════════════════════════════


class TestNovaFadeRouter:
    def test_generate_canonical_accepted(self, client: TestClient):
        resp = client.post(
            "/api/nova-fade/generate-canonical",
            json={"num_images": 20},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert "status" in body

    def test_train_identity_lora_accepted(self, client: TestClient):
        resp = client.post("/api/nova-fade/train-identity-lora")
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert "lora" in body

    def test_train_style_lora_accepted(self, client: TestClient):
        resp = client.post("/api/nova-fade/train-style-lora")
        assert resp.status_code == 202
        assert "task_id" in resp.json()

    def test_drift_test_returns_lora_path(self, client: TestClient):
        resp = client.post(
            "/api/nova-fade/drift-test",
            json={"lora_path": "/output/loras/novafade_id_v1.safetensors"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "task_id" in body
        assert "lora_path" in body

    def test_dj_video_accepted(self, client: TestClient):
        resp = client.post(
            "/api/nova-fade/dj-video",
            json={"mashup_id": "mashup-001", "theme": "sponsor_neon"},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert body["theme"] == "sponsor_neon"

    def test_nova_fade_status_has_constitution_version(self, client: TestClient):
        resp = client.get("/api/nova-fade/status")
        assert resp.status_code == 200
        body = resp.json()
        # Specific fields from the stub response
        assert "constitution_version" in body
        assert "identity_lora" in body


# ═════════════════════════════════════════════════════════════════════════════
# Task router  /api/tasks
# ═════════════════════════════════════════════════════════════════════════════


class TestTaskRouter:
    @staticmethod
    def _real_task_id(client: TestClient) -> str:
        """Upload a WAV and start analysis to get a real task_id."""
        upload = client.post(
            "/api/audio/upload",
            files={"file": ("song.wav", _minimal_wav(), "audio/wav")},
        )
        audio_id = upload.json()["audio_id"]
        analyze = client.post(
            "/api/audio/analyze",
            json={"audio_id": audio_id, "depth": "basic"},
        )
        return analyze.json()["task_id"]

    def test_get_task_returns_status(self, client: TestClient):
        task_id = self._real_task_id(client)
        resp = client.get(f"/api/tasks/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert "task_id" in body
        assert "status" in body

    def test_get_task_echoes_id(self, client: TestClient):
        task_id = self._real_task_id(client)
        body = client.get(f"/api/tasks/{task_id}").json()
        assert body["task_id"] == task_id

    def test_get_unknown_task_is_404(self, client: TestClient):
        resp = client.get("/api/tasks/no-such-task-xyz")
        assert resp.status_code == 404

    def test_list_active_tasks_returns_tasks(self, client: TestClient):
        resp = client.get("/api/tasks/")
        assert resp.status_code == 200
        body = resp.json()
        assert "tasks" in body
        assert isinstance(body["tasks"], list)

    def test_cancel_task_no_content(self, client: TestClient):
        task_id = self._real_task_id(client)
        resp = client.delete(f"/api/tasks/{task_id}")
        assert resp.status_code == 204

    def test_cancel_unknown_task_is_404(self, client: TestClient):
        resp = client.delete("/api/tasks/no-such-task-xyz")
        assert resp.status_code == 404


# ═════════════════════════════════════════════════════════════════════════════
# WebSocket  /api/tasks/ws/{task_id}
# ═════════════════════════════════════════════════════════════════════════════


class TestWebSocket:
    def test_websocket_connects(self, client: TestClient):
        with client.websocket_connect("/api/tasks/ws/task-001") as ws:
            # Connection established — no exception = success
            pass

    def test_websocket_receives_json(self, client: TestClient):
        with client.websocket_connect("/api/tasks/ws/task-001") as ws:
            data = ws.receive_json()
            assert isinstance(data, dict)

    def test_websocket_message_has_task_id(self, client: TestClient):
        with client.websocket_connect("/api/tasks/ws/task-001") as ws:
            data = ws.receive_json()
            assert data["task_id"] == "task-001"

    def test_websocket_message_has_status(self, client: TestClient):
        with client.websocket_connect("/api/tasks/ws/task-001") as ws:
            data = ws.receive_json()
            assert "status" in data


# ═════════════════════════════════════════════════════════════════════════════
# Full pipeline chain: audio → mashup → video
# ═════════════════════════════════════════════════════════════════════════════


class TestFullPipelineChain:
    """Simulate the two primary user workflows end-to-end through the API."""

    def test_path_a_original_song_to_video(self, client: TestClient):
        """Path A: Upload audio → analyze → plan video → generate."""
        # Step 1: upload (use a valid WAV so audio analysis writes the JSON cache)
        upload_resp = client.post(
            "/api/audio/upload",
            files={"file": ("song.wav", _minimal_wav(), "audio/wav")},
        )
        assert upload_resp.status_code == 201
        audio_id = upload_resp.json()["audio_id"]

        # Step 2: analyze
        analyze_resp = client.post(
            "/api/audio/analyze",
            json={"audio_id": audio_id, "depth": "full"},
        )
        assert analyze_resp.status_code == 200
        assert analyze_resp.json()["audio_id"] == audio_id

        # Step 3: plan video
        plan_resp = client.post(
            "/api/video/plan",
            json={"audio_id": audio_id, "style": "cinematic", "quality": "professional"},
        )
        assert plan_resp.status_code == 200
        plan_id = plan_resp.json()["plan_id"]
        assert plan_id

        # Step 4: generate
        gen_resp = client.post(
            "/api/video/generate",
            json={"plan_id": plan_id, "audio_id": audio_id},
        )
        assert gen_resp.status_code == 202

    def test_path_b_mashup_to_video(self, client: TestClient):
        """Path B: Ingest songs → create mashup → plan video."""
        # Step 1: ingest two songs
        ingest_a = client.post("/api/mashup/ingest", json={"source": "/tmp/song_a.mp3"})
        ingest_b = client.post("/api/mashup/ingest", json={"source": "/tmp/song_b.mp3"})
        assert ingest_a.status_code == 202
        assert ingest_b.status_code == 202

        # Step 2: create mashup
        mashup_resp = client.post(
            "/api/mashup/create",
            json={"song_a_id": "song-a-001", "song_b_id": "song-b-002", "mashup_type": "classic"},
        )
        assert mashup_resp.status_code == 202
        task_id = mashup_resp.json()["task_id"]

        # Step 3: poll task (stub returns immediately)
        task_resp = client.get(f"/api/tasks/{task_id}")
        assert task_resp.status_code == 200
        assert "status" in task_resp.json()

        # Step 4: plan video — use a real analysis id since /plan requires cached JSON
        mashup_audio_id = TestVideoRouter._real_analysis_id(client)
        plan_resp = client.post(
            "/api/video/plan",
            json={"audio_id": mashup_audio_id, "style": "synthwave", "quality": "basic"},
        )
        assert plan_resp.status_code == 200

    def test_health_gate_before_pipeline(self, client: TestClient):
        """Health check must pass before any pipeline work."""
        health = client.get("/api/system/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

    def test_lora_setup_before_video_plan(self, client: TestClient):
        """LoRA recommendations are available before planning a video."""
        resp = client.post(
            "/api/lora/recommend",
            json={"audio_id": "song-001", "style": "anime"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "available" in body

    def test_nova_fade_dj_path(self, client: TestClient):
        """Path B2: Create mashup → generate DJ video."""
        # Step 1: create mashup
        mashup_resp = client.post(
            "/api/mashup/create",
            json={"song_a_id": "song-x", "song_b_id": "song-y", "mashup_type": "conversational"},
        )
        assert mashup_resp.status_code == 202

        # Step 2: generate DJ video
        dj_resp = client.post(
            "/api/nova-fade/dj-video",
            json={"mashup_id": "mashup-xyz", "theme": "mashup_chaos"},
        )
        assert dj_resp.status_code == 202
        assert "task_id" in dj_resp.json()
