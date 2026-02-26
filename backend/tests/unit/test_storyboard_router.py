"""TDD tests for the storyboard router — written before the implementation.

Uses FastAPI TestClient with monkeypatched service + task manager so no GPU
or real filesystem I/O is needed.

Background tasks run synchronously inside TestClient, so service mock
assertions are valid immediately after the HTTP call.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.storyboard.types import (
    StoryboardScene,
    StoryboardState,
    VersionEntry,
)
from backend.services.storyboard.service import StoryboardService


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_state(storyboard_id: str = "sb-123", n_scenes: int = 2) -> StoryboardState:
    scenes = []
    for i in range(n_scenes):
        scenes.append(StoryboardScene(
            scene_idx=i,
            storyboard_prompt=f"Scene {i}: dramatic ocean",
            positive_prompt=f"cinematic film still, ocean, scene {i}",
            approved_version=None,
            versions=[
                VersionEntry(version=1, filename="v1.png", seed=i * 137, timestamp=_ts())
            ],
        ))
    return StoryboardState(
        storyboard_id=storyboard_id,
        style="cinematic",
        base_checkpoint=StoryboardService.SDXL_BASE,
        lora_names=[],
        status="complete",
        scenes=scenes,
    )


@pytest.fixture
def mock_svc(monkeypatch):
    svc = MagicMock(spec=StoryboardService)
    svc.get_state.return_value = None       # default: not found
    svc.approve.return_value = {}
    monkeypatch.setattr("backend.routers.storyboard._get_service", lambda: svc)
    return svc


@pytest.fixture
def mock_tm(monkeypatch):
    tm = MagicMock()
    tm.create_task.return_value = "task-abc-123"
    monkeypatch.setattr("backend.routers.storyboard._get_task_manager", lambda: tm)
    return tm


@pytest.fixture
def client(mock_svc, mock_tm):
    return TestClient(app)


def _generate_payload(n_scenes: int = 2) -> dict:
    return {
        "style": "cinematic",
        "lora_names": [],
        "scenes": [
            {
                "scene_idx": i,
                "storyboard_prompt": f"Scene {i}: dramatic ocean",
                "positive_prompt": f"cinematic film still, ocean {i}",
            }
            for i in range(n_scenes)
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# POST /generate
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateEndpoint:
    def test_returns_202(self, client):
        resp = client.post("/api/video/storyboard/generate", json=_generate_payload())
        assert resp.status_code == 202

    def test_response_has_task_id(self, client, mock_tm):
        resp = client.post("/api/video/storyboard/generate", json=_generate_payload())
        assert resp.json()["task_id"] == "task-abc-123"

    def test_response_has_storyboard_id(self, client):
        resp = client.post("/api/video/storyboard/generate", json=_generate_payload())
        assert "storyboard_id" in resp.json()
        assert len(resp.json()["storyboard_id"]) > 0  # non-empty UUID

    def test_response_status_is_queued(self, client):
        resp = client.post("/api/video/storyboard/generate", json=_generate_payload())
        assert resp.json()["status"] == "queued"

    def test_scene_count_in_response(self, client):
        resp = client.post("/api/video/storyboard/generate", json=_generate_payload(3))
        assert resp.json()["scene_count"] == 3

    def test_missing_style_returns_422(self, client):
        payload = _generate_payload()
        del payload["style"]
        resp = client.post("/api/video/storyboard/generate", json=payload)
        assert resp.status_code == 422

    def test_missing_scenes_returns_422(self, client):
        resp = client.post("/api/video/storyboard/generate", json={"style": "cinematic"})
        assert resp.status_code == 422

    def test_empty_scenes_returns_422(self, client):
        payload = {"style": "cinematic", "scenes": []}
        resp = client.post("/api/video/storyboard/generate", json=payload)
        assert resp.status_code == 422

    def test_background_task_calls_service_generate(self, client, mock_svc):
        client.post("/api/video/storyboard/generate", json=_generate_payload(2))
        mock_svc.generate_all_scenes.assert_called_once()

    def test_background_task_passes_style_and_lora_names(self, client, mock_svc):
        payload = _generate_payload(1)
        payload["lora_names"] = ["rob-character"]
        client.post("/api/video/storyboard/generate", json=payload)

        _, call_args, _ = mock_svc.generate_all_scenes.mock_calls[0]
        # generate_all_scenes(storyboard_id, scenes, style, lora_names)
        assert call_args[2] == "cinematic"
        assert call_args[3] == ["rob-character"]

    def test_task_created_with_generate_storyboard_type(self, client, mock_tm):
        client.post("/api/video/storyboard/generate", json=_generate_payload())
        mock_tm.create_task.assert_called_once()
        call_args = mock_tm.create_task.call_args[0]
        assert call_args[0] == "generate_storyboard"

    def test_background_task_completes_task_on_success(self, client, mock_tm, mock_svc):
        mock_svc.generate_all_scenes.return_value = None
        client.post("/api/video/storyboard/generate", json=_generate_payload(2))
        mock_tm.complete_task.assert_called_once()

    def test_background_task_fails_task_on_exception(self, client, mock_tm, mock_svc):
        mock_svc.generate_all_scenes.side_effect = RuntimeError("GPU OOM")
        client.post("/api/video/storyboard/generate", json=_generate_payload(1))
        mock_tm.fail_task.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# GET /{storyboard_id}/images
# ═══════════════════════════════════════════════════════════════════════════════


class TestImagesEndpoint:
    def test_returns_404_for_unknown_storyboard(self, client, mock_svc):
        mock_svc.get_state.return_value = None
        resp = client.get("/api/video/storyboard/does-not-exist/images")
        assert resp.status_code == 404

    def test_returns_200_for_known_storyboard(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-123")
        resp = client.get("/api/video/storyboard/sb-123/images")
        assert resp.status_code == 200

    def test_response_includes_storyboard_id(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-abc")
        resp = client.get("/api/video/storyboard/sb-abc/images")
        assert resp.json()["storyboard_id"] == "sb-abc"

    def test_response_includes_status(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-123")
        resp = client.get("/api/video/storyboard/sb-123/images")
        assert resp.json()["status"] == "complete"

    def test_response_includes_correct_scene_count(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-123", n_scenes=3)
        resp = client.get("/api/video/storyboard/sb-123/images")
        assert len(resp.json()["scenes"]) == 3

    def test_version_url_is_correct_format(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-url")
        resp = client.get("/api/video/storyboard/sb-url/images")
        url = resp.json()["scenes"][0]["versions"][0]["url"]
        assert url == "/api/video/storyboard/sb-url/img/scene_0/v1.png"

    def test_version_seed_is_included(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-seed")
        resp = client.get("/api/video/storyboard/sb-seed/images")
        seed = resp.json()["scenes"][0]["versions"][0]["seed"]
        assert seed == 0  # scene_idx 0 * 137

    def test_approved_version_none_when_not_approved(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-appr")
        resp = client.get("/api/video/storyboard/sb-appr/images")
        assert resp.json()["scenes"][0]["approved_version"] is None

    def test_approved_version_set_when_approved(self, client, mock_svc):
        state = _make_state("sb-appr2")
        state.scenes[0].approved_version = 1
        mock_svc.get_state.return_value = state
        resp = client.get("/api/video/storyboard/sb-appr2/images")
        assert resp.json()["scenes"][0]["approved_version"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# POST /{storyboard_id}/scene/{scene_idx}/regenerate
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegenerateEndpoint:
    def test_returns_404_for_unknown_storyboard(self, client, mock_svc):
        mock_svc.get_state.return_value = None
        resp = client.post(
            "/api/video/storyboard/no-such-id/scene/0/regenerate", json={}
        )
        assert resp.status_code == 404

    def test_returns_202_for_known_storyboard(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-regen")
        resp = client.post(
            "/api/video/storyboard/sb-regen/scene/0/regenerate", json={}
        )
        assert resp.status_code == 202

    def test_response_has_task_id(self, client, mock_svc, mock_tm):
        mock_svc.get_state.return_value = _make_state("sb-regen2")
        resp = client.post(
            "/api/video/storyboard/sb-regen2/scene/0/regenerate", json={}
        )
        assert resp.json()["task_id"] == "task-abc-123"

    def test_response_includes_scene_idx(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-regen3", n_scenes=3)
        resp = client.post(
            "/api/video/storyboard/sb-regen3/scene/1/regenerate", json={}
        )
        assert resp.json()["scene_idx"] == 1

    def test_background_task_calls_service(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-regen4")
        client.post(
            "/api/video/storyboard/sb-regen4/scene/0/regenerate", json={}
        )
        mock_svc.generate_single_scene.assert_called_once()

    def test_background_task_passes_prompt_override(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-regen5")
        client.post(
            "/api/video/storyboard/sb-regen5/scene/0/regenerate",
            json={"positive_prompt": "moonlit beach"},
        )
        call_kwargs = mock_svc.generate_single_scene.call_args[1]
        assert call_kwargs["prompt_override"] == "moonlit beach"

    def test_background_task_passes_seed(self, client, mock_svc):
        mock_svc.get_state.return_value = _make_state("sb-regen6")
        client.post(
            "/api/video/storyboard/sb-regen6/scene/0/regenerate",
            json={"seed": 42},
        )
        call_kwargs = mock_svc.generate_single_scene.call_args[1]
        assert call_kwargs["seed"] == 42

    def test_background_task_completes_task_on_success(self, client, mock_svc, mock_tm):
        mock_svc.get_state.return_value = _make_state("sb-regen7")
        mock_svc.generate_single_scene.return_value = MagicMock()
        client.post("/api/video/storyboard/sb-regen7/scene/0/regenerate", json={})
        mock_tm.complete_task.assert_called_once()

    def test_background_task_fails_task_on_exception(self, client, mock_svc, mock_tm):
        mock_svc.get_state.return_value = _make_state("sb-regen8")
        mock_svc.generate_single_scene.side_effect = RuntimeError("OOM")
        client.post("/api/video/storyboard/sb-regen8/scene/0/regenerate", json={})
        mock_tm.fail_task.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# POST /{storyboard_id}/approve
# ═══════════════════════════════════════════════════════════════════════════════


class TestApproveEndpoint:
    def test_returns_404_for_unknown_storyboard(self, client, mock_svc):
        mock_svc.approve.side_effect = LookupError("not found")
        resp = client.post(
            "/api/video/storyboard/no-such/approve",
            json={"selections": {"0": 1}},
        )
        assert resp.status_code == 404

    def test_returns_200_on_success(self, client, mock_svc):
        mock_svc.approve.return_value = {0: "/output/storyboard/sb/scene_0/v1.png"}
        resp = client.post(
            "/api/video/storyboard/sb-ok/approve",
            json={"selections": {"0": 1}},
        )
        assert resp.status_code == 200

    def test_response_has_storyboard_id(self, client, mock_svc):
        mock_svc.approve.return_value = {0: "/output/storyboard/sb/scene_0/v1.png"}
        resp = client.post(
            "/api/video/storyboard/sb-ok2/approve",
            json={"selections": {"0": 1}},
        )
        assert resp.json()["storyboard_id"] == "sb-ok2"

    def test_approved_paths_in_response(self, client, mock_svc):
        mock_svc.approve.return_value = {
            0: "/output/storyboard/sb/scene_0/v2.png",
            1: "/output/storyboard/sb/scene_1/v1.png",
        }
        resp = client.post(
            "/api/video/storyboard/sb-paths/approve",
            json={"selections": {"0": 2, "1": 1}},
        )
        paths = resp.json()["approved_paths"]
        assert paths["0"] == "/output/storyboard/sb/scene_0/v2.png"
        assert paths["1"] == "/output/storyboard/sb/scene_1/v1.png"

    def test_string_scene_idx_keys_converted_to_int_for_service(self, client, mock_svc):
        """JSON keys are always strings; router must convert to int before calling service."""
        mock_svc.approve.return_value = {}
        client.post(
            "/api/video/storyboard/sb-int/approve",
            json={"selections": {"0": 1, "2": 3}},
        )
        call_kwargs = mock_svc.approve.call_args[0]
        selections = call_kwargs[1]  # second positional arg
        assert all(isinstance(k, int) for k in selections.keys())

    def test_bad_scene_idx_returns_400(self, client, mock_svc):
        mock_svc.approve.side_effect = ValueError("scene_idx 99 not found")
        resp = client.post(
            "/api/video/storyboard/sb-bad/approve",
            json={"selections": {"99": 1}},
        )
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# GET /{storyboard_id}/img/{scene_dir}/{filename}
# ═══════════════════════════════════════════════════════════════════════════════


class TestServeImageEndpoint:
    def test_returns_404_for_missing_image(self, client):
        resp = client.get(
            "/api/video/storyboard/ghost-id/img/scene_0/v1.png"
        )
        assert resp.status_code == 404
