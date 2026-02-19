"""Tests for the SQLite-backed TaskManager."""
import time
from pathlib import Path

import pytest

from backend.services.shared.task_manager import TaskManager, TaskStatus, TaskState


@pytest.fixture
def task_db(tmp_dir):
    db_path = str(tmp_dir / "tasks.db")
    return TaskManager(db_path=db_path)


class TestTaskCreation:
    def test_create_returns_task_id(self, task_db):
        task_id = task_db.create_task("video_generation", {"style": "watercolor"})
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    def test_create_task_default_status_pending(self, task_db):
        task_id = task_db.create_task("mashup_creation", {})
        state = task_db.get_status(task_id)
        assert state.status == TaskStatus.PENDING

    def test_create_stores_task_type(self, task_db):
        task_id = task_db.create_task("lora_training", {"steps": 1500})
        state = task_db.get_status(task_id)
        assert state.task_type == "lora_training"

    def test_create_stores_params(self, task_db):
        params = {"song_id": "taylor_swift_shake_it_off", "style": "watercolor"}
        task_id = task_db.create_task("video_generation", params)
        state = task_db.get_status(task_id)
        assert state.params["song_id"] == "taylor_swift_shake_it_off"


class TestTaskProgress:
    def test_update_progress(self, task_db):
        task_id = task_db.create_task("video_generation", {})
        task_db.update_progress(task_id, stage="generating_scenes", percent=50.0, message="Halfway there")
        state = task_db.get_status(task_id)
        assert state.percent == 50.0
        assert state.stage == "generating_scenes"
        assert "Halfway there" in state.message

    def test_update_progress_changes_status_to_running(self, task_db):
        task_id = task_db.create_task("video_generation", {})
        task_db.update_progress(task_id, stage="init", percent=0.0, message="Starting")
        state = task_db.get_status(task_id)
        assert state.status == TaskStatus.RUNNING

    def test_mark_complete(self, task_db):
        task_id = task_db.create_task("video_generation", {})
        task_db.complete_task(task_id, result={"output": "video.mp4"})
        state = task_db.get_status(task_id)
        assert state.status == TaskStatus.COMPLETE
        assert state.result["output"] == "video.mp4"
        assert state.percent == 100.0

    def test_mark_failed(self, task_db):
        task_id = task_db.create_task("video_generation", {})
        task_db.fail_task(task_id, error="VRAM out of memory")
        state = task_db.get_status(task_id)
        assert state.status == TaskStatus.FAILED
        assert "VRAM" in state.error


class TestTaskPersistence:
    def test_task_survives_new_instance(self, tmp_dir):
        db_path = str(tmp_dir / "tasks.db")
        mgr1 = TaskManager(db_path=db_path)
        task_id = mgr1.create_task("video_generation", {"key": "value"})
        mgr1.update_progress(task_id, "generating", 25.0, "In progress")

        # New instance — simulates server restart
        mgr2 = TaskManager(db_path=db_path)
        state = mgr2.get_status(task_id)
        assert state.status == TaskStatus.RUNNING
        assert state.percent == 25.0

    def test_get_missing_task_returns_none(self, task_db):
        assert task_db.get_status("nonexistent-id-xyz") is None


class TestActiveTasks:
    def test_list_active_tasks(self, task_db):
        id1 = task_db.create_task("video_generation", {})
        id2 = task_db.create_task("mashup_creation", {})
        task_db.update_progress(id1, "running", 10.0, "Go")
        task_db.complete_task(id2, {})

        active = task_db.list_active()
        active_ids = [t.task_id for t in active]
        assert id1 in active_ids
        assert id2 not in active_ids

    def test_cancel_task(self, task_db):
        task_id = task_db.create_task("video_generation", {})
        task_db.cancel_task(task_id)
        state = task_db.get_status(task_id)
        assert state.status == TaskStatus.CANCELLED
