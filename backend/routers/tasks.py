"""Task router — background task state, polling, and WebSocket progress."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from backend.services.shared.task_manager import TaskManager, TaskState

logger = logging.getLogger("beat_studio.routers.tasks")
router = APIRouter()

# ── Paths / singletons ────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent.parent
_task_manager: Optional[TaskManager] = None

_WS_POLL_INTERVAL = 0.5   # seconds between DB polls for WebSocket updates
_WS_TIMEOUT       = 3600  # max WebSocket session duration (1 hour)


def _get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager(db_path=str(_BACKEND_DIR / "tasks.db"))
    return _task_manager


# ── Helpers ───────────────────────────────────────────────────────────────────

# TaskManager stores "complete"; frontend expects "completed"
_STATUS_MAP = {"complete": "completed"}

_TERMINAL_STATUSES = {"complete", "failed", "cancelled"}


def _state_to_dict(state: TaskState) -> Dict[str, Any]:
    raw_status = state.status.value
    return {
        "task_id":   state.task_id,
        "task_type": state.task_type,
        "status":    _STATUS_MAP.get(raw_status, raw_status),
        "progress":  state.percent,
        "stage":     state.stage,
        "message":   state.message,
        "result":    state.result,
        "error":     state.error,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

# NOTE: GET "/" must be defined before GET "/{task_id}" so FastAPI doesn't
# swallow the empty path as a task_id.

@router.get("/")
async def list_active_tasks() -> Dict[str, Any]:
    """List all pending and running tasks."""
    tasks = _get_task_manager().list_active()
    return {"tasks": [_state_to_dict(t) for t in tasks], "total": len(tasks)}


@router.get("/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """Return the current state of a background task.

    Status values: ``pending`` | ``running`` | ``completed`` | ``failed`` | ``cancelled``
    """
    state = _get_task_manager().get_status(task_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id!r} not found.",
        )
    return _state_to_dict(state)


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_task(task_id: str) -> None:
    """Cancel a pending or running task."""
    tm = _get_task_manager()
    state = tm.get_status(task_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id!r} not found.",
        )
    tm.cancel_task(task_id)


@router.websocket("/ws/{task_id}")
async def task_progress_ws(websocket: WebSocket, task_id: str) -> None:
    """Stream task progress over WebSocket.

    Polls the TaskManager every 500 ms and pushes ``ProgressEvent`` JSON:
    ``{task_id, progress, stage, message, status}``.

    Closes automatically when the task reaches a terminal state
    (completed / failed / cancelled) or after 1 hour.
    """
    await websocket.accept()
    tm = _get_task_manager()
    elapsed = 0.0

    try:
        while elapsed < _WS_TIMEOUT:
            state = tm.get_status(task_id)

            if state is None:
                await websocket.send_json({
                    "task_id": task_id,
                    "progress": 0,
                    "stage": "error",
                    "message": f"Task {task_id!r} not found.",
                    "status": "failed",
                })
                break

            raw_status = state.status.value
            await websocket.send_json({
                "task_id":  task_id,
                "progress": state.percent,
                "stage":    state.stage,
                "message":  state.message,
                "status":   _STATUS_MAP.get(raw_status, raw_status),
            })

            if raw_status in _TERMINAL_STATUSES:
                break

            await asyncio.sleep(_WS_POLL_INTERVAL)
            elapsed += _WS_POLL_INTERVAL

        await websocket.close()

    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected for task_id=%s", task_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("WebSocket error for task_id=%s: %s", task_id, exc)
