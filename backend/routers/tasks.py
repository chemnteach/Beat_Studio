"""Task router — background task state, polling, and WebSocket progress."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

router = APIRouter()


@router.get("/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """Get the current status of a background task."""
    return {
        "task_id": task_id,
        "status": "stub",
        "progress": 0,
        "result": None,
        "error": None,
    }


@router.get("/")
async def list_active_tasks() -> Dict[str, Any]:
    """List all currently active background tasks."""
    return {"tasks": [], "total": 0}


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_task(task_id: str) -> None:
    """Cancel a running background task."""
    return None


@router.websocket("/ws/{task_id}")
async def task_progress_ws(websocket: WebSocket, task_id: str) -> None:
    """WebSocket endpoint for real-time task progress updates."""
    await websocket.accept()
    try:
        # Production: stream progress events from task manager
        await websocket.send_json({
            "task_id": task_id,
            "status": "connected",
            "progress": 0,
        })
        await websocket.close()
    except WebSocketDisconnect:
        pass
