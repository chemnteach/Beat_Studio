"""Persistent SQLite-backed task manager for Beat Studio.

Improvement over BeatCanvas's ``active_tasks = {}`` (in-memory, lost on restart).
All task state survives server restarts.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskState:
    task_id: str
    task_type: str
    status: TaskStatus
    params: Dict[str, Any]
    stage: str = ""
    percent: float = 0.0
    message: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


class TaskManager:
    """SQLite-backed task manager.

    Creates the database and table on first use.  Thread-safe via the
    ``check_same_thread=False`` SQLite flag (suitable for FastAPI's async
    thread pool workers — each connection is created per-operation).
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id   TEXT PRIMARY KEY,
        task_type TEXT NOT NULL,
        status    TEXT NOT NULL DEFAULT 'pending',
        params    TEXT NOT NULL DEFAULT '{}',
        stage     TEXT NOT NULL DEFAULT '',
        percent   REAL NOT NULL DEFAULT 0.0,
        message   TEXT NOT NULL DEFAULT '',
        result    TEXT NOT NULL DEFAULT '{}',
        error     TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    )
    """

    def __init__(self, db_path: str = "backend/tasks.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── private ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(self._CREATE_TABLE)

    @staticmethod
    def _row_to_state(row: sqlite3.Row) -> TaskState:
        return TaskState(
            task_id=row["task_id"],
            task_type=row["task_type"],
            status=TaskStatus(row["status"]),
            params=json.loads(row["params"]),
            stage=row["stage"],
            percent=row["percent"],
            message=row["message"],
            result=json.loads(row["result"]),
            error=row["error"],
        )

    # ── public ───────────────────────────────────────────────────────────────

    def create_task(self, task_type: str, params: Dict[str, Any]) -> str:
        """Create a new task and return its ID."""
        import time
        task_id = str(uuid.uuid4())
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO tasks
                   (task_id, task_type, status, params, created_at, updated_at)
                   VALUES (?, ?, 'pending', ?, ?, ?)""",
                (task_id, task_type, json.dumps(params), now, now),
            )
        return task_id

    def update_progress(
        self,
        task_id: str,
        stage: str,
        percent: float,
        message: str,
    ) -> None:
        """Update task progress and set status to RUNNING."""
        import time
        with self._connect() as conn:
            conn.execute(
                """UPDATE tasks
                   SET status='running', stage=?, percent=?, message=?, updated_at=?
                   WHERE task_id=?""",
                (stage, percent, message, time.time(), task_id),
            )

    def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """Mark task as complete and store result."""
        import time
        with self._connect() as conn:
            conn.execute(
                """UPDATE tasks
                   SET status='complete', percent=100.0, result=?, updated_at=?
                   WHERE task_id=?""",
                (json.dumps(result), time.time(), task_id),
            )

    def fail_task(self, task_id: str, error: str) -> None:
        """Mark task as failed and store error message."""
        import time
        with self._connect() as conn:
            conn.execute(
                """UPDATE tasks
                   SET status='failed', error=?, updated_at=?
                   WHERE task_id=?""",
                (error, time.time(), task_id),
            )

    def cancel_task(self, task_id: str) -> None:
        """Cancel a task."""
        import time
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET status='cancelled', updated_at=? WHERE task_id=?",
                (time.time(), task_id),
            )

    def get_status(self, task_id: str) -> Optional[TaskState]:
        """Return current task state, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_state(row)

    def list_active(self) -> List[TaskState]:
        """Return all tasks that are pending or running."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE status IN ('pending', 'running') ORDER BY created_at"
            ).fetchall()
        return [self._row_to_state(r) for r in rows]
