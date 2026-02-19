"""Data types for the Beat Studio video generation engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class VideoClip:
    """A generated video clip ready for assembly."""
    file_path: str
    duration_sec: float
    width: int
    height: int
    fps: int
    scene_index: int
    backend_used: str = ""
    generation_time_sec: float = 0.0
    cost_usd: float = 0.0


@dataclass
class ScenePlan:
    """Execution plan for a single scene."""
    scene_index: int
    backend_name: str
    estimated_time_sec: float
    estimated_cost_usd: float
    resolution: Tuple[int, int] = (1920, 1080)
    notes: str = ""


@dataclass
class ExecutionPlan:
    """Complete video generation execution plan returned by ModelRouter."""
    scenes: List[ScenePlan]
    total_time_sec: float
    total_cost_usd: float
    primary_backend: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_local(self) -> bool:
        return self.total_cost_usd == 0.0
