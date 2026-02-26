"""Data types for the storyboard keyframe preview service."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SceneInput:
    """Input-only DTO: what the caller provides per scene for generation."""
    scene_idx: int
    storyboard_prompt: str     # Cinematic description (used for context; not sent to diffusion)
    positive_prompt: str       # Composed style prompt ready for diffusion


@dataclass
class VersionEntry:
    """One generated keyframe image for a scene."""
    version: int           # 1-indexed; v1 is the first generated image
    filename: str          # e.g. "v1.png" — relative to scene dir
    seed: int
    timestamp: str         # ISO-8601


@dataclass
class StoryboardScene:
    """Per-scene state: prompt + version history + approval."""
    scene_idx: int
    storyboard_prompt: str            # The cinematic description used for generation
    positive_prompt: str              # The composed positive prompt (style prefix + scene)
    approved_version: Optional[int]   # None until approved; 1-indexed
    versions: List[VersionEntry] = field(default_factory=list)

    # Max versions kept on disk before eviction (oldest dropped)
    MAX_VERSIONS: int = 5


@dataclass
class StoryboardState:
    """Full state for one storyboard session."""
    storyboard_id: str             # UUID assigned at generation time
    style: str                     # Style name e.g. "cinematic"
    base_checkpoint: str           # HF model ID from style
    lora_names: List[str]          # LoRAs applied during generation
    status: str                    # "generating" | "complete" | "failed"
    scenes: List[StoryboardScene] = field(default_factory=list)
    error: Optional[str] = None
