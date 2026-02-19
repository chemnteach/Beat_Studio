"""Data types for the Beat_Studio prompt generation engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NarrativeSection:
    """Visual direction for one section of the song."""
    section_index: int
    section_type: str          # "intro" | "verse" | "chorus" | "bridge" | "outro"
    start_sec: float
    end_sec: float
    visual_description: str    # Scene description for video generation
    emotional_tone: str        # "hopeful" | "melancholic" | "triumphant" | etc.
    energy_level: float        # 0.0-1.0
    is_climax: bool            # True for peak narrative moments
    key_lyric: str             # The most visually evocative lyric in this section
    themes: List[str]          # Visual themes: ["urban", "night", "rain"]


@dataclass
class NarrativeArc:
    """Complete visual narrative extracted from a song."""
    artist: str
    title: str
    overall_concept: str       # One-sentence description of the visual narrative
    color_palette: List[str]   # Dominant visual colors (e.g., ["deep blue", "gold"])
    mood_progression: str      # e.g., "starts hopeful → builds tension → cathartic release"
    visual_style_hint: str     # e.g., "cinematic, dramatic lighting, rain-soaked streets"
    sections: List[NarrativeSection]


@dataclass
class AnimationStyle:
    """Video generation style configuration."""
    name: str                  # "cinematic" | "anime" | "lofi" | "abstract" | "photorealistic"
    prefix: str                # Prompt prefix e.g. "cinematic film still, 35mm, "
    negative_prefix: str       # Style-specific negatives
    recommended_model: str     # "animatediff" | "wan26" | "sdxl_controlnet" | etc.
    cfg_scale: float           # 7.0 default
    steps: int                 # 20 default


@dataclass
class LoRAConfig:
    """LoRA configuration for prompt assembly."""
    name: str
    trigger_token: str
    weight: float              # 0.0-1.0
    lora_type: str             # "style" | "character" | "scene" | "identity"


@dataclass
class CinematographyProfile:
    """Cinematography-specific prompt tokens (from BeatCanvas CinematographyEngine)."""
    camera_movement: str       # "slow pan", "tracking shot", "static", etc.
    lighting: str              # "golden hour", "neon-lit", "overcast", etc.
    film_stock: str            # "35mm film grain", "8mm", "digital", etc.
    lens: str                  # "wide angle", "telephoto", "macro", etc.


@dataclass
class ScenePrompt:
    """Video generation prompt for a single scene."""
    scene_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    is_hero: bool
    energy_level: float
    positive: str              # Full positive prompt
    negative: str              # Full negative prompt
    style: str                 # Style name
    lora_names: List[str]      # Applied LoRAs
    transition_hint: str       # "cut" | "dissolve" | "fade" | "match_cut"
    cfg_scale: float
    steps: int


@dataclass
class ComposedPrompt:
    """Final assembled prompt ready for video generation."""
    positive: str
    negative: str
    cfg_scale: float
    steps: int
    model: str                 # Which backend to use
    nsfw: bool
