"""Data types for the Beat Studio LoRA management system."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LoRAEntry:
    """A single LoRA registration record."""
    name: str
    type: str                       # "character" | "scene" | "style" | "identity"
    trigger_token: str
    file_path: str                  # Relative to lora.base_path in settings
    weight: float = 0.8
    status: str = "available"       # "available" | "missing" | "downloading"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    source: str = "local"           # "local" | "huggingface" | "civitai"
    source_url: Optional[str] = None


@dataclass
class LoRAValidation:
    """Result of validating a LoRA registration."""
    name: str
    valid: bool
    file_exists: bool
    error: Optional[str] = None


@dataclass
class LoRASearchResult:
    """A LoRA found via online search (HuggingFace or Civitai)."""
    name: str
    source: str                     # "huggingface" | "civitai"
    url: str
    type: str
    description: str
    trigger_token: Optional[str] = None
    confidence: float = 0.0         # Relevance score 0.0-1.0


@dataclass
class LoRARecommendation:
    """Recommendations returned for a video project."""
    available: List[LoRAEntry]          # On-disk LoRAs matching the project
    downloadable: List[LoRASearchResult]  # Online LoRAs worth downloading
    trainable: List[str]                # Suggestions for new LoRAs to train


@dataclass
class LoRATrainingConfig:
    """Configuration for training a new LoRA."""
    dataset_path: str
    lora_type: str                  # "character" | "scene" | "style" | "identity"
    trigger_token: str
    output_name: str
    training_steps: int = 1500
    learning_rate: float = 5e-5
    rank: int = 16
    optimizer: str = "adamw8bit"
    resolution: int = 1024


@dataclass
class TrainingResult:
    """Result of a LoRA training run."""
    success: bool
    lora_path: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class DatasetResult:
    """Result of preparing a training dataset."""
    success: bool
    image_count: int
    caption_count: int
    dataset_path: str
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
