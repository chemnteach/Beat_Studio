"""Shared data types for the Nova Fade character pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ComposedScenePrompt:
    """A fully-assembled positive + negative prompt pair for a Nova Fade scene."""
    positive: str
    negative: str
    expression: str = ""
    gesture: str = ""
