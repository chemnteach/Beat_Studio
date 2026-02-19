"""CanonicalPrompts — the locked Nova Fade prompt library.

All expression and gesture prompts are built from the NovaFadeCharacter
constitution. This module provides a convenient composition interface so
callers never have to assemble prompts from scratch.
"""
from __future__ import annotations

import logging
from typing import List

from backend.services.nova_fade.character import NovaFadeCharacter
from backend.services.nova_fade.types import ComposedScenePrompt

logger = logging.getLogger("beat_studio.nova_fade.canonical_prompts")


class CanonicalPrompts:
    """Locked prompt library for Nova Fade scenes.

    Usage::

        cp = CanonicalPrompts()
        pos, neg = cp.get_expression_prompt("focused_intensity")
        prompt = cp.compose_scene_prompt(
            scene_description="DJ under neon lights",
            expression="mischievous_grin",
            gesture="crossfader_tap",
            character=NovaFadeCharacter(),
        )
    """

    def __init__(self) -> None:
        self._char = NovaFadeCharacter()

    # ── Expression prompts ─────────────────────────────────────────────────────

    def list_expressions(self) -> List[str]:
        """Return all valid expression names."""
        return list(NovaFadeCharacter.EXPRESSIONS)

    def get_expression_prompt(self, expression: str) -> tuple[str, str]:
        """Return ``(positive, negative)`` prompt pair for an expression.

        Args:
            expression: Must be one of :attr:`NovaFadeCharacter.EXPRESSIONS`.

        Raises:
            KeyError: If ``expression`` is not in the known list.
        """
        if expression not in NovaFadeCharacter.EXPRESSIONS:
            raise KeyError(f"Unknown expression: '{expression}'. "
                           f"Valid: {NovaFadeCharacter.EXPRESSIONS}")
        return self._char.get_canonical_header(expression=expression)

    # ── Gesture prompts ────────────────────────────────────────────────────────

    def list_gestures(self) -> List[str]:
        """Return all valid gesture names."""
        return list(NovaFadeCharacter.ALLOWED_GESTURES)

    def get_gesture_prompt(self, gesture: str) -> tuple[str, str]:
        """Return ``(positive, negative)`` prompt pair for a gesture.

        Args:
            gesture: Must be one of :attr:`NovaFadeCharacter.ALLOWED_GESTURES`.

        Raises:
            KeyError: If ``gesture`` is not in the known list.
        """
        if gesture not in NovaFadeCharacter.ALLOWED_GESTURES:
            raise KeyError(f"Unknown gesture: '{gesture}'. "
                           f"Valid: {NovaFadeCharacter.ALLOWED_GESTURES}")
        return self._char.get_canonical_header(gesture=gesture)

    # ── Scene composition ──────────────────────────────────────────────────────

    def compose_scene_prompt(
        self,
        scene_description: str,
        expression: str,
        gesture: str,
        character: NovaFadeCharacter,
    ) -> ComposedScenePrompt:
        """Compose a fully-assembled scene prompt.

        Combines:
        1. Canonical header (positive + negative from constitution)
        2. Scene description
        3. Expression phrase
        4. Gesture phrase

        Args:
            scene_description: Free-text description of the scene.
            expression: One of :attr:`NovaFadeCharacter.EXPRESSIONS`.
            gesture: One of :attr:`NovaFadeCharacter.ALLOWED_GESTURES`.
            character: :class:`NovaFadeCharacter` instance for constitution access.

        Returns:
            :class:`ComposedScenePrompt` ready to send to a video backend.
        """
        if expression not in NovaFadeCharacter.EXPRESSIONS:
            raise KeyError(f"Unknown expression: '{expression}'")
        if gesture not in NovaFadeCharacter.ALLOWED_GESTURES:
            raise KeyError(f"Unknown gesture: '{gesture}'")

        pos_header, neg_header = character.get_canonical_header(
            expression=expression, gesture=gesture
        )
        positive = f"{pos_header}, {scene_description}"
        negative = neg_header

        return ComposedScenePrompt(
            positive=positive,
            negative=negative,
            expression=expression,
            gesture=gesture,
        )
