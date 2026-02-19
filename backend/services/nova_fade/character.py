"""NovaFadeCharacter — programmatic enforcement of the Nova Fade Constitution v1.0."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger("beat_studio.nova_fade.character")


@dataclass
class ValidationResult:
    """Result of validating a prompt against the Nova Fade constitution."""
    valid: bool
    violations: List[str] = field(default_factory=list)


class NovaFadeCharacter:
    """Enforces Nova Fade Character Constitution v1.0.

    Immutable class-level constants define the character's allowed and
    forbidden attributes. Use ``validate_prompt()`` before sending any
    prompt to a generation backend.

    Usage::

        char = NovaFadeCharacter()
        result = char.validate_prompt("nova_fade_char DJ in studio")
        if not result.valid:
            raise ValueError(result.violations)

        pos, neg = char.get_canonical_header(expression="focused_intensity")
    """

    # ── Identity constants ─────────────────────────────────────────────────────

    EXPRESSIONS: List[str] = [
        "neutral_confident",
        "mischievous_grin",
        "focused_intensity",
        "delighted_joy",
        "drop_anticipation",
    ]

    ALLOWED_GESTURES: List[str] = [
        "left_deck_scratch",
        "right_deck_scratch",
        "crossfader_tap",
        "downbeat_head_nod",
        "spotlight_presentation",
    ]

    FORBIDDEN: List[str] = [
        "photorealistic",
        "realistic human skin",
        "anime",
        "manga",
        "age change",
        "different hairstyle",
        "different face",
        "acrobatics",
        "running",
        "aggressive dance",
        "physical combat",
    ]

    # ── Canonical prompt fragments ─────────────────────────────────────────────

    _POSITIVE_BASE = (
        "nova_fade_char, 3D cartoon stylized DJ woman, vibrant purple hair, "
        "silver geometric earrings, confident posture, DJ booth, professional studio"
    )

    _NEGATIVE_BASE = (
        "photorealistic, realistic human skin, anime, manga, different face, "
        "different hairstyle, age change, acrobatics, running, aggressive, combat, "
        "low quality, blurry, deformed, extra limbs, watermark, text"
    )

    _EXPRESSION_PHRASES = {
        "neutral_confident": "neutral confident expression, slight smile, relaxed",
        "mischievous_grin": "mischievous grin, playful smirk, knowing look",
        "focused_intensity": "focused intensity expression, brow furrowed slightly, in the zone",
        "delighted_joy": "delighted joy expression, wide smile, eyes lit up, elated",
        "drop_anticipation": "drop anticipation expression, building excitement, leaning in",
    }

    _GESTURE_PHRASES = {
        "left_deck_scratch": "left hand on left turntable deck, scratching vinyl record",
        "right_deck_scratch": "right hand on right turntable deck, scratching vinyl record",
        "crossfader_tap": "finger on crossfader, precision tap gesture",
        "downbeat_head_nod": "head nodding on the downbeat, feeling the rhythm",
        "spotlight_presentation": "arms raised, presenting to the crowd, spotlight moment",
    }

    # ── Public API ─────────────────────────────────────────────────────────────

    def validate_prompt(self, prompt: str) -> ValidationResult:
        """Check ``prompt`` against constitution constraints.

        Args:
            prompt: The raw text prompt to validate.

        Returns:
            :class:`ValidationResult` with ``valid=True`` and empty violations
            list if the prompt is clean, or ``valid=False`` with a list of
            violation descriptions.
        """
        prompt_lower = prompt.lower()
        violations: List[str] = []

        for forbidden in self.FORBIDDEN:
            if forbidden.lower() in prompt_lower:
                violations.append(
                    f"Forbidden term detected: '{forbidden}'"
                )

        return ValidationResult(valid=len(violations) == 0, violations=violations)

    def get_canonical_header(
        self,
        expression: Optional[str] = None,
        gesture: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Return the locked positive + negative prompt header for Nova Fade.

        Args:
            expression: One of :attr:`EXPRESSIONS`, or ``None`` for base header.
            gesture: One of :attr:`ALLOWED_GESTURES`, or ``None``.

        Returns:
            ``(positive_header, negative_header)`` — strings to prepend to any
            Nova Fade scene prompt.
        """
        parts = [self._POSITIVE_BASE]

        if expression and expression in self._EXPRESSION_PHRASES:
            parts.append(self._EXPRESSION_PHRASES[expression])
        elif expression:
            logger.warning("Unknown expression '%s' — using base header.", expression)

        if gesture and gesture in self._GESTURE_PHRASES:
            parts.append(self._GESTURE_PHRASES[gesture])
        elif gesture:
            logger.warning("Unknown gesture '%s' — ignoring.", gesture)

        positive = ", ".join(parts)
        negative = self._NEGATIVE_BASE
        return positive, negative
