"""PromptComposer — assembles the final ComposedPrompt for video generation.

Assembly order (canonical):
    LoRA triggers → style prefix → base prompt → cinematography tokens → quality tokens
"""
from __future__ import annotations

from typing import List, Optional

from backend.services.prompt.types import (
    AnimationStyle, CinematographyProfile, ComposedPrompt, LoRAConfig,
)

_QUALITY_TOKENS = "high quality, detailed, professional"
_SAFETY_NEGATIVES = "nsfw, nude, explicit, sexual, naked"
_NEGATIVE_BASE = (
    "low quality, blurry, distorted, ugly, bad anatomy, watermark, text, "
    "jpeg artifacts, noise"
)


class PromptComposer:
    """Assembles a ComposedPrompt ready for the video generation backend.

    The canonical assembly order keeps LoRA triggers first so the model
    attends to them before the semantic content, and quality tokens last
    as a persistent emphasis booster.
    """

    def compose(
        self,
        base_prompt: str,
        style: AnimationStyle,
        loras: Optional[List[LoRAConfig]] = None,
        cinematography: Optional[CinematographyProfile] = None,
        nsfw: bool = False,
    ) -> ComposedPrompt:
        """Assemble a ComposedPrompt.

        Args:
            base_prompt: Core scene description.
            style: AnimationStyle providing prefix, negatives, and model settings.
            loras: Optional LoRA configs whose trigger tokens are prepended.
            cinematography: Optional cinematography profile adding camera/lighting tokens.
            nsfw: If False (default), adds safety negatives. If True, marks nsfw=True.

        Returns:
            ComposedPrompt ready for the video generation backend.
        """
        loras = loras or []

        # ── Positive prompt ────────────────────────────────────────────────────
        parts: List[str] = []

        # 1. LoRA trigger tokens
        for lora in loras:
            parts.append(lora.trigger_token)

        # 2. Style prefix
        if style.prefix:
            parts.append(style.prefix.rstrip(", "))

        # 3. Base prompt
        parts.append(base_prompt)

        # 4. Cinematography tokens
        if cinematography:
            for token in (
                cinematography.camera_movement,
                cinematography.lighting,
                cinematography.film_stock,
                cinematography.lens,
            ):
                if token:
                    parts.append(token)

        # 5. Quality tokens (always last)
        parts.append(_QUALITY_TOKENS)

        positive = ", ".join(p.strip(", ") for p in parts if p.strip())

        # ── Negative prompt ────────────────────────────────────────────────────
        neg_parts: List[str] = []
        if style.negative_prefix:
            neg_parts.append(style.negative_prefix.rstrip(", "))
        neg_parts.append(_NEGATIVE_BASE)
        if not nsfw:
            neg_parts.append(_SAFETY_NEGATIVES)

        negative = ", ".join(p.strip(", ") for p in neg_parts if p.strip())

        return ComposedPrompt(
            positive=positive,
            negative=negative,
            cfg_scale=style.cfg_scale,
            steps=style.steps,
            model=style.recommended_model,
            nsfw=nsfw,
        )
