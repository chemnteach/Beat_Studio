"""PromptComposer — assembles the final ComposedPrompt for video generation.

Assembly order (canonical):
    LoRA triggers → style prefix → base prompt → cinematography tokens → quality tokens
"""
from __future__ import annotations

from typing import List, Optional


def derive_video_prompt(image_prompt: str) -> str:
    """Derive a video generation prompt from a storyboard image prompt.

    Removes static image style descriptors (watercolor, brush strokes, etc.) and
    LoRA trigger tokens, adds appropriate motion language, and preserves subject,
    mood, lighting, and framing. Result is under 150 words.

    Args:
        image_prompt: The positive prompt used for SDXL storyboard image generation.

    Returns:
        A video-ready prompt string.
    """
    import os
    import openai

    system = (
        "You are a video prompt specialist. Transform an image generation prompt into a "
        "video generation prompt. Follow these rules strictly:\n"
        "1. REMOVE all static image style terms: watercolor, oil painting, brush strokes, "
        "paper texture, pencil sketch, wet-on-wet, bleeding edges, painted, illustration, "
        "drawing, 2D art, digital painting, acrylic, charcoal, pastel, ink wash.\n"
        "2. REMOVE all LoRA trigger tokens — these are short underscore_separated or "
        "lowercase identifiers that don't describe visible scene content, such as rob_char, "
        "island_girl, novafade_char, crossfadeclub_style, beach_bar_ext, tiki_bar_int, "
        "boat_deck, bonfire_beach, ocean_underwater, stage_perf, beach_sunset, "
        "70s_film_style, michele_char, and similar tokens.\n"
        "3. ADD natural motion language appropriate to the scene: gentle breeze moving hair, "
        "waves rolling, camera slowly pushing in, drifting clouds, flickering candlelight, "
        "subtle sway, cinematic dolly, soft breath, or similar.\n"
        "4. PRESERVE: subject description, mood, lighting, framing, setting, emotional quality.\n"
        "5. KEEP the result under 150 words. Return plain text only — no bullets, no labels."
    )

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": image_prompt},
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

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
