"""ScenePromptGenerator — builds video generation prompts from narrative + scene timings."""
from __future__ import annotations

from typing import Dict, List, Optional

from backend.services.audio.types import SceneTiming
from backend.services.prompt.types import (
    AnimationStyle, LoRAConfig, NarrativeArc, NarrativeSection, ScenePrompt,
)

_QUALITY_TOKENS = "high quality, detailed, professional"
_NEGATIVE_BASE = (
    "low quality, blurry, distorted, ugly, bad anatomy, watermark, text, "
    "jpeg artifacts, noise"
)
_HERO_STEP_BONUS = 10


class ScenePromptGenerator:
    """Generates per-scene video prompts from a NarrativeArc and scene timings."""

    def generate_prompts(
        self,
        narrative: NarrativeArc,
        scenes: List[SceneTiming],
        style: AnimationStyle,
        loras: Optional[List[LoRAConfig]] = None,
        user_overrides: Optional[Dict[int, str]] = None,
    ) -> List[ScenePrompt]:
        """Generate a ScenePrompt for each SceneTiming.

        Args:
            narrative: Visual narrative arc from NarrativeAnalyzer.
            scenes: Beat-aligned scene timings from AudioAnalyzer.
            style: AnimationStyle controlling prompt prefix, negatives, and model settings.
            loras: Optional LoRA configurations whose trigger tokens are prepended.
            user_overrides: Map of scene_index → custom prompt text that replaces the
                            narrative-derived base description.

        Returns:
            List of ScenePrompt objects, one per scene.
        """
        loras = loras or []
        user_overrides = user_overrides or {}

        lora_triggers = [l.trigger_token for l in loras]
        lora_names = [l.name for l in loras]

        prompts: List[ScenePrompt] = []
        for scene in scenes:
            narrative_sec = self._find_narrative_section(narrative, scene)

            # ── Base scene description ─────────────────────────────────────────
            if scene.scene_index in user_overrides:
                base_desc = user_overrides[scene.scene_index]
            elif narrative_sec:
                base_desc = narrative_sec.visual_description
            else:
                base_desc = f"A {scene.section_type} scene, cinematic"

            # ── Positive prompt: LoRA triggers → style prefix → base → extra context → quality ──
            parts: List[str] = []
            parts.extend(lora_triggers)
            if style.prefix:
                parts.append(style.prefix.rstrip(", "))
            parts.append(base_desc)
            if narrative_sec:
                if narrative_sec.key_lyric:
                    parts.append(f'"{narrative_sec.key_lyric}"')
                if narrative_sec.themes:
                    parts.append(", ".join(narrative_sec.themes))
            parts.append(_QUALITY_TOKENS)

            positive = ", ".join(p.strip(", ") for p in parts if p.strip())

            # ── Negative prompt ────────────────────────────────────────────────
            neg_parts: List[str] = []
            if style.negative_prefix:
                neg_parts.append(style.negative_prefix.rstrip(", "))
            neg_parts.append(_NEGATIVE_BASE)
            negative = ", ".join(p.strip(", ") for p in neg_parts if p.strip())

            # ── Steps (hero scenes get a bonus) ───────────────────────────────
            steps = style.steps + (_HERO_STEP_BONUS if scene.is_hero else 0)

            # ── Transition hint ────────────────────────────────────────────────
            if scene.is_hero:
                transition = "match_cut"
            elif scene.energy_level < 0.3:
                transition = "dissolve"
            else:
                transition = "cut"

            prompts.append(ScenePrompt(
                scene_index=scene.scene_index,
                start_sec=scene.start_sec,
                end_sec=scene.end_sec,
                duration_sec=scene.duration_sec,
                is_hero=scene.is_hero,
                energy_level=scene.energy_level,
                positive=positive,
                negative=negative,
                style=style.name,
                lora_names=lora_names,
                transition_hint=transition,
                cfg_scale=style.cfg_scale,
                steps=steps,
            ))

        return prompts

    # ── internal ──────────────────────────────────────────────────────────────

    def _find_narrative_section(
        self,
        narrative: NarrativeArc,
        scene: SceneTiming,
    ) -> Optional[NarrativeSection]:
        """Return the best matching NarrativeSection for this scene timing.

        Tries by index first, then falls back to time-overlap.
        """
        if not narrative.sections:
            return None

        # Fast path: scene index within narrative sections
        if scene.scene_index < len(narrative.sections):
            return narrative.sections[scene.scene_index]

        # Time-overlap fallback
        best: Optional[NarrativeSection] = None
        best_overlap = 0.0
        for sec in narrative.sections:
            overlap = max(0.0, min(scene.end_sec, sec.end_sec) - max(scene.start_sec, sec.start_sec))
            if overlap > best_overlap:
                best_overlap = overlap
                best = sec

        return best
