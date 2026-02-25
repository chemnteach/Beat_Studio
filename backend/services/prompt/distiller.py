"""PromptDistiller — distills cinematic storyboards into diffusion-model generation prompts.

Uses a single batched Claude call to produce unique, under-60-word prompts for every clip,
grouped by narrative section so subdivided clips each get a different visual focus.
Falls back to heuristic stripping if the LLM is unavailable.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from backend.services.prompt.types import ScenePrompt

load_dotenv(Path.home() / ".claude" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent.parent / "backend" / ".env", override=False)

logger = logging.getLogger("beat_studio.prompt.distiller")

# Patterns for heuristic fallback — strip camera/editing language
_CAMERA_RE = re.compile(
    r"\b(dolly|tracking shot|tracking|pan|tilt|crane|POV|point.of.view|"
    r"cut to|match cut|dissolve|cross.cut|wide shot|close.up|medium shot|"
    r"slow motion|time.lapse|the camera|camera circles?|pull.back|push.in|"
    r"zoom shot|zoom)\b",
    re.IGNORECASE,
)
_SEQUENCE_RE = re.compile(
    r"\b(then|next|followed by|cut to|as we|meanwhile|finally|afterwards)\b",
    re.IGNORECASE,
)


class PromptDistiller:
    """Distills full storyboard descriptions into diffusion-model-ready generation prompts.

    Replaces ``.positive`` on each ScenePrompt with a unique, under-60-word prompt.
    The original cinematic description is preserved in ``.storyboard`` for reference.

    Usage::

        distiller = PromptDistiller()
        scene_prompts = distiller.distill(scene_prompts, style_prefix="watercolor painting, ")
    """

    def __init__(self, llm_provider: str = "anthropic"):
        self.llm_provider = llm_provider

    # ── public ────────────────────────────────────────────────────────────────

    def distill(
        self,
        prompts: List[ScenePrompt],
        style_prefix: str = "",
    ) -> List[ScenePrompt]:
        """Replace ``.positive`` on each ScenePrompt with a distilled generation prompt.

        Groups clips that share the same storyboard (clips subdivided from one
        narrative section). Calls the LLM once with all groups. Falls back
        to heuristic stripping if the LLM is unavailable.

        Args:
            prompts: ScenePrompt list from ScenePromptGenerator.
            style_prefix: AnimationStyle.prefix string (prepended to every prompt).

        Returns:
            The same list with ``.positive`` replaced on each item.
        """
        if not prompts:
            return prompts

        # Group clips by storyboard text — clips that share a storyboard are
        # subdivisions of the same narrative section and need unique visual focuses.
        groups: Dict[str, List[ScenePrompt]] = {}
        for p in prompts:
            groups.setdefault(p.storyboard, []).append(p)

        # Snapshot originals so we can detect clips the LLM response missed
        original_positive: Dict[int, str] = {id(p): p.positive for p in prompts}

        try:
            sections_data = [
                {"section_index": i, "n_clips": len(clips), "storyboard": sb}
                for i, (sb, clips) in enumerate(groups.items())
            ]
            raw = self._call_llm(self._build_prompt(sections_data, style_prefix))
            self._apply_results(raw["sections"], groups)

            # Find any clips the LLM response omitted and fall back on them
            missed: Dict[str, List[ScenePrompt]] = {}
            for p in prompts:
                if p.positive == original_positive[id(p)]:
                    missed.setdefault(p.storyboard, []).append(p)
            if missed:
                n_missed = sum(len(v) for v in missed.values())
                logger.warning(
                    "PromptDistiller: LLM response omitted %d clip(s) — applying heuristic fallback to those",
                    n_missed,
                )
                self._fallback_apply(missed, style_prefix)

            logger.info(
                "PromptDistiller: distilled %d sections → %d clips (%d via LLM, %d via fallback)",
                len(groups), len(prompts),
                len(prompts) - sum(len(v) for v in missed.values()) if missed else len(prompts),
                sum(len(v) for v in missed.values()) if missed else 0,
            )
        except Exception as exc:
            logger.warning(
                "PromptDistiller: LLM failed (%s) — heuristic fallback. "
                "Check ANTHROPIC_API_KEY.",
                exc,
            )
            self._fallback_apply(groups, style_prefix)

        return prompts

    # ── internal: prompt construction ─────────────────────────────────────────

    def _build_prompt(self, sections_data: List[Dict], style_prefix: str) -> str:
        sections_json = json.dumps(sections_data, indent=2)
        style_note = (
            f'Style prefix (include at the START of every prompt, EXACTLY as written):\n'
            f'"{style_prefix}"\n\n'
            if style_prefix else ""
        )
        return f"""You are distilling music video storyboard descriptions into diffusion model prompts.

{style_note}Rules for each distilled prompt:
- Under 60 words total (including the style prefix and quality tokens)
- Structure: [style prefix], [LoRA triggers], [subject], [setting], [lighting/mood], [key detail], high quality, detailed, professional
- NO camera directions (dolly, tracking shot, pan, tilt, crane, POV, "the camera", wide shot, close-up, pull-back, zoom)
- NO narrative sequencing (then, next, cut to, followed by, meanwhile, as we, finally)
- ONE moment, ONE static frame — not an edited sequence
- Each clip in a multi-clip section must have a DIFFERENT visual focus (different subject, setting, or moment)
- Preserve any LoRA trigger tokens found in the storyboard (snake_case identifiers like rob_char, beach_sunset, tiki_bar_int)

Sections to distill:
{sections_json}

Return ONLY valid JSON — no markdown fences, no commentary:
{{
  "sections": [
    {{
      "section_index": 0,
      "clips": [
        {{"clip_index": 0, "prompt": "style prefix, lora_trigger, subject in setting, lighting, detail, high quality, detailed, professional"}}
      ]
    }}
  ]
}}"""

    # ── internal: LLM call ────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        if self.llm_provider == "anthropic":
            return self._call_anthropic(prompt)
        return self._call_openai(prompt)

    def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        import anthropic
        import os
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._extract_json(message.content[0].text)

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        import openai
        import os
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
        )
        return self._extract_json(resp.choices[0].message.content)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    # ── internal: result application ──────────────────────────────────────────

    @staticmethod
    def _apply_results(
        sections: List[Dict],
        groups: Dict[str, List[ScenePrompt]],
    ) -> None:
        """Write distilled prompts back onto ScenePrompt objects."""
        group_items = list(groups.items())
        for sec in sections:
            idx = sec.get("section_index", 0)
            if idx >= len(group_items):
                continue
            _sb, clips = group_items[idx]
            for clip_data in sec.get("clips", []):
                ci = clip_data.get("clip_index", 0)
                if ci < len(clips):
                    clips[ci].positive = clip_data["prompt"]

    # ── internal: heuristic fallback ──────────────────────────────────────────

    @staticmethod
    def _fallback_apply(
        groups: Dict[str, List[ScenePrompt]],
        style_prefix: str,
    ) -> None:
        """Strip camera/sequence language from storyboard and truncate to 60 words.

        Each clip in a multi-clip section gets a different sentence from the
        storyboard so they're not identical.
        """
        quality_suffix = "high quality, detailed, professional"
        for storyboard, clips in groups.items():
            sentences = [s.strip() for s in re.split(r"[.!?]", storyboard) if s.strip()]
            for i, clip in enumerate(clips):
                base = sentences[i % len(sentences)] if sentences else storyboard
                base = _CAMERA_RE.sub("", base)
                base = _SEQUENCE_RE.sub("", base)
                base = " ".join(base.split())  # normalise whitespace

                # Budget: 60 words - style prefix words - quality suffix words
                style_words = len(style_prefix.split()) if style_prefix else 0
                suffix_words = len(quality_suffix.split())
                budget = max(10, 60 - style_words - suffix_words)
                truncated = " ".join(base.split()[:budget])

                prefix = style_prefix.rstrip(", ") if style_prefix else ""
                parts = [p for p in [prefix, truncated, quality_suffix] if p]
                clip.positive = ", ".join(parts)
