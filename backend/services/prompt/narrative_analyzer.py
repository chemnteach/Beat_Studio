"""NarrativeAnalyzer — converts song analysis into a visual narrative arc.

Uses an LLM to generate per-section visual descriptions that tell the song's story.
Falls back gracefully when LLM is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from backend.services.audio.types import SongAnalysis
from backend.services.prompt.types import NarrativeArc, NarrativeSection

# Load ~/.claude/.env so ANTHROPIC_API_KEY is available even when uvicorn is
# started without it exported in the shell.  override=False means an already-set
# variable always wins.
load_dotenv(Path.home() / ".claude" / ".env", override=False)
load_dotenv(Path(__file__).parent.parent.parent.parent / "backend" / ".env", override=False)

logger = logging.getLogger("beat_studio.prompt.narrative_analyzer")


class NarrativeAnalyzer:
    """Extracts a visual narrative arc from a SongAnalysis using LLM.

    The resulting NarrativeArc drives the ScenePromptGenerator to produce
    video generation prompts that tell the song's story visually.

    Usage::

        az = NarrativeAnalyzer()
        arc = az.analyze(song_analysis, user_concept="retro 80s dance club")
    """

    def __init__(self, llm_provider: str = "anthropic"):
        self.llm_provider = llm_provider

    # ── public ────────────────────────────────────────────────────────────────

    def analyze(
        self,
        analysis: SongAnalysis,
        user_concept: Optional[str] = None,
    ) -> NarrativeArc:
        """Generate a NarrativeArc from a SongAnalysis.

        Args:
            analysis: Completed SongAnalysis (depth="standard" or "full").
            user_concept: Optional creative direction from the user.

        Returns:
            NarrativeArc with per-section visual descriptions.
        """
        prompt = self._build_prompt(analysis, user_concept)
        try:
            raw = self._call_llm(prompt)
            arc = self._parse_response(raw, analysis)
            logger.info(
                "NarrativeAnalyzer: LLM succeeded — %d sections, concept=%r",
                len(arc.sections), arc.overall_concept[:60],
            )
            return arc
        except Exception as exc:
            logger.warning(
                "NarrativeAnalyzer: LLM call failed (%s) — using heuristic fallback. "
                "Check ANTHROPIC_API_KEY and increase max_tokens if needed.",
                exc,
            )
            return self._fallback_arc(analysis)

    # ── internal: LLM call ────────────────────────────────────────────────────

    def _build_prompt(self, analysis: SongAnalysis, user_concept: Optional[str]) -> str:
        sections_summary = "\n".join(
            f"  Section {i} ({s.section_type}, {s.start_sec:.0f}s-{s.end_sec:.0f}s, "
            f"energy={s.energy_level:.2f}, tone={s.emotional_tone}): '{s.lyrical_content[:120]}'"
            for i, s in enumerate(analysis.sections)
        )
        user_dir = (
            f"\n\nCREATIVE DIRECTION (prioritise this above all else):\n{user_concept}"
            if user_concept else ""
        )

        return f"""You are a music video creative director writing shot-by-shot visual descriptions for an AI video generation pipeline.

Song: {analysis.artist} - {analysis.title}
BPM: {analysis.bpm:.0f}, Key: {analysis.key}, Genre: {analysis.primary_genre}
Overall mood: {analysis.mood_summary}
Emotional arc: {analysis.emotional_arc}
Lyric excerpt (context only): {analysis.transcript[:400]}
{user_dir}

Sections and their lyrics (use lyrics to understand meaning and emotion — do NOT copy them into output):
{sections_summary}

INSTRUCTIONS:
- For each section write a "visual_description": describe what the camera sees in cinematic language.
  Use concrete details: location, subject, action, lighting, camera angle, colour temperature.
  Example: "Wide shot — Rob stands at a sun-drenched airport departure gate, suit jacket over his shoulder, staring at the tarmac through floor-to-ceiling glass. Warm golden light, 35mm grain."
  NEVER quote or paraphrase lyrics in visual_description.
- For "key_lyric": copy the single most visually evocative line from that section's lyrics above (one line only).
- For "themes": list 2-4 concrete visual motifs (e.g. "airport terminal", "golden hour", "solitude").
- overall_concept: one sentence capturing the whole video's visual story.
- mood_progression: one sentence describing how the emotion shifts across the song.
- color_palette: 3-5 dominant colour descriptors for the whole video.

Return ONLY valid JSON — no markdown fences, no commentary:
{{
  "overall_concept": "One sentence describing the complete visual narrative",
  "color_palette": ["color1", "color2", "color3"],
  "mood_progression": "How the emotional tone evolves across the song",
  "visual_style_hint": "Cinematic style descriptor",
  "sections": [
    {{
      "section_index": 0,
      "visual_description": "Cinematic scene description — what the camera sees, not what the lyrics say",
      "key_lyric": "One evocative line copied verbatim from the section lyrics",
      "themes": ["concrete visual motif", "another motif"]
    }}
  ]
}}"""

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call the configured LLM provider. Overridable for testing."""
        if self.llm_provider == "anthropic":
            return self._call_anthropic(prompt)
        return self._call_openai(prompt)

    def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        import anthropic
        import os
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text
        return self._extract_json(text)

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        import openai
        import os
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        text = resp.choices[0].message.content
        return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response (may be wrapped in markdown)."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    # ── internal: response parsing ────────────────────────────────────────────

    def _parse_response(
        self,
        raw: Dict[str, Any],
        analysis: SongAnalysis,
    ) -> NarrativeArc:
        """Convert raw LLM JSON into a NarrativeArc."""
        raw_sections = raw.get("sections", [])

        narrative_sections: List[NarrativeSection] = []
        for i, sec_info in enumerate(analysis.sections):
            raw_sec = raw_sections[i] if i < len(raw_sections) else {}
            narrative_sections.append(NarrativeSection(
                section_index=i,
                section_type=sec_info.section_type,
                start_sec=sec_info.start_sec,
                end_sec=sec_info.end_sec,
                visual_description=raw_sec.get(
                    "visual_description",
                    f"A {sec_info.section_type} scene with {sec_info.emotional_tone} atmosphere",
                ),
                emotional_tone=sec_info.emotional_tone,
                energy_level=sec_info.energy_level,
                is_climax=sec_info.section_type == "chorus" and sec_info.energy_level > 0.7,
                key_lyric=raw_sec.get("key_lyric", ""),
                themes=raw_sec.get("themes", sec_info.themes),
            ))

        return NarrativeArc(
            artist=analysis.artist,
            title=analysis.title,
            overall_concept=raw.get("overall_concept", f"A visual journey through '{analysis.title}'"),
            color_palette=raw.get("color_palette", ["vibrant", "dynamic"]),
            mood_progression=raw.get("mood_progression", analysis.emotional_arc),
            visual_style_hint=raw.get("visual_style_hint", "cinematic, professional"),
            sections=narrative_sections,
        )

    # ── fallback (no LLM) ─────────────────────────────────────────────────────

    def _fallback_arc(self, analysis: SongAnalysis) -> NarrativeArc:
        """Generate a basic NarrativeArc from audio analysis metadata alone."""
        _TONE_VISUALS = {
            "hopeful": "soft golden light, open spaces, upward movement",
            "melancholic": "blue-grey tones, rain, empty streets",
            "triumphant": "vibrant colours, crowd energy, dynamic movement",
            "defiant": "stark contrast, urban grit, strong silhouettes",
            "serene": "calm water, diffused light, slow movement",
            "reflective": "warm interiors, candlelight, close-up portraits",
            "neutral": "documentary style, natural light, honest framing",
        }
        _TONE_THEMES = {
            "hopeful": ["sunrise", "open road", "growth"],
            "melancholic": ["rain", "solitude", "memory"],
            "triumphant": ["celebration", "crowd", "victory"],
            "defiant": ["urban", "resistance", "strength"],
            "serene": ["nature", "peace", "water"],
            "reflective": ["introspection", "light", "intimacy"],
        }

        narrative_sections: List[NarrativeSection] = []
        for i, sec in enumerate(analysis.sections):
            tone = sec.emotional_tone or "neutral"
            visual = _TONE_VISUALS.get(tone, "cinematic scene, professional")
            themes = _TONE_THEMES.get(tone, sec.themes or ["music"])
            narrative_sections.append(NarrativeSection(
                section_index=i,
                section_type=sec.section_type,
                start_sec=sec.start_sec,
                end_sec=sec.end_sec,
                visual_description=(
                    f"{sec.section_type.capitalize()} scene — {visual}. "
                    f"Energy: {'high' if sec.energy_level > 0.6 else 'medium' if sec.energy_level > 0.3 else 'low'}."
                ),
                emotional_tone=tone,
                energy_level=sec.energy_level,
                is_climax=sec.section_type == "chorus" and sec.energy_level > 0.7,
                key_lyric=sec.lyrical_content[:60] if sec.lyrical_content else "",
                themes=themes,
            ))

        # Fallback if no sections
        if not narrative_sections:
            narrative_sections = [NarrativeSection(
                section_index=0,
                section_type="verse",
                start_sec=0.0,
                end_sec=analysis.duration_sec,
                visual_description=f"A cinematic music video for {analysis.artist} - {analysis.title}",
                emotional_tone="neutral",
                energy_level=analysis.energy_level,
                is_climax=False,
                key_lyric="",
                themes=[analysis.primary_genre or "music"],
            )]

        return NarrativeArc(
            artist=analysis.artist,
            title=analysis.title,
            overall_concept=f"A visual journey through '{analysis.title}' by {analysis.artist}",
            color_palette=["vibrant", "dynamic", "professional"],
            mood_progression=analysis.emotional_arc or analysis.mood_summary,
            visual_style_hint="cinematic, professional, high-quality music video",
            sections=narrative_sections,
        )
