"""DJVideoGenerator — Nova Fade DJ performance video pipeline.

Converts a mashup audio track + analysis into a continuous Nova Fade DJ
performance video by:

1. Building a DJTimeline from audio structure (sections, energy, beats)
2. Mapping sections to DJ actions (idle_bob, deck_scratch_L/R,
   crossfader_hit, drop_reaction)
3. Converting each action to a canonical Nova Fade scene prompt
4. Generating video clips via the video engine
5. Assembling into a continuous output video

All GPU-heavy steps are isolated in mockable internal methods.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from backend.services.nova_fade.canonical_prompts import CanonicalPrompts
from backend.services.nova_fade.character import NovaFadeCharacter
from backend.services.nova_fade.types import ComposedScenePrompt

logger = logging.getLogger("beat_studio.nova_fade.dj_video_generator")

# ── Valid DJ action types ─────────────────────────────────────────────────────
_VALID_ACTION_TYPES = frozenset({
    "idle_bob",
    "deck_scratch_L",
    "deck_scratch_R",
    "crossfader_hit",
    "drop_reaction",
})

# ── Valid themes ──────────────────────────────────────────────────────────────
_VALID_THEMES = frozenset({
    "sponsor_neon",
    "award_elegant",
    "mashup_chaos",
    "chill_lofi",
})

# ── Energy threshold for "active" sections ────────────────────────────────────
_HIGH_ENERGY_THRESHOLD = 0.75


@dataclass
class DJAction:
    """A single DJ action segment in the performance timeline.

    Attributes:
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        action_type: One of the five canonical DJ actions.
        expression: Nova Fade expression for this segment.
        note: Optional annotation (e.g., "chorus drop").
    """
    start_sec: float
    end_sec: float
    action_type: str
    expression: str
    note: str = ""

    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end_sec - self.start_sec


@dataclass
class DJTimeline:
    """Complete DJ performance timeline for a track.

    Attributes:
        actions: Ordered list of :class:`DJAction` covering the track.
        total_duration_sec: Total track duration in seconds.
        theme: Visual theme name (studio lighting preset).
    """
    actions: List[DJAction]
    total_duration_sec: float
    theme: str


class DJVideoGenerator:
    """Generates Nova Fade DJ performance videos for mashups.

    All generation and assembly calls are isolated in mockable internal
    methods (_generate_clips, _assemble_video) so unit tests work without
    a GPU or real video files.

    Usage::

        gen = DJVideoGenerator()
        timeline = gen.build_timeline(mashup_analysis, theme="sponsor_neon")
        prompts = gen.timeline_to_prompts(timeline)
        output = gen.generate(
            mashup_path="/path/to/mashup.wav",
            mashup_analysis=analysis,
            theme="sponsor_neon",
            output_path="/path/to/output.mp4",
        )
    """

    # ── Section type → primary action mapping ─────────────────────────────────
    _SECTION_ACTION_MAP = {
        "intro": "idle_bob",
        "verse": "idle_bob",
        "pre_chorus": "deck_scratch_L",
        "chorus": "crossfader_hit",
        "bridge": "deck_scratch_R",
        "outro": "idle_bob",
        "drop": "drop_reaction",
    }

    # ── Action → best expression mapping ─────────────────────────────────────
    _ACTION_EXPRESSION_MAP = {
        "idle_bob": "neutral_confident",
        "deck_scratch_L": "focused_intensity",
        "deck_scratch_R": "focused_intensity",
        "crossfader_hit": "mischievous_grin",
        "drop_reaction": "drop_anticipation",
    }

    def __init__(self) -> None:
        self._canonical = CanonicalPrompts()
        self._character = NovaFadeCharacter()

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_timeline(
        self,
        analysis: object,  # SongAnalysis-like — typed loosely to avoid circular import
        theme: str = "sponsor_neon",
    ) -> DJTimeline:
        """Build a DJ action timeline from song analysis.

        Args:
            analysis: A SongAnalysis-like object with ``.duration``,
                ``.bpm``, and ``.sections`` (list of section objects with
                ``.start``, ``.end``, ``.section_type``, ``.energy_level``).
            theme: One of the four canonical theme names.

        Returns:
            :class:`DJTimeline` with actions covering the full track duration.

        Raises:
            ValueError: If ``theme`` is not a valid theme name.
        """
        if theme not in _VALID_THEMES:
            raise ValueError(
                f"Invalid theme: '{theme}'. "
                f"Valid themes: {sorted(_VALID_THEMES)}"
            )

        sections = analysis.sections
        total_duration = float(analysis.duration)
        actions: List[DJAction] = []

        for section in sections:
            start = float(section.start)
            end = float(section.end)
            # Clip to actual song duration
            if start >= total_duration:
                break
            end = min(end, total_duration)
            stype = getattr(section, "section_type", "verse")
            energy = float(getattr(section, "energy_level", 0.5))

            # Override to drop_reaction for high-energy sections
            if energy >= _HIGH_ENERGY_THRESHOLD and stype in ("chorus", "drop"):
                action_type = "drop_reaction"
            else:
                action_type = self._SECTION_ACTION_MAP.get(stype, "idle_bob")

            expression = self._ACTION_EXPRESSION_MAP.get(
                action_type, "neutral_confident"
            )
            actions.append(DJAction(
                start_sec=start,
                end_sec=end,
                action_type=action_type,
                expression=expression,
                note=stype,
            ))

        # Ensure coverage up to total_duration
        if actions and abs(actions[-1].end_sec - total_duration) > 0.5:
            last_end = actions[-1].end_sec if actions else 0.0
            if last_end < total_duration:
                actions.append(DJAction(
                    start_sec=last_end,
                    end_sec=total_duration,
                    action_type="idle_bob",
                    expression="neutral_confident",
                    note="gap_fill",
                ))

        return DJTimeline(
            actions=actions,
            total_duration_sec=total_duration,
            theme=theme,
        )

    def timeline_to_prompts(
        self,
        timeline: DJTimeline,
    ) -> List[ComposedScenePrompt]:
        """Convert a DJTimeline to a list of canonical scene prompts.

        Each action in the timeline becomes one scene prompt using the
        Nova Fade canonical header + action description.

        Args:
            timeline: :class:`DJTimeline` from :meth:`build_timeline`.

        Returns:
            List of :class:`ComposedScenePrompt` in the same order as
            ``timeline.actions``.
        """
        prompts: List[ComposedScenePrompt] = []
        for action in timeline.actions:
            # Map action_type to canonical gesture
            gesture = self._action_to_gesture(action.action_type)
            scene_desc = (
                f"DJ studio {timeline.theme.replace('_', ' ')}, "
                f"{action.action_type.replace('_', ' ')} motion, "
                f"dynamic music energy"
            )
            prompt = self._canonical.compose_scene_prompt(
                scene_description=scene_desc,
                expression=action.expression,
                gesture=gesture,
                character=self._character,
            )
            prompts.append(prompt)
        return prompts

    def generate(
        self,
        mashup_path: str,
        mashup_analysis: object,
        theme: str = "sponsor_neon",
        output_path: str = "output/videos/nova_fade_dj.mp4",
    ) -> str:
        """Generate a complete Nova Fade DJ performance video.

        Args:
            mashup_path: Path to the mashup audio file.
            mashup_analysis: SongAnalysis-like object.
            theme: Visual theme name.
            output_path: Where to write the final video.

        Returns:
            Path to the generated video file.
        """
        logger.info("Building DJ timeline for theme='%s'", theme)
        timeline = self.build_timeline(mashup_analysis, theme=theme)
        prompts = self.timeline_to_prompts(timeline)

        logger.info("Generating %d clips", len(prompts))
        clips = self._generate_clips(prompts, timeline)

        logger.info("Assembling video → %s", output_path)
        result = self._assemble_video(clips, mashup_path, output_path)
        return result

    # ── Internal — mockable for unit tests ────────────────────────────────────

    def _generate_clips(
        self,
        prompts: List[ComposedScenePrompt],
        timeline: DJTimeline,
    ) -> List[str]:
        """Generate one video clip per timeline action.

        Production: routes to the video engine (AnimateDiff, WAN, etc.).
        Override/mock in tests.
        """
        raise NotImplementedError(
            "_generate_clips: requires video backend. Mock in tests."
        )

    def _assemble_video(
        self,
        clips: List[str],
        audio_path: str,
        output_path: str,
    ) -> str:
        """Concatenate clips and mux with audio.

        Production: uses FFmpeg. Override/mock in tests.
        """
        raise NotImplementedError(
            "_assemble_video: requires FFmpeg. Mock in tests."
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _action_to_gesture(action_type: str) -> str:
        """Map a DJ action type to the nearest canonical gesture."""
        _MAP = {
            "idle_bob": "downbeat_head_nod",
            "deck_scratch_L": "left_deck_scratch",
            "deck_scratch_R": "right_deck_scratch",
            "crossfader_hit": "crossfader_tap",
            "drop_reaction": "spotlight_presentation",
        }
        return _MAP.get(action_type, "downbeat_head_nod")
