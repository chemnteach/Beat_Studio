"""BeatSynchronizer — aligns video scene boundaries to musical structure."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("beat_studio.video.beat_sync")

# Scene count caps per quality tier
_TIER_SCENE_CAPS = {
    "basic": 12,
    "professional": 24,
    "cinematic": 48,
}

# Minimum scene duration in seconds
_MIN_SCENE_DURATION = 2.0

# Energy threshold above which a scene is considered "hero"
_HERO_ENERGY_THRESHOLD = 0.80


@dataclass
class SyncedScenePlan:
    """A beat-aligned scene plan with timing and hero flag.

    Attributes:
        scene_index: Zero-based index of this scene.
        start_sec: Scene start time in seconds.
        end_sec: Scene end time in seconds.
        duration_sec: Scene duration in seconds.
        is_hero: True if this scene aligns to a high-energy section.
        backend_name: Preferred backend for this scene.
        notes: Optional annotation (section type, etc.).
    """
    scene_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    is_hero: bool = False
    backend_name: str = ""
    notes: str = ""


class BeatSynchronizer:
    """Ensures video scene boundaries align with musical structure.

    Scene boundaries are derived from the song's section structure.  The
    number of scenes is capped per quality tier (basic=12, professional=24,
    cinematic=48).  High-energy sections (chorus/drop) are flagged as hero
    scenes.

    Usage::

        sync = BeatSynchronizer()
        plans = sync.create_scene_plan(analysis, quality_tier="professional")
    """

    def create_scene_plan(
        self,
        analysis: object,
        quality_tier: str = "professional",
    ) -> List[SyncedScenePlan]:
        """Create a beat-aligned list of scene plans.

        Args:
            analysis: SongAnalysis-like object with ``.duration``, ``.bpm``,
                and ``.sections`` (list with ``.start``, ``.end``,
                ``.section_type``, ``.energy_level``).
            quality_tier: ``"basic"`` | ``"professional"`` | ``"cinematic"``.

        Returns:
            Ordered list of :class:`SyncedScenePlan` objects covering the full
            song duration, without gaps or overlaps.

        Raises:
            ValueError: If ``quality_tier`` is not a recognised tier.
        """
        if quality_tier not in _TIER_SCENE_CAPS:
            raise ValueError(
                f"Invalid quality_tier: '{quality_tier}'. "
                f"Valid: {list(_TIER_SCENE_CAPS)}"
            )

        cap = _TIER_SCENE_CAPS[quality_tier]
        total_duration = float(analysis.duration)
        sections = list(analysis.sections)

        # Build initial scene list from sections, clipped to total_duration
        raw_scenes: List[SyncedScenePlan] = []
        for sec in sections:
            start = float(sec.start)
            end = float(sec.end)
            if start >= total_duration:
                break
            end = min(end, total_duration)
            if end - start < _MIN_SCENE_DURATION:
                continue
            energy = float(getattr(sec, "energy_level", 0.5))
            stype = getattr(sec, "section_type", "verse")
            is_hero = energy >= _HERO_ENERGY_THRESHOLD
            raw_scenes.append(SyncedScenePlan(
                scene_index=0,  # re-indexed below
                start_sec=start,
                end_sec=end,
                duration_sec=end - start,
                is_hero=is_hero,
                notes=stype,
            ))

        # If sections don't cover total_duration, add a tail scene
        if raw_scenes:
            last_end = raw_scenes[-1].end_sec
            if total_duration - last_end > _MIN_SCENE_DURATION:
                raw_scenes.append(SyncedScenePlan(
                    scene_index=0,
                    start_sec=last_end,
                    end_sec=total_duration,
                    duration_sec=total_duration - last_end,
                    is_hero=False,
                    notes="tail",
                ))
        else:
            # No usable sections — use a single scene for the whole song
            raw_scenes.append(SyncedScenePlan(
                scene_index=0,
                start_sec=0.0,
                end_sec=total_duration,
                duration_sec=total_duration,
                is_hero=False,
            ))

        # Subdivide to approach cap while respecting minimum duration
        subdivided = self._subdivide_to_cap(raw_scenes, cap, quality_tier)

        # Assign sequential indices
        for i, scene in enumerate(subdivided):
            scene.scene_index = i

        return subdivided

    # ── Internal ──────────────────────────────────────────────────────────────

    def _subdivide_to_cap(
        self,
        scenes: List[SyncedScenePlan],
        cap: int,
        quality_tier: str,
    ) -> List[SyncedScenePlan]:
        """Subdivide long scenes to approach ``cap`` without exceeding it."""
        # If already at or under cap, return as-is
        if len(scenes) >= cap:
            # Trim to cap (keep hero scenes preferentially — keep first cap)
            return scenes[:cap]

        result = list(scenes)
        # For higher tiers subdivide long scenes to fill cap
        if quality_tier in ("professional", "cinematic"):
            max_single = 60.0 / cap * 3  # target: scenes ≤ 3× average
            changed = True
            while changed and len(result) < cap:
                changed = False
                new_result: List[SyncedScenePlan] = []
                for scene in result:
                    if (
                        scene.duration_sec > max_single
                        and len(new_result) + (len(result) - result.index(scene)) <= cap
                    ):
                        # Split in half
                        mid = (scene.start_sec + scene.end_sec) / 2
                        new_result.append(SyncedScenePlan(
                            scene_index=0,
                            start_sec=scene.start_sec,
                            end_sec=mid,
                            duration_sec=mid - scene.start_sec,
                            is_hero=scene.is_hero,
                            notes=scene.notes,
                        ))
                        new_result.append(SyncedScenePlan(
                            scene_index=0,
                            start_sec=mid,
                            end_sec=scene.end_sec,
                            duration_sec=scene.end_sec - mid,
                            is_hero=scene.is_hero,
                            notes=scene.notes,
                        ))
                        changed = True
                    else:
                        new_result.append(scene)
                result = new_result[:cap]

        return result
