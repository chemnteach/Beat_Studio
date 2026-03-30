"""BeatSynchronizer — aligns video scene boundaries to musical structure."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("beat_studio.video.beat_sync")

# Scene count caps per quality tier
# professional: 40 supports 8 sections × 5 avg clips at ~6-8 sec each for a 4.5-min song
# cinematic: 60 matches PRD target of 48-60 hero+supporting scenes
_TIER_SCENE_CAPS = {
    "basic": 12,
    "professional": 40,
    "cinematic": 60,
}

# Minimum scene duration in seconds
_MIN_SCENE_DURATION = 2.0

# Legacy energy threshold — used only when sections have no lyrical content (standard depth)
_HERO_ENERGY_THRESHOLD = 0.80

# Multi-signal hero score threshold (full-depth analysis with lyrics)
_HERO_SCORE_THRESHOLD = 0.60

# Stop words for hook phrase extraction (no external dependency)
_STOP_WORDS: frozenset = frozenset({
    "i", "im", "ill", "ive", "id", "me", "my", "myself", "we", "our",
    "you", "your", "he", "she", "it", "its", "they", "them", "their",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare",
    "a", "an", "the", "and", "but", "or", "nor", "for", "yet", "so",
    "in", "on", "at", "to", "of", "by", "up", "as", "if",
    "not", "no", "oh", "ah", "yeah", "hey", "ooh", "uh",
    "with", "from", "into", "about", "than", "then", "when",
    "just", "now", "got", "get", "let", "like",
})

# Minimum energy delta (normalized) for inflection signal to score 1.0
_INFLECTION_DELTA_SCALE = 0.20


def _extract_hook_phrase(sections: list) -> Optional[str]:
    """Find the most repeated all-content-word 2–4 gram across the given sections.

    Returns the hook phrase string, or ``None`` if no phrase appears in at
    least two distinct sections.  Skips sections with no string
    ``lyrical_content``.
    """
    def _tokens(text: str) -> List[str]:
        return [re.sub(r"[^a-z]", "", w.lower()) for w in re.split(r"\s+", text) if w]

    def _is_content(w: str) -> bool:
        return bool(w) and w not in _STOP_WORDS and w.isalpha()

    phrase_sections: Dict[str, set] = {}
    for idx, sec in enumerate(sections):
        lyrical = getattr(sec, "lyrical_content", None)
        if not isinstance(lyrical, str) or not lyrical.strip():
            continue
        words = _tokens(lyrical)
        for n in range(2, 5):
            for i in range(len(words) - n + 1):
                gram = words[i : i + n]
                if all(_is_content(w) for w in gram):
                    phrase = " ".join(gram)
                    if phrase not in phrase_sections:
                        phrase_sections[phrase] = set()
                    phrase_sections[phrase].add(idx)

    candidates = [(ph, secs) for ph, secs in phrase_sections.items() if len(secs) >= 2]
    if not candidates:
        return None
    hook, _ = max(candidates, key=lambda x: (len(x[1]), len(x[0].split()), x[0]))
    return hook


def _compute_hero_scores(sections: list, total_duration: float) -> List[float]:
    """Compute a multi-signal hero score [0.0, 1.0] for each section.

    Four signals, each 0–1, averaged equally:

    1. **bridge_type** — section_type == "bridge" → 1.0
    2. **temporal** — midpoint falls in 50–75 % of total duration → 1.0
    3. **energy_inflection** — energy changes direction at its boundaries vs
       adjacent sections (local peak or trough), scaled by delta magnitude.
    4. **hook_first** — section contains the first occurrence of the repeated
       hook phrase extracted from all inner section lyrics → 1.0

    Intro and outro (first/last section or explicit type) always score 0.0.
    """
    if not sections:
        return []

    energies = [float(getattr(s, "energy_level", 0.5)) for s in sections]
    inner_sections = sections[1:-1] if len(sections) > 2 else sections

    # Identify the hook phrase and the first inner section that contains it
    hook_phrase = _extract_hook_phrase(inner_sections)
    first_hook_inner_idx: Optional[int] = None
    if hook_phrase:
        for inner_idx, sec in enumerate(inner_sections):
            lyrical = getattr(sec, "lyrical_content", None)
            if not isinstance(lyrical, str):
                continue
            cleaned = re.sub(r"[^a-z ]", " ", lyrical.lower())
            if hook_phrase in cleaned:
                first_hook_inner_idx = inner_idx
                break

    scores: List[float] = []
    n = len(sections)

    for i, sec in enumerate(sections):
        stype = str(getattr(sec, "section_type", "verse"))

        # Intro/outro are never hero
        if i == 0 or i == n - 1 or stype in ("intro", "outro"):
            scores.append(0.0)
            continue

        inner_idx = i - 1  # 0-based index within inner_sections
        energy = energies[i]

        # Section midpoint (handle both .start_sec/.start naming conventions)
        start = float(getattr(sec, "start_sec", getattr(sec, "start", 0.0)))
        end = float(getattr(sec, "end_sec", getattr(sec, "end", 0.0)))
        mid = (start + end) / 2.0

        # Signal 1 — bridge type
        s1 = 1.0 if stype == "bridge" else 0.0

        # Signal 2 — temporal position in 50–75 % window
        ratio = mid / total_duration if total_duration > 0 else 0.0
        s2 = 1.0 if 0.50 <= ratio <= 0.75 else 0.0

        # Signal 3 — energy inflection (local peak or trough vs neighbors)
        s3 = 0.0
        if 0 < i < n - 1:
            d_left = energy - energies[i - 1]   # positive = energy rose
            d_right = energy - energies[i + 1]  # positive = energy will fall
            if d_left * d_right > 0:             # same sign → inflection
                delta = min(abs(d_left), abs(d_right))
                s3 = min(1.0, delta / _INFLECTION_DELTA_SCALE)

        # Signal 4 — first occurrence of the hook phrase
        s4 = 1.0 if (first_hook_inner_idx is not None and inner_idx == first_hook_inner_idx) else 0.0

        scores.append((s1 + s2 + s3 + s4) / 4.0)

    return scores


def beat_aligned_clip_durations(
    start_sec: float,
    end_sec: float,
    is_hero: bool,
    beat_times: Optional[List[float]],
    downbeat_times: Optional[List[float]],
    max_clip_sec: float = 5.0,
) -> List[float]:
    """Return clip durations for a scene, snapped to musical beat boundaries.

    Hero sections:
        ≤ 10 s  → single clip covering the full section.
        > 10 s  → two clips maximum, split at the nearest downbeat to the midpoint.

    Regular sections:
        Split into clips of approximately ``max_clip_sec``, each boundary
        snapped to the nearest beat timestamp.  Falls back to arithmetic
        division when no beat timestamps fall inside the section.

    Args:
        start_sec: Scene start time.
        end_sec: Scene end time.
        is_hero: Whether this scene is a hero scene.
        beat_times: All detected beat timestamps for the song.
        downbeat_times: Subset of beat_times that are downbeats (beat 1 of bar).
        max_clip_sec: Maximum clip length for regular scenes.

    Returns:
        Ordered list of clip durations (positive floats) summing to
        ``end_sec - start_sec``.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        return [max(0.001, duration)]

    def _nearest_interior(target: float, timestamps: Optional[List[float]]) -> Optional[float]:
        interior = [t for t in (timestamps or []) if start_sec < t < end_sec]
        if not interior:
            return None
        return min(interior, key=lambda t: abs(t - target))

    if is_hero:
        if duration <= 10.0:
            return [duration]
        mid = (start_sec + end_sec) / 2.0
        split = _nearest_interior(mid, downbeat_times) or _nearest_interior(mid, beat_times)
        if split is not None:
            a, b = split - start_sec, end_sec - split
            if a >= _MIN_SCENE_DURATION and b >= _MIN_SCENE_DURATION:
                return [a, b]
        return [duration]

    # Regular scene — arithmetic n-splits snapped to beats
    import math
    n_clips = math.ceil(duration / max_clip_sec)
    if n_clips <= 1:
        return [duration]

    split_targets = [start_sec + duration * k / n_clips for k in range(1, n_clips)]
    boundaries = [start_sec]
    prev = start_sec
    for target in split_targets:
        snapped = _nearest_interior(target, beat_times)
        candidate = snapped if snapped is not None else target
        if candidate - prev >= _MIN_SCENE_DURATION and end_sec - candidate >= _MIN_SCENE_DURATION:
            boundaries.append(candidate)
            prev = candidate
    boundaries.append(end_sec)

    return [boundaries[k + 1] - boundaries[k] for k in range(len(boundaries) - 1)]


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
    number of scenes is capped per quality tier (basic=12, professional=40,
    cinematic=60).  High-energy sections (chorus/drop) are flagged as hero
    scenes.

    Usage::

        sync = BeatSynchronizer()
        plans = sync.create_scene_plan(analysis, quality_tier="professional")
    """

    def create_scene_plan(
        self,
        analysis: object,
        quality_tier: str = "professional",
        beat_times: Optional[List[float]] = None,
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

        # Determine whether lyrical content is available (full-depth analysis)
        has_lyrics = any(
            isinstance(getattr(s, "lyrical_content", None), str)
            and bool(getattr(s, "lyrical_content", "").strip())
            for s in sections
        )

        # Compute multi-signal hero scores when lyrics are present
        hero_scores: List[float] = (
            _compute_hero_scores(sections, total_duration) if has_lyrics else []
        )

        # Build initial scene list from sections, clipped to total_duration
        raw_scenes: List[SyncedScenePlan] = []
        sec_idx = 0
        for sec in sections:
            start = float(sec.start)
            end = float(sec.end)
            if start >= total_duration:
                break
            end = min(end, total_duration)
            if end - start < _MIN_SCENE_DURATION:
                sec_idx += 1
                continue
            energy = float(getattr(sec, "energy_level", 0.5))
            stype = getattr(sec, "section_type", "verse")
            if has_lyrics:
                is_hero = hero_scores[sec_idx] > _HERO_SCORE_THRESHOLD if sec_idx < len(hero_scores) else False
            else:
                is_hero = energy >= _HERO_ENERGY_THRESHOLD or stype in ("bridge", "drop")
            sec_idx += 1
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
        subdivided = self._subdivide_to_cap(raw_scenes, cap, quality_tier, beat_times=beat_times)

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
        beat_times: Optional[List[float]] = None,
    ) -> List[SyncedScenePlan]:
        """Subdivide long scenes to approach ``cap`` without exceeding it.

        Split points snap to the nearest beat when ``beat_times`` is provided,
        so clip boundaries land on actual musical beats rather than arbitrary midpoints.
        """
        if len(scenes) >= cap:
            return scenes[:cap]

        result = list(scenes)
        if quality_tier in ("professional", "cinematic"):
            # Target: no single clip longer than 3× the average clip duration
            max_single = (scenes[-1].end_sec - scenes[0].start_sec) / cap * 3
            max_single = max(max_single, _MIN_SCENE_DURATION * 2)
            changed = True
            while changed and len(result) < cap:
                changed = False
                new_result: List[SyncedScenePlan] = []
                for scene in result:
                    slots_remaining = cap - len(new_result) - (len(result) - result.index(scene))
                    if scene.duration_sec > max_single and slots_remaining >= 1:
                        mid = self._nearest_beat(
                            scene.start_sec, scene.end_sec, beat_times
                        )
                        # Guard: both halves must meet minimum duration
                        if (
                            mid - scene.start_sec >= _MIN_SCENE_DURATION
                            and scene.end_sec - mid >= _MIN_SCENE_DURATION
                        ):
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
                            continue
                    new_result.append(scene)
                result = new_result[:cap]

        return result

    @staticmethod
    def _nearest_beat(start: float, end: float, beat_times: Optional[List[float]]) -> float:
        """Return the beat timestamp nearest to the midpoint of [start, end].

        Falls back to the geometric midpoint when no beat_times are provided
        or no beat falls strictly inside the interval.
        """
        mid = (start + end) / 2
        if not beat_times:
            return mid
        interior = [b for b in beat_times if start < b < end]
        if not interior:
            return mid
        return min(interior, key=lambda b: abs(b - mid))
