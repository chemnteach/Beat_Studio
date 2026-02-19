"""Mashup curator agent — ported from AI_Mixer mixer/agents/curator.py.

Finds compatible song pairs using hybrid matching and recommends mashup types.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from backend.services.mashup.memory import MashupLibrary, _camelot_distance

logger = logging.getLogger("beat_studio.mashup.curator")


class CuratorError(Exception):
    """Curator operation failure."""


# ── Camelot wheel distance for camelot key compatibility ─────────────────────

def _bpm_compatible(bpm_a: float, bpm_b: float, tolerance: float = 0.05) -> bool:
    if bpm_a <= 0 or bpm_b <= 0:
        return False
    ratio = max(bpm_a, bpm_b) / min(bpm_a, bpm_b)
    return ratio <= 1 + tolerance or abs(ratio - 2.0) <= 0.05 or abs(ratio - 0.5) <= 0.05


# ── Compatibility scoring ─────────────────────────────────────────────────────

def calculate_compatibility_score(
    meta_a: Dict[str, Any],
    meta_b: Dict[str, Any],
    bpm_weight: float = 0.35,
    key_weight: float = 0.30,
    energy_weight: float = 0.20,
    genre_weight: float = 0.15,
) -> float:
    """Calculate mashup compatibility score (0-100) for a pair of songs.

    Weights:
        BPM  35%, Key 30%, Energy 20%, Genre 15%
    """
    score = 0.0

    # ── BPM component ─────────────────────────────────────────────────────────
    bpm_a = meta_a.get("bpm") or 0
    bpm_b = meta_b.get("bpm") or 0
    if bpm_a > 0 and bpm_b > 0:
        ratio = max(bpm_a, bpm_b) / min(bpm_a, bpm_b)
        if ratio <= 1.05:
            bpm_score = 100
        elif ratio <= 1.10:
            bpm_score = 80
        elif abs(ratio - 2.0) <= 0.05 or abs(ratio - 0.5) <= 0.05:
            bpm_score = 70  # 2× or ½× relationship
        elif ratio <= 1.25:
            bpm_score = 50
        else:
            bpm_score = max(0, 100 - int((ratio - 1.0) * 100))
    else:
        bpm_score = 50  # unknown
    score += bpm_weight * bpm_score

    # ── Key component ─────────────────────────────────────────────────────────
    camelot_a = meta_a.get("camelot", "")
    camelot_b = meta_b.get("camelot", "")
    if camelot_a and camelot_b:
        dist = _camelot_distance(camelot_a, camelot_b)
        key_score = max(0, 100 - dist * 20)
    else:
        key_score = 50
    score += key_weight * key_score

    # ── Energy component ──────────────────────────────────────────────────────
    energy_a = meta_a.get("energy_level") or 5
    energy_b = meta_b.get("energy_level") or 5
    energy_diff = abs(energy_a - energy_b)
    energy_score = max(0, 100 - energy_diff * 10)
    score += energy_weight * energy_score

    # ── Genre component ───────────────────────────────────────────────────────
    genre_a = (meta_a.get("primary_genre") or "").lower()
    genre_b = (meta_b.get("primary_genre") or "").lower()
    if genre_a and genre_b and genre_a == genre_b:
        genre_score = 100
    elif genre_a and genre_b:
        # Partial genre compatibility (pop-adjacent genres)
        pop_adjacent = {"pop", "r&b", "soul", "funk", "dance", "electronic", "hip-hop"}
        if genre_a in pop_adjacent and genre_b in pop_adjacent:
            genre_score = 60
        else:
            genre_score = 20
    else:
        genre_score = 40
    score += genre_weight * genre_score

    return round(score, 1)


# ── Mashup type recommendation ────────────────────────────────────────────────

def recommend_mashup_type(
    meta_a: Dict[str, Any],
    meta_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Recommend the best mashup type for a song pair.

    Returns {"mashup_type": str, "confidence": float, "reasoning": str}.
    """
    camelot_a = meta_a.get("camelot", "")
    camelot_b = meta_b.get("camelot", "")
    key_dist = _camelot_distance(camelot_a, camelot_b) if camelot_a and camelot_b else 3

    sections_a = meta_a.get("sections", [])
    sections_b = meta_b.get("sections", [])
    has_sections = bool(sections_a and sections_b)

    bpm_a = meta_a.get("bpm") or 0
    bpm_b = meta_b.get("bpm") or 0
    bpm_ok = _bpm_compatible(bpm_a, bpm_b) if bpm_a and bpm_b else False

    candidates: List[Dict[str, Any]] = []

    # ── CLASSIC — always viable ───────────────────────────────────────────────
    classic_conf = 0.5
    if bpm_ok:
        classic_conf += 0.2
    if key_dist <= 1:
        classic_conf += 0.2
    candidates.append({
        "mashup_type": "CLASSIC",
        "confidence": min(1.0, classic_conf),
        "reasoning": f"Vocal A + Instrumental B. BPM compatible={bpm_ok}, key distance={key_dist}.",
    })

    # ── ADAPTIVE HARMONY — recommended when keys clash ─────────────────────────
    if key_dist >= 2:
        candidates.append({
            "mashup_type": "ADAPTIVE_HARMONY",
            "confidence": min(1.0, 0.5 + key_dist * 0.08),
            "reasoning": f"Key distance={key_dist}. Pitch-shifting recommended to fix key clash.",
        })

    # ── ENERGY_MATCHED — recommended when sections available ──────────────────
    if has_sections:
        candidates.append({
            "mashup_type": "ENERGY_MATCHED",
            "confidence": 0.7,
            "reasoning": "Section-level metadata available. Energy-driven section selection enabled.",
        })

    # ── STEM_SWAP — always viable ─────────────────────────────────────────────
    candidates.append({
        "mashup_type": "STEM_SWAP",
        "confidence": 0.45,
        "reasoning": "Mix stems from both songs. Drums from A + vocals from B + bass from A.",
    })

    # Pick highest confidence
    best = max(candidates, key=lambda x: x["confidence"])
    return best


# ── find_match ────────────────────────────────────────────────────────────────

def find_match(
    target_song_id: str,
    criteria: str = "hybrid",
    max_results: int = 5,
    library: Optional[MashupLibrary] = None,
) -> List[Dict[str, Any]]:
    """Find compatible songs for mashup using hybrid matching.

    Args:
        target_song_id: The song to match against.
        criteria: "harmonic" | "semantic" | "hybrid"
        max_results: Number of results to return.
        library: MashupLibrary instance (creates default if None).

    Returns:
        List of match result dicts with id, metadata, compatibility_score,
        match_reasons, and recommended_mashup.
    """
    if library is None:
        library = MashupLibrary()

    target = library.get_song(target_song_id)
    if target is None:
        raise CuratorError(f"Song not found in library: {target_song_id}")

    if criteria == "harmonic":
        raw = library.query_harmonic(
            target_bpm=target["metadata"].get("bpm", 120.0),
            target_camelot=target["metadata"].get("camelot", "8B"),
            exclude_ids=[target_song_id],
            max_results=max_results,
        )
    elif criteria == "semantic":
        raw = library.query_semantic(
            mood_summary=target["metadata"].get("mood_summary", ""),
            exclude_ids=[target_song_id],
            max_results=max_results,
        )
    else:
        raw = library.query_hybrid(
            target_song_id=target_song_id,
            max_results=max_results,
        )

    results = []
    for item in raw:
        meta = item.get("metadata", {})
        compat = calculate_compatibility_score(target["metadata"], meta)
        recommendation = recommend_mashup_type(target["metadata"], meta)

        reasons = []
        bpm_a = target["metadata"].get("bpm") or 0
        bpm_b = meta.get("bpm") or 0
        if bpm_a and bpm_b and _bpm_compatible(bpm_a, bpm_b):
            reasons.append(f"BPM compatible ({bpm_a:.0f} ≈ {bpm_b:.0f})")
        camelot_a = target["metadata"].get("camelot", "")
        camelot_b = meta.get("camelot", "")
        if camelot_a and camelot_b and _camelot_distance(camelot_a, camelot_b) <= 1:
            reasons.append(f"Keys harmonically compatible ({camelot_a} ↔ {camelot_b})")

        results.append({
            "id": item["id"],
            "metadata": meta,
            "compatibility_score": compat,
            "match_reasons": reasons,
            "recommended_mashup": recommendation,
        })

    results.sort(key=lambda x: x["compatibility_score"], reverse=True)
    return results
