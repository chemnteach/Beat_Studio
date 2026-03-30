"""Audio signal analysis — ported from AI_Mixer mixer/audio/analysis.py.

Section detection, energy analysis, key estimation, Camelot wheel mapping.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("beat_studio.audio.analysis")

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None  # type: ignore[assignment]


def _adaptive_k(
    chroma: np.ndarray,
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    min_k: int = 4,
    max_k: int = 14,
) -> int:
    """Estimate the number of agglomerative boundary frames via novelty curve.

    Beat-synchronises the chroma to the musical grid, builds an affinity
    recurrence matrix, applies a checkerboard kernel (Foote 2000) along the
    diagonal to produce a novelty curve, and peak-picks to count structural
    transitions.  Returns a k suitable for librosa.segment.agglomerative,
    where k boundaries yield k-1 sections.
    """
    try:
        from scipy.ndimage import convolve as _ndimage_conv

        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        duration = float(librosa.get_duration(y=y, sr=sr))

        if len(beats) > 4:
            chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
        else:
            chroma_sync = chroma
            beat_times = np.linspace(0.0, duration, chroma.shape[1])

        n_frames = chroma_sync.shape[1]
        if n_frames < 8:
            return min_k

        # Affinity self-similarity matrix
        R = librosa.segment.recurrence_matrix(chroma_sync, mode='affinity', sym=True)

        # Checkerboard kernel (L=8: four 4×4 quadrants)
        half = 4
        ones = np.ones((half, half))
        kernel = np.block([[-ones, ones], [ones, -ones]])

        novelty_map = _ndimage_conv(
            R.astype(np.float32), kernel.astype(np.float32),
            mode='constant', cval=0.0,
        )
        novelty = np.diag(novelty_map)
        novelty = np.maximum(novelty, 0.0)
        if novelty.max() > 0:
            novelty /= novelty.max()

        win = max(1, n_frames // 20)
        peaks = librosa.util.peak_pick(
            novelty,
            pre_max=win, post_max=win,
            pre_avg=win * 3, post_avg=win * 3,
            delta=0.10, wait=win,
        )

        # Discard spurious peaks at song start / end (within 10s / 15s respectively)
        filtered = [
            p for p in peaks
            if 10.0 < float(beat_times[min(int(p), len(beat_times) - 1)]) < duration - 15.0
        ]

        # n filtered internal boundaries + start frame → n+1 boundaries → n sections
        # librosa.segment.agglomerative(chroma, k) returns k boundaries → k-1 sections
        # so pass k = n_filtered + 2 to get n_filtered + 1 sections
        k = len(filtered) + 2
        logger.info("Adaptive k: %d peaks (raw=%d) → k=%d", len(filtered), len(peaks), k)
        return max(min_k, min(max_k, k))

    except Exception as exc:
        logger.warning("Adaptive k estimation failed: %s — using default k=9", exc)
        return 9


def _novelty_boundaries(
    chroma: np.ndarray,
    y: np.ndarray,
    sr: int,
    duration: float,
    hop_length: int = 512,
) -> np.ndarray:
    """Return section boundary times (seconds) via checkerboard novelty peaks.

    Beats are used to synchronise the chroma before building the self-similarity
    matrix, which reduces noise and aligns boundaries to the musical grid.
    The result always includes 0.0 and duration as the first/last elements.
    Falls back to uniform 8-chunk segmentation on any failure.
    """
    from scipy.ndimage import convolve as _ndimage_conv

    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    if len(beats) > 4:
        chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
    else:
        chroma_sync = chroma
        beat_times = np.linspace(0.0, duration, chroma.shape[1])

    n_frames = chroma_sync.shape[1]
    if n_frames < 8:
        return np.array([0.0, duration])

    R = librosa.segment.recurrence_matrix(chroma_sync, mode='affinity', sym=True)

    half = 4
    ones = np.ones((half, half))
    kernel = np.block([[-ones, ones], [ones, -ones]])

    novelty_map = _ndimage_conv(
        R.astype(np.float32), kernel.astype(np.float32),
        mode='constant', cval=0.0,
    )
    novelty = np.diag(novelty_map)
    novelty = np.maximum(novelty, 0.0)
    if novelty.max() > 0:
        novelty /= novelty.max()

    win = max(1, n_frames // 20)
    peaks = librosa.util.peak_pick(
        novelty,
        pre_max=win, post_max=win,
        pre_avg=win * 3, post_avg=win * 3,
        delta=0.10, wait=win,
    )

    # Discard spurious peaks within 10s of song start or 15s of song end
    filtered = [
        p for p in peaks
        if 10.0 < float(beat_times[min(int(p), len(beat_times) - 1)]) < duration - 15.0
    ]
    peak_times = [float(beat_times[min(int(p), len(beat_times) - 1)]) for p in filtered]

    logger.info("Novelty boundaries: %d peaks (raw=%d)", len(filtered), len(peaks))
    return np.array([0.0] + peak_times + [duration])


def detect_sections(
    y: np.ndarray,
    sr: int,
    n_segments: int = 8,
) -> List[Tuple[float, float]]:
    """Detect section boundaries from a checkerboard novelty curve.

    Beat-synchronises the chroma, builds an affinity self-similarity matrix,
    applies a Foote (2000) checkerboard kernel to extract a novelty curve,
    and peak-picks to locate structural boundaries.  Agglomerative clustering
    is not used; the peaks themselves are the boundaries.

    Returns:
        List of (start_sec, end_sec) tuples.
    """
    try:
        duration = librosa.get_duration(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        boundary_times = _novelty_boundaries(chroma, y, sr, float(duration))

        raw = []
        for i in range(len(boundary_times) - 1):
            raw.append((float(boundary_times[i]), float(boundary_times[i + 1])))

        # Extend last section to cover full duration (librosa may leave a tail)
        if raw and raw[-1][1] < duration - 0.5:
            raw[-1] = (raw[-1][0], float(duration))

        # Merge sections shorter than MIN_SECTION_SEC into their longer neighbor.
        # Process inner sections first, then check first/last.
        MIN_SEC = 10.0
        changed = True
        while changed and len(raw) > 2:
            changed = False
            for i in range(1, len(raw) - 1):   # skip first and last
                if raw[i][1] - raw[i][0] < MIN_SEC:
                    prev_dur = raw[i - 1][1] - raw[i - 1][0]
                    next_dur = raw[i + 1][1] - raw[i + 1][0]
                    if next_dur >= prev_dur:
                        raw = raw[:i] + [(raw[i][0], raw[i + 1][1])] + raw[i + 2:]
                    else:
                        raw = raw[:i - 1] + [(raw[i - 1][0], raw[i][1])] + raw[i + 1:]
                    changed = True
                    break

        # Absorb a tiny first or last section into its neighbor
        if len(raw) > 1 and raw[0][1] - raw[0][0] < MIN_SEC:
            raw = [(raw[0][0], raw[1][1])] + raw[2:]
        if len(raw) > 1 and raw[-1][1] - raw[-1][0] < MIN_SEC:
            raw = raw[:-2] + [(raw[-2][0], raw[-1][1])]

        logger.info("Detected %d sections (after merge)", len(raw))
        return raw

    except Exception as exc:
        logger.error("Section detection failed: %s", exc)
        duration = librosa.get_duration(y=y, sr=sr)
        n_chunks = max(1, int(duration / 8))
        chunk_dur = duration / n_chunks
        return [(i * chunk_dur, min((i + 1) * chunk_dur, duration))
                for i in range(n_chunks)]


# Pre-defined inner-section patterns (between intro and outro) for common counts.
# Keyed by number of inner sections.
_INNER_PATTERNS: dict = {
    1: ["verse"],
    2: ["verse", "chorus"],
    3: ["verse", "chorus", "verse"],
    4: ["verse", "chorus", "verse", "chorus"],
    5: ["verse", "chorus", "verse", "chorus", "bridge"],
    6: ["verse", "chorus", "verse", "chorus", "bridge", "verse"],
    7: ["verse", "chorus", "verse", "chorus", "bridge", "verse", "chorus"],
}


def classify_section_type(
    section_idx: int,
    total_sections: int,
    energy_level: float,
    spectral_centroid: float,
) -> str:
    """Assign section type by position.

    First section is always intro, last is always outro.
    Inner sections use a pre-defined pattern scaled to the detected count.
    """
    if total_sections == 1:
        return "verse"
    if section_idx == 0:
        return "intro"
    if section_idx == total_sections - 1:
        return "outro"

    n_inner = total_sections - 2
    inner_idx = section_idx - 1

    if n_inner in _INNER_PATTERNS:
        return _INNER_PATTERNS[n_inner][inner_idx]

    # > 7 inner sections: alternate verse/chorus
    return "verse" if inner_idx % 2 == 0 else "chorus"


def analyze_section_energy(
    y: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> dict:
    """Return energy_level, spectral_centroid, tempo_stability for a section."""
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    section_audio = y[start_sample:end_sample]

    if len(section_audio) == 0:
        return {"energy_level": 0.0, "spectral_centroid": 0.0, "tempo_stability": 0.0}

    try:
        rms = librosa.feature.rms(y=section_audio)[0]
        energy_level = float(np.mean(rms))

        centroid = librosa.feature.spectral_centroid(y=section_audio, sr=sr)[0]
        spectral_centroid = float(np.mean(centroid))

        try:
            tempo, beats = librosa.beat.beat_track(y=section_audio, sr=sr)
            if len(beats) > 1:
                beat_times = librosa.frames_to_time(beats, sr=sr)
                beat_intervals = np.diff(beat_times)
                tempo_stability = float(np.exp(-np.var(beat_intervals) * 10))
            else:
                tempo_stability = 0.0
        except Exception as exc:
            logger.debug("tempo_stability calculation failed: %s — using 0.5", exc)
            tempo_stability = 0.5

        return {
            "energy_level": energy_level,
            "spectral_centroid": spectral_centroid,
            "tempo_stability": tempo_stability,
        }
    except Exception as exc:
        logger.error("Energy analysis failed: %s", exc)
        return {"energy_level": 0.0, "spectral_centroid": 0.0, "tempo_stability": 0.0}


def _merge_two_sections(a: "SectionInfo", b: "SectionInfo") -> "SectionInfo":  # type: ignore[name-defined]
    """Merge two adjacent sections into one, keeping the longer one's type."""
    from backend.services.audio.types import SectionInfo
    total = a.duration_sec + b.duration_sec
    energy = (a.energy_level * a.duration_sec + b.energy_level * b.duration_sec) / total
    centroid = (a.spectral_centroid * a.duration_sec + b.spectral_centroid * b.duration_sec) / total
    longer = a if a.duration_sec >= b.duration_sec else b
    vocal = "dense" if energy > 0.65 else ("medium" if energy > 0.30 else "sparse")
    return SectionInfo(
        section_type=longer.section_type,
        start_sec=min(a.start_sec, b.start_sec),
        end_sec=max(a.end_sec, b.end_sec),
        duration_sec=total,
        energy_level=energy,
        spectral_centroid=centroid,
        tempo_stability=(a.tempo_stability + b.tempo_stability) / 2,
        vocal_density=vocal,
        vocal_intensity=energy,
        lyrical_content="",
        emotional_tone="neutral",
        lyrical_function="narrative",
        themes=[],
    )


def _retype_section(sec: "SectionInfo", new_type: str) -> "SectionInfo":  # type: ignore[name-defined]
    """Return a copy of sec with section_type replaced."""
    from backend.services.audio.types import SectionInfo
    return SectionInfo(
        section_type=new_type,
        start_sec=sec.start_sec,
        end_sec=sec.end_sec,
        duration_sec=sec.duration_sec,
        energy_level=sec.energy_level,
        spectral_centroid=sec.spectral_centroid,
        tempo_stability=sec.tempo_stability,
        vocal_density=sec.vocal_density,
        vocal_intensity=sec.vocal_intensity,
        lyrical_content=sec.lyrical_content,
        emotional_tone=sec.emotional_tone,
        lyrical_function=sec.lyrical_function,
        themes=sec.themes,
    )


def post_process_sections(
    sections: List,
    total_duration: float,
    min_sec: float = 12.0,
) -> List:
    """Merge tiny/adjacent-same-type sections and re-label with alternating pattern.

    Steps:
      1. Merge sections shorter than min_sec into their longer neighbor.
      2. Merge consecutive sections of the same type.
      3. Re-label inner sections using energy rank + verse/chorus alternation.
    """
    if len(sections) <= 1:
        return sections

    # ── Step 1: Absorb tiny sections ──────────────────────────────────────────
    changed = True
    while changed and len(sections) > 1:
        changed = False
        for i, sec in enumerate(sections):
            if sec.duration_sec < min_sec:
                if i < len(sections) - 1:
                    sections = sections[:i] + [_merge_two_sections(sec, sections[i + 1])] + sections[i + 2:]
                else:
                    sections = sections[:i - 1] + [_merge_two_sections(sections[i - 1], sec)] + sections[i + 1:]
                changed = True
                break

    # ── Step 2: Merge adjacent same-type sections ──────────────────────────────
    changed = True
    while changed and len(sections) > 1:
        changed = False
        for i in range(len(sections) - 1):
            if sections[i].section_type == sections[i + 1].section_type:
                sections = sections[:i] + [_merge_two_sections(sections[i], sections[i + 1])] + sections[i + 2:]
                changed = True
                break

    if len(sections) <= 2:
        return sections

    # ── Step 3: Re-label inner sections ───────────────────────────────────────
    inner = sections[1:-1]
    if not inner:
        return sections

    energies = sorted(s.energy_level for s in inner)
    median_e = energies[len(energies) // 2]

    labeled: List = []
    for sec in inner:
        if sec.energy_level >= median_e:
            new_type = "chorus"
        elif sec.energy_level < median_e * 0.55:
            new_type = "bridge"
        else:
            new_type = "verse"
        labeled.append(_retype_section(sec, new_type))

    # Ensure first inner section is verse, not chorus
    if labeled and labeled[0].section_type == "chorus":
        labeled[0] = _retype_section(labeled[0], "verse")

    # Break up any run of 3+ same type by flipping the middle one
    # (treat pre_chorus as distinct — never flip it into a run-breaker)
    for i in range(1, len(labeled) - 1):
        prev_t = labeled[i - 1].section_type
        curr_t = labeled[i].section_type
        next_t = labeled[i + 1].section_type
        if prev_t == curr_t == next_t and curr_t != "pre_chorus":
            flip = "verse" if curr_t == "chorus" else "chorus"
            labeled[i] = _retype_section(labeled[i], flip)

    # ── Step 4: Identify pre_chorus ───────────────────────────────────────────
    # A verse or chorus immediately preceding a higher-energy chorus, with
    # energy ≥ 60% of the following chorus, is likely a pre_chorus.
    for i in range(len(labeled) - 1):
        curr = labeled[i]
        nxt = labeled[i + 1]
        if (curr.section_type in ("verse", "chorus")
                and nxt.section_type == "chorus"
                and curr.energy_level >= 0.60 * nxt.energy_level
                and curr.energy_level < nxt.energy_level):
            labeled[i] = _retype_section(curr, "pre_chorus")

    return [sections[0]] + labeled + [sections[-1]]


def relabel_by_lyrical_repetition(
    sections: List,
    similarity_threshold: float = 0.5,
) -> List:
    """Override section labels for repeated lyric blocks (likely choruses).

    Computes pairwise Jaccard similarity between inner section word sets.
    Any inner section that shares >= similarity_threshold overlap with at
    least one other inner section is relabeled 'chorus'. Intro and outro
    are never changed. Skips silently if no section has lyrical_content
    (e.g. standard depth where Whisper has not run).
    """
    if len(sections) <= 2:
        return sections

    inner = sections[1:-1]
    if not any(s.lyrical_content for s in inner):
        return sections

    def _word_set(text: str) -> set:
        return set(text.lower().split()) if text else set()

    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    word_sets = [_word_set(s.lyrical_content) for s in inner]
    n = len(inner)

    relabeled = list(inner)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _jaccard(word_sets[i], word_sets[j]) >= similarity_threshold:
                relabeled[i] = _retype_section(inner[i], "chorus")
                break  # already promoted

    return [sections[0]] + relabeled + [sections[-1]]


def relabel_by_hook_phrase(sections: List, min_sections: int = 2) -> List:
    """Re-label sections using the most repeated all-content-word hook phrase.

    Algorithm:
      1. Extract 2–4 word n-grams from inner section lyrics where ALL words
         are content words (not stop words).
      2. The hook is the phrase appearing in the most distinct inner sections;
         ties broken by phrase length (longer wins), then lexicographic.
      3. Inner sections containing the hook → 'chorus' (intro/outro protected).
      4. Inner sections immediately preceding a hook-chorus that aren't already
         'verse' → 'pre_chorus'.
      5. Inner sections between two choruses with no hook and energy below the
         median chorus energy → 'bridge'.
    """
    if len(sections) <= 2:
        return sections

    inner = sections[1:-1]
    if not any(s.lyrical_content for s in inner):
        return sections

    _STOP = {
        "i", "im", "ill", "ive", "id", "me", "my", "myself", "we", "our",
        "you", "your", "he", "she", "it", "its", "they", "them", "their",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare",
        "a", "an", "the",
        "and", "but", "or", "nor", "for", "yet", "so",
        "in", "on", "at", "to", "of", "by", "up", "as", "if",
        "not", "no", "nor",
        "oh", "ah", "yeah", "hey", "ooh", "uh",
        "with", "from", "into", "about", "than", "then", "when",
        "just", "now", "got", "get", "let", "like",
    }

    import re

    def _tokens(text: str) -> List[str]:
        return [re.sub(r"[^a-z]", "", w.lower()) for w in re.split(r"\s+", text) if w]

    def _is_content(w: str) -> bool:
        return bool(w) and w not in _STOP and w.isalpha()

    # Collect content-word n-grams per inner section
    phrase_section_sets: dict = {}
    for idx, sec in enumerate(inner):
        if not sec.lyrical_content:
            continue
        words = _tokens(sec.lyrical_content)
        for n in range(2, 5):
            for i in range(len(words) - n + 1):
                gram = words[i : i + n]
                if all(_is_content(w) for w in gram):
                    phrase = " ".join(gram)
                    if phrase not in phrase_section_sets:
                        phrase_section_sets[phrase] = set()
                    phrase_section_sets[phrase].add(idx)

    candidates = [
        (phrase, secs)
        for phrase, secs in phrase_section_sets.items()
        if len(secs) >= min_sections
    ]

    if not candidates:
        return sections

    hook, hook_section_idxs = max(
        candidates, key=lambda x: (len(x[1]), len(x[0].split()), x[0])
    )

    result = list(inner)

    # Sections containing the hook → chorus
    for i in range(len(result)):
        if i in hook_section_idxs:
            result[i] = _retype_section(result[i], "chorus")

    # Section immediately before a hook-chorus that isn't verse/chorus → pre_chorus
    for i in range(1, len(result)):
        if result[i].section_type == "chorus" and i in hook_section_idxs:
            prev = result[i - 1]
            if prev.section_type not in ("verse", "chorus", "pre_chorus"):
                result[i - 1] = _retype_section(prev, "pre_chorus")

    # Between two choruses with no hook and energy < median chorus → bridge
    chorus_energies = [s.energy_level for s in result if s.section_type == "chorus"]
    if len(chorus_energies) >= 2:
        median_chorus_e = sorted(chorus_energies)[len(chorus_energies) // 2]
        for i in range(1, len(result) - 1):
            if (
                result[i - 1].section_type == "chorus"
                and result[i + 1].section_type == "chorus"
                and result[i].section_type not in ("chorus", "pre_chorus")
                and i not in hook_section_idxs
                and result[i].energy_level < median_chorus_e
            ):
                result[i] = _retype_section(result[i], "bridge")

    return [sections[0]] + result + [sections[-1]]


def estimate_key(chroma: np.ndarray) -> str:
    """Estimate musical key using Krumhansl-Schmuckler profile matching."""
    chroma_mean = np.mean(chroma, axis=1)
    total = np.sum(chroma_mean)
    if total == 0:
        return "Unknown"
    chroma_mean = chroma_mean / total

    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_profile = major_profile / np.sum(major_profile)
    minor_profile = minor_profile / np.sum(minor_profile)

    pitch_classes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best_corr = -1.0
    best_key = "Unknown"

    for shift in range(12):
        rotated = np.roll(chroma_mean, shift)
        corr_maj = float(np.corrcoef(rotated, major_profile)[0, 1])
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = f"{pitch_classes[shift]}maj"
        corr_min = float(np.corrcoef(rotated, minor_profile)[0, 1])
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = f"{pitch_classes[shift]}min"

    return best_key


def key_to_camelot(key: str) -> str:
    """Convert musical key string to Camelot wheel notation."""
    camelot_map = {
        "Cmaj": "8B",  "Gmaj": "9B",  "Dmaj": "10B", "Amaj": "11B",
        "Emaj": "12B", "Bmaj": "1B",  "F#maj": "2B", "C#maj": "3B",
        "G#maj": "4B", "D#maj": "5B", "A#maj": "6B", "Fmaj": "7B",
        "Amin": "8A",  "Emin": "9A",  "Bmin": "10A", "F#min": "11A",
        "C#min": "12A","G#min": "1A", "D#min": "2A", "A#min": "3A",
        "Fmin": "4A",  "Cmin": "5A",  "Gmin": "6A",  "Dmin": "7A",
    }
    return camelot_map.get(key, "Unknown")
