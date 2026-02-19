"""StyleMapper — YAML-backed registry and recommendation engine for AnimationStyles."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from backend.services.prompt.types import AnimationStyle

logger = logging.getLogger("beat_studio.prompt.style_mapper")

# Path to the YAML style definitions (relative to this file)
_YAML_PATH = Path(__file__).parent.parent.parent / "config" / "animation_styles.yaml"

# Energy buckets
_HIGH = 0.6
_LOW = 0.3

# (energy_bucket, mood_lower) → style name
_MOOD_MAP: Dict[tuple, str] = {
    ("high", "energetic"): "cinematic",
    ("high", "triumphant"): "cinematic",
    ("high", "defiant"): "comic_book",
    ("high", "euphoric"): "anime",
    ("high", "excited"): "cinematic",
    ("high", "dark"): "synthwave",
    ("medium", "hopeful"): "cinematic",
    ("medium", "reflective"): "lofi",
    ("medium", "melancholic"): "lofi",
    ("medium", "serene"): "watercolor",
    ("medium", "romantic"): "cinematic",
    ("medium", "neutral"): "cinematic",
    ("low", "melancholic"): "lofi",
    ("low", "serene"): "lofi",
    ("low", "reflective"): "lofi",
    ("low", "peaceful"): "watercolor",
    ("low", "nostalgic"): "lofi",
    ("low", "neutral"): "lofi",
}

# genre → ordered style preferences (high energy, then low energy fallback)
_GENRE_MAP: Dict[str, List[str]] = {
    "hip-hop":     ["graffiti", "comic_book", "pop_art", "lofi"],
    "rap":         ["graffiti", "comic_book", "pop_art", "lofi"],
    "electronic":  ["synthwave", "psychedelic", "motion_graphics", "abstract"],
    "edm":         ["synthwave", "psychedelic", "motion_graphics"],
    "folk":        ["watercolor", "pencil_sketch", "impressionist", "lofi"],
    "acoustic":    ["watercolor", "pencil_sketch", "oil_painting", "lofi"],
    "classical":   ["oil_painting", "impressionist", "ink_wash", "watercolor"],
    "orchestral":  ["oil_painting", "impressionist", "cinematic"],
    "pop":         ["pop_art", "anime", "cel_animation", "cinematic"],
    "k-pop":       ["anime", "pop_art", "cel_animation"],
    "j-pop":       ["anime", "cel_animation", "pop_art"],
    "rock":        ["comic_book", "graffiti", "oil_painting"],
    "punk":        ["comic_book", "graffiti", "pop_art"],
    "jazz":        ["art_deco", "impressionist", "watercolor"],
    "blues":       ["oil_painting", "pencil_sketch", "impressionist"],
    "country":     ["watercolor", "pencil_sketch", "oil_painting"],
    "trop rock":   ["watercolor", "cel_animation", "impressionist"],
    "ambient":     ["ink_wash", "impressionist", "abstract", "lofi"],
    "lofi":        ["lofi", "isometric", "watercolor"],
    "synthwave":   ["synthwave", "abstract", "motion_graphics"],
    "vaporwave":   ["synthwave", "psychedelic", "abstract"],
    "world":       ["ink_wash", "ukiyo_e", "impressionist"],
    "chiptune":    ["pixel_art", "motion_graphics", "cel_animation"],
}


class StyleNotFoundError(KeyError):
    """Raised when a requested style name is not in the registry."""


class StyleMapper:
    """YAML-backed registry and recommendation engine for AnimationStyles.

    Styles are loaded from ``backend/config/animation_styles.yaml`` on first
    use. Falls back to a minimal hardcoded set if the file is not found.

    Usage::

        mapper = StyleMapper()
        style = mapper.get_style("cinematic")
        rec = mapper.recommend(energy_level=0.8, mood="triumphant", genre="pop")
        top3 = mapper.recommend_top(energy_level=0.6, mood="hopeful", genre="folk", n=3)
    """

    def __init__(self, yaml_path: Optional[str] = None):
        self._yaml_path = Path(yaml_path) if yaml_path else _YAML_PATH
        self._styles: Optional[Dict[str, AnimationStyle]] = None

    # ── public ────────────────────────────────────────────────────────────────

    def get_style(self, name: str) -> AnimationStyle:
        """Return the AnimationStyle for the given name.

        Raises:
            StyleNotFoundError: If no style with that name is registered.
        """
        styles = self._load()
        try:
            return styles[name]
        except KeyError:
            raise StyleNotFoundError(
                f"Style '{name}' not found. Available: {list(styles.keys())}"
            )

    def list_styles(self) -> List[str]:
        """Return all registered style names."""
        return list(self._load().keys())

    def recommend(
        self,
        energy_level: float,
        mood: str,
        genre: str = "",
    ) -> AnimationStyle:
        """Recommend a single AnimationStyle based on energy, mood, and genre.

        Args:
            energy_level: 0.0-1.0 energy value from audio analysis.
            mood: Mood string (e.g. "energetic", "melancholic", "triumphant").
            genre: Optional music genre (e.g. "hip-hop", "folk", "electronic").

        Returns:
            Recommended AnimationStyle instance.
        """
        return self.recommend_top(energy_level, mood, genre, n=1)[0]

    def recommend_top(
        self,
        energy_level: float,
        mood: str,
        genre: str = "",
        n: int = 3,
    ) -> List[AnimationStyle]:
        """Recommend top N AnimationStyles ranked by fit.

        Args:
            energy_level: 0.0-1.0 energy value.
            mood: Mood string.
            genre: Optional music genre.
            n: Number of recommendations to return.

        Returns:
            List of up to n AnimationStyle instances (no duplicates).
        """
        styles = self._load()
        candidates: List[str] = []

        # 1. Genre-based preferences (ordered)
        genre_lower = genre.lower().strip()
        if genre_lower in _GENRE_MAP:
            candidates.extend(_GENRE_MAP[genre_lower])

        # 2. Mood + energy fallback
        bucket = "high" if energy_level >= _HIGH else ("medium" if energy_level >= _LOW else "low")
        mood_key = (bucket, mood.lower())
        mood_style = _MOOD_MAP.get(mood_key)
        if mood_style:
            candidates.append(mood_style)

        # 3. Hard defaults
        default = "cinematic" if energy_level >= 0.5 else "lofi"
        candidates.append(default)

        # Deduplicate while preserving order, filter to known styles
        seen: set[str] = set()
        result: List[AnimationStyle] = []
        for name in candidates:
            if name not in seen and name in styles:
                seen.add(name)
                result.append(styles[name])
            if len(result) >= n:
                break

        # If we still need more, fill from all styles
        if len(result) < n:
            for name, style in styles.items():
                if name not in seen:
                    seen.add(name)
                    result.append(style)
                if len(result) >= n:
                    break

        return result[:n]

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, AnimationStyle]:
        """Load styles from YAML (cached after first load)."""
        if self._styles is not None:
            return self._styles

        if self._yaml_path.exists():
            self._styles = self._load_from_yaml(self._yaml_path)
        else:
            logger.warning(
                "animation_styles.yaml not found at %s — using minimal fallback",
                self._yaml_path,
            )
            self._styles = self._fallback_styles()

        logger.debug("Loaded %d animation styles", len(self._styles))
        return self._styles

    def _load_from_yaml(self, path: Path) -> Dict[str, AnimationStyle]:
        """Parse animation_styles.yaml into AnimationStyle instances."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}

        raw_styles = data.get("styles", {}) or {}
        result: Dict[str, AnimationStyle] = {}

        for name, cfg in raw_styles.items():
            result[name] = AnimationStyle(
                name=name,
                # YAML uses prompt_suffix; we use it as the style prefix in prompts
                prefix=cfg.get("prompt_suffix", "") + ", ",
                negative_prefix=cfg.get("negative", "") + ", ",
                recommended_model=cfg.get("recommended_backend", "animatediff"),
                cfg_scale=float(cfg.get("guidance_scale", 7.5)),
                steps=int(cfg.get("steps", 20)),
            )

        return result

    @staticmethod
    def _fallback_styles() -> Dict[str, AnimationStyle]:
        """Minimal hardcoded styles as fallback when YAML is missing."""
        return {
            "cinematic": AnimationStyle(
                name="cinematic",
                prefix="cinematic film still, 35mm, dramatic lighting, ",
                negative_prefix="cartoon, anime, illustration, ",
                recommended_model="animatediff",
                cfg_scale=7.5, steps=25,
            ),
            "anime": AnimationStyle(
                name="anime",
                prefix="anime style, vibrant colors, detailed linework, ",
                negative_prefix="photorealistic, live action, ",
                recommended_model="animatediff",
                cfg_scale=7.0, steps=20,
            ),
            "lofi": AnimationStyle(
                name="lofi",
                prefix="lofi aesthetic, warm tones, grain texture, ",
                negative_prefix="sharp, harsh lighting, clinical, ",
                recommended_model="animatediff",
                cfg_scale=6.5, steps=20,
            ),
            "abstract": AnimationStyle(
                name="abstract",
                prefix="abstract art, geometric shapes, flowing colors, ",
                negative_prefix="realistic, figurative, photographic, ",
                recommended_model="animatediff",
                cfg_scale=8.0, steps=25,
            ),
            "photorealistic": AnimationStyle(
                name="photorealistic",
                prefix="photorealistic, hyperdetailed, DSLR photograph, ",
                negative_prefix="cartoon, anime, drawing, painting, ",
                recommended_model="wan26_cloud",
                cfg_scale=7.0, steps=30,
            ),
        }
