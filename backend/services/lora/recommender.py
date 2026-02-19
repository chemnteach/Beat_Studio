"""LoRA Recommender — orchestrates registry + downloader to suggest LoRAs for a project."""
from __future__ import annotations

import logging
from typing import List

from backend.services.lora.downloader import LoRADownloader
from backend.services.lora.registry import LoRARegistry
from backend.services.lora.types import LoRAEntry, LoRARecommendation, LoRASearchResult
from backend.services.prompt.types import AnimationStyle, NarrativeArc

logger = logging.getLogger("beat_studio.lora.recommender")


class LoRARecommender:
    """Recommends LoRAs for a video project.

    For a given NarrativeArc + AnimationStyle, returns:
    - available: LoRAs already on disk that match the project
    - downloadable: Online LoRAs worth downloading
    - trainable: Text suggestions for new LoRAs to train

    Usage::

        recommender = LoRARecommender(registry, downloader)
        rec = recommender.recommend(narrative, style)
    """

    def __init__(self, registry: LoRARegistry, downloader: LoRADownloader):
        self.registry = registry
        self.downloader = downloader

    def recommend(
        self,
        narrative: NarrativeArc,
        style: AnimationStyle,
    ) -> LoRARecommendation:
        """Generate LoRA recommendations for a project.

        Args:
            narrative: The visual narrative arc.
            style: The selected animation style.

        Returns:
            LoRARecommendation with available, downloadable, trainable lists.
        """
        available = self._find_available(narrative, style)
        downloadable = self._find_downloadable(narrative, style)
        trainable = self._suggest_trainable(narrative, style, available)

        return LoRARecommendation(
            available=available,
            downloadable=downloadable,
            trainable=trainable,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _find_available(
        self, narrative: NarrativeArc, style: AnimationStyle
    ) -> List[LoRAEntry]:
        """Find on-disk LoRAs that match the project themes and style."""
        all_entries = self.registry.list_all()
        matched: List[LoRAEntry] = []

        # Collect all project themes
        project_themes: set[str] = set()
        for sec in narrative.sections:
            project_themes.update(t.lower() for t in sec.themes)
        project_themes.add(style.name.lower())

        for entry in all_entries:
            # Only include actually-available LoRAs
            validation = self.registry.validate(entry.name)
            if not validation.file_exists:
                continue

            entry_tags = {t.lower() for t in entry.tags}
            if entry_tags & project_themes:
                matched.append(entry)

        return matched

    def _find_downloadable(
        self, narrative: NarrativeArc, style: AnimationStyle
    ) -> List[LoRASearchResult]:
        """Search online for LoRAs that would benefit this project."""
        # Build a focused query from style + top themes
        all_themes: List[str] = []
        for sec in narrative.sections:
            all_themes.extend(sec.themes)

        # Deduplicate, keep top 3
        seen: set[str] = set()
        unique_themes: List[str] = []
        for t in all_themes:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique_themes.append(t)
            if len(unique_themes) >= 3:
                break

        query = f"{style.name} {' '.join(unique_themes)}".strip()
        try:
            return self.downloader.search(query, source="both")[:5]
        except Exception as exc:
            logger.warning("Download search failed: %s", exc)
            return []

    def _suggest_trainable(
        self,
        narrative: NarrativeArc,
        style: AnimationStyle,
        available: List[LoRAEntry],
    ) -> List[str]:
        """Suggest new LoRAs to train for unmatched project needs."""
        available_tags: set[str] = set()
        for entry in available:
            available_tags.update(t.lower() for t in entry.tags)

        suggestions: List[str] = []

        # Check if character is covered
        character_entries = self.registry.list_all(type_filter="character")
        if not character_entries:
            suggestions.append(
                "No character LoRA registered. Consider training a character LoRA "
                "for consistent on-screen talent."
            )

        # Check dominant unmatched themes
        theme_counts: dict[str, int] = {}
        for sec in narrative.sections:
            for theme in sec.themes:
                theme_counts[theme.lower()] = theme_counts.get(theme.lower(), 0) + 1

        for theme, count in sorted(theme_counts.items(), key=lambda x: -x[1]):
            if theme not in available_tags and count >= 2:
                suggestions.append(
                    f"A scene LoRA for '{theme}' would improve "
                    f"{count} sections in the narrative."
                )

        return suggestions
