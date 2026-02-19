"""LoRA Downloader — search and download LoRAs from HuggingFace and Civitai."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional
from urllib.request import urlretrieve

from backend.services.lora.types import LoRAEntry, LoRASearchResult
from backend.services.prompt.types import NarrativeArc

logger = logging.getLogger("beat_studio.lora.downloader")


class LoRADownloader:
    """Downloads LoRAs from HuggingFace Hub and Civitai.

    Usage::

        dl = LoRADownloader(base_path="output/loras")
        results = dl.search("cinematic style", source="both")
        entry = dl.download(url=results[0].url, name="my_lora", lora_type="style")
    """

    _HF_API = "https://huggingface.co/api/models"
    _CIVITAI_API = "https://civitai.com/api/v1/models"

    def __init__(self, base_path: str = "output/loras"):
        self._base = Path(base_path)

    # ── public ────────────────────────────────────────────────────────────────

    def search(self, query: str, source: str = "both") -> List[LoRASearchResult]:
        """Search for LoRAs matching the query.

        Args:
            query: Free-text search term.
            source: "huggingface" | "civitai" | "both"

        Returns:
            List of LoRASearchResult sorted by confidence descending.
        """
        results: List[LoRASearchResult] = []
        if source in ("huggingface", "both"):
            try:
                results.extend(self._search_huggingface(query))
            except Exception as exc:
                logger.warning("HuggingFace search failed: %s", exc)
        if source in ("civitai", "both"):
            try:
                results.extend(self._search_civitai(query))
            except Exception as exc:
                logger.warning("Civitai search failed: %s", exc)
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def download(
        self,
        url: str,
        name: str,
        lora_type: str,
        trigger_token: Optional[str] = None,
        weight: float = 0.8,
    ) -> LoRAEntry:
        """Download a LoRA file and return a LoRAEntry for the registry.

        Args:
            url: Direct download URL.
            name: Registry name for this LoRA.
            lora_type: "style" | "character" | "scene" | "identity"
            trigger_token: Trigger token for the LoRA (guessed from name if None).
            weight: Default application weight.

        Returns:
            LoRAEntry ready to be passed to LoRARegistry.register().
        """
        dest_dir = self._base / lora_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".safetensors" if ".safetensors" in url else ".pt"
        dest_file = dest_dir / f"{name}{suffix}"

        local_path = self._download_file(url, str(dest_file))

        return LoRAEntry(
            name=name,
            type=lora_type,
            trigger_token=trigger_token or name.upper().replace("-", "_"),
            file_path=str(Path(lora_type) / f"{name}{suffix}"),
            weight=weight,
            status="available",
            source="huggingface" if "huggingface" in url else "civitai",
            source_url=url,
        )

    def recommend_downloads(self, narrative: NarrativeArc) -> List[LoRASearchResult]:
        """Suggest LoRAs to download based on the project narrative.

        Extracts themes from narrative sections and searches for matching LoRAs.
        """
        all_themes: set[str] = set()
        for sec in narrative.sections:
            all_themes.update(sec.themes)

        query = " ".join(list(all_themes)[:5])  # top 5 themes as query
        if not query:
            query = narrative.overall_concept[:60]

        return self.search(query, source="both")

    # ── internal: override in tests ───────────────────────────────────────────

    def _search_huggingface(self, query: str) -> List[LoRASearchResult]:
        """Query HuggingFace API for LoRA models. Returns list of results."""
        try:
            import requests
            resp = requests.get(
                self._HF_API,
                params={"search": query, "filter": "lora", "limit": 10},
                timeout=10,
            )
            resp.raise_for_status()
            items = resp.json()
        except Exception as exc:
            logger.warning("HF API call failed: %s", exc)
            return []

        results = []
        for item in items:
            results.append(LoRASearchResult(
                name=item.get("modelId", "unknown"),
                source="huggingface",
                url=f"https://huggingface.co/{item.get('modelId', '')}",
                type="style",
                description=item.get("description") or "",
                confidence=0.5,
            ))
        return results

    def _search_civitai(self, query: str) -> List[LoRASearchResult]:
        """Query Civitai API for LoRA models. Returns list of results."""
        try:
            import requests
            resp = requests.get(
                self._CIVITAI_API,
                params={"query": query, "types": "LORA", "limit": 10},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Civitai API call failed: %s", exc)
            return []

        results = []
        for item in data.get("items", []):
            results.append(LoRASearchResult(
                name=item.get("name", "unknown"),
                source="civitai",
                url=f"https://civitai.com/models/{item.get('id', '')}",
                type="style",
                description=item.get("description") or "",
                confidence=0.4,
            ))
        return results

    def _download_file(self, url: str, dest: str) -> str:
        """Download a file from URL to dest path. Returns dest."""
        logger.info("Downloading LoRA from %s → %s", url, dest)
        urlretrieve(url, dest)
        return dest
