"""Tests for Phase 4 LoRA management (Registry, Downloader, Recommender, Trainer)."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import yaml

from backend.services.lora.types import (
    DatasetResult, LoRAEntry, LoRARecommendation, LoRASearchResult,
    LoRATrainingConfig, LoRAValidation, TrainingResult,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_registry_yaml(tmp_path: Path, entries: list | None = None) -> Path:
    """Write a minimal loras.yaml and return its path."""
    if entries is None:
        entries = [
            {
                "name": "test_style_v1",
                "type": "style",
                "trigger_token": "TESTSTYLE",
                "file_path": "style/test_style_v1.safetensors",
                "weight": 0.8,
                "status": "available",
                "description": "A test style LoRA",
                "tags": ["cinematic", "film"],
                "source": "local",
                "source_url": None,
            },
            {
                "name": "test_char_v1",
                "type": "character",
                "trigger_token": "TESTCHAR",
                "file_path": "character/test_char_v1.safetensors",
                "weight": 0.85,
                "status": "missing",
                "description": "A test character LoRA",
                "tags": ["character", "portrait"],
                "source": "local",
                "source_url": None,
            },
        ]
    p = tmp_path / "loras.yaml"
    p.write_text(yaml.dump({"loras": entries}))
    return p


def _make_narrative_arc():
    from backend.services.prompt.types import NarrativeArc, NarrativeSection
    sec = NarrativeSection(
        section_index=0, section_type="chorus", start_sec=0.0, end_sec=30.0,
        visual_description="neon city at night",
        emotional_tone="triumphant", energy_level=0.9, is_climax=True,
        key_lyric="we rise", themes=["urban", "night", "neon"],
    )
    return NarrativeArc(
        artist="Test", title="Song",
        overall_concept="Urban triumph",
        color_palette=["neon", "dark"],
        mood_progression="hopeful → triumphant",
        visual_style_hint="cinematic",
        sections=[sec],
    )


def _make_animation_style():
    from backend.services.prompt.types import AnimationStyle
    return AnimationStyle(
        name="cinematic",
        prefix="cinematic film still, ",
        negative_prefix="cartoon, anime, ",
        recommended_model="wan26_cloud",
        cfg_scale=7.5,
        steps=25,
    )


# ── LoRARegistry ──────────────────────────────────────────────────────────────

class TestLoRARegistryLoad:
    def test_loads_yaml_and_returns_entries(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        entries = reg.list_all()
        assert len(entries) == 2

    def test_list_all_returns_lora_entry_objects(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        entries = reg.list_all()
        assert all(isinstance(e, LoRAEntry) for e in entries)

    def test_list_all_type_filter_style(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        styles = reg.list_all(type_filter="style")
        assert len(styles) == 1
        assert styles[0].name == "test_style_v1"

    def test_list_all_type_filter_character(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        chars = reg.list_all(type_filter="character")
        assert len(chars) == 1
        assert chars[0].name == "test_char_v1"

    def test_get_by_name_returns_entry(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        entry = reg.get("test_style_v1")
        assert entry is not None
        assert entry.trigger_token == "TESTSTYLE"

    def test_get_missing_name_returns_none(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        assert reg.get("nonexistent_xyz") is None

    def test_empty_registry_file(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        p = tmp_path / "loras.yaml"
        p.write_text("loras: []\n")
        reg = LoRARegistry(str(p), base_path=str(tmp_path))
        assert reg.list_all() == []


class TestLoRARegistryRegister:
    def test_register_adds_entry(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        new_entry = LoRAEntry(
            name="new_scene_v1", type="scene",
            trigger_token="NEWSCENE",
            file_path="scene/new_scene_v1.safetensors",
        )
        reg.register(new_entry)
        assert reg.get("new_scene_v1") is not None

    def test_register_persists_to_yaml(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        yaml_path = _make_registry_yaml(tmp_path)
        reg = LoRARegistry(str(yaml_path), base_path=str(tmp_path))
        reg.register(LoRAEntry(
            name="persistent_v1", type="style",
            trigger_token="PERSISTENT",
            file_path="style/persistent_v1.safetensors",
        ))
        # Re-load from disk
        reg2 = LoRARegistry(str(yaml_path), base_path=str(tmp_path))
        assert reg2.get("persistent_v1") is not None

    def test_register_duplicate_name_updates(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        updated = LoRAEntry(
            name="test_style_v1", type="style",
            trigger_token="UPDATED_TRIGGER",
            file_path="style/test_style_v1.safetensors",
        )
        reg.register(updated)
        assert reg.get("test_style_v1").trigger_token == "UPDATED_TRIGGER"
        assert len(reg.list_all()) == 2  # No duplicate


class TestLoRARegistryValidate:
    def test_validate_missing_file_returns_invalid(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        result = reg.validate("test_style_v1")
        assert isinstance(result, LoRAValidation)
        assert not result.file_exists

    def test_validate_existing_file_returns_valid(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        # Create the file
        lora_file = tmp_path / "style" / "test_style_v1.safetensors"
        lora_file.parent.mkdir(parents=True)
        lora_file.write_bytes(b"\x00" * 128)
        result = reg.validate("test_style_v1")
        assert result.file_exists
        assert result.valid

    def test_validate_unknown_name_returns_invalid(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        reg = LoRARegistry(str(_make_registry_yaml(tmp_path)), base_path=str(tmp_path))
        result = reg.validate("does_not_exist")
        assert not result.valid
        assert result.error is not None


# ── LoRADownloader ─────────────────────────────────────────────────────────────

class TestLoRADownloader:
    def test_search_returns_list(self):
        from backend.services.lora.downloader import LoRADownloader
        dl = LoRADownloader()
        with patch.object(dl, "_search_huggingface", return_value=[
            LoRASearchResult(
                name="hf_lora", source="huggingface",
                url="https://hf.co/test/lora",
                type="style", description="A style lora",
                confidence=0.9,
            )
        ]):
            results = dl.search("cinematic style", source="huggingface")
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_search_civitai_returns_list(self):
        from backend.services.lora.downloader import LoRADownloader
        dl = LoRADownloader()
        with patch.object(dl, "_search_civitai", return_value=[
            LoRASearchResult(
                name="civitai_lora", source="civitai",
                url="https://civitai.com/models/1",
                type="style", description="A civitai lora",
                confidence=0.8,
            )
        ]):
            results = dl.search("urban graffiti", source="civitai")
        assert isinstance(results, list)

    def test_search_both_combines_results(self):
        from backend.services.lora.downloader import LoRADownloader
        dl = LoRADownloader()
        hf_result = LoRASearchResult("hf", "huggingface", "https://hf.co/x", "style", "desc")
        civ_result = LoRASearchResult("civ", "civitai", "https://civitai.com/1", "style", "desc")
        with patch.object(dl, "_search_huggingface", return_value=[hf_result]), \
             patch.object(dl, "_search_civitai", return_value=[civ_result]):
            results = dl.search("landscape", source="both")
        assert len(results) == 2

    def test_download_returns_lora_entry(self, tmp_path):
        from backend.services.lora.downloader import LoRADownloader
        dl = LoRADownloader(base_path=str(tmp_path))
        with patch.object(dl, "_download_file", return_value=str(tmp_path / "test.safetensors")):
            (tmp_path / "test.safetensors").write_bytes(b"\x00" * 64)
            entry = dl.download(
                url="https://example.com/lora.safetensors",
                name="test_download", lora_type="style",
            )
        assert isinstance(entry, LoRAEntry)
        assert entry.name == "test_download"

    def test_recommend_downloads_from_narrative(self):
        from backend.services.lora.downloader import LoRADownloader
        dl = LoRADownloader()
        narrative = _make_narrative_arc()
        fake_results = [
            LoRASearchResult("neon_city", "huggingface", "https://hf.co/neon", "scene",
                             "Neon city scenes", confidence=0.85)
        ]
        with patch.object(dl, "_search_huggingface", return_value=fake_results), \
             patch.object(dl, "_search_civitai", return_value=[]):
            results = dl.recommend_downloads(narrative)
        assert isinstance(results, list)


# ── LoRARecommender ───────────────────────────────────────────────────────────

class TestLoRARecommender:
    def _make_recommender(self, tmp_path):
        from backend.services.lora.registry import LoRARegistry
        from backend.services.lora.downloader import LoRADownloader
        from backend.services.lora.recommender import LoRARecommender

        # Add an available style lora tagged "cinematic"
        entries = [{
            "name": "cinematic_v1", "type": "style",
            "trigger_token": "CINEMATIC", "file_path": "style/cinematic_v1.safetensors",
            "weight": 0.8, "status": "available",
            "description": "Cinematic style", "tags": ["cinematic", "film"],
            "source": "local", "source_url": None,
        }]
        yaml_path = _make_registry_yaml(tmp_path, entries)
        # Create the file so validate passes
        lora_file = tmp_path / "style" / "cinematic_v1.safetensors"
        lora_file.parent.mkdir(parents=True, exist_ok=True)
        lora_file.write_bytes(b"\x00" * 64)

        registry = LoRARegistry(str(yaml_path), base_path=str(tmp_path))
        downloader = LoRADownloader(base_path=str(tmp_path))
        return LoRARecommender(registry=registry, downloader=downloader)

    def test_recommend_returns_recommendation_object(self, tmp_path):
        recommender = self._make_recommender(tmp_path)
        narrative = _make_narrative_arc()
        style = _make_animation_style()
        with patch.object(recommender.downloader, "_search_huggingface", return_value=[]), \
             patch.object(recommender.downloader, "_search_civitai", return_value=[]):
            rec = recommender.recommend(narrative, style)
        assert isinstance(rec, LoRARecommendation)

    def test_available_includes_matching_loras(self, tmp_path):
        recommender = self._make_recommender(tmp_path)
        narrative = _make_narrative_arc()
        style = _make_animation_style()  # style.name == "cinematic"
        with patch.object(recommender.downloader, "_search_huggingface", return_value=[]), \
             patch.object(recommender.downloader, "_search_civitai", return_value=[]):
            rec = recommender.recommend(narrative, style)
        # The "cinematic_v1" lora is tagged "cinematic" and style is "cinematic"
        assert any(e.name == "cinematic_v1" for e in rec.available)

    def test_trainable_has_suggestions(self, tmp_path):
        recommender = self._make_recommender(tmp_path)
        narrative = _make_narrative_arc()  # themes: urban, night, neon
        style = _make_animation_style()
        with patch.object(recommender.downloader, "_search_huggingface", return_value=[]), \
             patch.object(recommender.downloader, "_search_civitai", return_value=[]):
            rec = recommender.recommend(narrative, style)
        assert isinstance(rec.trainable, list)


# ── LoRATrainer ───────────────────────────────────────────────────────────────

class TestLoRATrainer:
    def test_train_returns_training_result(self, tmp_path):
        from backend.services.lora.trainer import LoRATrainer
        trainer = LoRATrainer()
        config = LoRATrainingConfig(
            dataset_path=str(tmp_path / "dataset"),
            lora_type="style",
            trigger_token="MYSTYLE",
            output_name="my_style_v1",
        )
        result = trainer.train(config)
        assert isinstance(result, TrainingResult)

    def test_train_stub_fails_without_gpu(self, tmp_path):
        from backend.services.lora.trainer import LoRATrainer
        trainer = LoRATrainer()
        config = LoRATrainingConfig(
            dataset_path=str(tmp_path / "dataset"),
            lora_type="character",
            trigger_token="MYCHAR",
            output_name="my_char_v1",
        )
        result = trainer.train(config)
        # Stub returns success=False (no GPU/dataset) or raises — both acceptable
        assert isinstance(result, TrainingResult)
        assert result.error is not None or not result.success

    def test_prepare_dataset_empty_dir(self, tmp_path):
        from backend.services.lora.trainer import LoRATrainer
        trainer = LoRATrainer()
        result = trainer.prepare_dataset(str(tmp_path / "empty"))
        assert isinstance(result, DatasetResult)
        assert not result.success  # no images

    def test_prepare_dataset_with_images(self, tmp_path):
        from backend.services.lora.trainer import LoRATrainer
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        for i in range(3):
            (dataset_dir / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        trainer = LoRATrainer()
        result = trainer.prepare_dataset(str(dataset_dir))
        assert isinstance(result, DatasetResult)
        assert result.image_count == 3
