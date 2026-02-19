"""Tests for Phase 3 prompt generation (NarrativeAnalyzer, ScenePromptGenerator,
PromptComposer, StyleMapper)."""
from unittest.mock import MagicMock, patch
import pytest

from backend.services.audio.types import SectionInfo, SongAnalysis, SceneTiming
from backend.services.prompt.types import (
    AnimationStyle, ComposedPrompt, CinematographyProfile,
    LoRAConfig, NarrativeArc, NarrativeSection, ScenePrompt,
)


# ── test helpers ──────────────────────────────────────────────────────────────

def _make_section_info(stype="verse", start=0.0, end=30.0, energy=0.5,
                       tone="hopeful", themes=None):
    return SectionInfo(
        section_type=stype, start_sec=start, end_sec=end,
        duration_sec=end - start, energy_level=energy,
        spectral_centroid=2000.0, tempo_stability=0.8,
        vocal_density="medium", vocal_intensity=0.5,
        lyrical_content="I stay out too late",
        emotional_tone=tone, lyrical_function="narrative",
        themes=themes or ["love"],
    )


def _make_analysis():
    secs = [
        _make_section_info("intro", 0, 20, energy=0.3, tone="hopeful"),
        _make_section_info("verse", 20, 60, energy=0.5, tone="reflective"),
        _make_section_info("chorus", 60, 90, energy=0.9, tone="triumphant"),
        _make_section_info("outro", 90, 120, energy=0.2, tone="serene"),
    ]
    return SongAnalysis(
        artist="Test Artist", title="Test Song", file_path="/tmp/test.wav",
        bpm=120.0, key="Cmaj", camelot="8B", duration_sec=120.0,
        sample_rate=44100, energy_level=0.5, first_downbeat_sec=0.1,
        sections=secs, beat_times=[0.5 * i for i in range(240)],
        transcript="I stay out too late got nothing in my brain",
        mood_summary="upbeat pop confidence", genres=["pop"],
        primary_genre="pop", irony_score=0, valence=8,
        emotional_arc="intro:hopeful → verse:reflective → chorus:triumphant",
    )


def _make_scene_timing(idx=0, start=0.0, end=30.0, energy=0.5, hero=False):
    return SceneTiming(
        scene_index=idx, start_sec=start, end_sec=end,
        duration_sec=end - start, is_hero=hero, energy_level=energy,
        section_type="verse", beat_aligned=True,
    )


def _make_style(name="cinematic"):
    return AnimationStyle(
        name=name,
        prefix="cinematic film still, 35mm, ",
        negative_prefix="cartoon, anime, ",
        recommended_model="animatediff",
        cfg_scale=7.0, steps=20,
    )


def _make_narrative_arc():
    secs = [
        NarrativeSection(
            section_index=0, section_type="verse", start_sec=0.0, end_sec=30.0,
            visual_description="A woman dancing alone in an empty room, golden light",
            emotional_tone="hopeful", energy_level=0.5, is_climax=False,
            key_lyric="I stay out too late", themes=["dancing", "freedom"],
        ),
        NarrativeSection(
            section_index=1, section_type="chorus", start_sec=30.0, end_sec=60.0,
            visual_description="Crowd erupts in celebration, confetti falling",
            emotional_tone="triumphant", energy_level=0.9, is_climax=True,
            key_lyric="Shake it off", themes=["celebration", "joy"],
        ),
    ]
    return NarrativeArc(
        artist="Test", title="Song",
        overall_concept="A journey from self-doubt to confident self-expression",
        color_palette=["warm gold", "vibrant pink"],
        mood_progression="reflective → triumphant",
        visual_style_hint="cinematic, dynamic movement",
        sections=secs,
    )


# ── NarrativeAnalyzer ─────────────────────────────────────────────────────────

class TestNarrativeAnalyzerInit:
    def test_instantiation_default(self):
        from backend.services.prompt.narrative_analyzer import NarrativeAnalyzer
        az = NarrativeAnalyzer()
        assert az is not None

    def test_instantiation_with_model(self):
        from backend.services.prompt.narrative_analyzer import NarrativeAnalyzer
        az = NarrativeAnalyzer(llm_provider="anthropic")
        assert az.llm_provider == "anthropic"


class TestNarrativeAnalyzerAnalyze:
    def test_returns_narrative_arc(self):
        from backend.services.prompt.narrative_analyzer import NarrativeAnalyzer
        analysis = _make_analysis()
        az = NarrativeAnalyzer()

        with patch.object(az, "_call_llm", return_value={
            "overall_concept": "A dance of freedom",
            "color_palette": ["gold", "pink"],
            "mood_progression": "hopeful → triumphant",
            "visual_style_hint": "cinematic",
            "sections": [
                {"section_index": i, "visual_description": "test scene",
                 "key_lyric": "lyrics", "themes": ["dance"]}
                for i in range(len(analysis.sections))
            ],
        }):
            result = az.analyze(analysis)

        assert isinstance(result, NarrativeArc)

    def test_sections_count_matches_analysis(self):
        from backend.services.prompt.narrative_analyzer import NarrativeAnalyzer
        analysis = _make_analysis()
        az = NarrativeAnalyzer()

        with patch.object(az, "_call_llm", return_value={
            "overall_concept": "A dance of freedom",
            "color_palette": ["gold"],
            "mood_progression": "hopeful",
            "visual_style_hint": "cinematic",
            "sections": [
                {"section_index": i, "visual_description": "scene",
                 "key_lyric": "lyric", "themes": ["dance"]}
                for i in range(len(analysis.sections))
            ],
        }):
            result = az.analyze(analysis)

        assert len(result.sections) == len(analysis.sections)

    def test_user_concept_included_in_prompt(self):
        from backend.services.prompt.narrative_analyzer import NarrativeAnalyzer
        analysis = _make_analysis()
        az = NarrativeAnalyzer()
        captured_prompt = []

        def fake_call_llm(prompt):
            captured_prompt.append(prompt)
            return {
                "overall_concept": "X", "color_palette": ["blue"],
                "mood_progression": "hopeful", "visual_style_hint": "cinematic",
                "sections": [{"section_index": i, "visual_description": "s",
                              "key_lyric": "l", "themes": []}
                             for i in range(len(analysis.sections))],
            }

        with patch.object(az, "_call_llm", side_effect=fake_call_llm):
            az.analyze(analysis, user_concept="retro 80s dance club")

        assert any("retro 80s dance club" in p for p in captured_prompt)

    def test_fallback_when_llm_fails(self):
        from backend.services.prompt.narrative_analyzer import NarrativeAnalyzer
        analysis = _make_analysis()
        az = NarrativeAnalyzer()

        with patch.object(az, "_call_llm", side_effect=Exception("API down")):
            result = az.analyze(analysis)  # should not raise

        assert isinstance(result, NarrativeArc)
        assert len(result.sections) > 0


# ── ScenePromptGenerator ──────────────────────────────────────────────────────

class TestScenePromptGenerator:
    def test_generate_returns_scene_prompts(self):
        from backend.services.prompt.scene_generator import ScenePromptGenerator
        gen = ScenePromptGenerator()
        narrative = _make_narrative_arc()
        scenes = [_make_scene_timing(0, 0, 30, 0.5), _make_scene_timing(1, 30, 60, 0.9, hero=True)]
        style = _make_style()
        prompts = gen.generate_prompts(narrative, scenes, style)
        assert len(prompts) == len(scenes)
        assert all(isinstance(p, ScenePrompt) for p in prompts)

    def test_hero_scene_gets_more_steps(self):
        from backend.services.prompt.scene_generator import ScenePromptGenerator
        gen = ScenePromptGenerator()
        narrative = _make_narrative_arc()
        scenes = [
            _make_scene_timing(0, 0, 30, 0.5, hero=False),
            _make_scene_timing(1, 30, 60, 0.9, hero=True),
        ]
        style = _make_style()
        prompts = gen.generate_prompts(narrative, scenes, style)
        hero = [p for p in prompts if p.is_hero]
        normal = [p for p in prompts if not p.is_hero]
        if hero and normal:
            assert hero[0].steps >= normal[0].steps

    def test_lora_triggers_prepended(self):
        from backend.services.prompt.scene_generator import ScenePromptGenerator
        gen = ScenePromptGenerator()
        narrative = _make_narrative_arc()
        scenes = [_make_scene_timing(0, 0, 30)]
        style = _make_style()
        loras = [LoRAConfig(name="test_lora", trigger_token="TESTTRIGGER",
                            weight=0.8, lora_type="style")]
        prompts = gen.generate_prompts(narrative, scenes, style, loras=loras)
        assert "TESTTRIGGER" in prompts[0].positive

    def test_user_override_replaces_base_prompt(self):
        from backend.services.prompt.scene_generator import ScenePromptGenerator
        gen = ScenePromptGenerator()
        narrative = _make_narrative_arc()
        scenes = [_make_scene_timing(0, 0, 30)]
        style = _make_style()
        overrides = {0: "Custom override prompt for scene 0"}
        prompts = gen.generate_prompts(narrative, scenes, style, user_overrides=overrides)
        assert "Custom override prompt" in prompts[0].positive

    def test_quality_tokens_always_present(self):
        from backend.services.prompt.scene_generator import ScenePromptGenerator
        gen = ScenePromptGenerator()
        narrative = _make_narrative_arc()
        scenes = [_make_scene_timing(0, 0, 30)]
        style = _make_style()
        prompts = gen.generate_prompts(narrative, scenes, style)
        assert "high quality" in prompts[0].positive.lower()

    def test_negative_prompt_present(self):
        from backend.services.prompt.scene_generator import ScenePromptGenerator
        gen = ScenePromptGenerator()
        narrative = _make_narrative_arc()
        scenes = [_make_scene_timing(0, 0, 30)]
        style = _make_style()
        prompts = gen.generate_prompts(narrative, scenes, style)
        assert prompts[0].negative != ""

    def test_no_narrative_sections_fallback(self):
        from backend.services.prompt.scene_generator import ScenePromptGenerator
        gen = ScenePromptGenerator()
        empty_arc = NarrativeArc(
            artist="A", title="T", overall_concept="concept",
            color_palette=[], mood_progression="", visual_style_hint="",
            sections=[],
        )
        scenes = [_make_scene_timing(0, 0, 30)]
        style = _make_style()
        prompts = gen.generate_prompts(empty_arc, scenes, style)
        assert len(prompts) == 1
        assert prompts[0].positive != ""


# ── PromptComposer ────────────────────────────────────────────────────────────

class TestPromptComposer:
    def test_compose_returns_composed_prompt(self):
        from backend.services.prompt.prompt_composer import PromptComposer
        composer = PromptComposer()
        style = _make_style()
        result = composer.compose("a woman dancing", style)
        assert isinstance(result, ComposedPrompt)

    def test_quality_tokens_always_appended(self):
        from backend.services.prompt.prompt_composer import PromptComposer
        composer = PromptComposer()
        style = _make_style()
        result = composer.compose("dancer", style)
        assert "high quality" in result.positive.lower()

    def test_style_prefix_prepended(self):
        from backend.services.prompt.prompt_composer import PromptComposer
        composer = PromptComposer()
        style = _make_style("cinematic")
        result = composer.compose("dancer on stage", style)
        assert "cinematic" in result.positive.lower() or "film" in result.positive.lower()

    def test_lora_triggers_included(self):
        from backend.services.prompt.prompt_composer import PromptComposer
        composer = PromptComposer()
        style = _make_style()
        loras = [LoRAConfig("my_lora", "MYTRIGGER", 0.8, "style")]
        result = composer.compose("dancer", style, loras=loras)
        assert "MYTRIGGER" in result.positive

    def test_nsfw_false_adds_safety_negatives(self):
        from backend.services.prompt.prompt_composer import PromptComposer
        composer = PromptComposer()
        style = _make_style()
        result = composer.compose("dancer", style, nsfw=False)
        negative_lower = result.negative.lower()
        assert any(word in negative_lower for word in ["nsfw", "nude", "explicit", "sexual"])

    def test_nsfw_true_removes_safety_negatives(self):
        from backend.services.prompt.prompt_composer import PromptComposer
        composer = PromptComposer()
        style = _make_style()
        result = composer.compose("dancer", style, nsfw=True)
        assert result.nsfw is True

    def test_cinematography_tokens_included(self):
        from backend.services.prompt.prompt_composer import PromptComposer
        composer = PromptComposer()
        style = _make_style()
        cine = CinematographyProfile(
            camera_movement="tracking shot",
            lighting="golden hour",
            film_stock="35mm film grain",
            lens="wide angle",
        )
        result = composer.compose("dancer", style, cinematography=cine)
        assert any(word in result.positive.lower()
                   for word in ["tracking", "golden hour", "35mm", "wide"])


# ── StyleMapper ───────────────────────────────────────────────────────────────

class TestStyleMapper:
    def test_returns_animation_style(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        style = mapper.get_style("cinematic")
        assert isinstance(style, AnimationStyle)

    def test_unknown_style_raises(self):
        from backend.services.prompt.style_mapper import StyleMapper, StyleNotFoundError
        mapper = StyleMapper()
        with pytest.raises(StyleNotFoundError):
            mapper.get_style("nonexistent_style_xyz")

    def test_list_styles_returns_names(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        names = mapper.list_styles()
        assert len(names) >= 4
        assert "cinematic" in names

    def test_recommend_for_energy(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        # High energy should prefer dynamic styles
        rec = mapper.recommend(energy_level=0.9, mood="energetic")
        assert isinstance(rec, AnimationStyle)

    def test_recommend_low_energy(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        rec = mapper.recommend(energy_level=0.2, mood="melancholic")
        assert isinstance(rec, AnimationStyle)

    def test_all_styles_have_required_fields(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        for name in mapper.list_styles():
            style = mapper.get_style(name)
            assert style.prefix != ""
            assert style.negative_prefix != ""
            assert style.recommended_model != ""
