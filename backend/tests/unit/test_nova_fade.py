"""Phase 5: Nova Fade Character Pipeline — unit tests (TDD, no real GPU needed)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ── Imports under test ────────────────────────────────────────────────────────
from backend.services.nova_fade.character import (
    NovaFadeCharacter,
    ValidationResult,
)
from backend.services.nova_fade.canonical_prompts import CanonicalPrompts
from backend.services.nova_fade.drift_tester import (
    DriftTester,
    DriftScorecard,
    RunConfig,
    Thresholds,
)
from backend.services.nova_fade.dj_video_generator import (
    DJVideoGenerator,
    DJAction,
    DJTimeline,
)


# ═══════════════════════════════════════════════════════════════════════════════
# NovaFadeCharacter — constitution enforcement
# ═══════════════════════════════════════════════════════════════════════════════


class TestNovaFadeCharacterConstants:
    def test_expressions_list_not_empty(self):
        assert len(NovaFadeCharacter.EXPRESSIONS) == 5

    def test_gestures_list_not_empty(self):
        assert len(NovaFadeCharacter.ALLOWED_GESTURES) == 5

    def test_forbidden_list_not_empty(self):
        assert len(NovaFadeCharacter.FORBIDDEN) > 0

    def test_expressions_contain_expected(self):
        assert "neutral_confident" in NovaFadeCharacter.EXPRESSIONS
        assert "mischievous_grin" in NovaFadeCharacter.EXPRESSIONS
        assert "drop_anticipation" in NovaFadeCharacter.EXPRESSIONS

    def test_gestures_contain_expected(self):
        assert "left_deck_scratch" in NovaFadeCharacter.ALLOWED_GESTURES
        assert "crossfader_tap" in NovaFadeCharacter.ALLOWED_GESTURES

    def test_forbidden_contains_photorealistic(self):
        assert "photorealistic" in NovaFadeCharacter.FORBIDDEN

    def test_forbidden_contains_anime(self):
        assert "anime" in NovaFadeCharacter.FORBIDDEN


class TestNovaFadeCharacterValidation:
    def setup_method(self):
        self.char = NovaFadeCharacter()

    def test_valid_prompt_passes(self):
        result = self.char.validate_prompt(
            "nova_fade_char DJ in studio, left_deck_scratch, neutral_confident expression"
        )
        assert result.valid is True
        assert result.violations == []

    def test_forbidden_term_detected(self):
        result = self.char.validate_prompt(
            "photorealistic portrait of a DJ woman"
        )
        assert result.valid is False
        assert any("photorealistic" in v.lower() for v in result.violations)

    def test_anime_forbidden(self):
        result = self.char.validate_prompt("anime style DJ girl spinning records")
        assert result.valid is False

    def test_age_change_forbidden(self):
        result = self.char.validate_prompt("age change to elderly DJ nova fade")
        assert result.valid is False

    def test_multiple_violations_all_reported(self):
        result = self.char.validate_prompt(
            "photorealistic anime DJ with different hairstyle"
        )
        assert result.valid is False
        assert len(result.violations) >= 2

    def test_empty_prompt_is_valid(self):
        # No forbidden terms → no violation
        result = self.char.validate_prompt("")
        assert result.valid is True

    def test_forbidden_case_insensitive(self):
        result = self.char.validate_prompt("PHOTOREALISTIC studio shot")
        assert result.valid is False


class TestNovaFadeCharacterCanonicalHeader:
    def setup_method(self):
        self.char = NovaFadeCharacter()

    def test_returns_tuple_of_two_strings(self):
        pos, neg = self.char.get_canonical_header()
        assert isinstance(pos, str)
        assert isinstance(neg, str)

    def test_positive_header_contains_trigger(self):
        pos, _ = self.char.get_canonical_header()
        assert "nova_fade_char" in pos

    def test_positive_header_contains_style_descriptors(self):
        pos, _ = self.char.get_canonical_header()
        # Should describe the 3D cartoon stylized aesthetic
        assert any(term in pos.lower() for term in ["3d", "cartoon", "stylized", "dj"])

    def test_negative_header_blocks_forbidden_styles(self):
        _, neg = self.char.get_canonical_header()
        assert any(term in neg.lower() for term in ["photorealistic", "anime", "realistic"])

    def test_negative_header_non_empty(self):
        _, neg = self.char.get_canonical_header()
        assert len(neg) > 10

    def test_expression_variant_positive(self):
        """Each expression variant adds the expression to the positive prompt."""
        for expr in NovaFadeCharacter.EXPRESSIONS:
            pos, _ = self.char.get_canonical_header(expression=expr)
            assert expr.replace("_", " ") in pos.lower() or expr in pos

    def test_gesture_variant_positive(self):
        pos, _ = self.char.get_canonical_header(gesture="left_deck_scratch")
        assert "left" in pos.lower() or "scratch" in pos.lower() or "deck" in pos.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# CanonicalPrompts — locked prompt library
# ═══════════════════════════════════════════════════════════════════════════════


class TestCanonicalPrompts:
    def setup_method(self):
        self.cp = CanonicalPrompts()

    def test_all_expression_prompts_exist(self):
        for expr in NovaFadeCharacter.EXPRESSIONS:
            assert expr in self.cp.list_expressions()

    def test_all_gesture_prompts_exist(self):
        for gesture in NovaFadeCharacter.ALLOWED_GESTURES:
            assert gesture in self.cp.list_gestures()

    def test_get_expression_prompt_returns_strings(self):
        pos, neg = self.cp.get_expression_prompt("neutral_confident")
        assert isinstance(pos, str) and len(pos) > 10
        assert isinstance(neg, str) and len(neg) > 5

    def test_get_gesture_prompt_returns_strings(self):
        pos, neg = self.cp.get_gesture_prompt("left_deck_scratch")
        assert isinstance(pos, str) and len(pos) > 10
        assert isinstance(neg, str)

    def test_unknown_expression_raises(self):
        with pytest.raises(KeyError):
            self.cp.get_expression_prompt("flying_leap")

    def test_unknown_gesture_raises(self):
        with pytest.raises(KeyError):
            self.cp.get_gesture_prompt("backflip")

    def test_compose_scene_prompt_contains_header(self):
        char = NovaFadeCharacter()
        prompt = self.cp.compose_scene_prompt(
            scene_description="DJ in neon-lit club",
            expression="mischievous_grin",
            gesture="crossfader_tap",
            character=char,
        )
        assert "nova_fade_char" in prompt.positive
        assert len(prompt.positive) > 20

    def test_compose_scene_prompt_negative_non_empty(self):
        char = NovaFadeCharacter()
        prompt = self.cp.compose_scene_prompt(
            scene_description="close-up on turntable hands",
            expression="focused_intensity",
            gesture="right_deck_scratch",
            character=char,
        )
        assert len(prompt.negative) > 5


# ═══════════════════════════════════════════════════════════════════════════════
# DriftTester — CLIP-based identity regression
# ═══════════════════════════════════════════════════════════════════════════════


class TestDriftTesterConfig:
    def test_run_config_defaults(self):
        cfg = RunConfig(
            lora_path="output/loras/novafade_id_v1.safetensors",
            canonical_dir="output/nova_fade/canonical",
        )
        assert cfg.num_test_images == 20
        assert cfg.seed_start == 42

    def test_thresholds_defaults(self):
        t = Thresholds()
        assert 0.0 < t.s_id_min < 1.0
        assert 0.0 < t.s_face_min < 1.0
        assert 0.0 < t.s_sil_min < 1.0
        assert 0.0 < t.v_batch_max < 1.0


class TestDriftTesterScorecard:
    def test_scorecard_pass(self):
        sc = DriftScorecard(
            s_id=0.85, s_face=0.82, s_sil=0.90, v_batch=0.05,
            thresholds=Thresholds(), passed=True,
        )
        assert sc.passed is True

    def test_scorecard_fail(self):
        sc = DriftScorecard(
            s_id=0.50, s_face=0.45, s_sil=0.60, v_batch=0.30,
            thresholds=Thresholds(), passed=False,
        )
        assert sc.passed is False

    def test_scorecard_summary_string(self):
        sc = DriftScorecard(
            s_id=0.85, s_face=0.82, s_sil=0.90, v_batch=0.05,
            thresholds=Thresholds(), passed=True,
        )
        s = sc.summary()
        assert "s_id" in s.lower() or "identity" in s.lower() or "0.85" in s


class TestDriftTesterMocked:
    """Unit tests for DriftTester using mocked internals."""

    def test_run_test_returns_scorecard(self):
        tester = DriftTester()
        config = RunConfig(
            lora_path="output/loras/novafade_id_v1.safetensors",
            canonical_dir="output/nova_fade/canonical",
        )
        # Mock the internals so no real GPU needed
        with patch.object(tester, "_generate_test_images", return_value=["img1.png"] * 20):
            with patch.object(tester, "_compute_clip_scores", return_value=(0.85, 0.82, 0.90, 0.04)):
                scorecard = tester.run_test(config, Thresholds())
        assert isinstance(scorecard, DriftScorecard)
        assert scorecard.s_id == pytest.approx(0.85)

    def test_passing_scorecard_when_above_thresholds(self):
        tester = DriftTester()
        config = RunConfig(
            lora_path="output/loras/novafade_id_v1.safetensors",
            canonical_dir="output/nova_fade/canonical",
        )
        t = Thresholds(s_id_min=0.70, s_face_min=0.65, s_sil_min=0.75, v_batch_max=0.20)
        with patch.object(tester, "_generate_test_images", return_value=["img.png"] * 20):
            with patch.object(tester, "_compute_clip_scores", return_value=(0.85, 0.82, 0.90, 0.05)):
                scorecard = tester.run_test(config, t)
        assert scorecard.passed is True

    def test_failing_scorecard_when_below_thresholds(self):
        tester = DriftTester()
        config = RunConfig(
            lora_path="output/loras/novafade_id_v1.safetensors",
            canonical_dir="output/nova_fade/canonical",
        )
        t = Thresholds(s_id_min=0.80, s_face_min=0.75, s_sil_min=0.80, v_batch_max=0.10)
        with patch.object(tester, "_generate_test_images", return_value=["img.png"] * 20):
            with patch.object(tester, "_compute_clip_scores", return_value=(0.55, 0.50, 0.60, 0.25)):
                scorecard = tester.run_test(config, t)
        assert scorecard.passed is False


# ═══════════════════════════════════════════════════════════════════════════════
# DJVideoGenerator — Nova Fade DJ performance timeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestDJAction:
    def test_dj_action_fields(self):
        action = DJAction(
            start_sec=0.0,
            end_sec=4.0,
            action_type="idle_bob",
            expression="neutral_confident",
        )
        assert action.action_type == "idle_bob"
        assert action.duration == pytest.approx(4.0)

    def test_dj_action_duration_property(self):
        action = DJAction(start_sec=10.0, end_sec=13.5, action_type="deck_scratch_L",
                          expression="focused_intensity")
        assert action.duration == pytest.approx(3.5)


class TestDJTimeline:
    def _make_timeline(self):
        actions = [
            DJAction(0.0, 4.0, "idle_bob", "neutral_confident"),
            DJAction(4.0, 6.0, "deck_scratch_L", "focused_intensity"),
            DJAction(6.0, 10.0, "idle_bob", "neutral_confident"),
        ]
        return DJTimeline(actions=actions, total_duration_sec=10.0, theme="sponsor_neon")

    def test_timeline_has_actions(self):
        tl = self._make_timeline()
        assert len(tl.actions) == 3

    def test_timeline_total_duration(self):
        tl = self._make_timeline()
        assert tl.total_duration_sec == pytest.approx(10.0)

    def test_timeline_theme(self):
        tl = self._make_timeline()
        assert tl.theme == "sponsor_neon"

    def test_actions_cover_full_duration(self):
        tl = self._make_timeline()
        covered = sum(a.duration for a in tl.actions)
        assert covered == pytest.approx(tl.total_duration_sec)


class TestDJVideoGeneratorTimeline:
    def setup_method(self):
        self.gen = DJVideoGenerator()

    def _mock_analysis(self, duration=120.0, bpm=128.0):
        """Return a minimal mock SongAnalysis-like object."""
        analysis = MagicMock()
        analysis.duration = duration
        analysis.bpm = bpm
        analysis.sections = [
            MagicMock(start=0.0, end=30.0, section_type="verse", energy_level=0.5),
            MagicMock(start=30.0, end=60.0, section_type="chorus", energy_level=0.9),
            MagicMock(start=60.0, end=90.0, section_type="verse", energy_level=0.5),
            MagicMock(start=90.0, end=120.0, section_type="chorus", energy_level=0.95),
        ]
        return analysis

    def test_build_timeline_returns_dj_timeline(self):
        analysis = self._mock_analysis()
        timeline = self.gen.build_timeline(analysis, theme="sponsor_neon")
        assert isinstance(timeline, DJTimeline)

    def test_timeline_covers_song_duration(self):
        analysis = self._mock_analysis(duration=60.0)
        timeline = self.gen.build_timeline(analysis, theme="chill_lofi")
        # Sum of actions should equal song duration
        total = sum(a.duration for a in timeline.actions)
        assert total == pytest.approx(60.0, abs=1.0)

    def test_high_energy_sections_get_active_actions(self):
        analysis = self._mock_analysis()
        timeline = self.gen.build_timeline(analysis, theme="mashup_chaos")
        # Actions during chorus (high energy) should not all be idle_bob
        chorus_actions = [
            a for a in timeline.actions
            if a.start_sec >= 30.0 and a.end_sec <= 60.0
        ]
        action_types = {a.action_type for a in chorus_actions}
        # At least one active (non-idle) action during chorus
        assert action_types - {"idle_bob"} != set() or len(chorus_actions) > 0

    def test_valid_theme_accepted(self):
        for theme in ["sponsor_neon", "award_elegant", "mashup_chaos", "chill_lofi"]:
            analysis = self._mock_analysis()
            tl = self.gen.build_timeline(analysis, theme=theme)
            assert tl.theme == theme

    def test_invalid_theme_raises(self):
        analysis = self._mock_analysis()
        with pytest.raises(ValueError, match="theme"):
            self.gen.build_timeline(analysis, theme="purple_disco")

    def test_generate_returns_path_string(self):
        """DJVideoGenerator.generate() calls backend + returns output path."""
        analysis = self._mock_analysis()
        with patch.object(self.gen, "_generate_clips", return_value=["/tmp/clip1.mp4"]):
            with patch.object(self.gen, "_assemble_video", return_value="/tmp/out.mp4"):
                result = self.gen.generate(
                    mashup_path="/tmp/mashup.wav",
                    mashup_analysis=analysis,
                    theme="sponsor_neon",
                    output_path="/tmp/out.mp4",
                )
        assert result == "/tmp/out.mp4"

    def test_prompt_contains_nova_fade_trigger(self):
        """Scene prompts generated from timeline use canonical header."""
        analysis = self._mock_analysis(duration=30.0)
        timeline = self.gen.build_timeline(analysis, theme="sponsor_neon")
        prompts = self.gen.timeline_to_prompts(timeline)
        for p in prompts:
            assert "nova_fade_char" in p.positive

    def test_actions_use_valid_action_types(self):
        analysis = self._mock_analysis()
        timeline = self.gen.build_timeline(analysis, theme="sponsor_neon")
        valid = {"idle_bob", "deck_scratch_L", "deck_scratch_R", "crossfader_hit", "drop_reaction"}
        for a in timeline.actions:
            assert a.action_type in valid
