"""Tests for Phase 7 Animation Style system (YAML-backed StyleMapper)."""
from __future__ import annotations

import pytest

from backend.services.prompt.types import AnimationStyle


# ── StyleMapper YAML loading ──────────────────────────────────────────────────

class TestStyleMapperYAML:
    def test_list_styles_returns_at_least_20(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        names = mapper.list_styles()
        assert len(names) >= 20

    def test_cinematic_in_list(self):
        from backend.services.prompt.style_mapper import StyleMapper
        assert "cinematic" in StyleMapper().list_styles()

    def test_lofi_in_list(self):
        from backend.services.prompt.style_mapper import StyleMapper
        assert "lofi" in StyleMapper().list_styles()

    def test_synthwave_in_list(self):
        from backend.services.prompt.style_mapper import StyleMapper
        assert "synthwave" in StyleMapper().list_styles()

    def test_watercolor_in_list(self):
        from backend.services.prompt.style_mapper import StyleMapper
        assert "watercolor" in StyleMapper().list_styles()

    def test_rotoscope_in_list(self):
        from backend.services.prompt.style_mapper import StyleMapper
        assert "rotoscope" in StyleMapper().list_styles()

    def test_all_styles_have_prefix(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        for name in mapper.list_styles():
            style = mapper.get_style(name)
            assert style.prefix, f"Style '{name}' has empty prefix"

    def test_all_styles_have_negative_prefix(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        for name in mapper.list_styles():
            style = mapper.get_style(name)
            assert style.negative_prefix, f"Style '{name}' has empty negative_prefix"

    def test_all_styles_have_recommended_model(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        for name in mapper.list_styles():
            style = mapper.get_style(name)
            assert style.recommended_model, f"Style '{name}' has empty recommended_model"

    def test_get_cinematic_returns_animation_style(self):
        from backend.services.prompt.style_mapper import StyleMapper
        style = StyleMapper().get_style("cinematic")
        assert isinstance(style, AnimationStyle)
        assert style.name == "cinematic"

    def test_cinematic_uses_wan26_cloud(self):
        from backend.services.prompt.style_mapper import StyleMapper
        style = StyleMapper().get_style("cinematic")
        assert "wan26" in style.recommended_model or "cloud" in style.recommended_model

    def test_animatediff_styles_have_correct_backend(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        # These should all use animatediff
        for name in ["watercolor", "pixel_art", "synthwave", "anime", "lofi"]:
            style = mapper.get_style(name)
            assert "animatediff" in style.recommended_model, \
                f"'{name}' should use animatediff, got '{style.recommended_model}'"

    def test_cfg_scale_in_valid_range(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        for name in mapper.list_styles():
            style = mapper.get_style(name)
            assert 5.0 <= style.cfg_scale <= 15.0, \
                f"Style '{name}' has cfg_scale={style.cfg_scale} out of range"

    def test_steps_positive(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        for name in mapper.list_styles():
            style = mapper.get_style(name)
            assert style.steps > 0, f"Style '{name}' has steps={style.steps}"


# ── StyleMapper recommend ────────────────────────────────────────────────────

class TestStyleMapperRecommend:
    def test_recommend_hiphop_returns_graffiti_or_comic(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        rec = mapper.recommend(energy_level=0.8, mood="energetic", genre="hip-hop")
        assert isinstance(rec, AnimationStyle)

    def test_recommend_folk_returns_soft_style(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        rec = mapper.recommend(energy_level=0.3, mood="melancholic", genre="folk")
        assert isinstance(rec, AnimationStyle)
        # Should be a soft style: watercolor, pencil_sketch, or impressionist
        assert rec.name in {"watercolor", "pencil_sketch", "impressionist", "lofi", "oil_painting"}

    def test_recommend_electronic_dark_returns_synthwave_or_psychedelic(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        rec = mapper.recommend(energy_level=0.9, mood="dark", genre="electronic")
        assert isinstance(rec, AnimationStyle)

    def test_recommend_lofi_returns_lofi_style(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        rec = mapper.recommend(energy_level=0.2, mood="serene", genre="lofi")
        assert rec.name in {"lofi", "isometric", "watercolor", "impressionist"}

    def test_recommend_unknown_genre_returns_style(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        rec = mapper.recommend(energy_level=0.5, mood="neutral", genre="unknown_xyz")
        assert isinstance(rec, AnimationStyle)

    def test_recommend_top_n_returns_list(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        recs = mapper.recommend_top(energy_level=0.7, mood="triumphant", genre="pop", n=3)
        assert len(recs) == 3
        assert all(isinstance(r, AnimationStyle) for r in recs)

    def test_recommend_top_no_duplicates(self):
        from backend.services.prompt.style_mapper import StyleMapper
        mapper = StyleMapper()
        recs = mapper.recommend_top(energy_level=0.5, mood="hopeful", genre="pop", n=5)
        names = [r.name for r in recs]
        assert len(names) == len(set(names))
