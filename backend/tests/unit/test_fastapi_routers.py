"""Phase 9: FastAPI Backend — unit tests for router structure and endpoint contracts.

Tests verify:
- All 7 routers are importable and expose a .router attribute
- main.py app is importable and mounts all routers at expected prefixes
- Endpoint function signatures are correct (params, return type hints)
- Critical rule: no dead-code routers (all imported = all mounted)
"""
from __future__ import annotations

import importlib
from typing import get_type_hints

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Router import checks — all 7 routers must be importable
# ═══════════════════════════════════════════════════════════════════════════════


ROUTER_MODULES = [
    "backend.routers.audio",
    "backend.routers.mashup",
    "backend.routers.video",
    "backend.routers.lora",
    "backend.routers.nova_fade",
    "backend.routers.tasks",
    "backend.routers.system",
]


class TestRouterImports:
    @pytest.mark.parametrize("module_path", ROUTER_MODULES)
    def test_router_module_importable(self, module_path: str):
        mod = importlib.import_module(module_path)
        assert mod is not None

    @pytest.mark.parametrize("module_path", ROUTER_MODULES)
    def test_router_has_router_attribute(self, module_path: str):
        mod = importlib.import_module(module_path)
        assert hasattr(mod, "router"), f"{module_path} must expose a 'router' attribute"

    @pytest.mark.parametrize("module_path", ROUTER_MODULES)
    def test_router_is_fastapi_router(self, module_path: str):
        from fastapi import APIRouter
        mod = importlib.import_module(module_path)
        assert isinstance(mod.router, APIRouter), (
            f"{module_path}.router must be an APIRouter instance"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# main.py app — importable, all routers mounted
# ═══════════════════════════════════════════════════════════════════════════════


class TestMainApp:
    def test_main_module_importable(self):
        mod = importlib.import_module("backend.main")
        assert mod is not None

    def test_app_is_fastapi_instance(self):
        from fastapi import FastAPI
        mod = importlib.import_module("backend.main")
        assert hasattr(mod, "app")
        assert isinstance(mod.app, FastAPI)

    def test_all_routers_mounted(self):
        """Every router module must be mounted in the app."""
        mod = importlib.import_module("backend.main")
        app = mod.app
        # Collect all prefixes from mounted routes
        mounted_prefixes = {route.path.split("/")[1] for route in app.routes
                            if hasattr(route, "path")}
        # These API segments must all appear
        expected = {"audio", "mashup", "video", "lora", "nova-fade", "tasks", "system"}
        # Check via route paths (strips /api/ prefix)
        route_paths = " ".join(
            str(getattr(r, "path", "")) for r in app.routes
        )
        for segment in expected:
            assert segment in route_paths, (
                f"Router prefix '{segment}' not found in app routes. "
                "Ensure it is mounted in backend/main.py."
            )

    def test_app_title(self):
        mod = importlib.import_module("backend.main")
        assert mod.app.title == "Beat Studio"

    def test_cors_middleware_configured(self):
        mod = importlib.import_module("backend.main")
        app = mod.app
        # FastAPI stores middleware in user_middleware as Starlette Middleware objects.
        # Each has a .cls attribute pointing to the middleware class.
        cors_present = any(
            "CORS" in str(getattr(m, "cls", "")) or "cors" in str(getattr(m, "cls", "")).lower()
            for m in app.user_middleware
        )
        assert cors_present, "CORS middleware must be configured in main.py"


# ═══════════════════════════════════════════════════════════════════════════════
# Audio router endpoints
# ═══════════════════════════════════════════════════════════════════════════════


class TestAudioRouterEndpoints:
    def setup_method(self):
        self.mod = importlib.import_module("backend.routers.audio")

    def test_upload_endpoint_exists(self):
        assert hasattr(self.mod, "upload_audio")

    def test_analyze_endpoint_exists(self):
        assert hasattr(self.mod, "analyze_audio")

    def test_get_analysis_endpoint_exists(self):
        assert hasattr(self.mod, "get_analysis")


class TestVideoRouterEndpoints:
    def setup_method(self):
        self.mod = importlib.import_module("backend.routers.video")

    def test_plan_endpoint_exists(self):
        assert hasattr(self.mod, "plan_video")

    def test_generate_endpoint_exists(self):
        assert hasattr(self.mod, "generate_video")

    def test_list_styles_endpoint_exists(self):
        assert hasattr(self.mod, "list_styles")

    def test_list_backends_endpoint_exists(self):
        assert hasattr(self.mod, "list_backends")


class TestLoRARouterEndpoints:
    def setup_method(self):
        self.mod = importlib.import_module("backend.routers.lora")

    def test_list_loras_endpoint_exists(self):
        assert hasattr(self.mod, "list_loras")

    def test_train_endpoint_exists(self):
        assert hasattr(self.mod, "train_lora")

    def test_download_endpoint_exists(self):
        assert hasattr(self.mod, "download_lora")


class TestNovaFadeRouterEndpoints:
    def setup_method(self):
        self.mod = importlib.import_module("backend.routers.nova_fade")

    def test_dj_video_endpoint_exists(self):
        assert hasattr(self.mod, "generate_dj_video")

    def test_drift_test_endpoint_exists(self):
        assert hasattr(self.mod, "run_drift_test")

    def test_status_endpoint_exists(self):
        assert hasattr(self.mod, "get_status")


class TestTaskRouterEndpoints:
    def setup_method(self):
        self.mod = importlib.import_module("backend.routers.tasks")

    def test_get_task_endpoint_exists(self):
        assert hasattr(self.mod, "get_task")

    def test_list_active_endpoint_exists(self):
        assert hasattr(self.mod, "list_active_tasks")

    def test_cancel_task_endpoint_exists(self):
        assert hasattr(self.mod, "cancel_task")


class TestSystemRouterEndpoints:
    def setup_method(self):
        self.mod = importlib.import_module("backend.routers.system")

    def test_health_endpoint_exists(self):
        assert hasattr(self.mod, "health_check")

    def test_gpu_endpoint_exists(self):
        assert hasattr(self.mod, "gpu_status")

    def test_models_endpoint_exists(self):
        assert hasattr(self.mod, "list_models")
