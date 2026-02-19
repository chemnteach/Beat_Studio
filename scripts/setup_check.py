#!/usr/bin/env python3
"""Beat Studio — Environment Setup Checker

Validates that all required tools, models, and API keys are present before
running Beat Studio for the first time.

Usage:
    python scripts/setup_check.py            # full check
    python scripts/setup_check.py --models   # model paths only
    python scripts/setup_check.py --quick    # skip slow model checks
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed.  Run: pip install pyyaml")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
CONFIG_DIR = REPO_ROOT / "backend" / "config"
CHECKPOINTS_YAML = CONFIG_DIR / "checkpoints.yaml"
HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def ok(msg: str) -> str:
    return f"  {GREEN}✓{RESET}  {msg}"


def warn(msg: str) -> str:
    return f"  {YELLOW}⚠{RESET}  {msg}"


def err(msg: str) -> str:
    return f"  {RED}✗{RESET}  {msg}"


def section(title: str) -> None:
    print(f"\n{BOLD}{BLUE}━━ {title} ━━{RESET}")


# ── Result accumulator ────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    passed: int = 0
    warned: int = 0
    failed: int = 0
    messages: List[str] = field(default_factory=list)

    def add(self, level: str, msg: str) -> None:
        self.messages.append((level, msg))
        if level == "ok":
            self.passed += 1
        elif level == "warn":
            self.warned += 1
        else:
            self.failed += 1

    def print_summary(self) -> None:
        section("Summary")
        for level, msg in self.messages:
            if level == "ok":
                print(ok(msg))
            elif level == "warn":
                print(warn(msg))
            else:
                print(err(msg))

        print()
        total = self.passed + self.warned + self.failed
        print(f"  {GREEN}{self.passed}{RESET} passed  "
              f"{YELLOW}{self.warned}{RESET} warnings  "
              f"{RED}{self.failed}{RESET} failed  "
              f"({total} checks)")

    @property
    def exit_code(self) -> int:
        return 0 if self.failed == 0 else 1


# ── Individual checks ─────────────────────────────────────────────────────────


def check_python_version(result: CheckResult) -> None:
    section("Python")
    major, minor = sys.version_info[:2]
    ver = f"{major}.{minor}"
    if (major, minor) >= (3, 10):
        print(ok(f"Python {ver}"))
        result.add("ok", f"Python {ver}")
    else:
        print(err(f"Python {ver} — need 3.10+"))
        result.add("fail", f"Python {ver} — need 3.10+")


def check_system_dependencies(result: CheckResult) -> None:
    section("System dependencies")

    tools = {
        "ffmpeg": ("ffmpeg", "Required for audio/video conversion"),
        "git":    ("git",    "Required for model downloads"),
    }
    for name, (cmd, purpose) in tools.items():
        path = shutil.which(cmd)
        if path:
            print(ok(f"{name}: {path}"))
            result.add("ok", f"{name} found")
        else:
            print(err(f"{name} not found — {purpose}"))
            result.add("fail", f"{name} missing")


def check_python_packages(result: CheckResult, quick: bool) -> None:
    section("Python packages")

    required = [
        "fastapi", "uvicorn", "pydantic", "yaml", "dotenv",
        "torch", "diffusers", "transformers", "peft",
        "librosa", "whisper", "pydub", "soundfile",
        "chromadb", "sentence_transformers",
        "anthropic", "openai",
        "pytest",
    ]
    import_aliases = {
        "yaml": "yaml",
        "dotenv": "dotenv",
    }

    if quick:
        required = ["fastapi", "uvicorn", "pydantic", "torch", "chromadb", "pytest"]
        print(warn("Quick mode — checking essential packages only"))
        result.add("warn", "Package check in quick mode")

    for pkg in required:
        module = import_aliases.get(pkg, pkg)
        try:
            importlib.import_module(module)
            print(ok(f"{pkg}"))
            result.add("ok", f"Package: {pkg}")
        except ImportError:
            print(err(f"{pkg} not installed"))
            result.add("fail", f"Package missing: {pkg}")


def check_gpu(result: CheckResult) -> None:
    section("GPU / CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / 1e9
            print(ok(f"GPU: {name} ({total_gb:.1f} GB VRAM)"))
            result.add("ok", f"GPU: {name} ({total_gb:.1f} GB)")

            if total_gb < 6:
                print(warn("< 6 GB VRAM — only basic AnimateDiff tier available"))
                result.add("warn", "Low VRAM — limited model support")
            elif total_gb < 12:
                print(warn("< 12 GB VRAM — WAN 2.6 local will be unavailable"))
                result.add("warn", "WAN 2.6 local requires 12 GB+ VRAM")
            else:
                print(ok(f"VRAM sufficient for all local models"))
                result.add("ok", "VRAM sufficient")
        else:
            print(warn("No CUDA GPU detected — cloud backends only for video generation"))
            result.add("warn", "No CUDA GPU — cloud required for video generation")
    except ImportError:
        print(err("torch not installed"))
        result.add("fail", "torch missing — cannot check GPU")


def check_models(result: CheckResult) -> None:
    section("Models")

    if not CHECKPOINTS_YAML.exists():
        print(err(f"checkpoints.yaml not found at {CHECKPOINTS_YAML}"))
        result.add("fail", "checkpoints.yaml missing")
        return

    with open(CHECKPOINTS_YAML) as f:
        data = yaml.safe_load(f)

    hf_cache = HF_CACHE / "hub"

    for model in data.get("models", []):
        name = model["name"]
        status = model.get("status", "optional")
        cache_subdir = model.get("cache_subdir")
        note = model.get("note", "")

        if cache_subdir:
            model_path = hf_cache / cache_subdir
            found = model_path.exists()
        else:
            # No known cache path — mark as unknown
            found = None

        if found is True:
            msg = f"{name}: found"
            print(ok(msg))
            result.add("ok", msg)
        elif found is False:
            if status == "required":
                msg = f"{name}: MISSING (required)"
                print(err(msg))
                if note:
                    print(f"         Note: {note}")
                install = model.get("install_cmd")
                if install:
                    print(f"         Install: {install}")
                result.add("fail", msg)
            elif status == "recommended":
                msg = f"{name}: not found (recommended)"
                print(warn(msg))
                result.add("warn", msg)
            else:
                msg = f"{name}: not installed (optional)"
                print(f"  {RESET}○  {msg}")
                result.add("ok", msg)
        else:
            # No cache_subdir — can't verify automatically
            msg = f"{name}: path unknown ({note or status})"
            print(warn(msg))
            result.add("warn", f"{name}: cannot auto-verify")


def check_lora_registry(result: CheckResult) -> None:
    section("LoRA registry")

    loras_yaml = CONFIG_DIR / "loras.yaml"
    if not loras_yaml.exists():
        print(warn("loras.yaml not found — registry empty (normal on first run)"))
        result.add("warn", "loras.yaml missing")
        return

    with open(loras_yaml) as f:
        data = yaml.safe_load(f) or {}

    loras = data.get("loras", [])
    print(ok(f"Registry loaded: {len(loras)} LoRA(s) registered"))
    result.add("ok", f"LoRA registry: {len(loras)} entries")

    missing = []
    for lora in loras:
        path_str = lora.get("file_path") or lora.get("path")
        if path_str and not Path(path_str).exists():
            missing.append(lora.get("name", path_str))

    if missing:
        for name in missing:
            print(warn(f"  LoRA file missing: {name}"))
        result.add("warn", f"{len(missing)} LoRA file(s) missing from disk")
    else:
        if loras:
            print(ok("All registered LoRA files found on disk"))
            result.add("ok", "All LoRA files present")


def check_api_keys(result: CheckResult) -> None:
    section("API keys")

    keys = {
        "ANTHROPIC_API_KEY": "Anthropic Claude (primary LLM provider)",
        "OPENAI_API_KEY":    "OpenAI (fallback LLM provider)",
        "RUNPOD_API_KEY":    "RunPod (cloud video generation)",
    }
    for var, desc in keys.items():
        val = os.environ.get(var, "")
        if val:
            masked = val[:4] + "..." + val[-4:] if len(val) > 8 else "****"
            print(ok(f"{var}: {masked}  ({desc})"))
            result.add("ok", f"{var} set")
        else:
            is_required = var in ("ANTHROPIC_API_KEY",)
            if is_required:
                print(warn(f"{var}: not set — {desc}"))
                result.add("warn", f"{var} missing (optional but recommended)")
            else:
                print(f"  {RESET}○  {var}: not set (optional) — {desc}")
                result.add("ok", f"{var} not set (optional)")


def check_directories(result: CheckResult) -> None:
    section("Data directories")

    dirs = {
        "uploads":          REPO_ROOT / "backend" / "data" / "uploads",
        "generated_images": REPO_ROOT / "backend" / "data" / "generated_images",
        "generated_videos": REPO_ROOT / "backend" / "data" / "generated_videos",
        "library_cache":    REPO_ROOT / "backend" / "data" / "library_cache",
        "output/videos":    REPO_ROOT / "output" / "videos",
        "output/loras":     REPO_ROOT / "output" / "loras",
        "nova_fade/canonical": REPO_ROOT / "output" / "nova_fade" / "canonical",
    }
    for name, path in dirs.items():
        if path.exists():
            print(ok(f"{name}: {path}"))
            result.add("ok", f"Dir: {name}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(ok(f"{name}: created {path}"))
            result.add("ok", f"Dir created: {name}")


def check_config_files(result: CheckResult) -> None:
    section("Configuration files")

    configs = {
        "settings.yaml":          CONFIG_DIR / "settings.yaml",
        "checkpoints.yaml":       CONFIG_DIR / "checkpoints.yaml",
        "animation_styles.yaml":  CONFIG_DIR / "animation_styles.yaml",
        "nova_fade_constitution.yaml": CONFIG_DIR / "nova_fade_constitution.yaml",
    }
    for name, path in configs.items():
        if path.exists():
            print(ok(f"{name}"))
            result.add("ok", f"Config: {name}")
        else:
            print(warn(f"{name}: not found at {path}"))
            result.add("warn", f"Config missing: {name}")


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beat Studio environment setup checker"
    )
    parser.add_argument("--models", action="store_true", help="Check model paths only")
    parser.add_argument("--quick", action="store_true", help="Skip slow checks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = CheckResult()

    print(f"\n{BOLD}Beat Studio — Setup Checker{RESET}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"HF cache:  {HF_CACHE}")

    if args.models:
        check_models(result)
    else:
        check_python_version(result)
        check_system_dependencies(result)
        check_python_packages(result, quick=args.quick)
        check_gpu(result)
        check_models(result)
        check_lora_registry(result)
        check_api_keys(result)
        check_directories(result)
        check_config_files(result)

    result.print_summary()
    print()

    if result.failed == 0 and result.warned == 0:
        print(f"{GREEN}{BOLD}✓ All checks passed — Beat Studio is ready!{RESET}\n")
    elif result.failed == 0:
        print(f"{YELLOW}{BOLD}⚠ Setup complete with warnings — Beat Studio will run "
              f"but some features may be limited.{RESET}\n")
    else:
        print(f"{RED}{BOLD}✗ {result.failed} check(s) failed — resolve errors before "
              f"running Beat Studio.{RESET}\n")

    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
