"""Shared test fixtures for Beat Studio."""
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import yaml


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent.parent


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_settings(tmp_dir: Path) -> Path:
    """Write a minimal settings.yaml to a temp dir and return its path."""
    settings = {
        "paths": {
            "library_cache": str(tmp_dir / "library_cache"),
            "uploads": str(tmp_dir / "uploads"),
            "generated_images": str(tmp_dir / "generated_images"),
            "generated_videos": str(tmp_dir / "generated_videos"),
            "output_videos": str(tmp_dir / "output" / "videos"),
            "output_loras": str(tmp_dir / "output" / "loras"),
            "models_dir": str(tmp_dir / "models"),
        },
        "audio": {"sample_rate": 44100, "bit_depth": 16, "channels": 2, "default_format": "wav"},
        "models": {
            "whisper_size": "base",
            "demucs_model": "htdemucs",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "llm": {
            "primary_provider": "anthropic",
            "fallback_provider": "openai",
            "anthropic_model": "claude-sonnet-4-6",
            "openai_model": "gpt-4-turbo-preview",
            "max_retries": 3,
            "timeout": 30,
        },
        "mashup": {
            "chroma_collection": "tiki_library",
            "chroma_db_path": str(tmp_dir / "chroma"),
            "bpm_tolerance": 0.05,
            "max_stretch_ratio": 1.2,
            "default_match_criteria": "hybrid",
            "max_candidates": 5,
            "weight_bpm": 0.35,
            "weight_key": 0.30,
            "weight_energy": 0.20,
            "weight_genre": 0.15,
            "min_compatibility": 0.5,
            "max_pairs": 10,
        },
        "engineer": {
            "default_quality": "high",
            "vocal_attenuation_db": -2,
            "fade_duration_sec": 4.0,
            "normalize_lufs": -14,
        },
        "logging": {"level": "DEBUG", "file": str(tmp_dir / "test.log")},
        "vram": {"baseline_threshold_gb": 1.5, "budget_gb": 12.0},
        "lora": {
            "registry_path": str(tmp_dir / "loras.yaml"),
            "base_path": str(tmp_dir / "loras"),
        },
    }
    cfg_path = tmp_dir / "settings.yaml"
    cfg_path.write_text(yaml.dump(settings))
    return cfg_path


@pytest.fixture
def temp_chroma_dir(tmp_dir: Path) -> Path:
    """Isolated ChromaDB directory for each test."""
    d = tmp_dir / "chroma"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Audio fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_audio_file(tmp_dir: Path) -> Path:
    """Write a minimal valid WAV file for tests that need a real file path.

    The file contains 0.5 seconds of silence at 44100 Hz, 16-bit mono.
    No external dependencies (librosa, soundfile) required.
    """
    sample_rate = 44100
    num_samples = sample_rate // 2  # 0.5 seconds
    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = num_samples * block_align
    chunk_size = 36 + data_size

    path = tmp_dir / "test_audio.wav"
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", chunk_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))           # subchunk1 size
        f.write(struct.pack("<H", 1))            # PCM format
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        # data chunk (silence)
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# GPU / hardware mocks
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_gpu():
    """Patch torch.cuda so tests run without a real GPU.

    Yields a MagicMock configured as a 24 GB CUDA device.
    """
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090 (mock)"), \
         patch("torch.cuda.get_device_properties") as mock_props, \
         patch("torch.cuda.memory_allocated", return_value=int(2e9)), \
         patch("torch.cuda.empty_cache"), \
         patch("torch.cuda.synchronize"):
        mock_props.return_value.total_memory = int(24e9)
        yield mock_props


@pytest.fixture
def no_gpu():
    """Patch torch.cuda so tests run in CPU-only mode."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Video backend mock
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_video_backend(tmp_dir: Path):
    """Return a minimal VideoBackend-compatible mock.

    ``generate_clip`` writes an empty file and returns its path so that
    downstream assembler code can treat it as a real clip.
    """
    clip_path = tmp_dir / "mock_clip.mp4"
    clip_path.write_bytes(b"")

    backend = MagicMock()
    backend.name.return_value = "mock_backend"
    backend.vram_required_gb.return_value = 0.0
    backend.supports_style.return_value = True
    backend.is_available.return_value = True
    backend.estimated_time_per_scene.return_value = 0.01
    backend.estimated_cost_per_scene.return_value = 0.0
    backend.generate_clip.return_value = str(clip_path)
    backend.generate_batch.return_value = [str(clip_path)]
    backend.kill.return_value = None
    return backend


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI TestClient
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def api_client():
    """Session-scoped FastAPI TestClient for integration tests.

    Prefer the module-scoped ``client`` fixture in individual test
    modules; use this when a single client should persist across the
    whole test run.
    """
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c
