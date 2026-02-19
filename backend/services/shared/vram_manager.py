"""VRAM manager — kill-and-revive pattern for Beat Studio.

Rule: Only ONE model loaded in GPU VRAM at a time.  Before loading a new model,
the current one must be fully killed and VRAM must drop below the baseline
threshold.

Ported and extended from BeatCanvas ``backend/src/local/vram_manager.py``
(496 lines).  Key improvements:
- No in-memory state that survives only the current process (StatefulConfig
  is separate from VRAM state)
- Explicit ``set_pipeline`` replaces scattered ``self.pipe = ...`` patterns
- Single ``kill()`` method with correct gc+cuda sequence
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("beat_studio.vram_manager")

# ── optional torch import ─────────────────────────────────────────────────────
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    torch = None  # type: ignore[assignment]


@dataclass
class VRAMStatus:
    cuda_available: bool
    total_gb: float
    free_gb: float
    used_gb: float
    current_model: Optional[str]
    baseline_ok: bool  # True if free_gb >= baseline_threshold_gb


class VRAMManager:
    """Manages GPU VRAM by enforcing one-model-at-a-time discipline.

    Usage::

        vram = VRAMManager(budget_gb=12.0, baseline_threshold_gb=1.5)

        # Load a model
        pipe = load_animatediff()
        vram.set_pipeline(pipe, "animatediff_lightning")

        # Later, before loading a different model:
        vram.kill()               # ensures VRAM is freed
        pipe2 = load_wan26()
        vram.set_pipeline(pipe2, "wan26_local")
    """

    def __init__(self, budget_gb: float = 12.0, baseline_threshold_gb: float = 1.5):
        self.budget_gb = budget_gb
        self.baseline_threshold_gb = baseline_threshold_gb
        self._current_pipeline: Any = None
        self._current_model_name: Optional[str] = None

    # ── public ───────────────────────────────────────────────────────────────

    @property
    def current_model_name(self) -> Optional[str]:
        return self._current_model_name

    def set_pipeline(self, pipeline: Any, model_name: str) -> None:
        """Register a newly loaded pipeline.  Kills any existing pipeline first."""
        if self._current_pipeline is not None:
            logger.info("Killing existing pipeline '%s' before loading '%s'",
                        self._current_model_name, model_name)
            self.kill()
        self._current_pipeline = pipeline
        self._current_model_name = model_name
        logger.info("Pipeline '%s' registered in VRAMManager.", model_name)

    def kill(self) -> None:
        """Release all GPU resources for the current pipeline.

        Sequence (matches BeatCanvas's validated kill order):
        1. unload_lora_weights()  — prevent weight contamination
        2. Delete pipeline reference
        3. gc.collect() × 3
        4. torch.cuda.empty_cache() + synchronize()
        5. Log VRAM after kill

        Safe to call when nothing is loaded.
        """
        if self._current_pipeline is not None:
            name = self._current_model_name or "unknown"
            logger.info("Killing pipeline '%s' …", name)

            # Step 1 — unload LoRA weights if present
            if hasattr(self._current_pipeline, "unload_lora_weights"):
                try:
                    self._current_pipeline.unload_lora_weights()
                except Exception as exc:
                    logger.warning("unload_lora_weights() raised: %s", exc)

            # Step 2 — drop reference
            self._current_pipeline = None
            self._current_model_name = None

        # Step 3 — garbage collect (× 3 per BeatCanvas pattern)
        for _ in range(3):
            gc.collect()

        # Step 4 — CUDA cleanup
        if CUDA_AVAILABLE and torch is not None:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as exc:
                logger.warning("CUDA cleanup raised: %s", exc)

        # Step 5 — log result
        status = self.get_status()
        if status.cuda_available:
            logger.info(
                "After kill: %.2f GB free / %.2f GB total (baseline=%.2f GB, ok=%s)",
                status.free_gb,
                status.total_gb,
                self.baseline_threshold_gb,
                status.baseline_ok,
            )
            if not status.baseline_ok:
                logger.warning(
                    "VRAM did not reach baseline after kill! "
                    "free=%.2f GB, required>=%.2f GB",
                    status.free_gb,
                    self.baseline_threshold_gb,
                )

    def get_status(self) -> VRAMStatus:
        """Return current VRAM usage snapshot."""
        if not CUDA_AVAILABLE or torch is None:
            return VRAMStatus(
                cuda_available=False,
                total_gb=0.0,
                free_gb=0.0,
                used_gb=0.0,
                current_model=self._current_model_name,
                baseline_ok=True,  # no GPU constraint
            )

        try:
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            free = total - allocated
        except Exception:
            total = free = allocated = 0.0

        return VRAMStatus(
            cuda_available=True,
            total_gb=round(total, 2),
            free_gb=round(free, 2),
            used_gb=round(allocated, 2),
            current_model=self._current_model_name,
            baseline_ok=free >= self.baseline_threshold_gb,
        )
