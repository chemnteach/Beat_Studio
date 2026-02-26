"""StoryboardService — SDXL keyframe generation for storyboard previews.

All LoRAs in loras.yaml are SDXL 1.0, so this service always uses
StableDiffusionXLPipeline regardless of the animation style's SD 1.5
base_checkpoint (which is only used by AnimateDiff).

Resolution: 1024x576 (16:9 at SDXL native scale).
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import torch
from diffusers import StableDiffusionXLPipeline

from backend.services.lora.registry import LoRARegistry
from backend.services.prompt.style_mapper import StyleMapper
from backend.services.shared.vram_manager import VRAMManager
from backend.services.storyboard.state import StoryboardStateStore
from backend.services.storyboard.types import (
    SceneInput,
    StoryboardScene,
    StoryboardState,
    VersionEntry,
)

logger = logging.getLogger("beat_studio.storyboard.service")

_BACKEND_DIR = Path(__file__).parent.parent.parent
_DEFAULT_LORAS_YAML = _BACKEND_DIR / "config" / "loras.yaml"
_DEFAULT_LORA_BASE = _BACKEND_DIR.parent / "output" / "loras"
_DEFAULT_STORYBOARD_BASE = _BACKEND_DIR.parent / "output" / "storyboard"


class StoryboardService:
    """Generate SDXL keyframe previews for storyboard approval.

    Each scene gets one image at v1 during ``generate_all_scenes``.
    Subsequent calls to ``generate_single_scene`` append v2, v3 … up to
    MAX_VERSIONS (5), evicting the oldest.

    Pipeline is loaded once per call and killed (VRAM freed) in a finally block.
    """

    SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
    PREVIEW_WIDTH = 1024
    PREVIEW_HEIGHT = 576
    SEED_STRIDE = 137   # deterministic spread across scenes (same as nova_fade)

    def __init__(
        self,
        state_store: Optional[StoryboardStateStore] = None,
        vram_manager: Optional[VRAMManager] = None,
        base_checkpoint: str = SDXL_BASE,
        loras_yaml: Optional[str] = None,
        lora_base: Optional[str] = None,
        device: str = "cuda",
    ) -> None:
        self._store = state_store or StoryboardStateStore()
        self._vm = vram_manager or VRAMManager(budget_gb=12.0)
        self._base_checkpoint = base_checkpoint
        self._loras_yaml = Path(loras_yaml) if loras_yaml else _DEFAULT_LORAS_YAML
        self._lora_base = Path(lora_base) if lora_base else _DEFAULT_LORA_BASE
        self._device = device

    # ── public ────────────────────────────────────────────────────────────────

    def generate_all_scenes(
        self,
        storyboard_id: str,
        scenes: List[SceneInput],
        style_name: str,
        lora_names: List[str],
    ) -> None:
        """Generate v1 keyframe for every scene. Runs synchronously (call from background task).

        Creates the StoryboardState entry before loading the pipeline, so the
        router can immediately mark the task as 'generating'.
        """
        style = StyleMapper().get_style(style_name)

        state = StoryboardState(
            storyboard_id=storyboard_id,
            style=style_name,
            base_checkpoint=self._base_checkpoint,
            lora_names=lora_names,
            status="generating",
            scenes=[
                StoryboardScene(
                    scene_idx=s.scene_idx,
                    storyboard_prompt=s.storyboard_prompt,
                    positive_prompt=s.positive_prompt,
                    approved_version=None,
                )
                for s in scenes
            ],
        )
        self._store.create(state)

        pipe = None
        try:
            pipe, _adapter_names, _default_weights, trigger_tokens = self._load_pipeline(lora_names)
            # Trigger tokens prepended to every scene prompt (matches ScenePromptGenerator behaviour)
            trigger_prefix = ", ".join(trigger_tokens) + ", " if trigger_tokens else ""

            for i, scene in enumerate(scenes):
                seed = i * self.SEED_STRIDE
                prompt = trigger_prefix + style.prefix + scene.positive_prompt
                negative = style.negative_prefix

                scene_dir = self._store.scene_dir(storyboard_id, scene.scene_idx, create=True)
                filename = "v1.png"

                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    width=self.PREVIEW_WIDTH,
                    height=self.PREVIEW_HEIGHT,
                    num_inference_steps=style.steps,
                    guidance_scale=style.cfg_scale,
                    generator=torch.Generator(self._device).manual_seed(seed),
                ).images[0]

                image.save(scene_dir / filename)

                entry = VersionEntry(
                    version=1,
                    filename=filename,
                    seed=seed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self._store.append_version(storyboard_id, scene.scene_idx, entry)

            self._store.update_status(storyboard_id, status="complete")

        except Exception as exc:
            logger.exception("Storyboard generation failed for %s: %s", storyboard_id, exc)
            self._store.update_status(storyboard_id, status="failed", error=str(exc))
            raise
        finally:
            if pipe is not None:
                try:
                    self._vm.kill()
                except Exception as exc:
                    logger.warning(
                        "VRAM cleanup failed after storyboard generation: %s", exc
                    )

    def generate_single_scene(
        self,
        storyboard_id: str,
        scene_idx: int,
        prompt_override: Optional[str],
        seed: Optional[int],
        lora_names: List[str],
        lora_weights: Optional[Dict[str, float]] = None,
    ) -> VersionEntry:
        """Regenerate one scene, appending a new version. Returns the new VersionEntry.

        Args:
            lora_weights: Optional per-LoRA weight overrides, keyed by registry name
                (e.g. ``{"rob-character": 0.3}``). LoRAs not in the dict use their
                registry default weight. Stored in the resulting VersionEntry.
        """
        state = self._store.load(storyboard_id)
        if state is None:
            raise LookupError(f"Storyboard '{storyboard_id}' not found.")

        scene = self._get_scene(state, scene_idx)
        style = StyleMapper().get_style(state.style)

        positive_prompt = prompt_override if prompt_override is not None else scene.positive_prompt
        actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

        next_version = (max(v.version for v in scene.versions) + 1) if scene.versions else 1
        filename = f"v{next_version}.png"

        pipe = None
        try:
            pipe, adapter_names, default_weights, trigger_tokens = self._load_pipeline(lora_names)
            trigger_prefix = ", ".join(trigger_tokens) + ", " if trigger_tokens else ""

            # Apply per-scene weight overrides when the caller provides them
            if lora_weights is not None and adapter_names:
                orig_names = [n.replace("_", "-") for n in adapter_names]
                override_w = [
                    lora_weights.get(orig, dw)
                    for orig, dw in zip(orig_names, default_weights)
                ]
                pipe.set_adapters(adapter_names, adapter_weights=override_w)

            scene_dir = self._store.scene_dir(storyboard_id, scene_idx, create=True)
            prompt = trigger_prefix + style.prefix + positive_prompt
            negative = style.negative_prefix

            image = pipe(
                prompt=prompt,
                negative_prompt=negative,
                width=self.PREVIEW_WIDTH,
                height=self.PREVIEW_HEIGHT,
                num_inference_steps=style.steps,
                guidance_scale=style.cfg_scale,
                generator=torch.Generator(self._device).manual_seed(actual_seed),
            ).images[0]

            image.save(scene_dir / filename)

            entry = VersionEntry(
                version=next_version,
                filename=filename,
                seed=actual_seed,
                timestamp=datetime.now(timezone.utc).isoformat(),
                lora_weights=lora_weights or {},
            )
            self._store.append_version(storyboard_id, scene_idx, entry)
            return entry

        except Exception as exc:
            logger.exception(
                "Single-scene regeneration failed for %s/scene_%d: %s",
                storyboard_id, scene_idx, exc,
            )
            raise
        finally:
            if pipe is not None:
                try:
                    self._vm.kill()
                except Exception as exc:
                    logger.warning(
                        "VRAM cleanup failed after single-scene regen: %s", exc
                    )

    def get_state(self, storyboard_id: str) -> Optional[StoryboardState]:
        """Return current StoryboardState, or None if not found."""
        return self._store.load(storyboard_id)

    def approve(
        self,
        storyboard_id: str,
        selections: Dict[int, int],
    ) -> Dict[int, str]:
        """Record approved version per scene. Returns {scene_idx: abs_path}."""
        return self._store.set_approved(storyboard_id, selections)

    # ── internal ──────────────────────────────────────────────────────────────

    def _load_pipeline(self, lora_names: List[str]) -> tuple:
        """Load SDXL pipeline, apply LoRAs, register with VRAMManager.

        Returns ``(pipe, adapter_names, default_weights, trigger_tokens)`` where:
        - ``adapter_names`` — diffusers adapter identifiers (underscores) for each
          loaded LoRA, in load order
        - ``default_weights`` — registry weight for each adapter (same order)
        - ``trigger_tokens`` — trigger strings to prepend to every positive prompt
        """
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self._base_checkpoint,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self._device)

        self._vm.set_pipeline(pipe, "sdxl_storyboard")
        adapter_names, default_weights, trigger_tokens = self._load_loras(pipe, lora_names)
        return pipe, adapter_names, default_weights, trigger_tokens

    def _load_loras(
        self,
        pipe: StableDiffusionXLPipeline,
        lora_names: List[str],
    ) -> tuple:
        """Load each LoRA by name from the registry. Unknown names are skipped.

        Returns ``(adapter_names, default_weights, trigger_tokens)``:
        - ``adapter_names``   — diffusers adapter identifiers (underscores), in load order
        - ``default_weights`` — registry weight for each adapter (same order)
        - ``trigger_tokens``  — trigger strings for all loaded LoRAs, in load order

        The caller is responsible for applying weight overrides via
        ``pipe.set_adapters`` after this returns.
        """
        if not lora_names:
            return [], [], []

        if not self._loras_yaml.exists():
            logger.warning(
                "loras.yaml not found at %s — skipping LoRA loading", self._loras_yaml
            )
            return [], [], []

        registry = LoRARegistry(
            registry_path=str(self._loras_yaml),
            base_path=str(self._lora_base),
        )

        adapter_names: List[str] = []
        adapter_weights: List[float] = []
        trigger_tokens: List[str] = []

        for name in lora_names:
            entry = registry.get(name)
            if entry is None:
                logger.warning("LoRA '%s' not found in registry — skipping", name)
                continue

            abs_path = str(self._lora_base / entry.file_path)
            adapter_name = name.replace("-", "_")

            try:
                pipe.load_lora_weights(abs_path, adapter_name=adapter_name)
                adapter_names.append(adapter_name)
                adapter_weights.append(float(entry.weight))
                if entry.trigger_token:
                    trigger_tokens.append(entry.trigger_token)
            except Exception as exc:
                logger.warning("Failed to load LoRA '%s': %s — skipping", name, exc)

        if len(adapter_names) > 1:
            pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

        return adapter_names, adapter_weights, trigger_tokens

    @staticmethod
    def _get_scene(state: StoryboardState, scene_idx: int) -> StoryboardScene:
        for s in state.scenes:
            if s.scene_idx == scene_idx:
                return s
        raise ValueError(
            f"scene_idx {scene_idx} not found in storyboard '{state.storyboard_id}'."
        )
