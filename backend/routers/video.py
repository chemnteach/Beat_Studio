"""Video router — execution plan, generation, style management."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import yaml
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.services.video.beat_sync import BeatSynchronizer
from backend.services.video.cost_estimator import CostEstimator
from backend.services.video.model_router import ModelRouter, NoBackendAvailableError
from backend.services.shared.task_manager import TaskManager

logger = logging.getLogger("beat_studio.routers.video")
router = APIRouter()

_BACKEND_DIR = Path(__file__).parent.parent
_ANALYSIS_DIR = _BACKEND_DIR / "data" / "analysis"
_STYLES_YAML = _BACKEND_DIR / "config" / "animation_styles.yaml"

_beat_sync: Optional[BeatSynchronizer] = None
_model_router: Optional[ModelRouter] = None
_cost_estimator: Optional[CostEstimator] = None
_task_manager: Optional[TaskManager] = None
_beat_sync_lock = threading.Lock()
_model_router_lock = threading.Lock()
_cost_estimator_lock = threading.Lock()
_task_manager_lock = threading.Lock()


def _get_beat_sync() -> BeatSynchronizer:
    global _beat_sync
    if _beat_sync is not None:
        return _beat_sync
    with _beat_sync_lock:
        if _beat_sync is None:
            _beat_sync = BeatSynchronizer()
    return _beat_sync


def _get_model_router() -> ModelRouter:
    global _model_router
    if _model_router is not None:
        return _model_router
    with _model_router_lock:
        if _model_router is None:
            _model_router = ModelRouter()
    return _model_router


def _get_cost_estimator() -> CostEstimator:
    global _cost_estimator
    if _cost_estimator is not None:
        return _cost_estimator
    with _cost_estimator_lock:
        if _cost_estimator is None:
            _cost_estimator = CostEstimator()
    return _cost_estimator


def _get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is not None:
        return _task_manager
    with _task_manager_lock:
        if _task_manager is None:
            _task_manager = TaskManager(db_path=str(_BACKEND_DIR / "tasks.db"))
    return _task_manager


def _load_analysis_for_sync(audio_id: str) -> SimpleNamespace:
    """Load cached SongAnalysis JSON and adapt to BeatSynchronizer duck-type.

    BeatSynchronizer expects:
      analysis.duration  (float)
      analysis.bpm       (float)
      analysis.sections  (list of objects with .start, .end, .section_type, .energy_level)
    SongAnalysis JSON stores duration_sec, start_sec, end_sec — so we remap.
    """
    path = _ANALYSIS_DIR / f"{audio_id}.json"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis found for audio_id {audio_id!r}. "
                   "Run POST /api/audio/analyze first.",
        )

    data = json.loads(path.read_text())

    sections = []
    for s in data.get("sections", []):
        sections.append(SimpleNamespace(
            start=s.get("start_sec", 0.0),
            end=s.get("end_sec", 0.0),
            section_type=s.get("section_type", "verse"),
            energy_level=s.get("energy_level", 0.5),
        ))

    return SimpleNamespace(
        duration=data.get("duration_sec", 0.0),
        bpm=data.get("bpm", 120.0),
        sections=sections,
        beat_times=data.get("beat_times", []),
    )


def _load_full_analysis(audio_id: str):
    """Load cached SongAnalysis JSON as a proper SongAnalysis dataclass.

    Used by the generate pipeline, which needs real SectionInfo objects
    (emotional_tone, lyrical_content, themes, etc.) for NarrativeAnalyzer.
    Falls back to sensible defaults for any missing fields.
    """
    path = _ANALYSIS_DIR / f"{audio_id}.json"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis found for audio_id {audio_id!r}. "
                   "Run POST /api/audio/analyze first.",
        )

    data = json.loads(path.read_text())

    from backend.services.audio.types import SectionInfo, SongAnalysis

    sections = []
    for s in data.get("sections", []):
        start = s.get("start_sec", 0.0)
        end = s.get("end_sec", 0.0)
        sections.append(SectionInfo(
            section_type=s.get("section_type", "verse"),
            start_sec=start,
            end_sec=end,
            duration_sec=end - start,
            energy_level=s.get("energy_level", 0.5),
            spectral_centroid=s.get("spectral_centroid", 2000.0),
            tempo_stability=s.get("tempo_stability", 0.8),
            vocal_density=s.get("vocal_density", "medium"),
            vocal_intensity=s.get("vocal_intensity", 0.5),
            lyrical_content=s.get("lyrical_content", ""),
            emotional_tone=s.get("emotional_tone", "neutral"),
            lyrical_function=s.get("lyrical_function", "narrative"),
            themes=s.get("themes", []),
        ))

    return SongAnalysis(
        artist=data.get("artist", "Unknown"),
        title=data.get("title", "Unknown"),
        file_path=data.get("file_path", ""),
        bpm=data.get("bpm", 120.0),
        key=data.get("key", "C major"),
        camelot=data.get("camelot", "8B"),
        duration_sec=data.get("duration_sec", 0.0),
        sample_rate=data.get("sample_rate", 44100),
        energy_level=data.get("energy_level", 0.5),
        first_downbeat_sec=data.get("first_downbeat_sec", 0.0),
        sections=sections,
        beat_times=data.get("beat_times", []),
        transcript=data.get("transcript", ""),
        mood_summary=data.get("mood_summary", ""),
        genres=data.get("genres", []),
        primary_genre=data.get("primary_genre", ""),
        emotional_arc=data.get("emotional_arc", ""),
    )


# ── Quality tier → encoder quality mapping ─────────────────────────────────────
_QUALITY_TO_ENC = {"basic": "draft", "professional": "standard", "cinematic": "broadcast"}


# ── Background worker ──────────────────────────────────────────────────────────

def _run_generate_video(
    task_id: str,
    audio_id: str,
    style: str,
    quality: str,
    resolution: Tuple[int, int],
    creative_direction: str = "",
    scene_indices: Optional[List[int]] = None,
    user_overrides: Optional[Dict[str, str]] = None,
    lora_names: Optional[List[str]] = None,
    backend: str = "local",
    runpod_model: Optional[str] = None,
    approved_image_paths: Optional[List[str]] = None,
) -> None:
    """Background worker: full video generation pipeline."""
    tm = _get_task_manager()
    try:
        from backend.services.audio.types import SceneTiming
        from backend.services.lora.registry import LoRARegistry
        from backend.services.prompt.narrative_analyzer import NarrativeAnalyzer
        from backend.services.prompt.scene_generator import ScenePromptGenerator
        from backend.services.prompt.style_mapper import StyleMapper
        from backend.services.prompt.types import ComposedPrompt, LoRAConfig
        from backend.services.video.assembler import VideoAssembler
        from backend.services.video.encoder import VideoEncoder
        from backend.services.video.transition_engine import TransitionEngine

        # 1. Load full SongAnalysis (for NarrativeAnalyzer) + simple namespace (for BeatSynchronizer)
        tm.update_progress(task_id, "loading_analysis", 5.0, "Loading audio analysis…")
        analysis = _load_full_analysis(audio_id)
        sync_ns = _load_analysis_for_sync(audio_id)

        # 2. Beat-aligned scene plan
        tm.update_progress(task_id, "planning_scenes", 10.0, "Creating scene plan…")
        synced_scenes = _get_beat_sync().create_scene_plan(
            sync_ns, quality_tier=quality, beat_times=sync_ns.beat_times or None,
        )
        if scene_indices:
            idx_set = set(scene_indices)
            synced_scenes = [s for s in synced_scenes if s.scene_index in idx_set]
            logger.info("Test run: generating %d specific scene(s): %s", len(synced_scenes), sorted(idx_set))

        # 3. Select video generation backend
        try:
            if backend == "runpod":
                from backend.services.video.backends.runpod_client import RunPodBackend
                video_backend = RunPodBackend(model_name=runpod_model or "skyreels_v3_r2v")
            else:
                video_backend = _get_model_router().select_backend(
                    style=style, quality=quality, local_preferred=True,
                )
        except Exception as exc:
            tm.fail_task(task_id, f"No video backend available: {exc}")
            return

        # 4. Narrative analysis (LLM or heuristic fallback)
        tm.update_progress(task_id, "analyzing_narrative", 15.0, "Analyzing narrative arc…")
        narrative = NarrativeAnalyzer().analyze(
            analysis, user_concept=creative_direction or None,
        )

        # 5. Animation style
        try:
            animation_style = StyleMapper().get_style(style)
        except Exception as exc:
            logger.warning("Unknown style %r (%s) — falling back to 'cinematic'", style, exc)
            animation_style = StyleMapper().get_style("cinematic")

        # 6. Convert SyncedScenePlan → SceneTiming (ScenePromptGenerator needs energy_level + section_type)
        def _parse_section_type(notes: str) -> str:
            for stype in ("intro", "verse", "chorus", "bridge", "outro"):
                if stype in notes.lower():
                    return stype
            return "verse"

        scene_timings = [
            SceneTiming(
                scene_index=s.scene_index,
                start_sec=s.start_sec,
                end_sec=s.end_sec,
                duration_sec=s.duration_sec,
                is_hero=s.is_hero,
                energy_level=1.0 if s.is_hero else 0.5,
                section_type=_parse_section_type(s.notes),
                beat_aligned=True,
            )
            for s in synced_scenes
        ]

        # 7. Load available LoRAs from registry
        lora_registry = LoRARegistry(
            str(_BACKEND_DIR / "config" / "loras.yaml"),
            base_path=str(_BACKEND_DIR.parent / "output" / "loras"),
        )
        requested_names: Optional[set] = set(lora_names) if lora_names else None
        active_loras: List[LoRAConfig] = []
        for e in lora_registry.list_all():
            if e.status != "available":
                continue
            if requested_names is not None and e.name not in requested_names:
                continue  # user didn't select this LoRA
            abs_path = lora_registry._base / e.file_path  # noqa: SLF001
            if not abs_path.exists():
                logger.warning("LoRA '%s' registered but file missing: %s", e.name, abs_path)
                continue
            active_loras.append(LoRAConfig(
                name=e.name,
                trigger_token=e.trigger_token,
                weight=e.weight,
                lora_type=e.type,
                file_path=str(abs_path),
            ))
        logger.info("Active LoRAs for generation (%d selected): %s",
                    len(active_loras), [la.name for la in active_loras])

        # 8. Generate per-scene prompts.
        # user_overrides keys arrive as strings from the frontend; ScenePromptGenerator
        # needs int keys so user text is used as base_desc (style prefix is still prepended).
        tm.update_progress(task_id, "generating_prompts", 20.0, "Building scene prompts…")
        int_overrides: Dict[int, str] = {
            int(k): v for k, v in (user_overrides or {}).items()
            if k.isdigit() and v.strip()
        }
        if int_overrides:
            logger.info("Applying %d user-reviewed prompt(s) via ScenePromptGenerator", len(int_overrides))
        scene_prompts = ScenePromptGenerator().generate_prompts(
            narrative, scene_timings, animation_style,
            loras=active_loras, user_overrides=int_overrides,
        )

        # 9. Wrap as ComposedPrompts for the backend API — carry LoRA configs per scene
        lora_by_name: Dict[str, LoRAConfig] = {la.name: la for la in active_loras}
        composed = [
            ComposedPrompt(
                positive=sp.positive,
                negative=sp.negative,
                cfg_scale=sp.cfg_scale,
                steps=sp.steps,
                model=video_backend.name(),
                nsfw=False,
                base_checkpoint=animation_style.base_checkpoint,
                lora_configs=[lora_by_name[n] for n in sp.lora_names if n in lora_by_name],
            )
            for sp in scene_prompts
        ]

        # 9. Generate video clips
        image_paths = approved_image_paths or []
        clips = []
        for i, (cp, scene) in enumerate(zip(composed, synced_scenes)):
            pct = 25.0 + (i / max(len(composed), 1)) * 50.0
            tm.update_progress(
                task_id, "generating_clips", pct,
                f"Generating clip {i + 1}/{len(composed)}…",
            )
            if i < len(image_paths):
                cp.init_image_path = image_paths[i]
            clip = video_backend.generate_clip(cp, scene.duration_sec, resolution, fps=24, seed=i)
            clip.scene_index = scene.scene_index
            clips.append(clip)

        # 10. Select transitions between consecutive scenes
        transition_engine = TransitionEngine()
        transitions = [
            transition_engine.select_transition(
                synced_scenes[i], synced_scenes[i + 1], video_backend, quality,
            )
            for i in range(len(synced_scenes) - 1)
        ]

        # 11. Assemble clips + audio
        tm.update_progress(task_id, "assembling", 80.0, "Assembling video…")
        video_id = uuid.uuid4().hex[:8]
        out_dir = _BACKEND_DIR.parent / "output" / "video" / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        assembled_path = str(out_dir / "assembled.mp4")
        scene_durations = [s.duration_sec for s in synced_scenes]
        VideoAssembler().assemble(
            clips, transitions,
            audio_path=analysis.file_path,
            output_path=assembled_path,
            resolution=resolution,
            scene_durations=scene_durations,
        )

        # 12. Encode for platform
        tm.update_progress(task_id, "encoding", 90.0, "Encoding final video…")
        final_path = str(out_dir / "final.mp4")
        enc_quality = _QUALITY_TO_ENC.get(quality, "standard")
        VideoEncoder().encode(assembled_path, final_path, quality=enc_quality, platform="youtube")

        tm.complete_task(task_id, {"video_id": video_id, "path": final_path})

    except Exception as exc:
        # Background tasks run after the 202 response is sent — never re-raise.
        # HTTPException detail is still useful as a fail_task message.
        logger.exception("Video generation failed: %s", exc)
        tm.fail_task(task_id, str(exc))


class SectionOverride(BaseModel):
    section_type: str
    start_sec: float
    end_sec: float
    lyrical_content: str = ""


class PromptsRequest(BaseModel):
    audio_id: str
    style: str = "cinematic"
    quality: str = "professional"
    creative_direction: str = ""
    sections: Optional[List[SectionOverride]] = None   # user-edited sections from frontend


class PlanRequest(BaseModel):
    audio_id: str
    style: str = "cinematic"
    quality: str = "professional"
    local_preferred: bool = True
    resolution: List[int] = [1920, 1080]


class GenerateRequest(BaseModel):
    plan_id: str
    audio_id: str
    style: str = "cinematic"
    quality: str = "professional"
    resolution: List[int] = [1920, 1080]
    creative_direction: str = ""
    scene_indices: Optional[List[int]] = None  # If set, generate only these scene indices (test run)
    user_overrides: Dict[str, str] = {}  # scene_index (as string) → reviewed positive prompt
    lora_names: List[str] = []           # LoRA names to activate; empty = no LoRAs loaded
    backend: str = "local"               # "local" | "runpod"
    runpod_model: Optional[str] = None   # model name when backend="runpod"
    approved_image_paths: List[str] = [] # ordered storyboard keyframe paths (index 0 = scene 0)


class SceneEditRequest(BaseModel):
    video_id: str
    scene_index: int
    new_prompt: str


@router.post("/prompts")
async def generate_prompts(request: PromptsRequest) -> Dict[str, Any]:
    """Generate per-section AI prompts from narrative analysis.

    Runs NarrativeAnalyzer (Claude) + ScenePromptGenerator synchronously.
    Accepts optional user-edited sections with lyrical content to override
    what's in the cached analysis JSON.
    """
    from backend.services.audio.types import SectionInfo, SceneTiming
    from backend.services.prompt.narrative_analyzer import NarrativeAnalyzer
    from backend.services.prompt.scene_generator import ScenePromptGenerator
    from backend.services.prompt.style_mapper import StyleMapper

    analysis = _load_full_analysis(request.audio_id)

    # Override sections with user-edited versions if provided
    if request.sections:
        from backend.services.audio.types import SectionInfo as SI
        analysis.sections = [
            SI(
                section_type=s.section_type,
                start_sec=s.start_sec,
                end_sec=s.end_sec,
                duration_sec=s.end_sec - s.start_sec,
                energy_level=0.5,
                spectral_centroid=2000.0,
                tempo_stability=0.8,
                vocal_density="medium",
                vocal_intensity=0.5,
                lyrical_content=s.lyrical_content,
                emotional_tone="neutral",
                lyrical_function="narrative",
                themes=[],
            )
            for s in request.sections
        ]

    # Build sync namespace from the (possibly user-overridden) analysis sections
    # so BeatSynchronizer respects the user's 8-section breakdown, not the cached 5-section auto-detect.
    sync_ns = SimpleNamespace(
        duration=analysis.duration_sec,
        bpm=analysis.bpm,
        sections=[
            SimpleNamespace(
                start=s.start_sec,
                end=s.end_sec,
                section_type=s.section_type,
                energy_level=s.energy_level,
            )
            for s in analysis.sections
        ],
    )
    beat_scenes = _get_beat_sync().create_scene_plan(
        sync_ns,
        quality_tier=request.quality,
        beat_times=analysis.beat_times,
    )

    scene_timings = [
        SceneTiming(
            scene_index=s.scene_index,
            start_sec=s.start_sec,
            end_sec=s.end_sec,
            duration_sec=s.duration_sec,
            is_hero=s.is_hero,
            energy_level=1.0 if s.is_hero else 0.5,
            section_type=_parse_section_type_from_notes(s.notes),
            beat_aligned=True,
        )
        for s in beat_scenes
    ]

    try:
        animation_style = StyleMapper().get_style(request.style)
    except Exception:
        animation_style = StyleMapper().get_style("cinematic")

    narrative = NarrativeAnalyzer().analyze(
        analysis,
        user_concept=request.creative_direction or None,
    )

    scene_prompts = ScenePromptGenerator().generate_prompts(
        narrative, scene_timings, animation_style, loras=[], user_overrides={},
    )

    # Distill storyboards → short (<60 word) generation prompts, one unique prompt per clip
    from backend.services.prompt.distiller import PromptDistiller
    scene_prompts = PromptDistiller().distill(scene_prompts, animation_style.prefix)

    # Mark first clip after bridge as hero (boat deck / first romantic moment)
    prev_was_bridge = False
    for sp, st in zip(scene_prompts, scene_timings):
        if st.section_type == "bridge":
            prev_was_bridge = True
        elif prev_was_bridge:
            sp.is_hero = True
            prev_was_bridge = False
            break

    # Mark final clip as hero (resolution / last shot of the song)
    if scene_prompts:
        scene_prompts[-1].is_hero = True

    return {
        "overall_concept": narrative.overall_concept,
        "color_palette":   narrative.color_palette,
        "mood_progression": narrative.mood_progression,
        "prompts": [
            {
                "scene_index":   sp.scene_index,
                "section_type":  scene_timings[sp.scene_index].section_type if sp.scene_index < len(scene_timings) else "verse",
                "start_sec":     sp.start_sec,
                "end_sec":       sp.end_sec,
                "duration_sec":  sp.duration_sec,
                "is_hero":       sp.is_hero,
                "storyboard":    sp.storyboard,
                "positive":      sp.positive,
                "negative":      sp.negative,
                "transition":    sp.transition_hint,
            }
            for sp in scene_prompts
        ],
    }


def _parse_section_type_from_notes(notes: str) -> str:
    for stype in ("intro", "verse", "pre_chorus", "chorus", "bridge", "outro"):
        if stype in notes.lower():
            return stype
    return "verse"


@router.post("/plan")
async def plan_video(request: PlanRequest) -> Dict[str, Any]:
    """Generate a video execution plan (backend selection, cost, time estimate).

    Loads the cached audio analysis, creates beat-aligned scenes, selects
    the best available backend, and returns cost and time estimates.
    """
    analysis = _load_analysis_for_sync(request.audio_id)
    resolution = tuple(request.resolution[:2]) if len(request.resolution) >= 2 else (1920, 1080)

    # Beat-aligned scene plan
    scenes = _get_beat_sync().create_scene_plan(analysis, quality_tier=request.quality)

    # Backend selection
    try:
        backend = _get_model_router().select_backend(
            style=request.style,
            quality=request.quality,
            local_preferred=request.local_preferred,
        )
        backend_name = backend.name()
        is_local = backend.vram_required_gb() > 0
    except NoBackendAvailableError as exc:
        logger.warning("No backend available: %s", exc)
        backend_name = "none"
        is_local = False

    # Cost and time estimates
    estimator = _get_cost_estimator()
    num_scenes = len(scenes)

    if backend_name != "none":
        try:
            total_cost = estimator.estimate_total(backend, num_scenes, resolution)
            total_time = estimator.estimate_total_time(backend, num_scenes, resolution)
        except Exception as exc:
            logger.warning("Cost estimation failed: %s", exc)
            total_cost = 0.0
            total_time = 0.0
    else:
        total_cost = 0.0
        total_time = 0.0

    plan_id = str(uuid.uuid4())

    return {
        "plan_id":            plan_id,
        "audio_id":           request.audio_id,
        "backend":            backend_name,
        "style":              request.style,
        "quality":            request.quality,
        "num_scenes":         num_scenes,
        "estimated_time_sec": round(total_time, 1),
        "estimated_cost_usd": round(total_cost, 4),
        "is_local":           is_local,
        "scenes": [
            {
                "scene_index": s.scene_index,
                "start_sec":   round(s.start_sec, 3),
                "end_sec":     round(s.end_sec, 3),
                "duration_sec": round(s.duration_sec, 3),
                "is_hero":     s.is_hero,
                "notes":       s.notes,
            }
            for s in scenes
        ],
    }


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED)
async def generate_video(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """Start video generation as a background task."""
    style = request.style
    quality = request.quality
    raw_res = request.resolution
    resolution: Tuple[int, int] = (raw_res[0], raw_res[1])

    task_id = _get_task_manager().create_task("generate_video", {
        "audio_id": request.audio_id,
        "plan_id": request.plan_id,
        "style": style,
        "quality": quality,
    })
    background_tasks.add_task(
        _run_generate_video, task_id, request.audio_id, style, quality, resolution,
        request.creative_direction,
        request.scene_indices or None,
        request.user_overrides or {},
        request.lora_names or None,
        request.backend,
        request.runpod_model,
        request.approved_image_paths or [],
    )
    return {"task_id": task_id, "status": "queued", "plan_id": request.plan_id}


@router.post("/scene/edit", status_code=status.HTTP_202_ACCEPTED)
async def edit_scene(request: SceneEditRequest) -> Dict[str, Any]:
    """Edit a single scene prompt and regenerate that scene."""
    raise HTTPException(status_code=501, detail="Scene editing not yet implemented")


@router.get("/styles")
async def list_styles() -> Dict[str, Any]:
    """List all available animation styles from animation_styles.yaml."""
    try:
        raw = yaml.safe_load(_STYLES_YAML.read_text())
    except Exception as exc:
        logger.warning("Failed to load animation_styles.yaml: %s", exc)
        return {"styles": [], "total": 0}

    styles = []
    for name, data in raw.get("styles", {}).items():
        styles.append({
            "name":                name,
            "display_name":        data.get("display_name"),
            "category":            data.get("category"),
            "recommended_backend": data.get("recommended_backend"),
            "best_for":            data.get("best_for"),
            "guidance_scale":      data.get("guidance_scale"),
            "steps":               data.get("steps"),
        })

    return {"styles": styles, "total": len(styles)}


@router.get("/backends")
async def list_backends() -> Dict[str, Any]:
    """List all video generation backends with availability and metadata."""
    router = _get_model_router()
    backends = []
    for b in router._discover_backends():
        try:
            available = b.is_available()
        except Exception as exc:
            logger.debug("Backend %s.is_available() failed: %s", b.name(), exc)
            available = False
        try:
            vram = b.vram_required_gb()
        except Exception as exc:
            logger.debug("Backend %s.vram_required_gb() failed: %s", b.name(), exc)
            vram = 0.0
        try:
            cost = b.estimated_cost_per_scene()
        except Exception as exc:
            logger.debug("Backend %s.estimated_cost_per_scene() failed: %s", b.name(), exc)
            cost = 0.0
        backends.append({
            "name":          b.name(),
            "available":     available,
            "is_local":      vram > 0,
            "vram_required_gb": vram,
            "cost_per_scene": cost,
        })
    return {"backends": backends}


@router.get("/frames")
async def list_clip_frames(video_id: Optional[str] = None):
    """List extracted first-frame PNGs.

    If video_id is given, return only frames for the clips in that run's concat.txt.
    Otherwise return all frames.
    """
    import re
    frames_dir = _BACKEND_DIR.parent / "output" / "video" / "clips" / "frames"
    if not frames_dir.exists():
        return {"frames": []}

    if video_id:
        concat = _BACKEND_DIR.parent / "output" / "video" / video_id / "concat.txt"
        if concat.exists():
            clip_ids = re.findall(r"clip_([0-9a-f]+)\.mp4", concat.read_text())
            names = [f"clip_{cid}_frame0.png" for cid in clip_ids]
            # Return only frames that actually exist (may not be extracted yet)
            return {"frames": [n for n in names if (frames_dir / n).exists()]}

    files = sorted(frames_dir.glob("*.png"))
    return {"frames": [f.name for f in files]}


@router.get("/frames/{filename}")
async def serve_clip_frame(filename: str):
    """Serve a single extracted frame PNG."""
    frames_dir = (_BACKEND_DIR.parent / "output" / "video" / "clips" / "frames").resolve()
    path = (frames_dir / filename).resolve()
    if not path.is_relative_to(frames_dir) or not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail=f"Frame {filename} not found")
    return FileResponse(str(path), media_type="image/png")


_ALLOWED_PLATFORMS = {"youtube", "instagram", "tiktok"}
_VIDEO_OUTPUT_DIR = None  # resolved lazily after _BACKEND_DIR is set


@router.get("/download/{video_id}")
async def download_video(video_id: str, platform: str = "youtube"):
    """Download a generated video by ID."""
    if platform not in _ALLOWED_PLATFORMS:
        raise HTTPException(status_code=400, detail=f"Invalid platform. Allowed: {sorted(_ALLOWED_PLATFORMS)}")
    video_root = (_BACKEND_DIR.parent / "output" / "video").resolve()
    final = (video_root / video_id / "final.mp4").resolve()
    if not final.is_relative_to(video_root) or not final.exists():
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return FileResponse(
        str(final),
        media_type="video/mp4",
        filename=f"video_{video_id}_{platform}.mp4",
    )
