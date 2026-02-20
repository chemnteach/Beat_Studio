"""Video router — execution plan, generation, style management."""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import yaml
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
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


def _get_beat_sync() -> BeatSynchronizer:
    global _beat_sync
    if _beat_sync is None:
        _beat_sync = BeatSynchronizer()
    return _beat_sync


def _get_model_router() -> ModelRouter:
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router


def _get_cost_estimator() -> CostEstimator:
    global _cost_estimator
    if _cost_estimator is None:
        _cost_estimator = CostEstimator()
    return _cost_estimator


def _get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
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
) -> None:
    """Background worker: full video generation pipeline."""
    tm = _get_task_manager()
    try:
        from backend.services.audio.types import SceneTiming
        from backend.services.prompt.narrative_analyzer import NarrativeAnalyzer
        from backend.services.prompt.scene_generator import ScenePromptGenerator
        from backend.services.prompt.style_mapper import StyleMapper
        from backend.services.prompt.types import ComposedPrompt
        from backend.services.video.assembler import VideoAssembler
        from backend.services.video.encoder import VideoEncoder
        from backend.services.video.transition_engine import TransitionEngine

        # 1. Load full SongAnalysis (for NarrativeAnalyzer) + simple namespace (for BeatSynchronizer)
        tm.update_progress(task_id, "loading_analysis", 5.0, "Loading audio analysis…")
        analysis = _load_full_analysis(audio_id)
        sync_ns = _load_analysis_for_sync(audio_id)

        # 2. Beat-aligned scene plan
        tm.update_progress(task_id, "planning_scenes", 10.0, "Creating scene plan…")
        synced_scenes = _get_beat_sync().create_scene_plan(sync_ns, quality_tier=quality)

        # 3. Select video generation backend
        try:
            backend = _get_model_router().select_backend(
                style=style, quality=quality, local_preferred=True,
            )
        except Exception as exc:
            tm.fail_task(task_id, f"No video backend available: {exc}")
            return

        # 4. Narrative analysis (LLM or heuristic fallback)
        tm.update_progress(task_id, "analyzing_narrative", 15.0, "Analyzing narrative arc…")
        narrative = NarrativeAnalyzer().analyze(analysis, user_concept=style)

        # 5. Animation style
        try:
            animation_style = StyleMapper().get_style(style)
        except Exception:
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

        # 7. Generate per-scene prompts
        tm.update_progress(task_id, "generating_prompts", 20.0, "Building scene prompts…")
        scene_prompts = ScenePromptGenerator().generate_prompts(
            narrative, scene_timings, animation_style, loras=[], user_overrides={},
        )

        # 8. Wrap as ComposedPrompts for the backend API
        composed = [
            ComposedPrompt(
                positive=sp.positive,
                negative=sp.negative,
                cfg_scale=sp.cfg_scale,
                steps=sp.steps,
                model=backend.name(),
                nsfw=False,
            )
            for sp in scene_prompts
        ]

        # 9. Generate video clips
        clips = []
        for i, (cp, scene) in enumerate(zip(composed, synced_scenes)):
            pct = 25.0 + (i / max(len(composed), 1)) * 50.0
            tm.update_progress(
                task_id, "generating_clips", pct,
                f"Generating clip {i + 1}/{len(composed)}…",
            )
            clip = backend.generate_clip(cp, scene.duration_sec, resolution, fps=24, seed=i)
            clip.scene_index = scene.scene_index
            clips.append(clip)

        # 10. Select transitions between consecutive scenes
        transition_engine = TransitionEngine()
        transitions = [
            transition_engine.select_transition(
                synced_scenes[i], synced_scenes[i + 1], backend, quality,
            )
            for i in range(len(synced_scenes) - 1)
        ]

        # 11. Assemble clips + audio
        tm.update_progress(task_id, "assembling", 80.0, "Assembling video…")
        video_id = uuid.uuid4().hex[:8]
        out_dir = _BACKEND_DIR.parent / "output" / "video" / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        assembled_path = str(out_dir / "assembled.mp4")
        VideoAssembler().assemble(
            clips, transitions,
            audio_path=analysis.file_path,
            output_path=assembled_path,
            resolution=resolution,
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


class SceneEditRequest(BaseModel):
    video_id: str
    scene_index: int
    new_prompt: str


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
    # Use style/quality/resolution from the GenerateRequest if available,
    # otherwise fall back to sensible defaults.
    style = getattr(request, "style", "cinematic")
    quality = getattr(request, "quality", "professional")
    raw_res = getattr(request, "resolution", [1920, 1080])
    resolution: Tuple[int, int] = (raw_res[0], raw_res[1]) if raw_res else (1920, 1080)

    task_id = _get_task_manager().create_task("generate_video", {
        "audio_id": request.audio_id,
        "plan_id": request.plan_id,
        "style": style,
        "quality": quality,
    })
    background_tasks.add_task(
        _run_generate_video, task_id, request.audio_id, style, quality, resolution,
    )
    return {"task_id": task_id, "status": "queued", "plan_id": request.plan_id}


@router.post("/scene/edit", status_code=status.HTTP_202_ACCEPTED)
async def edit_scene(request: SceneEditRequest) -> Dict[str, Any]:
    """Edit a single scene prompt and regenerate that scene."""
    return {
        "task_id":     "stub",
        "video_id":    request.video_id,
        "scene_index": request.scene_index,
        "status":      "queued",
    }


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
    """List available video generation backends."""
    available = _get_model_router().list_available_backends()
    return {"backends": available}


@router.get("/download/{video_id}")
async def download_video(video_id: str) -> Dict[str, str]:
    """Download a generated video by ID."""
    return {"video_id": video_id, "status": "stub", "url": ""}
