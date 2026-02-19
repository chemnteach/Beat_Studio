# Beat_Studio — Implementation Plan v1.0

**Date**: 2026-02-18
**Status**: DRAFT — Awaiting annotation cycle
**Parent folder**: Same level as `beatcanvas/` and `AI_Mixer/`
**Project folder**: `Beat_Studio/`

---

## Table of Contents

1. [Vision & Scope](#1-vision--scope)
2. [Architecture Overview](#2-architecture-overview)
3. [Project Structure](#3-project-structure)
4. [Phase 1: Foundation](#4-phase-1-foundation)
5. [Phase 2: Audio Engine](#5-phase-2-audio-engine)
6. [Phase 3: Song Analysis & Prompt Generation](#6-phase-3-song-analysis--prompt-generation)
7. [Phase 4: LoRA Management System](#7-phase-4-lora-management-system)
8. [Phase 5: Nova Fade Character Pipeline](#8-phase-5-nova-fade-character-pipeline)
9. [Phase 6: Video Generation Engine](#9-phase-6-video-generation-engine)
10. [Phase 7: Animation Style System](#10-phase-7-animation-style-system)
11. [Phase 8: Video Assembly & Continuity](#11-phase-8-video-assembly--continuity)
12. [Phase 9: FastAPI Backend](#12-phase-9-fastapi-backend)
13. [Phase 10: React Frontend](#13-phase-10-react-frontend)
14. [Phase 11: Testing & Quality](#14-phase-11-testing--quality)
15. [Phase 12: Model & Dependency Setup](#15-phase-12-model--dependency-setup)
16. [Reference Documents](#16-reference-documents)
17. [Architectural Decisions](#17-architectural-decisions)
18. [Known Constraints & Hard Rules](#18-known-constraints--hard-rules)
19. [Todo List](#19-todo-list)

---

## 1. Vision & Scope

Beat_Studio is a unified AI-powered music video production application that combines intelligent audio mashup creation with professional video generation. It merges the proven capabilities of AI_Mixer (audio mashup pipeline) and BeatCanvas (AI video generation) into a single, high-quality application — not by copying, but by smartly leveraging the work done in both projects to build something better.

### Two Primary Workflows

**Path A: Original Music → Music Video**
Upload a song → analyze audio + lyrics → generate scene prompts → select animation/photorealistic style → generate continuous video synced to the song.

**Path B: Mashup → Music Video**
Use the audio mashup engine (8 mashup types, semantic matching, stem separation) to create a mashup → then either:
- **(B1)** Generate a music video for the mashup using the same video pipeline as Path A
- **(B2)** Generate a DJ character video featuring Nova Fade performing the mashup, using the real video generation pipeline (not Blender)

### Core Principles

- **Quality over speed** — the output is a continuous, professional video. Never a slideshow.
- **No Ken Burns effects** — explicitly excluded from the application.
- **No Blender** — all video generation uses AI video models (local or cloud).
- **User decides** — present options with cost/time/quality tradeoffs. The user picks the final path.
- **Model-agnostic video backend** — the architecture supports swapping generation backends as better models emerge.
- **Smart reuse** — leverage AI_Mixer and BeatCanvas code intelligently, not copy blindly.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        React / TypeScript Frontend                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  Upload   │  │ Mashup   │  │ Video    │  │ LoRA Management  │   │
│  │  & Analyze│  │ Workshop │  │ Studio   │  │ & Training       │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ REST + WebSocket
┌───────────────────────────▼─────────────────────────────────────────┐
│                        FastAPI Backend                               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Orchestrator Layer                         │    │
│  │  Routes requests → manages task state → WebSocket progress   │    │
│  └──────────┬──────────────┬───────────────┬───────────────────┘    │
│             │              │               │                         │
│  ┌──────────▼──────┐ ┌────▼─────────┐ ┌──▼──────────────────┐     │
│  │  Audio Engine    │ │ Prompt Engine│ │ Video Engine         │     │
│  │                  │ │              │ │                      │     │
│  │ • Mashup Pipeline│ │ • Song       │ │ • Model Router       │     │
│  │   (8 types)      │ │   Analysis   │ │ • Local Backends     │     │
│  │ • Stem Separation│ │ • Lyric      │ │   - AnimateDiff      │     │
│  │ • ChromaDB Memory│ │   Extraction │ │   - WAN 2.6 local    │     │
│  │ • Curator/Match  │ │ • Scene      │ │   - CogVideoX        │     │
│  │ • Ingestion      │ │   Prompting  │ │   - SVD              │     │
│  │                  │ │ • User Input │ │   - SDXL + ControlNet│     │
│  └──────────────────┘ │   Merge      │ │ • Cloud Backends     │     │
│                       │ • Style      │ │   - WAN 2.6 RunPod   │     │
│                       │   Selection  │ │   - SkyReels RunPod  │     │
│                       └──────────────┘ │ • Assembly Pipeline   │     │
│                                        │ • VRAM Manager        │     │
│  ┌────────────────────────────────┐    └──────────────────────┘     │
│  │  LoRA Engine                   │                                  │
│  │ • Registry & Discovery         │    ┌──────────────────────┐     │
│  │ • Training Pipeline            │    │ Nova Fade Pipeline    │     │
│  │ • Download from HuggingFace/   │    │ • Character LoRA     │     │
│  │   Civitai                      │    │ • Drift Detection    │     │
│  │ • Recommendation Engine        │    │ • Canonical Prompts  │     │
│  │ • Drift Testing                │    │ • DJ Video Generator │     │
│  └────────────────────────────────┘    └──────────────────────┘     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Shared Services                                              │   │
│  │  • Config Manager  • VRAM Manager  • Task Queue  • Logging   │   │
│  │  • Hardware Detection  • Cost Estimator  • Model Downloader  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Difference from Source Projects

- **AI_Mixer** used a linear agent pipeline (Ingestion → Analyst → Curator → Engineer) with LangGraph orchestration. Beat_Studio preserves this pipeline for the audio/mashup path but wraps it in the FastAPI task system.
- **BeatCanvas** had a 2,315-line monolith `main.py`. Beat_Studio breaks this into properly separated routers and service modules.
- **Both** had their own audio analysis. Beat_Studio uses a single unified analysis module that serves both mashup matching and video scene generation.

---

## 3. Project Structure

```
Beat_Studio/
├── backend/
│   ├── main.py                         # FastAPI app, router mounting, CORS, startup
│   ├── routers/
│   │   ├── audio.py                    # Upload, analyze, ingest endpoints
│   │   ├── mashup.py                   # Mashup creation, matching, library
│   │   ├── video.py                    # Video generation, assembly, download
│   │   ├── lora.py                     # LoRA management, training, registry
│   │   ├── nova_fade.py                # Nova Fade specific endpoints
│   │   ├── tasks.py                    # Task status, WebSocket progress
│   │   └── system.py                   # Health, GPU status, model inventory
│   │
│   ├── services/
│   │   ├── audio/
│   │   │   ├── analyzer.py             # Unified audio analysis (librosa)
│   │   │   ├── transcriber.py          # Whisper transcription + word timings
│   │   │   └── section_detector.py     # Agglomerative clustering, section typing
│   │   │
│   │   ├── mashup/
│   │   │   ├── ingestion.py            # Local file + YouTube ingestion
│   │   │   ├── analyst.py              # 8-step analysis pipeline
│   │   │   ├── curator.py              # Matching, compatibility scoring
│   │   │   ├── engineer.py             # 8 mashup type implementations
│   │   │   └── memory.py               # ChromaDB client, schema, queries
│   │   │
│   │   ├── prompt/
│   │   │   ├── scene_generator.py      # LLM-driven scene prompt generation
│   │   │   ├── style_mapper.py         # Animation style → prompt modifiers
│   │   │   ├── narrative_analyzer.py   # Song narrative arc analysis
│   │   │   └── prompt_composer.py      # Final prompt assembly with quality tokens
│   │   │
│   │   ├── video/
│   │   │   ├── model_router.py         # Selects best backend for task
│   │   │   ├── backends/
│   │   │   │   ├── base.py             # Abstract VideoBackend interface
│   │   │   │   ├── animatediff.py      # AnimateDiff-Lightning local
│   │   │   │   ├── wan26_local.py      # WAN 2.6 local execution
│   │   │   │   ├── wan26_cloud.py      # WAN 2.6 RunPod execution
│   │   │   │   ├── skyreels_cloud.py   # SkyReels RunPod stitching
│   │   │   │   ├── cogvideox.py        # CogVideoX local
│   │   │   │   ├── svd.py              # Stable Video Diffusion local
│   │   │   │   ├── sdxl_controlnet.py  # SDXL + ControlNet (rotoscope)
│   │   │   │   ├── mochi.py            # Mochi local (evaluate)
│   │   │   │   └── ltx_video.py        # LTX-Video local (evaluate)
│   │   │   ├── assembler.py            # Continuous video assembly
│   │   │   ├── transition_engine.py    # Smooth inter-scene transitions
│   │   │   ├── beat_sync.py            # Beat-aligned scene timing
│   │   │   └── encoder.py             # FFmpeg final encoding
│   │   │
│   │   ├── lora/
│   │   │   ├── registry.py             # YAML-based LoRA inventory
│   │   │   ├── trainer.py              # LoRA training pipeline (SDXL)
│   │   │   ├── downloader.py           # HuggingFace / Civitai download
│   │   │   ├── recommender.py          # Recommend LoRAs for a project
│   │   │   └── drift_test.py           # CLIP-based drift detection
│   │   │
│   │   ├── nova_fade/
│   │   │   ├── character.py            # Nova Fade character management
│   │   │   ├── canonical_prompts.py    # Locked prompt headers
│   │   │   ├── dj_video_generator.py   # DJ performance video pipeline
│   │   │   └── identity_lora.py        # Identity + Style LoRA management
│   │   │
│   │   └── shared/
│   │       ├── config.py               # Unified config manager
│   │       ├── vram_manager.py         # GPU memory management, kill-and-revive
│   │       ├── hardware_detector.py    # GPU capability detection
│   │       ├── cost_estimator.py       # Cloud cost + time estimates
│   │       ├── task_manager.py         # Background task queue + state
│   │       ├── model_downloader.py     # Download models from HF/Civitai
│   │       └── logging.py             # Structured logging
│   │
│   ├── config/
│   │   ├── settings.yaml               # Main application config
│   │   ├── checkpoints.yaml            # Model standards registry
│   │   ├── loras.yaml                  # LoRA registry
│   │   ├── animation_styles.yaml       # All animation style definitions
│   │   ├── nova_fade_constitution.yaml # Nova Fade identity constraints
│   │   └── themes/                     # Visual themes (from AI_Mixer)
│   │
│   ├── models/                         # Local model files (gitignored)
│   ├── data/
│   │   ├── uploads/
│   │   ├── generated_images/
│   │   ├── generated_videos/
│   │   └── library_cache/              # ChromaDB + audio cache
│   │
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── conftest.py
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                     # Root — tab navigation
│   │   ├── components/
│   │   │   ├── AudioUpload.tsx
│   │   │   ├── SongAnalysis.tsx
│   │   │   ├── MashupWorkshop.tsx      # Mashup creation workflow
│   │   │   ├── MashupLibrary.tsx       # ChromaDB library browser
│   │   │   ├── VideoStudio.tsx         # Video generation workflow
│   │   │   ├── SceneEditor.tsx         # Per-scene prompt editing
│   │   │   ├── StyleSelector.tsx       # Animation style picker
│   │   │   ├── LoRAManager.tsx         # LoRA browse, train, download
│   │   │   ├── NovaFadeStudio.tsx      # Nova Fade DJ video workflow
│   │   │   ├── ExecutionPlanner.tsx    # Local vs cloud decision UI
│   │   │   ├── ProgressTracker.tsx     # Real-time generation progress
│   │   │   ├── VideoPreview.tsx        # Final video player + download
│   │   │   ├── CostEstimator.tsx       # Cost/time display widget
│   │   │   └── HardwareStatus.tsx      # GPU/VRAM status display
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts         # WebSocket progress hook
│   │   │   └── useTaskPolling.ts       # Fallback polling hook
│   │   └── types/
│   │       └── index.ts                # Shared TypeScript types
│   │
│   ├── package.json
│   └── tsconfig.json
│
├── output/
│   ├── videos/                         # Final rendered videos
│   ├── loras/                          # Trained LoRA files
│   └── nova_fade/
│       ├── canonical/                  # Nova Fade reference images
│       ├── drift_runs/                 # Drift test results
│       └── datasets/                   # Training datasets
│
├── config.yaml                         # Top-level user config
├── requirements.txt
├── pyproject.toml
├── plan.md                             # This document
├── research.md                         # Research findings (generated separately)
└── README.md
```

---

## 4. Phase 1: Foundation

### 4.1 Project Scaffolding

Create the directory structure above. Initialize:
- `pyproject.toml` with project metadata and dependencies
- `requirements.txt` with pinned versions
- Frontend with `create-react-app` or Vite + TypeScript
- Git repo with `.gitignore` (models/, data/, output/)

### 4.2 Configuration System

Build a unified config manager that merges:
- AI_Mixer's `ConfigManager` pattern (dot-notation access, singleton)
- BeatCanvas's env_loader (priority: local `.env` overrides global)

```python
# backend/services/shared/config.py

class Config:
    """Unified configuration with dot-notation access."""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self._data = self._load_yaml(config_path)
        self._load_env()  # .env overrides
    
    def get(self, key: str, default=None):
        """Dot-notation access: config.get('video.backends.wan26.vram_required')"""
        keys = key.split(".")
        val = self._data
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val
    
    def get_path(self, key: str) -> Path:
        return Path(self.get(key))
```

**Reference**: AI_Mixer `mixer/config.py` for the singleton pattern and dot-notation access. BeatCanvas `backend/src/utils/env_loader.py` for the env priority chain.

### 4.3 Logging

Structured logging with both file and console output. Adapt AI_Mixer's `MixerLogger` pattern but add structured JSON output for the task system.

### 4.4 Hardware Detection

```python
# backend/services/shared/hardware_detector.py

class HardwareProfile:
    gpu_name: str              # e.g., "NVIDIA RTX 4090"
    vram_total_gb: float       # e.g., 24.0
    vram_available_gb: float   # current free VRAM
    cuda_available: bool
    cuda_version: str
    cpu_cores: int
    ram_gb: float
    
    def can_run_local(self, model: str) -> bool:
        """Check if a specific model can run locally."""
        vram_requirements = {
            "animatediff_lightning": 5.6,
            "wan26_local": 12.0,
            "cogvideox": 14.0,
            "svd": 7.5,
            "sdxl_controlnet": 8.0,
            "sdxl_lora_training": 10.0,
            "demucs_stems": 8.0,
        }
        required = vram_requirements.get(model, float("inf"))
        return self.vram_total_gb >= required
    
    def recommend_execution(self, task_type: str) -> ExecutionPlan:
        """Returns local/cloud recommendation with cost/time estimates."""
        ...
```

**Reference**: BeatCanvas `backend/src/local/vram_manager.py` for VRAM thresholds. Extend with explicit model-to-VRAM mapping.

### 4.5 Task Manager

Persistent task state (not in-memory like BeatCanvas). Use SQLite for task persistence so state survives server restarts.

```python
# backend/services/shared/task_manager.py

class TaskManager:
    """Persistent task state with WebSocket progress broadcasting."""
    
    def create_task(self, task_type: str, params: dict) -> str:
        """Returns task_id, persists to SQLite."""
    
    def update_progress(self, task_id: str, stage: str, percent: float, message: str):
        """Updates task state and broadcasts via WebSocket."""
    
    def get_status(self, task_id: str) -> TaskStatus:
        """Returns current task state."""
```

**Improvement over BeatCanvas**: BeatCanvas used `active_tasks = {}` (in-memory dict) — all state lost on restart. Beat_Studio uses SQLite.

---

## 5. Phase 2: Audio Engine

### 5.1 Unified Audio Analysis

Merge the audio analysis from both projects into a single service. AI_Mixer's analysis is deeper (section-level semantic analysis, emotional arc, lyrical function) while BeatCanvas has scene timing generation. Beat_Studio needs both.

```python
# backend/services/audio/analyzer.py

class AudioAnalyzer:
    """Unified audio analysis serving both mashup and video pipelines."""
    
    def analyze(self, audio_path: str, depth: str = "full") -> SongAnalysis:
        """
        depth="basic": BPM, key, duration, energy, mood (fast, for previews)
        depth="standard": + sections, beats, scene timings
        depth="full": + transcription, word timings, semantic analysis,
                        emotional arc, lyrical functions per section
        """
    
    def get_scene_timings(self, analysis: SongAnalysis, 
                          target_count: int = None,
                          quality_tier: str = "professional") -> List[SceneTiming]:
        """Generate video scene boundaries from audio analysis.
        
        Uses beat-aligned boundaries (not evenly spaced).
        Hero scenes identified by energy peaks + structural position.
        """
    
    def get_mashup_metadata(self, analysis: SongAnalysis) -> SongMetadata:
        """Extract AI_Mixer-compatible metadata for ChromaDB storage."""
```

**What to take from AI_Mixer**:
- Section detection via agglomerative clustering (`mixer/audio/analysis.py`)
- Per-section semantic analysis (lyrical function, themes, emotional tone)
- Emotional arc generation
- Camelot wheel key mapping
- Word-level Whisper timestamps

**What to take from BeatCanvas**:
- Scene timing generation with min/max duration constraints (`analyzer.py` lines for `get_scene_timings`)
- Hero scene identification (top 25% energy, structural position)
- Beat-aligned scene boundaries
- Mood classification (but enhance beyond BeatCanvas's simple threshold approach)

**What to improve**:
- BeatCanvas's mood classification was a simple centroid+energy threshold (5 moods). Replace with LLM-based mood analysis from AI_Mixer's semantic layer for richer mood vocabulary.
- AI_Mixer's section detection targeted 4-16 segments. For video, we need finer granularity. Make segment count configurable based on use case.
- Combine BeatCanvas's `enhanced_scene_timings` (hero vs standard scenes) with AI_Mixer's section-level semantic data to produce scenes that are both musically and narratively meaningful.

### 5.2 Mashup Pipeline

Port AI_Mixer's complete mashup pipeline as a service module. This includes all 4 agents and supporting infrastructure.

```python
# backend/services/mashup/ — port from AI_Mixer

# ingestion.py  ← from mixer/agents/ingestion.py
# analyst.py    ← from mixer/agents/analyst.py  
# curator.py    ← from mixer/agents/curator.py
# engineer.py   ← from mixer/agents/engineer.py
# memory.py     ← from mixer/memory/ (client, schema, queries)
```

**What to preserve exactly**:
- All 8 mashup types and their implementations (1,614 lines of engineer.py)
- ChromaDB memory system with hybrid matching (60% harmonic + 40% semantic)
- Compatibility scoring (BPM 35%, Key 30%, Energy 20%, Genre 15%)
- Camelot wheel logic
- The full type system (SongMetadata, SectionMetadata, MashupType)

**What to refactor**:
- Replace AI_Mixer's standalone Streamlit UI with FastAPI endpoints
- Replace AI_Mixer's CLI with API calls from the React frontend
- The LangGraph workflow orchestration can be simplified — the FastAPI task system handles orchestration now. Keep the agent functions but remove the LangGraph dependency unless it adds clear value.
- Connect the mashup output to the video pipeline: when a mashup is created, it becomes available as input to Path B (mashup → video).

**What to drop**:
- Blender-based Crossfade Club video system (replaced by real video pipeline)
- The `director/`, `studio/`, `encoder/`, `batch/` folders from AI_Mixer
- Streamlit UI (`mixer_ui.py`)

### 5.3 Audio Processing

Port AI_Mixer's audio processing layer for stem separation and manipulation.

```python
# Processing functions needed (from mixer/audio/processing.py):
# - separate_stems()      → Demucs
# - time_stretch()         → pyrubberband
# - align_to_downbeat()
# - mix_tracks()
# - pitch_shift()          → librosa (max ±6 semitones)
# - normalize_lufs()       → -14 LUFS broadcast standard
# - export_audio()
```

**Reference**: AI_Mixer `mixer/audio/processing.py` (474 lines). Port directly, these are well-tested (170+ unit tests).

---

## 6. Phase 3: Song Analysis & Prompt Generation

This is where Beat_Studio adds significant new capability beyond either source project. The goal: analyze a song deeply and generate scene-by-scene prompts that tell the song's story visually.

### 6.1 Narrative Analysis

```python
# backend/services/prompt/narrative_analyzer.py

class NarrativeAnalyzer:
    """Analyzes song lyrics + audio to extract a visual narrative arc."""
    
    def analyze(self, analysis: SongAnalysis, user_concept: str = None) -> NarrativeArc:
        """
        Uses LLM to:
        1. Identify the story/theme of the song from lyrics
        2. Map emotional progression across sections
        3. Identify key visual moments (chorus climax, bridge reflection, etc.)
        4. Incorporate user's creative direction if provided
        
        Returns a NarrativeArc with per-section visual direction.
        """
```

**What to take from BeatCanvas**: 
- `ConceptGenerator` approach (GPT-4 with music analysis summary + user prompt) from `backend/src/storyboard/conceptor.py`
- `StoryboardGenerator` scene description generation from `backend/src/storyboard/generator.py`
- `NarrativeAnalyzerAI` section recommendation logic from `backend/src/storyboard/narrative_analyzer_ai.py`

**What to improve**:
- BeatCanvas generated concepts and storyboards as separate steps with separate LLM calls. Unify into a single narrative analysis that feeds directly into scene prompting.
- BeatCanvas's `ConceptGenerator` returned high-level style info (color_palette, mood_progression). Beat_Studio should go deeper: per-section visual descriptions that reference specific lyrics and emotional beats.
- Incorporate AI_Mixer's richer section metadata (lyrical function, themes, emotional tone) to make prompts more narratively grounded.

### 6.2 Scene Prompt Generator

```python
# backend/services/prompt/scene_generator.py

class ScenePromptGenerator:
    """Generates video generation prompts for each scene."""
    
    def generate_prompts(self, 
                         narrative: NarrativeArc,
                         scenes: List[SceneTiming],
                         style: AnimationStyle,
                         loras: List[LoRAConfig],
                         user_overrides: Dict[int, str] = None) -> List[ScenePrompt]:
        """
        For each scene:
        1. Base prompt from narrative arc + section lyrics
        2. Style modifiers from animation style config
        3. LoRA trigger tokens prepended
        4. Quality tokens appended
        5. Negative prompt assembled
        6. User override applied if provided for this scene
        
        Returns ScenePrompt with: positive, negative, style, loras, 
        timing, transition_hint, energy_level, is_hero
        """
```

### 6.3 Prompt Composition

```python
# backend/services/prompt/prompt_composer.py

class PromptComposer:
    """Final prompt assembly with quality tokens and safety."""
    
    MANDATORY_QUALITY_TOKENS = "high quality, detailed, professional"
    
    def compose(self, 
                base_prompt: str,
                style: AnimationStyle,
                loras: List[LoRAConfig],
                cinematography: Optional[CinematographyProfile] = None,
                nsfw: bool = False) -> ComposedPrompt:
        """
        Assembly order:
        1. LoRA trigger tokens
        2. Style prefix
        3. Base prompt (from scene generator)
        4. Cinematography tokens (camera, lighting, film stock) if applicable
        5. Quality tokens
        6. NSFW tokens if enabled (model-specific)
        
        Negative prompt:
        1. Style-specific negatives
        2. Quality negatives (blur, low res, etc.)
        3. NSFW negatives if SFW mode
        """
```

**What to take from BeatCanvas**:
- `CinematographyEngine` and `OpticsCatalog` for photorealistic prompts (`backend/src/cinematography/`)
- `PromptComposer` mandatory quality tokens pattern
- `StyleLogic` for style-specific prompt prefixes and detection

**NSFW handling**: BeatCanvas had models capable of NSFW content (lustify_v2, ponyDiffusionV6XL). Beat_Studio treats NSFW as a prompt modifier + model selection flag. When NSFW is requested:
- Route to appropriate checkpoint (e.g., STANDARD_ANATOMY or STANDARD_ACTION with appropriate trigger tags)
- Add NSFW-appropriate prompt tokens
- Remove SFW safety negatives
- This is part of the pipeline, not a separate system. It's activated through prompting and model selection.

---

## 7. Phase 4: LoRA Management System

### 7.1 LoRA Registry

```python
# backend/services/lora/registry.py

class LoRARegistry:
    """Central registry for all available LoRAs."""
    
    def __init__(self, registry_path: str = "config/loras.yaml"):
        self._registry = self._load(registry_path)
    
    def list_all(self, type_filter: str = None) -> List[LoRAEntry]:
        """List all registered LoRAs, optionally filtered by type.
        Types: character, scene, style, identity (Nova Fade)
        """
    
    def recommend_for_project(self, 
                               narrative: NarrativeArc,
                               style: AnimationStyle) -> LoRARecommendation:
        """
        Returns:
        - available: LoRAs already on disk that match the project
        - downloadable: LoRAs available on HuggingFace/Civitai that would help
        - trainable: Suggests new LoRAs that should be created
          (e.g., "A beach sunset LoRA would improve scenes 3-7")
        
        Recommendation logic:
        1. Match scene themes to available scene LoRAs
        2. Match characters to available character LoRAs
        3. Match style to available style LoRAs
        4. For unmatched themes, search HuggingFace/Civitai
        5. For highly specific needs, recommend training
        """
    
    def register(self, entry: LoRAEntry) -> None:
        """Add a new LoRA to the registry."""
    
    def validate(self, name: str) -> LoRAValidation:
        """Check that the LoRA file exists and is loadable."""
```

**What to take from BeatCanvas**:
- `backend/config/loras.yaml` registry format (name, type, trigger, file, weight, status)
- `SDXLLoRAGenerator` LoRA loading and stacking pattern from `backend/src/assets/sdxl_lora_generator.py`

**What to fix from BeatCanvas**:
- Scene LoRA file path bug: registry expected final files but only checkpoint files existed. Beat_Studio validates paths on registration.
- The `fuse_lora()` pattern may cause GPU memory leaks. Use `set_adapters()` with PEFT instead of fusing weights when possible.
- LoRAs not in registry but trained (70s-film-retro, boat-deck, etc.) should be auto-discovered.

### 7.2 LoRA Trainer

```python
# backend/services/lora/trainer.py

class LoRATrainer:
    """Train new LoRAs from image datasets."""
    
    def train(self, config: LoRATrainingConfig) -> TrainingResult:
        """
        Trains an SDXL LoRA from a captioned image dataset.
        
        Config includes:
        - dataset_path: folder of images + captions
        - lora_type: character | scene | style
        - trigger_token: e.g., "novafade_char"
        - training_steps: default 1500
        - learning_rate: default 5e-5
        - rank: default 16
        - optimizer: adamw8bit
        - resolution: 1024
        
        Returns: path to trained LoRA, training metrics
        """
    
    def prepare_dataset(self, 
                        images_dir: str,
                        captions: Dict[str, str] = None,
                        auto_caption: bool = False) -> DatasetResult:
        """
        Prepares training dataset:
        - Validates image quality and resolution
        - Generates/validates caption files
        - Auto-caption via BLIP-2 if requested
        - Applies acceptance gate (for Nova Fade identity LoRAs)
        """
```

**What to take from BeatCanvas**: The LoRA training approach used for Rob and Michele characters — 25 images, 1500 steps, rank 16, adamw8bit. This is documented in the research and datasets exist as reference.

**What to improve**: 
- BeatCanvas's captions were all identical ("a photo of ohwx man, portrait"). Beat_Studio should support richer, per-image captions.
- Add auto-captioning via BLIP-2 for convenience.
- Add acceptance gates for identity LoRAs (per Nova Fade protocol).

### 7.3 LoRA Downloader

```python
# backend/services/lora/downloader.py

class LoRADownloader:
    """Download LoRAs from HuggingFace and Civitai."""
    
    def search(self, query: str, source: str = "both") -> List[LoRASearchResult]:
        """Search HuggingFace and Civitai for relevant LoRAs."""
    
    def download(self, url: str, name: str, type: str) -> LoRAEntry:
        """Download, validate, and register a LoRA."""
    
    def recommend_downloads(self, narrative: NarrativeArc) -> List[LoRASearchResult]:
        """Based on the project narrative, suggest LoRAs to download."""
```

### 7.4 LoRA Recommender

The recommender is a key differentiator. When a user starts a video project, the system should:

1. Analyze the narrative arc and scene descriptions
2. Check what LoRAs are available locally
3. Suggest available LoRAs that match (with confidence scores)
4. Search online for downloadable LoRAs that would help
5. Suggest new LoRAs to train if nothing suitable exists
6. Prefer existing/downloadable over training from scratch (if quality supports it)

---

## 8. Phase 5: Nova Fade Character Pipeline

### 8.1 Character Constitution (Enforced in Code)

The Nova Fade Character Constitution v1.0 is not just a document — it must be enforced programmatically.

```python
# backend/services/nova_fade/character.py

class NovaFadeCharacter:
    """Enforces Nova Fade Character Constitution v1.0."""
    
    # Immutable identity constraints
    EXPRESSIONS = [
        "neutral_confident",
        "mischievous_grin", 
        "focused_intensity",
        "delighted_joy",
        "drop_anticipation"
    ]
    
    ALLOWED_GESTURES = [
        "left_deck_scratch",
        "right_deck_scratch",
        "crossfader_tap",
        "downbeat_head_nod",
        "spotlight_presentation"
    ]
    
    FORBIDDEN = [
        "photorealistic", "realistic human skin", "anime", "manga",
        "age change", "different hairstyle", "different face",
        "acrobatics", "running", "aggressive dance", "physical combat"
    ]
    
    def validate_prompt(self, prompt: str) -> ValidationResult:
        """Check prompt against constitution constraints."""
    
    def get_canonical_header(self) -> Tuple[str, str]:
        """Returns (positive_header, negative_header) per v1.0 spec."""
```

### 8.2 Canonical Image Generation

Nova Fade's reference images need to be generated as part of project setup.

```
Required canonical image set:
├── structural/
│   ├── front_neutral.png
│   ├── three_quarter_neutral.png
│   ├── side_view.png
│   └── back_view.png
├── face/
│   ├── front_closeup.png
│   └── three_quarter_closeup.png
├── expressions/
│   ├── neutral_confident.png
│   ├── mischievous_grin.png
│   ├── focused_intensity.png
│   ├── delighted_joy.png
│   └── drop_anticipation.png
├── dj_poses/
│   ├── hands_left_deck.png
│   ├── hands_right_deck.png
│   ├── crossfader_tap.png
│   └── hero_pose.png
├── silhouettes/
│   ├── front_silhouette.png
│   └── three_quarter_silhouette.png
├── palette/
│   └── color_swatches.png
└── studio_anchor/
    ├── booth_front.png
    └── booth_three_quarter.png
```

**Process**:
1. Use SDXL with the canonical prompt header to generate initial images
2. Curate: select best outputs that match constitution constraints
3. Iterate until reference set is complete and consistent
4. Freeze as `CanonicalSet_v1`
5. Train Identity LoRA (`novafade_id_v1`) from canonical set
6. Train Style LoRA (`crossfadeclub_style_v1`) from extended set
7. Run drift test baseline
8. Lock and version

### 8.3 Drift Detection

Port the drift detection automation plan directly into the application.

```python
# backend/services/lora/drift_test.py
# (Port from Nova Fade Drift Test v1 skeleton)

class DriftTester:
    """CLIP-based identity regression testing for Nova Fade."""
    
    def run_test(self, config: RunConfig, thresholds: Thresholds) -> DriftScorecard:
        """
        Generates 20 test images with fixed seeds and prompts.
        Computes: S_id, S_face, S_sil, V_batch
        Returns pass/fail scorecard + diff strip + run log
        """
    
    def schedule_weekly(self):
        """Set up weekly drift monitoring."""
```

**Reference**: The complete Python skeleton provided in the Nova Fade LoRA Training Protocol document. Port directly, it is production-ready.

### 8.4 DJ Video Generator

When Path B2 is selected (mashup → DJ video), Nova Fade performs the mashup.

```python
# backend/services/nova_fade/dj_video_generator.py

class DJVideoGenerator:
    """Generates Nova Fade DJ performance videos for mashups."""
    
    def generate(self, 
                 mashup_path: str,
                 mashup_analysis: SongAnalysis,
                 theme: str = "sponsor_neon",
                 style: str = "3d_cartoon") -> str:
        """
        1. Analyze mashup audio for beat grid, drops, section changes
        2. Generate DJ action timeline:
           - idle_bob during verses
           - deck_scratch on transitions
           - crossfader_tap on stem switches
           - drop_reaction on energy spikes
        3. Map actions to Nova Fade expressions
        4. Generate scene prompts with canonical header + action descriptions
        5. Generate video clips via video engine
        6. Assemble continuous DJ performance video
        """
```

**What to take from AI_Mixer Crossfade Club**:
- Timeline concept from `director/timeline.py` (beat grid, avatar actions, camera paths)
- Event detection from `director/events.py` (drops, section changes)
- 5 avatar actions (idle_bob, deck_scratch_L, deck_scratch_R, crossfader_hit, drop_reaction)
- 4 theme presets (sponsor_neon, award_elegant, mashup_chaos, chill_lofi)

**What changes**:
- Instead of Blender rendering, each "frame" of the timeline becomes a video generation prompt
- Nova Fade's gestures are described in prompts, not driven by Blender rig
- The video engine generates continuous clips of Nova Fade performing
- Assembly uses the same transition engine as Path A

---

## 9. Phase 6: Video Generation Engine

This is the core differentiator. A model-agnostic engine that selects the best available backend for each task.

### 9.1 Video Backend Interface

```python
# backend/services/video/backends/base.py

from abc import ABC, abstractmethod

class VideoBackend(ABC):
    """Abstract interface for all video generation backends."""
    
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    def vram_required_gb(self) -> float: ...
    
    @abstractmethod
    def supports_style(self, style: str) -> bool: ...
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend can run (models downloaded, GPU sufficient)."""
    
    @abstractmethod
    def estimated_time_per_scene(self, resolution: Tuple[int, int]) -> float:
        """Seconds per scene at given resolution."""
    
    @abstractmethod
    def estimated_cost_per_scene(self) -> float:
        """USD per scene (0.0 for local)."""
    
    @abstractmethod
    def generate_clip(self, 
                      prompt: ComposedPrompt,
                      duration_sec: float,
                      resolution: Tuple[int, int],
                      fps: int = 24,
                      seed: int = -1) -> VideoClip:
        """Generate a single video clip."""
    
    @abstractmethod
    def generate_batch(self, 
                       prompts: List[ComposedPrompt],
                       durations: List[float],
                       resolution: Tuple[int, int],
                       fps: int = 24) -> List[VideoClip]:
        """Generate multiple clips (may be parallelizable)."""
    
    @abstractmethod
    def kill(self) -> None:
        """Release all GPU resources."""
```

### 9.2 Model Router

```python
# backend/services/video/model_router.py

class ModelRouter:
    """Selects the best video backend for a given task."""
    
    def __init__(self):
        self.backends = self._discover_backends()
        self.hardware = HardwareProfile.detect()
    
    def select_backend(self, 
                       style: str,
                       quality: str,
                       local_preferred: bool = True) -> VideoBackend:
        """
        Selection logic:
        
        1. Filter backends that support the requested style
        2. Filter by availability (models downloaded, GPU sufficient)
        3. If local_preferred and local options exist:
           a. Rank by quality for this style
           b. Select highest quality that fits in VRAM
        4. If no local option or cloud preferred:
           a. Select cloud backend (WAN 2.6 RunPod or SkyReels)
        5. Return selected backend + alternatives with cost/time comparison
        """
    
    def get_execution_plan(self, 
                           scenes: List[ScenePrompt],
                           style: str,
                           quality: str) -> ExecutionPlan:
        """
        Returns a complete execution plan:
        - Which backend for each scene (may mix local + cloud)
        - Estimated total time
        - Estimated total cost
        - VRAM management plan (kill/revive sequence)
        - Alternative plans ranked by cost/quality/time
        
        The user reviews and approves the plan before execution.
        """
```

### 9.3 Backend Implementations

#### AnimateDiff-Lightning (Local)
- **Best for**: Fast animated clips, stylized content
- **Resolution**: 576×1024 (portrait)
- **Frames**: 16 per generation
- **VRAM**: ~5.6GB
- **Strengths**: Fast (4-step distilled), good style variety
- **Limitations**: SD 1.5 base (lower quality than SDXL), 75-token CLIP limit, short clips

**Reference**: BeatCanvas `backend/src/cinematography/animatediff_generator.py` and `backend/src/video/animatediff_pipeline.py`. Port the generator, improve the pipeline with better scene continuity.

#### WAN 2.6 Local
- **Best for**: Photorealistic and high-quality animated video
- **Resolution**: up to 720p locally (VRAM dependent)
- **VRAM**: 12GB+ recommended
- **Strengths**: Best quality for local generation
- **Limitations**: Slow, high VRAM

#### WAN 2.6 Cloud (RunPod)
- **Best for**: Photorealistic video, 1080p output
- **Resolution**: up to 1080p
- **Cost**: RunPod GPU time
- **Strengths**: No local VRAM constraint, highest quality
- **Recommended for**: All photorealistic video, high-quality animated when local GPU insufficient

**Reference**: BeatCanvas `backend/src/cinematography/wan26_cloud_generator.py`

#### SkyReels V2 DF (RunPod)
- **Best for**: Seamless scene stitching, eliminating visible cuts
- **Used**: As a post-processing step after WAN 2.6 scene generation
- **Critical for**: The "continuous video" requirement

**Reference**: BeatCanvas `backend/src/cinematography/skyreels_df_generator.py`

#### CogVideoX (Local)
- **Best for**: Text-to-video, good motion quality
- **VRAM**: 14-16GB
- **Status**: Evaluate during development — BeatCanvas had a stub but never integrated

#### SVD - Stable Video Diffusion (Local)
- **Best for**: Image-to-video (animate a still image)
- **VRAM**: ~7.5GB
- **Use case**: Generate a key frame with SDXL, then animate with SVD

**Reference**: BeatCanvas `backend/src/local/video_generator.py` (638 lines)

#### SDXL + ControlNet (Local, Rotoscoping)
- **Best for**: Applying artistic style to existing video/images
- **VRAM**: ~8GB
- **Use case**: One animation approach among many, not the default for all animation
- **Important**: This is rotoscoping specifically — applying an art style to source footage. It requires source footage as input.

**Reference**: BeatCanvas `backend/src/animation/rotoscope_generator.py`

#### Mochi / LTX-Video (Local, Evaluate)
- **Status**: Evaluate during development
- **Include**: Backend interface stubs so they can be added easily

### 9.4 VRAM Manager

Port and extend BeatCanvas's kill-and-revive pattern.

```python
# backend/services/shared/vram_manager.py

class VRAMManager:
    """Kill-and-revive VRAM management. Only ONE model loaded at a time."""
    
    def kill_current(self) -> None:
        """
        1. Unload LoRA weights
        2. Delete pipeline
        3. gc.collect() × 3
        4. torch.cuda.empty_cache() + synchronize()
        5. Verify VRAM < 1.5GB baseline
        """
    
    def load_backend(self, backend: VideoBackend) -> None:
        """Kill current, then load requested backend."""
    
    def get_vram_status(self) -> VRAMStatus:
        """Current VRAM usage, free VRAM, loaded model."""
```

**Reference**: BeatCanvas `backend/src/local/vram_manager.py` (496 lines). The kill-and-revive pattern is solid — port it directly.

---

## 10. Phase 7: Animation Style System

### 10.1 Style Definitions

All animation styles defined in a single config file. Each style specifies prompt modifiers, recommended models, and generation parameters.

```yaml
# backend/config/animation_styles.yaml

styles:
  # === Traditional Animation ===
  cel_animation:
    display_name: "Cel / Hand-Drawn Animation"
    category: "traditional"
    prompt_suffix: "traditional cel animation, clean line work, flat color fills, hand-drawn look, animation keyframe"
    negative: "photorealistic, 3D render, CGI, blurry, low quality"
    recommended_backend: "animatediff"
    recommended_checkpoint: null
    recommended_loras: []
    controlnet_conditioning_scale: 0.7
    guidance_scale: 7.5
    best_for: "Classic animated music videos, cartoon aesthetics"
    
  watercolor:
    display_name: "Watercolor Animation"
    category: "traditional"
    prompt_suffix: "watercolor painting, soft bleeding edges, wet-on-wet technique, delicate brush strokes, paper texture"
    negative: "photorealistic, sharp edges, digital, vector, flat color"
    recommended_backend: "animatediff"
    controlnet_conditioning_scale: 0.6
    guidance_scale: 7.0
    best_for: "Soft, emotional, folk or acoustic music"
    
  ink_wash:
    display_name: "Ink Wash / Sumi-e"
    category: "traditional"
    prompt_suffix: "sumi-e ink wash painting, East Asian brush technique, minimal strokes, zen aesthetic, rice paper texture"
    negative: "colorful, photorealistic, digital, cartoon, western style"
    recommended_backend: "animatediff"
    guidance_scale: 8.0
    best_for: "Ambient, meditative, world music"
    
  pencil_sketch:
    display_name: "Pencil Sketch Animation"
    category: "traditional"
    prompt_suffix: "pencil sketch animation, graphite on paper, hand-drawn hatching, raw textured look, sketch lines visible"
    negative: "color, photorealistic, clean lines, digital, polished"
    recommended_backend: "animatediff"
    guidance_scale: 7.5
    best_for: "Raw, emotional, singer-songwriter"

  # === Modern/Digital ===
  motion_graphics:
    display_name: "Motion Graphics / Flat Design"
    category: "modern"
    prompt_suffix: "flat design motion graphics, bold geometric shapes, smooth vector animation, minimal, clean"
    negative: "photorealistic, texture, noise, hand-drawn, sketch"
    recommended_backend: "animatediff"
    guidance_scale: 7.0
    best_for: "Electronic, pop, educational"
    
  collage_mixed_media:
    display_name: "Collage / Mixed Media"
    category: "modern"
    prompt_suffix: "mixed media collage, layered paper cutouts, found materials, textured surfaces, handmade craft aesthetic"
    negative: "photorealistic, clean, digital, smooth, uniform"
    recommended_backend: "animatediff"
    guidance_scale: 7.0
    best_for: "Indie, experimental, alternative"
    
  pixel_art:
    display_name: "Pixel Art Animation"
    category: "modern"
    prompt_suffix: "pixel art animation, retro game aesthetic, 16-bit style, crisp pixels, limited color palette"
    negative: "photorealistic, smooth, high resolution, anti-aliased, blurry"
    recommended_backend: "animatediff"
    guidance_scale: 8.0
    best_for: "Chiptune, retro, electronic, game soundtracks"
    
  low_poly_3d:
    display_name: "Low-Poly 3D"
    category: "modern"
    prompt_suffix: "low-poly 3D render, geometric faceted shapes, stylized, minimal polygon count, flat shading"
    negative: "photorealistic, organic, smooth, high-poly, subsurface scattering"
    recommended_backend: "animatediff"
    guidance_scale: 7.5
    best_for: "Electronic, ambient, synthpop"
    
  isometric:
    display_name: "Isometric Animation"
    category: "modern"
    prompt_suffix: "isometric view, fixed angle, geometric, diorama style, tilt-shift aesthetic, miniature world"
    negative: "perspective, photorealistic, fisheye, wide angle"
    recommended_backend: "animatediff"
    guidance_scale: 7.5
    best_for: "Lofi, chill, ambient"

  # === Stylized Realism ===
  rotoscope:
    display_name: "Rotoscope Animation"
    category: "stylized_realism"
    prompt_suffix: "rotoscope animation, traced from live footage, stylized outlines, painted over film"
    negative: "photorealistic, cartoon, flat color, low quality"
    recommended_backend: "sdxl_controlnet"
    requires_source_footage: true
    controlnet_conditioning_scale: 0.8
    guidance_scale: 7.5
    best_for: "Narrative music videos, A-ha Take On Me style"
    
  oil_painting:
    display_name: "Oil Painting Animation"
    category: "stylized_realism"
    prompt_suffix: "oil painting animation, thick impasto brushstrokes, rich color depth, Loving Vincent style, canvas texture"
    negative: "photorealistic, flat, digital, clean, vector, cartoon"
    recommended_backend: "animatediff"
    guidance_scale: 7.0
    best_for: "Classical, orchestral, dramatic ballads"
    
  comic_book:
    display_name: "Comic Book / Graphic Novel"
    category: "stylized_realism"
    prompt_suffix: "comic book art style, bold ink outlines, halftone dot pattern, dramatic shading, graphic novel panel"
    negative: "photorealistic, soft, blurry, watercolor, pastel"
    recommended_backend: "animatediff"
    guidance_scale: 8.0
    best_for: "Rock, punk, action-heavy narratives"
    
  pop_art:
    display_name: "Pop Art"
    category: "stylized_realism"
    prompt_suffix: "pop art style, bold primary colors, screen print aesthetic, Andy Warhol inspired, Ben-Day dots"
    negative: "photorealistic, muted colors, realistic shading, dark"
    recommended_backend: "animatediff"
    guidance_scale: 7.5
    best_for: "Pop, dance, upbeat anthems"
    
  ukiyo_e:
    display_name: "Ukiyo-e (Japanese Woodblock)"
    category: "stylized_realism"
    prompt_suffix: "ukiyo-e woodblock print style, flat color areas, strong outlines, Japanese traditional art, wave patterns"
    negative: "photorealistic, 3D, digital, western, modern"
    recommended_backend: "animatediff"
    guidance_scale: 8.0
    best_for: "World music, Japanese-influenced, contemplative"

  # === Abstract/Experimental ===
  synthwave:
    display_name: "Synthwave / Neon"
    category: "abstract"
    prompt_suffix: "synthwave aesthetic, neon glow, retro-futuristic, grid lines, purple and cyan, chrome reflections"
    negative: "natural, organic, daytime, muted colors, realistic"
    recommended_backend: "animatediff"
    guidance_scale: 7.5
    best_for: "Synthwave, retrowave, electronic, vaporwave"
    
  graffiti:
    display_name: "Graffiti / Street Art"
    category: "abstract"
    prompt_suffix: "graffiti street art style, spray paint texture, urban wall, bold tags, dripping paint, concrete surface"
    negative: "clean, digital, photorealistic, smooth, pastel"
    recommended_backend: "animatediff"
    guidance_scale: 7.5
    best_for: "Hip-hop, rap, urban, breakbeat"
    
  art_deco:
    display_name: "Art Deco"
    category: "abstract"
    prompt_suffix: "art deco style, geometric elegance, gold leaf accents, 1920s glamour, symmetrical patterns, luxury"
    negative: "modern, casual, rough, organic, photorealistic"
    recommended_backend: "animatediff"
    guidance_scale: 8.0
    best_for: "Jazz, swing, electro-swing, cabaret"
    
  impressionist:
    display_name: "Impressionist"
    category: "abstract"
    prompt_suffix: "impressionist painting style, Monet inspired, visible brushstrokes, dappled light, soft focus, plein air"
    negative: "photorealistic, sharp, digital, cartoon, flat color"
    recommended_backend: "animatediff"
    guidance_scale: 7.0
    best_for: "Classical, romantic, ambient, soft pop"
    
  psychedelic:
    display_name: "Psychedelic / Liquid"
    category: "abstract"
    prompt_suffix: "psychedelic art, flowing morphing shapes, kaleidoscopic patterns, vivid saturated colors, liquid motion, trippy"
    negative: "photorealistic, muted, realistic, static, sharp edges"
    recommended_backend: "animatediff"
    guidance_scale: 6.5
    best_for: "Psychedelic rock, EDM, trance, acid house"

  # === Photorealistic ===
  photorealistic:
    display_name: "Photorealistic"
    category: "photorealistic"
    prompt_suffix: "photorealistic, cinematic, 4K, professional cinematography"
    negative: "cartoon, anime, drawing, painting, illustration, low quality"
    recommended_backend: "wan26_cloud"
    guidance_scale: 7.5
    best_for: "Any genre requiring live-action aesthetic"
    note: "Cloud execution recommended for quality"
    
  cinematic:
    display_name: "Cinematic"
    category: "photorealistic"
    prompt_suffix: "cinematic film, anamorphic lens, shallow depth of field, film grain, color graded"
    negative: "cartoon, anime, flat, illustration, amateur, webcam"
    recommended_backend: "wan26_cloud"
    guidance_scale: 7.5
    best_for: "Dramatic narratives, any genre"
    note: "Cloud execution recommended for quality"
```

### 10.2 Style Selector Logic

```python
# backend/services/prompt/style_mapper.py

class StyleMapper:
    """Maps styles to generation parameters and recommends styles for songs."""
    
    def recommend_styles(self, analysis: SongAnalysis) -> List[StyleRecommendation]:
        """
        Based on genre, mood, energy, and tempo, recommend top 3-5 styles.
        
        Example mappings:
        - Folk/acoustic + calm mood → watercolor, pencil_sketch, impressionist
        - Hip-hop + energetic → graffiti, comic_book, pop_art
        - Electronic + dark → synthwave, psychedelic, low_poly_3d
        - Classical + emotional → oil_painting, impressionist, ink_wash
        - Trop rock + upbeat → watercolor, cel_animation (+ Nova Fade DJ option)
        """
    
    def get_style_config(self, style_name: str) -> AnimationStyleConfig:
        """Returns full config for a style including prompts, models, parameters."""
```

---

## 11. Phase 8: Video Assembly & Continuity

This phase is critical for the "continuous video, not a slideshow" requirement.

### 11.1 Beat Synchronization

```python
# backend/services/video/beat_sync.py

class BeatSynchronizer:
    """Ensures video scenes align with musical structure."""
    
    def create_scene_plan(self, 
                          analysis: SongAnalysis,
                          quality_tier: str) -> List[ScenePlan]:
        """
        Maps scenes to song structure:
        - Scene boundaries fall on beats or downbeats
        - Hero scenes aligned to choruses/drops
        - Scene duration respects musical phrases
        - Transitions timed to beat grid
        
        quality_tier affects scene count:
        - basic: 12 scenes
        - professional: 24 scenes
        - cinematic: 48 scenes
        """
```

### 11.2 Transition Engine

```python
# backend/services/video/transition_engine.py

class TransitionEngine:
    """Creates smooth, invisible transitions between generated clips."""
    
    TRANSITION_TYPES = {
        "morph": "Latent space interpolation between end of clip A and start of clip B",
        "crossfade": "Opacity blend (simplest, fallback)",
        "motion_blend": "Optical flow-based blending",
        "skyreels_stitch": "SkyReels Diffusion Forcing (highest quality, cloud)",
        "temporal_interp": "RAFT/RIFE frame interpolation across boundary",
    }
    
    def select_transition(self, 
                          scene_a: ScenePlan, 
                          scene_b: ScenePlan,
                          backend: VideoBackend) -> TransitionConfig:
        """
        Select transition type based on:
        - Scene similarity (same setting = subtle morph, different = crossfade)
        - Energy change (big drop = hard cut allowed, gradual = smooth blend)
        - Available backend capabilities
        - Quality tier
        """
    
    def apply_transition(self, 
                         clip_a: VideoClip, 
                         clip_b: VideoClip,
                         config: TransitionConfig) -> VideoClip:
        """Apply the transition, returning the blended boundary frames."""
```

**Critical for quality**: The viewer should never perceive individual clips. Techniques:
1. **End-frame seeding**: Use the last frame of clip A as the init image for clip B (for image-to-video backends like SVD)
2. **Prompt continuity**: Carry visual elements from scene A into scene B's prompt during transitions
3. **SkyReels DF stitching**: For cloud pipeline, use SkyReels to ensure seamless flow
4. **RAFT/RIFE interpolation**: Generate intermediate frames between clip boundaries

**What BeatCanvas had**: Simple crossfade transitions (0.75s) in the MoviePy assembler, plus Ken Burns effects on static images. Both are insufficient. Beat_Studio must go much further.

**What to avoid**: BeatCanvas's RAFT interpolation was disabled due to blur issues. Evaluate whether this is fixable or if SkyReels stitching is the better path.

### 11.3 Video Assembler

```python
# backend/services/video/assembler.py

class VideoAssembler:
    """Assembles generated clips into a continuous final video."""
    
    def assemble(self, 
                 clips: List[VideoClip],
                 transitions: List[TransitionConfig],
                 audio_path: str,
                 output_path: str,
                 resolution: Tuple[int, int] = (1920, 1080),
                 fps: int = 24) -> str:
        """
        Assembly pipeline:
        1. Upscale clips to target resolution if needed
        2. Apply transitions between consecutive clips
        3. Sync to audio (stretch/trim clips to match scene timings)
        4. Apply final audio track
        5. Apply global color grading if configured
        6. Encode with FFmpeg (H.264, CRF 18)
        
        No Ken Burns effects. No static images held on screen.
        Every frame is generated or interpolated.
        """
```

**Reference**: BeatCanvas `backend/src/video/assembler.py` (721 lines) for MoviePy assembly patterns. Use MoviePy 2.x API. But replace all Ken Burns/static image handling with continuous clip assembly.

### 11.4 Encoder

```python
# backend/services/video/encoder.py

class VideoEncoder:
    """FFmpeg-based final encoding."""
    
    PRESETS = {
        "draft": {"crf": 23, "preset": "fast"},
        "standard": {"crf": 18, "preset": "medium"},
        "broadcast": {"crf": 15, "preset": "slow"},
    }
    
    PLATFORMS = {
        "tiktok":  {"resolution": (1080, 1920), "bitrate": "5M",  "audio": "192k"},
        "reels":   {"resolution": (1080, 1920), "bitrate": "5M",  "audio": "192k"},
        "shorts":  {"resolution": (1080, 1920), "bitrate": "5M",  "audio": "192k"},
        "youtube": {"resolution": (1920, 1080), "bitrate": "8M",  "audio": "320k"},
        "standard":{"resolution": (1920, 1080), "bitrate": "8M",  "audio": "320k"},
    }
```

**Reference**: AI_Mixer `encoder/platform.py` for platform specs. BeatCanvas assembler for encoding settings. BeatCanvas had NVENC disabled (`gpu_available = False`). Investigate and enable if possible for faster encoding.

---

## 12. Phase 9: FastAPI Backend

### 12.1 Router Structure

Break BeatCanvas's 2,315-line monolith into clean routers.

```python
# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Beat Studio", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], ...)

# Mount routers
from routers import audio, mashup, video, lora, nova_fade, tasks, system

app.include_router(audio.router,     prefix="/api/audio",     tags=["Audio"])
app.include_router(mashup.router,    prefix="/api/mashup",    tags=["Mashup"])
app.include_router(video.router,     prefix="/api/video",     tags=["Video"])
app.include_router(lora.router,      prefix="/api/lora",      tags=["LoRA"])
app.include_router(nova_fade.router, prefix="/api/nova-fade", tags=["Nova Fade"])
app.include_router(tasks.router,     prefix="/api/tasks",     tags=["Tasks"])
app.include_router(system.router,    prefix="/api/system",    tags=["System"])
```

**Critical fix from BeatCanvas**: BeatCanvas had `api_animation_endpoints.py` with a router that was never mounted. Beat_Studio must not have dead code routers. All routers are mounted in `main.py`.

### 12.2 Key Endpoints

#### Audio Router
```
POST   /api/audio/upload          Upload audio file
POST   /api/audio/analyze         Analyze uploaded audio (returns structure + lyrics)
GET    /api/audio/analysis/{id}   Get cached analysis
```

#### Mashup Router
```
POST   /api/mashup/ingest         Ingest song (local or YouTube)
POST   /api/mashup/ingest/batch   Batch ingest from folder or playlist
GET    /api/mashup/library        List all songs in ChromaDB
GET    /api/mashup/library/search Search library
POST   /api/mashup/match          Find compatible matches for a song
POST   /api/mashup/create         Create a mashup (specify type or auto-recommend)
GET    /api/mashup/types          List all 8 mashup types with descriptions
```

#### Video Router
```
POST   /api/video/plan            Generate execution plan (scenes, backend, cost/time)
POST   /api/video/generate        Start video generation (background task)
POST   /api/video/scene/edit      Edit a single scene prompt and regenerate
GET    /api/video/styles          List all animation styles
GET    /api/video/backends        List available video backends + status
GET    /api/video/download/{id}   Download generated video
```

#### LoRA Router
```
GET    /api/lora/list             List all registered LoRAs
POST   /api/lora/recommend        Get LoRA recommendations for a project
POST   /api/lora/train            Start LoRA training (background task)
POST   /api/lora/download         Download LoRA from HF/Civitai
POST   /api/lora/register         Register an existing LoRA file
DELETE /api/lora/{name}           Remove a LoRA from registry
```

#### Nova Fade Router
```
POST   /api/nova-fade/generate-canonical   Generate canonical reference images
POST   /api/nova-fade/train-identity-lora  Train identity LoRA
POST   /api/nova-fade/train-style-lora     Train style LoRA
POST   /api/nova-fade/drift-test           Run drift detection
POST   /api/nova-fade/dj-video             Generate DJ performance video
GET    /api/nova-fade/status               Character system status
```

#### Task Router
```
GET    /api/tasks/{id}            Get task status (polling)
WS     /ws/{task_id}              WebSocket progress stream
GET    /api/tasks/active          List active tasks
DELETE /api/tasks/{id}            Cancel a task
```

#### System Router
```
GET    /api/system/health         Health check
GET    /api/system/gpu            GPU status and VRAM
GET    /api/system/models         Installed model inventory
POST   /api/system/models/install Install a recommended model
```

---

## 13. Phase 10: React Frontend

### 13.1 Application Structure

Four main tabs following BeatCanvas's tab pattern:

```
┌────────────────────────────────────────────────────┐
│  Beat Studio                                        │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │
│  │Upload│ │Mashup│ │Video │ │LoRA  │ │System│    │
│  │& Song│ │Work- │ │Studio│ │Manage│ │Status│    │
│  │      │ │shop  │ │      │ │      │ │      │    │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘    │
└────────────────────────────────────────────────────┘
```

### 13.2 Tab 1: Upload & Song Analysis

- Audio upload (drag-drop, supports mp3/wav/flac/m4a)
- YouTube URL input
- Real-time analysis display:
  - Waveform visualization
  - Beat markers
  - Section boundaries with labels
  - Lyrics display with word timestamps
  - Mood/energy/key/BPM summary
- "Create Video" button → goes to Video Studio with this song
- "Create Mashup" button → goes to Mashup Workshop

### 13.3 Tab 2: Mashup Workshop

- Song library browser (ChromaDB)
- Compatibility matcher (select song, see matches with scores)
- Mashup type selector (all 8 types with descriptions)
- Auto-recommend mashup type
- Mashup creation with progress
- Mashup preview (audio player)
- "Make Video from Mashup" → goes to Video Studio
- "Make DJ Video" → goes to Nova Fade DJ video flow

### 13.4 Tab 3: Video Studio

The main creative workspace. Stages:

1. **Song/Mashup loaded** — shows analysis summary
2. **Creative direction** — user describes their vision (text input)
3. **Style selection** — grid of animation styles with recommendations highlighted
4. **LoRA recommendations** — system suggests LoRAs, user confirms
5. **Execution plan** — shows local vs cloud options with cost/time
   - User selects preferred path
6. **Scene editor** — per-scene prompt review and editing
   - Scene list with auto-generated prompts
   - Edit any scene's prompt
   - Preview individual scenes (optional, costs time)
7. **Generation** — real-time progress with WebSocket updates
   - Stage-by-stage progress bars
   - Estimated time remaining
8. **Preview & Download** — video player, download options

### 13.5 Tab 4: LoRA Manager

- Browse installed LoRAs (character, scene, style categories)
- Search HuggingFace / Civitai
- Download with progress
- Training interface:
  - Upload images
  - Configure training (trigger token, steps, type)
  - Monitor training progress
  - Run drift test on trained LoRA
- Nova Fade section:
  - Canonical image generation
  - Identity/Style LoRA status
  - Drift test history

### 13.6 Tab 5: System Status

- GPU status (VRAM usage, model loaded)
- Installed models inventory
- Model installation interface
- RunPod connection status
- Disk usage (models, cache, output)

---

## 14. Phase 11: Testing & Quality

### 14.1 Testing Strategy

Port AI_Mixer's test patterns (170+ tests) and extend. Target coverage:

| Module | Target Coverage |
|--------|----------------|
| Audio analysis | 90%+ |
| Mashup pipeline | 90%+ (port AI_Mixer's tests) |
| Prompt generation | 85%+ |
| Video backends | 80%+ (mock GPU for unit tests) |
| LoRA management | 85%+ |
| Nova Fade | 90%+ |
| API routers | 80%+ |
| Assembly/transitions | 85%+ |

### 14.2 Test Infrastructure

```python
# backend/tests/conftest.py

@pytest.fixture
def temp_chroma(tmp_path):
    """Isolated ChromaDB for each test."""
    # Port from AI_Mixer tests/conftest.py
    
@pytest.fixture
def mock_gpu():
    """Mock GPU for unit tests that don't need real inference."""
    
@pytest.fixture
def sample_audio(tmp_path):
    """Generate a short test audio file."""
    
@pytest.fixture
def mock_backend():
    """Mock video backend that returns placeholder clips."""
```

**Reference**: AI_Mixer `tests/conftest.py` for ChromaDB fixtures. Extend with video-specific mocks.

---

## 15. Phase 12: Model & Dependency Setup

### 15.1 Required Models (Installation Recommendations)

#### Core Models (Required)
| Model | Purpose | Size | VRAM | Source |
|-------|---------|------|------|--------|
| AnimateDiff-Lightning | Animated video generation | ~2GB | 5.6GB | ByteDance/AnimateDiff-Lightning |
| epiCRealism (SD 1.5) | AnimateDiff base model | ~4GB | 5.6GB | emilianJR/epiCRealism |
| SDXL Base 1.0 | Image gen, LoRA training | ~6.5GB | 8GB | stabilityai/stable-diffusion-xl-base-1.0 |
| Whisper base | Audio transcription | ~150MB | 1GB | openai/whisper-base |
| Demucs htdemucs | Stem separation | ~300MB | 8GB | facebook/demucs |
| all-MiniLM-L6-v2 | Embedding for ChromaDB | ~80MB | <1GB | sentence-transformers |
| open_clip ViT-B-32 | Drift detection | ~350MB | <1GB | laion2b_s34b_b79k |

#### Recommended Models (Quality Enhancement)
| Model | Purpose | Size | VRAM | Source |
|-------|---------|------|------|--------|
| RealVisXL V5.0 | Photorealistic image gen | 6.5GB | 8GB | Already on disk (BeatCanvas) |
| Juggernaut-XL-v9 | High-quality SDXL | 6.7GB | 8GB | Already on disk |
| ControlNet Canny SDXL | Rotoscope pipeline | ~5GB | 8GB | diffusion-edge/controlnet-canny-sdxl-1.0 |
| CogVideoX | Alternative video gen | ~10GB | 14GB | Evaluate during development |
| SVD XT | Image-to-video | ~8GB | 7.5GB | stabilityai/stable-video-diffusion-img2vid-xt |

#### NSFW-Capable Models (Optional, On Request)
| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| lustify_v2 | NSFW image gen | 4.8GB | Already on disk (BeatCanvas) |
| ponyDiffusionV6XL | NSFW with trigger tags | 6.5GB | Already on disk (BeatCanvas) |

#### Cloud Services (Optional)
| Service | Purpose | Cost |
|---------|---------|------|
| RunPod WAN 2.6 | Photorealistic video | GPU time + RunPod fees |
| RunPod SkyReels | Scene stitching | GPU time + RunPod fees |
| OpenAI API | GPT-4 for concept/storyboard generation, Whisper | Per-token |
| Anthropic API | Claude for semantic analysis | Per-token |

### 15.2 Existing Assets to Migrate

From BeatCanvas `output/loras/`:
- rob-character.safetensors (82MB) — character LoRA
- michele-character.safetensors (82MB) — character LoRA
- Scene LoRA checkpoints (need renaming to final files):
  - tiki-bar-interior, beach-sunset, bonfire-beach (in registry)
  - 70s-film-retro, beach-bar-exterior, boat-deck, ocean-underwater, stage-performance (not in registry)

From BeatCanvas `backend/models/`:
- RealVisXL_V5.0_fp16.safetensors
- lustify_v2.safetensors
- ponyDiffusionV6XL.safetensors
- sdxl_lightning_4step.safetensors
- Style LoRAs: cinematic_grit_v1, pony_anatomy_v2, urban_atmosphere

From BeatCanvas `datasets/`:
- rob-character/ (25 PNG + 25 TXT)
- michele-character/ (25 PNG + 25 TXT)
- 70s-film-retro/ (30 images + cache)

### 15.3 Setup Script

Create a setup script that:
1. Detects GPU and VRAM
2. Lists required vs installed models
3. Offers to download missing models
4. Validates LoRA file paths
5. Tests ChromaDB connection
6. Validates FFmpeg installation
7. Tests API key availability (OpenAI, Anthropic, RunPod)
8. Reports system readiness

---

## 16. Reference Documents

These documents exist in the source projects and should be consulted during implementation. They inform but do not constrain the final product.

| Document | Location | Purpose |
|----------|----------|---------|
| AI_Mixer research.md | `AI_Mixer/research.md` | Complete AI_Mixer codebase analysis |
| BeatCanvas research.md | `beatcanvas/research.md` | Complete BeatCanvas codebase analysis |
| BeatCanvas cross-analysis | `AI_Mixer/thoughts/shared/research/2026-02-16-beatcanvas-comprehensive-analysis.md` | AI_Mixer's analysis of BeatCanvas capabilities |
| Crossfade Club PRD | `AI_Mixer/docs/crossfade-club/CROSSFADE_CLUB_PRD.md` | Product requirements for visual DJ system |
| AI_Mixer CONTINUITY.md | `AI_Mixer/CONTINUITY.md` | Architecture decisions and rationale |
| AI_Mixer session handoffs | `AI_Mixer/thoughts/shared/handoffs/general/` | 15 session handoff YAML files |
| BeatCanvas testing status | `beatcanvas/TESTING_STATUS.md`, `beatcanvas/FINAL_TEST_STATUS.md` | Test coverage information |
| Nova Fade Character Constitution | Provided in this plan (Phase 5) | Character identity constraints |
| Nova Fade LoRA Training Protocol | Provided in this plan (Phase 5) | Training and drift detection |

---

## 17. Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| UI Framework | React + TypeScript | Follows BeatCanvas; richer interactivity than Streamlit |
| Backend Framework | FastAPI | Follows BeatCanvas; async, WebSocket support |
| Router structure | Separate routers per domain | Fix BeatCanvas's monolith problem |
| Task persistence | SQLite | Fix BeatCanvas's in-memory task loss |
| Audio analysis | Unified service | Merge AI_Mixer depth + BeatCanvas scene timing |
| Mashup engine | Port from AI_Mixer | Production-ready, 170+ tests |
| Vector DB | ChromaDB 0.4.22 | Port from AI_Mixer; proven, pinned version |
| Video architecture | Model-agnostic backend interface | Future-proof, swap models easily |
| VRAM management | Kill-and-revive singleton | Port from BeatCanvas; proven pattern |
| Animation styles | YAML config | Easy to add new styles without code changes |
| Transitions | Multi-strategy engine | Critical for "no slideshow" requirement |
| LoRA management | Built-in training + download + registry | Full lifecycle management |
| Nova Fade | Enforced character constitution | IP asset protection |
| Drift detection | CLIP embedding + silhouette IoU | Automated brand regression |
| No Ken Burns | Hard rule | Explicitly excluded |
| No Blender | Hard rule | Replaced by AI video generation |
| NSFW | Pipeline flag, not separate system | Activated through prompting + model selection |
| LLM provider | Anthropic primary, OpenAI fallback | Port from AI_Mixer pattern |
| Embedding model | all-MiniLM-L6-v2 | Port from AI_Mixer; proven for semantic matching |

---

## 18. Known Constraints & Hard Rules

1. **No Ken Burns effects** — explicitly forbidden
2. **No Blender rendering** — all video via AI generation models
3. **No slideshow feel** — continuous video with smooth transitions
4. **Quality first** — user decides cost/time tradeoffs, but quality is the default
5. **ChromaDB 0.4.22** — do not upgrade (breaking changes); port from AI_Mixer
6. **Nova Fade constitution** — programmatically enforced, deviations require version increment
7. **One model in VRAM at a time** — kill-and-revive pattern
8. **User decides execution path** — system recommends, user approves
9. **Smart reuse, not copy** — leverage source projects intelligently

---

## 19. Todo List

### Phase 1: Foundation
- [x] Create project directory structure
- [x] Initialize pyproject.toml and requirements.txt
- [ ] Initialize React frontend with Vite + TypeScript
- [x] Implement unified ConfigManager
- [x] Implement structured logging
- [x] Implement HardwareDetector
- [x] Implement TaskManager with SQLite persistence
- [x] Implement VRAM Manager (port from BeatCanvas)
- [x] Set up git repo with .gitignore

### Phase 2: Audio Engine
- [x] Implement unified AudioAnalyzer
- [x] Port AI_Mixer section detection
- [x] Port AI_Mixer Whisper transcription with word timings
- [x] Implement scene timing generation (from BeatCanvas)
- [x] Port mashup ingestion agent
- [x] Port mashup analyst agent
- [x] Port mashup curator agent
- [x] Port mashup engineer (all 8 types)
- [x] Port ChromaDB memory system
- [x] Port audio processing (stems, stretch, pitch, mix)
- [x] Port AI_Mixer mashup tests (153 total, all passing)

### Phase 3: Prompt Generation
- [x] Implement NarrativeAnalyzer
- [x] Implement ScenePromptGenerator
- [x] Implement PromptComposer
- [x] Implement StyleMapper with recommendation logic
- [x] Implement NSFW prompt modifier (via PromptComposer.compose(nsfw=True/False))
- [x] Write prompt generation tests (26 tests, all passing)

### Phase 4: LoRA Management
- [x] Implement LoRA Registry (YAML-based) — registry.py + loras.yaml
- [x] Implement LoRA Trainer (SDXL training pipeline) — stub, real training needs GPU + ai-toolkit
- [x] Implement LoRA Downloader (HuggingFace + Civitai) — downloader.py (mockable API calls)
- [x] Implement LoRA Recommender — recommender.py (orchestrates registry + downloader)
- [ ] Migrate existing LoRAs from BeatCanvas — deferred (no BeatCanvas LoRA files yet available)
- [ ] Fix scene LoRA checkpoint renaming — deferred (tied to BeatCanvas migration)
- [x] Write LoRA management tests (25 tests, all passing)

### Phase 5: Nova Fade Character
- [x] Implement NovaFadeCharacter (constitution enforcement) — services/nova_fade/character.py
- [x] Implement canonical prompt headers — services/nova_fade/canonical_prompts.py
- [ ] Generate canonical reference image set (requires GPU + SDXL — deferred to setup)
- [ ] Train novafade_id_v1 Identity LoRA (requires canonical images — deferred)
- [ ] Train crossfadeclub_style_v1 Style LoRA (deferred)
- [x] Implement DriftTester — services/nova_fade/drift_tester.py (production: GPU + CLIP)
- [ ] Run baseline drift test (requires trained LoRA — deferred)
- [x] Implement DJVideoGenerator — services/nova_fade/dj_video_generator.py
- [x] Write Nova Fade tests (51 tests, all passing)

### Phase 6: Video Generation Engine
- [x] Implement VideoBackend abstract interface — backends/base.py
- [x] Implement ModelRouter — model_router.py (style/quality/local routing, NoBackendAvailableError)
- [x] Implement AnimateDiff backend — backends/animatediff.py
- [x] Implement WAN 2.6 local backend — backends/wan26_local.py
- [x] Implement WAN 2.6 cloud backend — backends/wan26_cloud.py (RunPod)
- [x] Implement SkyReels cloud backend — backends/skyreels.py (RunPod)
- [x] Implement CogVideoX backend (stub) — backends/cogvideox.py
- [x] Implement SVD backend — backends/svd.py
- [x] Implement SDXL+ControlNet backend — backends/sdxl_controlnet.py (rotoscope)
- [x] Stub Mochi and LTX-Video backends — backends/mochi.py, backends/ltx_video.py
- [x] Implement CostEstimator — cost_estimator.py
- [x] Write video backend tests (46 tests, all passing)

### Phase 7: Animation Styles
- [x] Create animation_styles.yaml with all style definitions (24 styles, 5 categories)
- [x] Implement style recommendation logic — genre + mood + energy → ranked suggestions
- [x] Test prompt generation for each style (25 tests, all passing)
- [x] Validate style-to-model mapping — all styles validated in test_animation_styles.py

### Phase 8: Video Assembly
- [x] Implement BeatSynchronizer — services/video/beat_sync.py (basic/professional/cinematic tiers)
- [x] Implement TransitionEngine (morph, crossfade, motion_blend, skyreels_stitch) — services/video/transition_engine.py
- [x] Implement VideoAssembler (continuous assembly, no Ken Burns) — services/video/assembler.py
- [x] Implement VideoEncoder (FFmpeg, platform presets) — services/video/encoder.py
- [x] Write assembly pipeline tests (38 tests, all passing)
- [ ] Validate "no slideshow" requirement with sample outputs (requires real video — deferred to integration)

### Phase 9: FastAPI Backend
- [x] Create main.py with router mounting — backend/main.py (7 routers, all mounted)
- [x] Implement audio router — backend/routers/audio.py
- [x] Implement mashup router — backend/routers/mashup.py
- [x] Implement video router — backend/routers/video.py
- [x] Implement LoRA router — backend/routers/lora.py
- [x] Implement Nova Fade router — backend/routers/nova_fade.py
- [x] Implement task router + WebSocket — backend/routers/tasks.py
- [x] Implement system router — backend/routers/system.py
- [x] Write API tests (45 tests, all passing) — test_fastapi_routers.py

### Phase 10: React Frontend
- [x] Scaffold React app with tabs (Vite + TypeScript + Vitest)
- [x] Implement AudioUpload component
- [x] Implement SongAnalysis display
- [x] Implement MashupWorkshop
- [x] Implement MashupLibrary
- [x] Implement VideoStudio (main creative workspace)
- [x] Implement SceneEditor
- [x] Implement StyleSelector
- [x] Implement LoRAManager
- [x] Implement NovaFadeStudio
- [x] Implement ExecutionPlanner (local vs cloud UI)
- [x] Implement ProgressTracker
- [x] Implement VideoPreview
- [x] Implement CostEstimator display
- [x] Implement HardwareStatus display
- [x] Implement WebSocket hook (useWebSocket + useTaskPolling fallback)
- [x] Write frontend tests (36 tests, all passing)

### Phase 11: Testing & Quality
- [x] Write API integration tests (53 tests, FastAPI TestClient)
- [x] Write frontend component tests (36 tests, Vitest + Testing Library)
- [x] Add mock_gpu, sample_audio_file, mock_video_backend fixtures to conftest.py
- [x] Add session-scoped api_client fixture for integration tests
- [x] Fix router response type annotations (Dict[str, str] → Dict[str, Any] where needed)
- [ ] Performance benchmarking (deferred — requires real models)

### Phase 12: Setup & Models
- [x] Create setup script (scripts/setup_check.py — GPU, models, API keys, dirs)
- [x] Create model inventory YAML (backend/config/checkpoints.yaml — 16 models)
- [x] Create README.md with full setup and architecture documentation
- [ ] Migrate BeatCanvas models and LoRAs (deferred — needs file access)
- [ ] Test RunPod cloud connection (deferred — needs API key)

---

*Plan generated: 2026-02-18*
*Status: DRAFT — ready for annotation cycle*
*Next step: Review this document, add inline notes, send back for refinement*
