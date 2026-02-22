# Beat_Studio Architecture Document

**Version:** 1.0
**Last Updated:** 2026-02-22
**Status:** Production

---

## 1. System Overview

Beat_Studio is a unified AI-powered music video production platform that combines intelligent audio mashup creation with professional neural video generation. It consolidates capabilities from two predecessor projects:

- **AI_Mixer**: 8 mashup types, ChromaDB semantic memory, LLM-driven analysis
- **BeatCanvas**: Cinematography engine, multi-provider video generation, VRAM management

### High-Level Architecture

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
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                         7 Routers                              │  │
│  │  audio │ mashup │ video │ lora │ nova_fade │ tasks │ system   │  │
│  └────────────────────────────┬──────────────────────────────────┘  │
│                               │                                      │
│  ┌────────────────────────────▼──────────────────────────────────┐  │
│  │                       Service Layer                            │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │  │
│  │  │  audio  │ │ mashup  │ │ prompt  │ │  video  │ │  lora   │ │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │  │
│  │  ┌─────────┐ ┌─────────────────────────────────────────────┐ │  │
│  │  │nova_fade│ │                  shared                      │ │  │
│  │  └─────────┘ └─────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Persistence Layer                           │  │
│  │  SQLite (tasks.db) │ ChromaDB (library) │ File System (cache) │  │
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Two-Layer Backend Pattern

### Routers (HTTP Adapters)

Routers in `backend/routers/` are thin HTTP adapters. They:
1. Validate input via Pydantic models
2. Create a `TaskManager` task entry
3. Dispatch slow work via `BackgroundTasks`
4. Return immediately with `task_id`

Routers contain **no business logic**.

```python
# Example: audio router pattern
@router.post("/analyze")
async def analyze_audio(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    task_id = _get_task_manager().create_task(task_type="audio_analysis", params={...})
    background_tasks.add_task(_run_analysis_worker, request.audio_id, task_id)
    return {"task_id": task_id, "status": "queued"}
```

### Services (Business Logic)

Services in `backend/services/` contain all business logic, organized by domain:

| Domain | Path | Purpose |
|--------|------|---------|
| `audio/` | analyzer, types, analysis, processing | Audio analysis, stem separation |
| `mashup/` | ingestion, memory, curator, engineer, analyst | 8 mashup types, ChromaDB |
| `prompt/` | narrative_analyzer, scene_generator, style_mapper | LLM-driven prompts |
| `video/` | backends/, beat_sync, assembler, encoder | 8 video backends |
| `lora/` | registry, trainer, downloader, recommender | LoRA management |
| `nova_fade/` | character, canonical_prompts, dj_video_generator | DJ character system |
| `shared/` | task_manager, vram_manager, config, hardware_detector | Cross-cutting concerns |

---

## 3. Component Architecture

### 3.1 Audio Engine

```
┌─────────────────────────────────────────────────────────────────┐
│                        Audio Engine                              │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ AudioAnalyzer│───▶│ SongAnalysis │───▶│  ChromaDB    │      │
│  │              │    │  (dataclass) │    │  (upsert)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Analysis Depth Levels                                    │   │
│  │  • basic: BPM, key, camelot, energy, first_downbeat      │   │
│  │  • standard: + section boundaries, beat times            │   │
│  │  • full: + Whisper transcription, LLM semantic analysis  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Demucs     │    │ Pyrubberband │    │   Whisper    │      │
│  │ (stem sep)   │    │ (time-stretch│    │ (transcript) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

**Key Classes:**
- `AudioAnalyzer`: Unified analyzer with configurable depth
- `SongAnalysis`: Dataclass with all extracted features
- `MashupLibrary`: ChromaDB wrapper for semantic search

### 3.2 Mashup Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                       Mashup Pipeline                            │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Ingestion│───▶│ Analyst  │───▶│ Curator  │───▶│ Engineer │  │
│  │          │    │          │    │          │    │          │  │
│  │ • YouTube│    │ • Signal │    │ • Match  │    │ • Stems  │  │
│  │ • Local  │    │ • Whisper│    │ • Score  │    │ • Mix    │  │
│  │ • Convert│    │ • LLM    │    │ • Rank   │    │ • Export │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  8 Mashup Types                                           │   │
│  │  1. Classic        5. Theme Fusion                        │   │
│  │  2. Stem Swap      6. Semantic-Aligned                    │   │
│  │  3. Energy Match   7. Role-Aware                          │   │
│  │  4. Adaptive Harmony  8. Conversational                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  ChromaDB Query Modes                                     │   │
│  │  • query_harmonic(): BPM + Camelot wheel                  │   │
│  │  • query_semantic(): Vector similarity (mood/vibe)        │   │
│  │  • query_hybrid(): 60% harmonic + 40% semantic (RRF)      │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Video Engine

```
┌─────────────────────────────────────────────────────────────────┐
│                        Video Engine                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     Model Router                          │   │
│  │  Selects best backend based on:                           │   │
│  │  • Style requirements    • Available VRAM                 │   │
│  │  • Quality preference    • Cost budget                    │   │
│  └───────────────────────────┬──────────────────────────────┘   │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │   LOCAL     │      │   LOCAL     │      │   CLOUD     │     │
│  │ AnimateDiff │      │  WAN 2.6    │      │  RunPod     │     │
│  │   (5.6GB)   │      │  (12GB)     │      │ WAN/SkyReels│     │
│  └─────────────┘      └─────────────┘      └─────────────┘     │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │    SVD      │      │ SDXL+CtrlNet│      │  CogVideoX  │     │
│  │   (7.5GB)   │      │   (8GB)     │      │   (14GB)    │     │
│  └─────────────┘      └─────────────┘      └─────────────┘     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Assembly Pipeline                                        │   │
│  │  BeatSync → SceneGen → Transitions → Encode → Export     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Backend Interface:**
```python
class VideoBackend(ABC):
    @abstractmethod
    async def generate_scene(self, prompt: str, duration: float, ...) -> Path:
        pass

    @abstractmethod
    def get_vram_requirement(self) -> float:
        pass

    @abstractmethod
    def supports_style(self, style: str) -> bool:
        pass
```

### 3.4 VRAM Management

```
┌─────────────────────────────────────────────────────────────────┐
│                      VRAM Manager                                │
│                                                                  │
│  CONSTRAINT: Only ONE model loaded in VRAM at a time            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Kill-and-Revive Pattern                                  │   │
│  │                                                           │   │
│  │  1. Check current VRAM usage                              │   │
│  │  2. If model loaded: unload + gc.collect() + empty_cache  │   │
│  │  3. Wait for VRAM to reach baseline (1.5GB threshold)     │   │
│  │  4. Load new model                                        │   │
│  │  5. Execute generation                                    │   │
│  │  6. Keep loaded for potential reuse (lazy unload)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Budget: 12GB total │ Baseline: 1.5GB │ Available: 10.5GB       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 Task Management

```
┌─────────────────────────────────────────────────────────────────┐
│                      Task Manager                                │
│                                                                  │
│  SQLite-backed (tasks.db) — persists across server restarts     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task Lifecycle                                           │   │
│  │                                                           │   │
│  │  create_task() → update_progress() → complete_task()     │   │
│  │                          │                                │   │
│  │                          └──────→ fail_task()             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Client Access:                                                  │
│  • GET /api/tasks/{task_id} — poll status                       │
│  • WS /api/tasks/ws/{task_id} — real-time progress              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow

### 4.1 Audio Analysis Flow

```
Upload (MP3/WAV/FLAC)
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Validate   │────▶│  Analyze    │────▶│   Cache     │
│  Format     │     │  (librosa)  │     │  JSON       │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Whisper    │ (if depth=full)
                    │ Transcribe  │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  LLM        │ (if depth=full)
                    │  Semantics  │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ SongAnalysis│ → data/analysis/{audio_id}.json
                    └─────────────┘
```

### 4.2 Video Generation Flow

```
SongAnalysis + Style + LoRAs
    │
    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  BeatSync   │────▶│   Scene     │────▶│   Prompt    │
│  Timing     │     │  Generator  │     │  Composer   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────┐
│                     Model Router                         │
│  Select backend → load model → generate scenes          │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Transition │────▶│  Assemble   │────▶│   Encode    │
│  Engine     │     │  Scenes     │     │  (FFmpeg)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        output/{video_id}.mp4
```

---

## 5. Persistence Layer

### 5.1 SQLite (tasks.db)

```sql
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    status TEXT DEFAULT 'queued',  -- queued, running, completed, failed
    params TEXT,                    -- JSON
    stage TEXT,
    percent REAL DEFAULT 0.0,
    message TEXT,
    result TEXT,                    -- JSON
    error TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### 5.2 ChromaDB (tiki_library)

```python
# Collection schema
{
    "id": str,           # sanitized song ID
    "embedding": [...],  # 384-dim (all-MiniLM-L6-v2)
    "metadata": {
        "source": str,       # "local" or "youtube"
        "path": str,         # absolute path to WAV
        "artist": str,
        "title": str,
        "sample_rate": int,  # always 44100
        "bpm": float,
        "key": str,          # e.g., "C major"
        "camelot": str,      # e.g., "8B"
        "energy_level": float,  # 0-10 scale (NOT 0-1)
        "genre": str,
        "mood": str,
    }
}
```

### 5.3 File System Cache

```
backend/data/
├── uploads/              # Original uploaded files
├── analysis/             # {audio_id}.json — cached SongAnalysis
├── generated_images/     # Interim frames (if needed)
├── generated_videos/     # Raw scene clips
├── library_cache/        # ChromaDB persistence + audio cache
│   └── chroma/
└── mashups/              # Exported mashup audio files

output/
├── videos/               # Final exported videos
├── loras/                # Trained/downloaded LoRA weights
└── nova_fade/
    └── canonical/        # Reference images for Nova Fade
```

---

## 6. Configuration Architecture

### 6.1 YAML Configuration Files

| File | Purpose |
|------|---------|
| `settings.yaml` | Main app config (audio, models, LLM, mashup, video, VRAM) |
| `animation_styles.yaml` | 24 animation styles with prompt modifiers |
| `checkpoints.yaml` | Model inventory with paths and VRAM requirements |
| `loras.yaml` | LoRA registry (name, type, trigger, weight, source) |
| `nova_fade_constitution.yaml` | Character constraints (expressions, gestures, forbidden) |

### 6.2 Configuration Hierarchy

```
settings.yaml
├── audio:
│   ├── sample_rate: 44100
│   ├── bit_depth: 16
│   └── format: wav
├── models:
│   ├── whisper_size: base
│   ├── demucs_model: htdemucs
│   └── embedding_model: all-MiniLM-L6-v2
├── llm:
│   ├── primary_provider: anthropic
│   ├── fallback_provider: openai
│   └── retries: 3
├── mashup:
│   ├── collection_name: tiki_library  # DO NOT CHANGE
│   ├── bpm_tolerance: 0.05
│   └── weights: {bpm: 0.35, key: 0.30, energy: 0.20, genre: 0.15}
├── video:
│   ├── fps: 24
│   ├── resolution: [1920, 1080]
│   └── scene_counts: {basic: 12, standard: 24, cinematic: 48}
└── vram:
    ├── baseline_threshold: 1.5
    └── budget: 12.0
```

---

## 7. API Structure

### 7.1 Router Endpoints

| Router | Key Endpoints |
|--------|---------------|
| `/api/audio` | POST /upload, POST /analyze, GET /analysis/{id} |
| `/api/mashup` | POST /ingest, GET /library, POST /match, POST /create |
| `/api/video` | POST /plan, POST /generate, GET /video/{id}, GET /styles |
| `/api/lora` | GET /list, POST /recommend, POST /train, POST /download |
| `/api/nova-fade` | GET /status, POST /train-identity, POST /dj-video |
| `/api/tasks` | GET /status/{id}, GET /all, WS /ws/{id} |
| `/api/system` | GET /health, GET /gpu, GET /models |

### 7.2 WebSocket Protocol

```json
// Client connects to: ws://host/api/tasks/ws/{task_id}
// Server sends progress updates:
{
    "task_id": "abc-123",
    "status": "running",
    "stage": "generating_scene_3",
    "percent": 45.5,
    "message": "Generating scene 3 of 24..."
}
```

---

## 8. Module Dependencies

```
shared/
    ↑
    ├── audio/ ←── mashup/ (uses AudioAnalyzer)
    │     ↑
    │     └── prompt/ (uses SongAnalysis)
    │           ↑
    │           └── video/ (uses scene prompts)
    │                 ↑
    ├── lora/ ←───────┘ (video backends use LoRAs)
    │     ↑
    └── nova_fade/ (uses lora/, video/)
```

---

## 9. Security & Constraints

### 9.1 Hard Rules

- **No Ken Burns effects** — all video via AI generation
- **No Blender** — 100% neural network generation
- **No slideshows** — must be continuous 24fps video
- **ChromaDB 0.4.22** — pinned, do not upgrade (breaking changes)
- **One VRAM model** — kill-and-revive, never two simultaneously
- **Nova Fade constitution** — programmatically enforced, version tracked

### 9.2 Input Validation

- Audio formats: MP3, WAV, FLAC, M4A, OGG, AAC
- Max file size: 100MB (configurable)
- Prompt length: 2000 chars max
- All paths sanitized before filesystem access

---

## 10. Testing Architecture

### 10.1 Test Organization

```
backend/tests/
├── unit/                 # 409 tests — no external deps
│   ├── test_audio_analyzer.py
│   ├── test_mashup_engineer.py
│   ├── test_video_backends.py
│   └── ...
├── integration/          # 53 tests — real services
│   └── test_api_integration.py
└── conftest.py           # Shared fixtures

frontend/src/
└── components/
    ├── *.test.tsx        # 36 tests — Vitest
    └── ...
```

### 10.2 Test Commands

```bash
# All backend tests
python -m pytest backend/tests/ -q

# Unit only (fast)
python -m pytest backend/tests/unit/ -q

# Integration only
python -m pytest backend/tests/integration/ -q

# Frontend tests
cd frontend && npm test -- --run
```

---

## 11. Deployment Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                      Development Machine                         │
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Vite      │     │   Uvicorn   │     │   SQLite    │       │
│  │   :5173     │────▶│   :8000     │────▶│  tasks.db   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│                             │                                    │
│                             ▼                                    │
│                      ┌─────────────┐                            │
│                      │  ChromaDB   │                            │
│                      │  (embedded) │                            │
│                      └─────────────┘                            │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    GPU (NVIDIA)                           │   │
│  │  AnimateDiff │ WAN 2.6 │ SVD │ SDXL │ Whisper │ Demucs   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  External APIs:                                                  │
│  • Anthropic Claude (semantic analysis)                         │
│  • OpenAI GPT-4 (fallback)                                      │
│  • RunPod (cloud video generation)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Evolution from Source Projects

| Aspect | AI_Mixer | BeatCanvas | Beat_Studio |
|--------|----------|------------|-------------|
| Architecture | Linear agent pipeline | 2,315-line main.py | 7 modular routers |
| Task Persistence | In-memory | In-memory | SQLite |
| Video | Blender 3D | MoviePy + cloud | 8 neural backends |
| Audio Analysis | Separate analyzer | Separate analyzer | Unified AudioAnalyzer |
| Configuration | YAML | Hardcoded | YAML with TypedDict |
| Tests | 170+ | ~35% coverage | 498 tests |

---

## Appendix: Key File Locations

```
backend/
├── main.py                              # FastAPI app entry
├── routers/
│   └── {audio,mashup,video,lora,nova_fade,tasks,system}.py
├── services/
│   ├── audio/analyzer.py                # Unified audio analysis
│   ├── mashup/engineer.py               # 8 mashup implementations
│   ├── video/model_router.py            # Backend selection
│   ├── video/backends/base.py           # Abstract interface
│   ├── shared/task_manager.py           # SQLite task queue
│   └── shared/vram_manager.py           # GPU memory management
├── config/
│   ├── settings.yaml                    # Main config
│   ├── animation_styles.yaml            # 24 styles
│   └── nova_fade_constitution.yaml      # Character rules
└── tests/
    ├── unit/                            # 409 tests
    └── integration/                     # 53 tests
```
