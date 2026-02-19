# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Backend — run from repo root with venv active
uvicorn backend.main:app --reload --port 8000

# All backend tests
python -m pytest backend/tests/ -q

# Unit tests only (fast, no GPU)
python -m pytest backend/tests/unit/ -q

# Integration tests only
python -m pytest backend/tests/integration/ -q

# Single test
python -m pytest backend/tests/unit/test_audio_analyzer.py::TestAudioAnalyzer::test_basic_analysis -v

# Frontend (from frontend/)
npm install
npm run dev        # dev server at localhost:5173
npm test -- --run  # Vitest, single pass

# Environment validation
python scripts/setup_check.py
python scripts/setup_check.py --models  # model paths only
```

## Architecture

### Two-Layer Backend

**Routers** (`backend/routers/`) are thin HTTP adapters — they validate input, create a `TaskManager` task, dispatch slow work via `BackgroundTasks`, and return immediately. They do not contain business logic.

**Services** (`backend/services/`) contain all business logic and are used directly by routers. Services are grouped by domain:

| Domain | Path | Status |
|---|---|---|
| audio/ | `analyzer.py`, `types.py`, `analysis.py` | Wired |
| mashup/ | `ingestion.py`, `memory.py`, `curator.py`, `engineer.py` | Wired |
| video/ | `backends/`, `beat_sync.py`, `assembler.py`, `encoder.py` | Stub |
| lora/ | `registry.py`, `trainer.py`, `downloader.py`, `recommender.py` | Stub |
| nova_fade/ | `character.py`, `canonical_prompts.py`, `dj_video_generator.py` | Stub |
| shared/ | `task_manager.py`, `vram_manager.py`, `hardware_detector.py`, `config.py` | Complete |

### BackgroundTasks Pattern

All CPU-heavy operations (audio analysis, ingest, mashup creation) follow this pattern:

```python
# 1. Router creates a task entry
task_id = _get_task_manager().create_task(task_type="...", params={...})

# 2. Dispatch sync function to thread pool
background_tasks.add_task(_run_worker, ..., task_id=task_id)

# 3. Return immediately
return {"task_id": task_id, "status": "queued"}

# 4. Worker calls update_progress() / complete_task() / fail_task()
```

The client polls `GET /api/tasks/{task_id}` or connects to `ws://...//api/tasks/ws/{task_id}` for progress.

### Module-Level Singletons

Routers initialize expensive objects once per process using module-level globals:

```python
_task_manager: Optional[TaskManager] = None
_analyzer: Optional[AudioAnalyzer] = None

def _get_analyzer() -> AudioAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = AudioAnalyzer(sample_rate=44100, whisper_model="base")
    return _analyzer
```

`MashupLibrary` (ChromaDB) and `TaskManager` (SQLite) are both singletons. **Both the audio and mashup routers share the same `tasks.db`** — they instantiate separate `TaskManager` objects but point at the same file, which is safe because `TaskManager` uses `check_same_thread=False`.

### AudioAnalyzer Depth Levels

`AudioAnalyzer.analyze(path, artist, title, depth)` returns a `SongAnalysis` dataclass:

- `"basic"` — BPM, key, camelot, energy, first downbeat. Fast, no heavy models.
- `"standard"` — adds section boundaries and beat times via librosa.
- `"full"` — adds Whisper transcription, LLM semantic analysis (mood, genre, themes, valence).

`SongAnalysis` is serialized to JSON with `dataclasses.asdict()` and cached at `backend/data/analysis/{audio_id}.json`.

### MashupLibrary / ChromaDB

`MashupLibrary` in `backend/services/mashup/memory.py` wraps a ChromaDB persistent collection (`tiki_library`). **Pinned at chromadb==0.4.22** — do not upgrade.

Metadata stored per song must include required fields: `source`, `path`, `artist`, `title`, `sample_rate`. The `energy_level` field must be on a **0–10 scale** (multiply the `SongAnalysis.energy_level` 0–1 float by 10 before storing).

Three query modes: `query_harmonic()` (BPM + Camelot wheel), `query_semantic()` (ChromaDB vector search), `query_hybrid()` (60% harmonic + 40% semantic via RRF).

### Ingest → Library Pipeline

When `POST /api/mashup/ingest` is called, the background worker does three things in sequence:
1. `ingest_song(source)` — download/convert to WAV, return `{id, path, ...}`
2. `AudioAnalyzer.analyze(path, depth="standard")` — extract BPM/key/sections
3. `MashupLibrary.upsert_song(...)` — index in ChromaDB for future matching

Songs must be ingested before they can be matched or used in mashup creation.

### Mashup Engineer

`backend/services/mashup/engineer.py` implements all 8 mashup types. Each `create_*_mashup()` function:
- Takes `song_a_id`, `song_b_id` (or `stem_config` for stem_swap), `output_path`, and an optional `library` instance
- Loads audio via `_load_song_audio()` which reads the `path` field from library metadata
- Calls Demucs for stem separation and pyrubberband for time-stretching
- Returns the output file path as a string

`create_stem_swap_mashup()` takes a `stem_config: Dict[str, str]` mapping stem names to song IDs instead of two positional song IDs.

### TaskManager

`backend/services/shared/task_manager.py` — SQLite-backed, persists across server restarts.

```python
task_id = tm.create_task(task_type="audio_analysis", params={...})
tm.update_progress(task_id, stage="analyzing", percent=50.0, message="...")
tm.complete_task(task_id, result={"key": "value"})
tm.fail_task(task_id, error="...")
```

Default db path: `backend/tasks.db` (excluded from git).

### VRAM Management

Only one model is loaded in VRAM at a time — **kill-and-revive pattern**. `VRAMManager` in `backend/services/shared/vram_manager.py` handles model lifecycle. This is a hard constraint; do not load two heavy models simultaneously.

### Integration Tests

`backend/tests/integration/test_api_integration.py` uses a module-scoped `TestClient`. Tests that exercise endpoints requiring a previously-uploaded/ingested resource must do that setup inline — the test client scope means fixtures can't do async setup.

The `_minimal_wav()` helper at the top of the integration test file generates a valid 0.5-second silence WAV with no external dependencies. Use it for any test that needs a real audio file upload.

Stubs still in place: `tasks`, `video`, `lora`, `nova_fade` routers all return placeholder data. Integration tests for those endpoints test shape/status only, not real results.

## Constraints

- **No Ken Burns, no Blender, no slideshows** — all video via AI generation backends
- **chromadb==0.4.22** — pinned, do not upgrade
- **One VRAM model at a time** — kill-and-revive, never load two simultaneously
- **Nova Fade constitution** (`backend/config/nova_fade_constitution.yaml`) is programmatically enforced; changes require version increment
- **All routers must be mounted in `backend/main.py`** — no orphan router modules
