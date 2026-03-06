# Continuity Ledger - Beat_Studio

**Last Updated:** 2026-03-06
**Project:** Beat_Studio - Unified AI music video production platform
**Current Phase:** RunPod Worker Fixed — Local Paths + V3 Subprocess — Ready to Deploy

---

## Goal

Production-ready music video generation platform combining:
- 8 mashup types from AI_Mixer
- Neural video backends (no Ken Burns, no slideshows)
- Nova Fade DJ character system

## Constraints

- ChromaDB pinned at 0.4.22
- One VRAM model at a time (kill-and-revive)
- VRAM budget: 12 GB total, 1.5 GB baseline
- Nova Fade constitution is programmatically enforced
- Desktop GPU (Quadro M2000, 4GB) insufficient for SDXL — use laptop (12GB) or cloud
- Desktop system RAM: 15 GB — insufficient for FramePack (needs 30+ GB for model in CPU RAM)

## State

- Done:
  - [x] Phase 1-9: Backend services (audio, mashup, video, lora, nova_fade, shared)
  - [x] Phase 10: React Frontend (36 tests)
  - [x] Phase 11: Integration Tests (56 tests)
  - [x] Phase 12: Setup & Models
  - [x] All routers wired to real services
  - [x] Deep codebase research (Beat_Studio, AI_Mixer, BeatCanvas)
  - [x] Migrated thoughts/handoffs from AI_Mixer to Beat_Studio
  - [x] LoRA trainer per-image caption support (write_captions, _load_caption)
  - [x] Imported 12 LoRAs (identity, style, character, scene)
  - [x] Imported test song (5bb5790e-9a46-4bcd-987e-f1c82f33f7dd.wav)
  - [x] venv setup with all dependencies (NumPy <2.0 for ChromaDB compat)
  - [x] Fixed video generation pipeline (style prefix, beat_times, LoRA loading, download)
  - [x] First real test run (cb217758): 4-scene render, diagnosed issues
  - [x] Fixed energy level showing 100% (bad /0.1 divisor from security review)
  - [x] Fixed single-LoRA set_adapters skipping weight application
  - [x] Storyboard stage: full SDXL keyframe preview system (TDD, 88 tests)
  - [x] Per-scene LoRA weight sliders (default 0.7, presets: Style/Balanced/LoRA)
  - [x] Progress bar during SDXL generation (polls image count every 3s)
  - [x] Prompt override textarea in regen panel (pre-populated, highlights red when edited)
  - [x] LoRA auto-selection also matches by descriptive tags (scene LoRAs)
  - [x] ZIP download (scene_01.png … scene_18.png)
  - [x] ZIP upload + snapshot system (create/list/restore snapshots, upload external ZIPs)
  - [x] FramePack backend (framepack.py): fp8+cpu_offload, 4k+1 snap, I2V via init_image_path, 41 tests
  - [x] init_image_path added to ComposedPrompt (backward-compatible, default "")
  - [x] checkpoints.yaml: framepack_f1_transformer (25.7 GB) + framepack_hunyuan_base (4.7 GB)
  - [x] RunPod worker: runpod_worker/ (handler, models, Dockerfile) — 5 model comparison
  - [x] RunPodBackend: proper VideoBackend subclass, polls job, writes MP4 to disk
  - [x] Assembler duration fix: loops short clips (-stream_loop), trims long clips
  - [x] Fixed approved storyboard paths discarded in VideoStudio (TDD, 9 tests)
  - [x] GenerateRequest: added backend, runpod_model, approved_image_paths fields
  - [x] RunPod worker models.py: local /workspace/models paths, Wan2.2-I2V-A14B ID fix, V3 subprocess
- Now: [→] Craig deploys worker → runs comparison → picks winner
- Next: Wire winner as default backend, run full Island Girl video

## Key Decisions

- `blender_first_not_sdxl`: SDXL /generate-canonical is NOT used for Nova Fade identity LoRA — produces inconsistent characters. Blender renders are the only source of truth for identity.
- `sdxl_canonical_repurposed`: POST /generate-canonical repurposed for STYLE LoRA (Crossfade Club aesthetic) once identity LoRA exists.
- `quality_over_speed`: User explicitly prioritizes quality. Time is not a constraint.
- `laptop_for_inference`: Desktop GPU (4GB) can't run SDXL; use laptop (12GB) for inference testing.
- `lora_names_default_empty`: lora_names=[] means no LoRAs loaded; user must explicitly select. Avoids all-trigger chaos.
- `lora_unload_finally`: Always unload LoRA weights in try/finally so they never bleed between clips.
- `user_overrides_via_scene_generator`: Pass as int_overrides to ScenePromptGenerator so style prefix wraps user text.
- `beat_times_in_namespace`: Added to _load_analysis_for_sync so generation worker matches prompts endpoint behavior.
- `file_path_in_lora_config`: Added file_path field to LoRAConfig so AnimateDiff backend uses exact registry path, not fuzzy match.
- `framepack_needs_30gb_ram`: FramePack transformer (25.7 GB) lives in CPU RAM. Desktop (15 GB) cannot run it. Cloud (A100 80 GB) or laptop with 32+ GB needed.
- `runpod_hf_home_on_volume`: Dockerfile sets HF_HOME=/runpod-volume/hf_cache so all model weights persist on the network volume across pod restarts.
- `runpod_one_model_at_a_time`: Worker caches one model in VRAM at a time; requesting a different model frees the current one first (same kill-and-revive pattern as local backend).
- `approved_paths_as_ordered_list`: Frontend converts Record<string, string> (scene_index → path) to sorted list before sending to backend, so backend indexing is positional.

## Blockers

- None currently (Craig needs to set up RunPod account + credits to run the comparison)

## Open Questions / Known Issues

- `watercolor_inconsistent`: In run cb217758, scene02 had watercolor style but scene03 was photorealistic — checkpoint swap not firing consistently for all clips. Suspect `_current_checkpoint` caching. Needs investigation.
- `beat_times_alignment`: Did beat_times fix actually align scene indices between prompts stage and generation worker? Unconfirmed.
- `dreamshaper_download`: Does DreamShaper-8 download cleanly from HF on the system, or needs pre-caching?
- `scene_editor_overrides`: SceneEditor prompt edits not yet wired back through user_overrides — only prompts-stage edits are wired.
- `skyreels_v2_i2v_pipeline_class`: models.py uses WanImageToVideoPipeline as placeholder — verify exact class from Skywork/SkyReels-V2-I2V-14B-720P model card before deploying.
- `skyreels_v3_r2v_pipeline`: RESOLVED — uses subprocess via generate_video.py (clone SkyworkAI/SkyReels-V3, copy script to /workspace/models/SkyReels-V3-R2V-14B/).
- `runpod_winner_selection`: Craig picks winning model after reviewing 15 comparison clips (scene_03, scene_11, scene_17 × 5 models). Then delete losers from network volume.

## Working Set

**Branch:** main
**Test Command:** `python -m pytest backend/tests/ -q`
**Frontend Tests:** `npm test -- --run` (from frontend/)
**Key Files:**
- `backend/routers/storyboard.py` — storyboard router (snapshot, upload, download, regen)
- `backend/services/storyboard/service.py` — SDXL keyframe generation
- `backend/services/storyboard/state.py` — JSON-backed state store, version eviction
- `backend/services/storyboard/types.py` — StoryboardState, StoryboardScene, VersionEntry
- `frontend/src/components/StoryboardPreview.tsx` — full storyboard UI
- `backend/config/loras.yaml` — 12 LoRAs registered
- `output/loras/` — LoRA weights organized by type
- `runpod_worker/src/handler.py` — RunPod job handler
- `runpod_worker/src/models.py` — 5-model loader + generator
- `backend/services/video/backends/runpod_client.py` — RunPodBackend (VideoBackend subclass)
- `scripts/runpod_compare.py` — 15-clip comparison runner

## Test Status

- Backend: 598 tests (includes 41 FramePack + 6 new GenerateRequest tests)
- Frontend: 55 tests (includes 3 new VideoStudio tests)
- Total: 653 tests passing

## LoRAs Available

| Type | Name | Trigger Token |
|------|------|---------------|
| identity | nova_fade_id_v1 | `novafade_char` |
| style | crossfadeclub_style_v1 | `crossfadeclub_style` |
| style | 70s-film-retro | `70s_film_style` |
| character | rob-character | `rob_char` |
| character | michele-character | `michele_char` |
| scene | beach-bar-exterior | `beach_bar_ext` |
| scene | beach-sunset | `beach_sunset` |
| scene | boat-deck | `boat_deck` |
| scene | bonfire-beach | `bonfire_beach` |
| scene | ocean-underwater | `ocean_underwater` |
| scene | stage-performance | `stage_perf` |
| scene | tiki-bar-interior | `tiki_bar_int` |

All trained on **SDXL Base 1.0**.

## Session 2026-02-25: Video Generation Pipeline Fixes

**What was done:**
- Fixed video download (JSON → FileResponse, videoId extraction from task result)
- Added ClipFrameViewer: first frame PNGs per clip, scoped to video_id via concat.txt
- Ran first real 4-scene test (run cb217758)
- Fixed style prefix stripped (user_overrides routed through ScenePromptGenerator int_overrides)
- Fixed beat_times mismatch (added to _load_analysis_for_sync namespace)
- Wired LoRA loading into AnimateDiff pipeline (load_lora_weights per clip, unload in finally)
- Added LoRA selection UI with checkboxes; lora_names in GenerateRequest
- Fixed LoRA stage bypass (Continue → lora stage, not direct to plan)

## Session 2026-02-26–28: Storyboard Stage

**What was done:**
- Fixed energy level showing 100% (security review introduced bad /0.1 divisor on RMS)
- Fixed single-LoRA set_adapters bug (was skipping weight application for len == 1)
- Added regen spinner animation (.spin CSS + @keyframes in index.css)
- Built complete Storyboard stage (TDD, 88 tests across state/service/router/frontend):
  - SDXL 1024×576 keyframe generation
  - VersionEntry with lora_weights, source ("generated"|"upload"), seed
  - Version carousel (up to MAX_VERSIONS=5), evicts oldest
  - Per-scene LoRA weight sliders (only shown if trigger token in prompt)
  - Weight presets: Style/Balanced/LoRA
  - Progress bar during generation (polls image count every 3s)
  - Keyboard navigation (← →)
  - Optimistic version placeholder on regen
  - Prompt override textarea (pre-populated, red highlight when edited, reset button)
  - Auto LoRA selection by trigger token AND descriptive tags
  - ZIP download: GET /api/video/storyboard/{id}/download → scene_01.png … scene_18.png
  - ZIP upload + snapshots: POST /upload, POST /snapshot, GET /snapshots, POST /snapshots/{sid}/restore

## Session 2026-03-01: RunPod Cloud Backend

**What was done:**
- Investigated FramePack RAM: needs 30+ GB system RAM; desktop (15 GB) ruled out
- Built RunPod serverless worker (runpod_worker/): handler, 5-model loader, Dockerfile
  - Models: framepack, skyreels_v2_i2v, skyreels_v2_df, wan22_i2v, skyreels_v3_r2v
  - One-at-a-time VRAM caching; HF_HOME on network volume for persistence
- Built RunPodBackend (VideoBackend subclass): submits job, polls, writes MP4 to disk
- Built scripts/runpod_compare.py: 15-clip comparison (3 scenes × 5 models), saves to output/comparison/
- Fixed assembler duration bug: loops short clips via ffmpeg -stream_loop, trims long clips
- Added scene_durations optional parameter to VideoAssembler.assemble (backward-compatible)
- TDD: fixed onApprove discarding approvedPaths in VideoStudio.tsx (9 tests: 6 backend + 3 frontend)
- Added backend/runpod_model/approved_image_paths to GenerateRequest

**Craig's next steps:**
1. Create RunPod account, add $25 credits, get API key
2. Create 200 GB network volume (beat-studio-models)
3. Spin up temp A100 pod, download all 5 models + HunyuanVideo base + flux_redux_bfl
4. Deploy serverless endpoint (GitHub or Docker Hub)
5. Run: python scripts/runpod_compare.py --storyboard-zip path/to/approved.zip
6. Review 15 clips, pick winner, delete losers from network volume

## Session 2026-03-06: RunPod Worker — Local Paths + V3 Subprocess Fix

**What was done:**
- Fixed runpod_worker/src/models.py — 4 changes:
  1. Constants now use `Path("/workspace/models")` for all local model dirs
  2. Wan 2.2 repo ID corrected: `Wan2.2-I2V-A14B` (was `Wan2.2-I2V-14B-720P`)
  3. Added `_SKYREELS_V3_SCRIPT` constant for generate_video.py path
  4. `_load_skyreels_v3_r2v()` returns sentinel dict (script path + model_id) instead of pipeline
  5. `_gen_skyreels_v3_r2v()` runs generate_video.py via subprocess, returns MP4 bytes directly
  6. `load_and_generate()` early-returns for V3 before the `_GENERATORS` dispatch

**Craig's next steps (unchanged):**
1. Create RunPod account, add credits, get API key + endpoint ID
2. Create 200 GB network volume (beat-studio-models), download 5 models
3. For V3: clone SkyworkAI/SkyReels-V3, copy generate_video.py to /workspace/models/SkyReels-V3-R2V-14B/
4. Deploy serverless endpoint
5. Run: python scripts/runpod_compare.py --storyboard-zip path/to/approved.zip
6. Review 15 clips, pick winner
