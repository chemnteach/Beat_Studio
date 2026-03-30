# Continuity Ledger - Beat_Studio

**Last Updated:** 2026-03-30
**Project:** Beat_Studio - Unified AI music video production platform
**Current Phase:** Hero Detection + Video Blockers Fixed — Improve Boundary Detection

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
  - [x] video.py fully wired: RunPodBackend selection, init_image_path per clip, scene_durations to assembler
  - [x] Docker v20: FramePack ndarray fix, built+pushed from Windows
  - [x] Docker v21: Wan2.2 configuration.json check + AutoPipelineForImage2Video
  - [x] Docker v22: Fix Wan2.2 loader to DiffusionPipeline (diffusers 0.34.0 compat)
  - [x] RunPod endpoint updated to v22, timeout 1800s, purged
  - [x] SkyReels V3 comparison: 691s, 72 frames, 2.7MB — OK
  - [x] FramePack comparison: 517s, 90 frames, 1.6MB — OK
  - [x] WAN 2.2: failed all runs (model_index.json missing on network volume)
  - [x] Winner selected: SkyReels V3
  - [x] derive_video_prompt() in prompt_composer.py — GPT-4-turbo strips image style/LoRA tokens, adds motion language
  - [x] video_prompt field added to StoryboardScene (persisted, editable independently via PATCH endpoint)
  - [x] Video prompt side-by-side editor in StoryboardPreview.tsx
  - [x] Multi-clip strategy: ceil(duration/8) clips per scene, each duration/num_clips seconds
  - [x] VideoAssembler.assemble() accepts clips_per_scene — transitions only at scene boundaries
  - [x] RunPodBackend.generate_clip() clamps to MAX_CLIP_SEC=8 with warning
  - [x] compare_models.py: better error handling (missing output key, handler errors)
  - [x] Discovered RunPod payload size limit (~10MB): 7s V3 clips (~24MB base64) exceed it
  - [x] RunPod endpoint timeout raised to 2400s
  - [x] 617/619 unit tests passing (2 pre-existing failures from real creds in backend/.env)
  - [x] R2 upload in worker: MP4 → Cloudflare R2, handler returns video_url (SigV4 fix included)
  - [x] Fixed RunPod handler: generator→regular return (serverless standard doesn't capture yields)
  - [x] Fixed SkyReels-V3 HF repo ID: Skywork/SkyReels-V3-Reference2Video (not SkyReels-V3-R2V-14B)
  - [x] Added --low_vram to V3 subprocess: FP8 quant + block offload, fits 14B on 24GB GPU
  - [x] Fixed HF_HOME=/runpod-volume/hf_cache (was /workspace/hf_cache — container disk too small)
  - [x] Validated full end-to-end pipeline (v33): 7s watercolor scene, 168 frames, 1570s, R2 upload OK
  - [x] Validated ref_images pipeline (v33): 3 CC0 images → 5s clip, 120 frames, 1138s
  - [x] TF32 measured: no speedup on FP8-quantized workload (kept in code, harmless)
  - [x] Cleaned up Kijai FP8 files (~41GB freed); added delete_dirs maintenance op (v35)
  - [x] derive_video_prompt() watercolor suffix: REMOVED — style is user choice, not a hardcoded default
  - [x] Docker v35 deployed to endpoint 2zo31rfwfbzsz8; volume has warm 28GB model cache
  - [x] Audio segmentation overhaul: novelty boundaries, lyrics alignment, relabel_by_hook_phrase (2-4 gram n-gram extraction, promotes chorus/pre_chorus/bridge)
  - [x] Hero moment detection: 4-signal scoring (bridge type, temporal 50-75%, energy inflection, first hook phrase); has_lyrics gate; fallback to energy threshold for standard-depth
  - [x] Beat-aligned clip splitting: beat_aligned_clip_durations; hero snaps to downbeats; regular snaps to beats; _MAX_CLIP_SEC 8→5; lyrical_content wired through _load_analysis_for_sync
  - [x] ref_image_paths wiring: GenerateRequest field → _run_generate_video → ComposedPrompt.ref_image_paths (RunPodBackend already encoded it — gap closed)
  - [x] Default backend: animatediff → skyreels_r2v in style_mapper.py fallback styles and VideoStudio.tsx initial state
  - [x] VideoStudio.tsx: ref portrait photo file input (max 3), thumbnail previews, ref_image_paths in POST body
- Now: [→] Improve boundary detection — investigate novelty curve near 160s for Island Girl GT bridge onset
- Next: Upload endpoint for ref portrait images (currently blob URLs won't reach RunPod worker as valid paths)

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
- `video_backend_local_var`: Renamed local `backend` var in _run_generate_video to `video_backend` to avoid shadowing the `backend: str` parameter.
- `scene_durations_now_wired`: VideoAssembler.assemble() now receives scene_durations from synced_scenes so clips loop/trim to exact beat-aligned durations.
- `skyreels_v3_winner`: SkyReels V3 selected as production video backend after 3-model comparison. FramePack ok but smaller file, WAN 2.2 never ran (missing model files).
- `multi_clip_per_scene`: Scenes split into ceil(duration/8) clips of equal duration. No looping. clips_per_scene tracks grouping for assembler. Transitions only at scene boundaries.
- `runpod_payload_limit`: RunPod serverless output payload ~10MB max. 3s V3 clip (7.6MB raw / ~10MB b64) barely fits. 7s clip (~18MB raw / ~24MB b64) silently dropped — COMPLETED with no output key. Solution: R2 upload from worker, return URL. RESOLVED in v24.
- `v3_generation_time`: SkyReels V3 at 480P takes ~3 min/sec of video on A100. Warm cache: 5s clip = 1138s, 7s clip = 1570s. Cold start (first 28GB download) adds ~26 min.
- `r2_sigv4_required`: boto3 with Cloudflare R2 requires explicit `config=botocore.config.Config(signature_version="s3v4")`. Default SigV4 negotiation fails.
- `generator_vs_return`: RunPod standard serverless does NOT capture generator yields in output[]. Handler must be a regular function returning a single dict.
- `hf_cache_on_volume`: HF_HOME must point to /runpod-volume/hf_cache. Container disk is ~20GB; SkyReels-V3-Reference2Video is 28GB.
- `kijai_fp8_unusable`: ComfyUI-format FP8 safetensors from Kijai/WanVideo_comfy_fp8_scaled are incompatible with SkyReels-V3 generate_video.py. Use --low_vram for equivalent VRAM savings.
- `tf32_no_benefit`: TF32 provides no speedup when model runs with FP8 quantization (--low_vram).
- `watercolor_suffix_removed`: derive_video_prompt() no longer appends any style suffix. Style is selected by the user via StyleSelector; the prompt derivation step only strips image-specific terms and adds motion language.
- `hero_score_has_lyrics_gate`: Multi-signal hero scoring only activates when at least one section has string lyrical_content. Fallback to energy threshold preserves all existing unit test compatibility with MagicMock sections.
- `skyreels_r2v_new_default`: animatediff replaced by skyreels_r2v as default recommended_model in style_mapper.py fallback styles and as initial selectedBackend in VideoStudio.tsx.
- `ref_image_paths_wired`: ref_image_paths now flows from GenerateRequest → _run_generate_video → every ComposedPrompt. RunPodBackend already encoded it via getattr. Upload-to-server wiring is a separate task (blob URLs from file input won't survive to the worker).

## Blockers

- None. R2 upload unblocked long clip delivery.

## Open Questions / Known Issues

- `watercolor_inconsistent`: In run cb217758, scene02 had watercolor style but scene03 was photorealistic — checkpoint swap not firing consistently for all clips. Suspect `_current_checkpoint` caching. Needs investigation.
- `beat_times_alignment`: Did beat_times fix actually align scene indices between prompts stage and generation worker? Unconfirmed.
- `dreamshaper_download`: Does DreamShaper-8 download cleanly from HF on the system, or needs pre-caching?
- `scene_editor_overrides`: SceneEditor prompt edits not yet wired back through user_overrides — only prompts-stage edits are wired.
- `skyreels_v2_i2v_pipeline_class`: models.py uses WanImageToVideoPipeline as placeholder — verify exact class from Skywork/SkyReels-V2-I2V-14B-720P model card before deploying.
- `skyreels_v3_r2v_pipeline`: RESOLVED — uses subprocess via generate_video.py (clone SkyworkAI/SkyReels-V3, copy script to /workspace/models/SkyReels-V3-R2V-14B/).
- `runpod_winner_selection`: RESOLVED — SkyReels V3 selected. WAN 2.2 never ran (missing model files). FramePack ok but V3 chosen for quality.
- `r2_upload_wired`: RESOLVED — worker uploads MP4 to R2, returns video_url. Backend RunPodBackend downloads via URL.

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
- `scripts/compare_models.py` — 3-model quick comparison (single image + prompt)

## Test Status

- Backend: ~620 tests (617/619 passing pre-R2; 2 pre-existing failures from real creds in backend/.env)
- Frontend: 55 tests
- Total: ~675 tests

## Docker / RunPod State

| Key | Value |
|-----|-------|
| Current image | `chemnteach/beat-studio-worker:v35` |
| Endpoint ID | `2zo31rfwfbzsz8` |
| Template ID | `bgr4o5wxbs` |
| Timeout | 3600s |
| HF cache | `/runpod-volume/hf_cache/models--Skywork--SkyReels-V3-Reference2Video` (~28GB, warm) |
| models_dir | empty (Kijai FP8 deleted) |
| R2 env vars | confirmed on endpoint |

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

## Session 2026-03-07: Video Router — Full RunPod Wiring

**What was done:**
- Wired 4 connections in backend/routers/video.py to complete end-to-end RunPod flow:
  1. Added `backend`, `runpod_model`, `approved_image_paths` params to `_run_generate_video`
  2. Step 3: instantiates `RunPodBackend(model_name=...)` when `backend=="runpod"`, else `ModelRouter`
  3. Renamed local `backend` → `video_backend` throughout to avoid param shadowing
  4. Step 9: each `ComposedPrompt` gets `init_image_path = approved_image_paths[i]`
  5. Step 11: `VideoAssembler.assemble()` receives `scene_durations` from `synced_scenes`
  6. Route handler forwards all three new fields to background task
- Integration test run: 20.8 hours, 15 failed (all pre-existing stubs), 41 passed — no regressions

## Session 2026-03-16: Dev Environment Setup + First Comparison Attempt

**What was done:**
- Onboarded to project on new Windows/WSL2 machine
- Verified all dependencies installed (Python 3.12 venv, Node 20, ffmpeg, rubberband)
- Created `backend/.env` with ANTHROPIC_API_KEY, OPENAI_API_KEY, RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
- Built `scripts/compare_models.py` — standalone 3-model comparison (submits to RunPod, polls, saves MP4s)
- Ran first comparison attempt — all 3 models failed with fixable errors:
  - SkyReels V3: `executionTimeout exceeded` at 622s (endpoint timeout too low)
  - WAN 2.2: `No model_index.json` at `/runpod-volume/models/Wan2.2-I2V-A14B` (incomplete download)
  - FramePack: `append_data requires ndarray` in `frames_to_mp4_bytes` (PIL Images not converted)
- Fixed FramePack ndarray bug in `runpod_worker/src/models.py` (added `np.array(frame)` conversion)
- Investigated version history: v19 is current production, our fix should be v20 (not v17)
- All 16 RunPod backend unit tests pass

**Craig's next steps:**
1. Build/push Docker v20 from Windows PowerShell
2. Update RunPod endpoint: image tag → v20, execution timeout → 1800s
3. Spin up pod, re-download WAN 2.2 to network volume, verify model_index.json
4. Re-run `scripts/compare_models.py`

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

## Sessions 2026-03-19 to 2026-03-20: R2 Upload + Full Pipeline Validation (v24–v35)

**What was done:**
- Implemented R2 upload in worker: MP4 bytes → Cloudflare R2 → presigned URL returned as `video_url`
  - SigV4 fix: explicit `botocore.config.Config(signature_version="s3v4")` required for R2
  - Explicit env var validation on startup (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME)
- Fixed generator handler: RunPod standard serverless doesn't capture yield outputs → converted to regular return
- Fixed SkyReels-V3 HF repo ID: `Skywork/SkyReels-V3-Reference2Video` (was `Skywork/SkyReels-V3-R2V-14B`)
- Added `--low_vram` to V3 subprocess: FP8 weight-only quant + block offload, fits 14B on 24GB GPU
- Fixed HF_HOME=/runpod-volume/hf_cache (was /workspace/hf_cache — only 21GB free, model is 28GB)
- Added watercolor suffix to derive_video_prompt() system prompt — appended to every video prompt
- Added `list_models_dir` and `delete_dirs` maintenance ops to handler
- Validated end-to-end pipeline:
  - v33 scene11: 7s, 480x720, 168 frames, 1570s, R2 upload confirmed
  - v33 ref_images: 3 CC0 images, 5s, 120 frames, 1138s
- TF32 test (v34): no speedup on FP8-quantized workload (1679s vs 1570s baseline)
- Cleaned up ~41GB Kijai FP8 files from network volume (v35)
- GraphQL saveTemplate mutation used to update endpoint image without dashboard access

**Next:**
1. Wire `skyreels_v3_r2v` as default model in backend video router (runpod_client.py or video.py)
2. Evaluate output/v33_test_scene11.mp4 and output/v33_ref_images_test.mp4
3. Character consistency test with real same-person portraits (2–3 angles)

## Session 2026-03-29: Audio Segmentation + Hook Phrase Detection

**What was done:**
- Audio segmentation overhaul (committed as a09efdb): novelty-based boundary detection, Whisper lyrics alignment, 5-stage relabeling pipeline
- Added `relabel_by_hook_phrase()`: extracts most-repeated all-content-word 2-4 gram across inner sections; promotes to chorus/pre_chorus/bridge; 7 unit tests
- Wired `relabel_by_hook_phrase` as step 5d in analyzer.py after `relabel_by_lyrical_repetition`
- Validated with Island Girl Whisper transcription: "island girl" correctly wins as hook (5 inner sections) over "old man" (4)
- Found boundary is the bottleneck: GT bridge [160-190s] falls inside single pipeline section [132.8-170.5s]

## Session 2026-03-30: Hero Detection + Video Pipeline Blockers

**What was done:**
- Multi-signal hero scoring in beat_sync.py: 4 signals (bridge type, temporal 50-75%, energy inflection, first hook phrase), threshold 0.6, has_lyrics gate
- beat_aligned_clip_durations: hero→downbeat split, regular→beat snap, _MAX_CLIP_SEC 8→5
- Wired lyrical_content through _load_analysis_for_sync; beat_aligned_clip_durations into video.py clip loop
- Removed hardcoded watercolor suffix from derive_video_prompt() system prompt
- Changed default backend animatediff → skyreels_r2v (style_mapper.py + VideoStudio.tsx)
- Wired ref_image_paths: GenerateRequest field → _run_generate_video → ComposedPrompt (was already encoded by RunPodBackend)
- Added portrait reference photo file input in VideoStudio.tsx
- 21 new unit tests (TestBeatAlignedClipDurations, TestComputeHeroScores, TestExtractHookPhrase)
- 666/668 tests passing (2 pre-existing failures)

**Next:**
1. Investigate novelty curve boundary sensitivity near 160s (GT bridge onset for Island Girl)
2. Add server-side upload endpoint for portrait ref images (blob URLs don't survive to RunPod worker)
3. Character consistency test with real same-person portrait photos
