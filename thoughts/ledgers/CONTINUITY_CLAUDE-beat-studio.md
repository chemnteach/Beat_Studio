# Continuity Ledger - Beat_Studio

**Last Updated:** 2026-02-22
**Project:** Beat_Studio - Unified AI music video production platform
**Current Phase:** Full Video Cycle Testing

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
- Now: [→] Full video cycle testing on laptop (12GB VRAM)
- Next: Test LoRA inference, then full video generation pipeline

## Key Decisions

- `blender_first_not_sdxl`: SDXL /generate-canonical is NOT used for Nova Fade identity LoRA — produces inconsistent characters. Blender renders are the only source of truth for identity.
- `sdxl_canonical_repurposed`: POST /generate-canonical repurposed for STYLE LoRA (Crossfade Club aesthetic) once identity LoRA exists.
- `quality_over_speed`: User explicitly prioritizes quality. Time is not a constraint.
- `laptop_for_inference`: Desktop GPU (4GB) can't run SDXL; use laptop (12GB) for inference testing.

## Blockers

- None currently

## Open Questions

- UNCONFIRMED: Which video backend to use for full cycle test?

## Working Set

**Branch:** main
**Test Command:** `python -m pytest backend/tests/ -q`
**Key Files:**
- `backend/services/lora/trainer.py` - caption support complete
- `backend/config/loras.yaml` - 12 LoRAs registered
- `output/loras/` - LoRA weights organized by type

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

## Test Status

- Backend: 469 tests (413 unit + 56 integration)
- Frontend: 36 tests
- Total: 505 tests passing

## Session 2026-02-22: Environment Setup & LoRA Import

**What was done:**
- Set up venv, installed all dependencies (~4GB)
- Fixed NumPy 2.x / ChromaDB 0.4.22 incompatibility (downgraded to numpy<2.0)
- Verified all 505 tests pass
- Fixed LoRA trainer to support per-image .txt sidecar captions
- Added write_captions() helper and _load_caption() method
- Imported 12 LoRAs from SD card (identity, style, character, scene)
- Imported test song (5bb5790e-9a46-4bcd-987e-f1c82f33f7dd.wav)
- Updated loras.yaml registry with all LoRAs and trigger tokens

**Next on laptop:**
- Test LoRA inference: `rob_char, man standing on a beach, golden hour lighting`
- Run full video generation pipeline
