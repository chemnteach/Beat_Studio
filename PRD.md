# Beat_Studio Product Requirements Document

**Version:** 1.0
**Last Updated:** 2026-02-22
**Status:** Production

---

## 1. Executive Summary

Beat_Studio is an AI-powered music video production platform that combines intelligent audio mashup creation with professional neural video generation. It enables users to transform songs into continuous, beat-synchronized videos through a streamlined workflow.

### Vision

Create the most accessible tool for producing professional-quality AI music videos — not slideshows with Ken Burns effects, but genuine continuous video synchronized to music.

### Core Value Proposition

- **For musicians**: Turn any song into a music video without film crews or budgets
- **For DJs**: Create mashups with semantic intelligence, then visualize them
- **For content creators**: Produce platform-ready videos (TikTok, YouTube, Reels)

---

## 2. User Workflows

### Workflow A: Original Music → Music Video

```
1. Upload song (MP3/WAV/FLAC)
2. Analyze audio (BPM, key, sections, mood)
3. Optional: Add Whisper transcription + LLM semantic analysis
4. Select animation style (24 options)
5. Apply optional LoRA (character, style, scene)
6. Preview execution plan (scenes, timing, cost)
7. Generate continuous video via selected backend
8. Export final video with audio
```

### Workflow B: Mashup Creation → Music Video

```
1. Ingest songs into library (local files or YouTube)
2. Match songs via semantic + harmonic similarity
3. Select mashup type (8 available)
4. Create mashup audio
5. Generate video (same as Workflow A)
   OR
   Generate DJ video featuring Nova Fade character
```

---

## 3. Feature Requirements

### 3.1 Audio Analysis (P0 - Must Have)

| Feature | Description | Status |
|---------|-------------|--------|
| BPM Detection | Accurate tempo extraction via librosa | Complete |
| Key Detection | Musical key + Camelot wheel mapping | Complete |
| Section Detection | Verse/chorus/bridge boundaries | Complete |
| Energy Analysis | 0-10 scale energy level per section | Complete |
| Beat Timing | Precise beat timestamps for sync | Complete |
| Whisper Transcription | Lyrics extraction with word timings | Complete |
| LLM Semantic Analysis | Mood, genre, themes, valence | Complete |

**Analysis Depth Levels:**
- `basic`: BPM, key, energy, first downbeat (fast, no heavy models)
- `standard`: + section boundaries, beat times
- `full`: + transcription, semantic analysis

### 3.2 Mashup Engine (P0 - Must Have)

| Mashup Type | Description | Complexity |
|-------------|-------------|------------|
| Classic | Vocals A + Instrumental B | Low |
| Stem Swap | Mix stems from 3+ songs | Low |
| Energy Match | Align by energy curves | Medium |
| Adaptive Harmony | Auto-fix key clashes via pitch-shift | Medium |
| Theme Fusion | Filter by lyrical themes | Medium |
| Semantic-Aligned | Question→answer pairing | High |
| Role-Aware | Vocals as lead/harmony/call/response | High |
| Conversational | Dialogue-style with silence gaps | High |

**Matching Modes:**
- Harmonic: BPM tolerance ±5%, Camelot wheel compatibility
- Semantic: Vector similarity via ChromaDB embeddings
- Hybrid: 60% harmonic + 40% semantic (recommended)

### 3.3 Video Generation (P0 - Must Have)

| Backend | VRAM | Quality | Speed | Cost |
|---------|------|---------|-------|------|
| AnimateDiff-Lightning | 5.6 GB | Good | Fast | Free |
| WAN 2.6 Local | 12 GB | Excellent | Slow | Free |
| WAN 2.6 Cloud | N/A | Excellent | Medium | ~$0.05/scene |
| SkyReels Cloud | N/A | Good | Fast | ~$0.03/scene |
| SVD | 7.5 GB | Good | Medium | Free |
| SDXL + ControlNet | 8 GB | Good | Medium | Free |
| CogVideoX | 14 GB | Excellent | Slow | Free |

**Model Router** automatically selects best backend based on:
- Style requirements (some styles only work with certain backends)
- Available VRAM
- User quality/speed preference
- Cost budget

### 3.4 Animation Styles (P0 - Must Have)

24 styles available, each with:
- Prompt suffix modifiers
- Negative prompt tokens
- Recommended backend
- Guidance scale
- Inference steps

**Categories:**
- Traditional: Cel animation, watercolor, ink wash, pencil sketch
- Digital: Motion graphics, pixel art, low-poly 3D, isometric
- Artistic: Oil painting, impressionist, ukiyo-e, pop art
- Modern: Synthwave, graffiti, psychedelic, photorealistic

### 3.5 LoRA System (P1 - Should Have)

| Feature | Description | Status |
|---------|-------------|--------|
| Registry | YAML-based LoRA inventory | Complete |
| Training | SDXL LoRA training pipeline (rank 16, AdamW8bit) | Complete |
| Download | HuggingFace + Civitai support | Complete |
| Recommendation | Match LoRAs to song + style | Complete |
| Drift Testing | CLIP-based identity regression | Complete |

**LoRA Types:**
- Character: Specific person/figure
- Style: Visual aesthetic
- Scene: Environment/setting
- Identity: Nova Fade character

### 3.6 Nova Fade DJ Character (P1 - Should Have)

| Feature | Description | Status |
|---------|-------------|--------|
| Character Definition | Constitution with expressions, gestures, forbidden terms | Complete |
| Canonical Prompts | Immutable prompt templates | Complete |
| Identity LoRA | Blender-trained character consistency | In Progress |
| Style LoRA | Crossfade Club aesthetic | Planned |
| DJ Video Generation | Character performing mashup | Planned |
| Drift Testing | CLIP-based identity verification | Complete |

**Constitution Rules:**
- 5 allowed expressions (neutral_confident, subtle_smile, focused_intensity, knowing_smirk, energized)
- 5 allowed gestures (deck_touch, crossfader_slide, headphone_adjust, pointing_up, arm_raised)
- Forbidden attributes enforced programmatically

### 3.7 Frontend UI (P0 - Must Have)

**5 Tabs:**
1. **Upload & Analyze**: Drag-drop audio, analysis display
2. **Mashup Workshop**: Library management, matching, creation
3. **Video Studio**: Style selection, execution planning, generation
4. **LoRA Manager**: Registry, training, downloads, recommendations
5. **System Status**: GPU info, model inventory, health checks

**Real-time Features:**
- WebSocket progress updates
- Task polling fallback
- Cost estimation before generation

---

## 4. Non-Functional Requirements

### 4.1 Performance

| Metric | Requirement |
|--------|-------------|
| Audio analysis (basic) | < 5 seconds |
| Audio analysis (full) | < 2 minutes |
| Scene generation (local) | 10-60 seconds each |
| Scene generation (cloud) | 5-30 seconds each |
| Video assembly | < 2 minutes for 24 scenes |

### 4.2 Scalability

- Single-user local deployment (primary use case)
- No distributed processing required
- SQLite sufficient for task persistence

### 4.3 Reliability

- Task state persists across server restarts
- Graceful degradation if cloud APIs unavailable
- Automatic fallback from primary to secondary LLM

### 4.4 Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| VRAM | 6 GB | 12 GB |
| Storage | 50 GB | 200 GB |
| GPU | NVIDIA GTX 1060 | NVIDIA RTX 3080+ |

---

## 5. Constraints and Rules

### 5.1 Hard Constraints

| Constraint | Rationale |
|------------|-----------|
| No Ken Burns effects | Quality standard — not a slideshow |
| No Blender rendering | Neural-only video generation |
| No slideshows | Must be continuous video |
| ChromaDB 0.4.22 | Breaking API changes in newer versions |
| One VRAM model at a time | 12GB budget, kill-and-revive pattern |

### 5.2 Design Principles

- **Quality over speed**: User expects professional output
- **User decides**: Present options with tradeoffs, don't auto-select
- **Model-agnostic**: Backend swappable as better models emerge
- **Local-first**: Minimize cloud API costs where possible
- **Smart reuse**: Leverage AI_Mixer and BeatCanvas patterns

---

## 6. Integration Points

### 6.1 External APIs

| Service | Purpose | Required |
|---------|---------|----------|
| Anthropic Claude | Semantic analysis, scene prompts | Yes |
| OpenAI GPT-4 | Fallback LLM | Optional |
| RunPod | Cloud video generation | Optional |
| HuggingFace | Model/LoRA downloads | Optional |
| Civitai | LoRA downloads | Optional |

### 6.2 Local Models

| Model | Purpose | Size |
|-------|---------|------|
| Whisper (base) | Transcription | ~150 MB |
| Demucs (htdemucs) | Stem separation | ~1 GB |
| AnimateDiff-Lightning | Video generation | ~5 GB |
| WAN 2.6 | Video generation | ~8 GB |
| SDXL | Image/LoRA base | ~6 GB |
| sentence-transformers | Embeddings | ~100 MB |

---

## 7. Data Requirements

### 7.1 Supported Input Formats

| Type | Formats |
|------|---------|
| Audio | MP3, WAV, FLAC, M4A, OGG, AAC |
| Reference Images | PNG, JPEG, WebP |
| LoRA Weights | .safetensors, .pt |

### 7.2 Output Formats

| Type | Format | Specs |
|------|--------|-------|
| Video | MP4 (H.264 + AAC) | 1920x1080, 24fps |
| Mashup Audio | MP3 or WAV | 44.1kHz, 16-bit |
| Analysis Cache | JSON | Dataclass serialization |

### 7.3 Persistence

| Data | Storage | Lifetime |
|------|---------|----------|
| Task state | SQLite (tasks.db) | Until cleared |
| Song library | ChromaDB | Persistent |
| Analysis cache | JSON files | Persistent |
| Generated videos | File system | User-managed |

---

## 8. Success Criteria

### 8.1 Functional Success

- [ ] Generate 4-minute music video end-to-end
- [ ] All 8 mashup types produce valid audio
- [ ] Style selection affects visual output
- [ ] LoRA application changes character/style
- [ ] Real-time progress visible in UI

### 8.2 Quality Success

- [ ] Video is continuous (no slideshow frames)
- [ ] Beat synchronization perceptibly correct
- [ ] Scene transitions smooth (no jarring cuts)
- [ ] Audio-visual mood alignment

### 8.3 Performance Success

- [ ] Full pipeline < 15 minutes for 4-minute video
- [ ] No VRAM OOM crashes
- [ ] Server stable for 8+ hour sessions

---

## 9. Roadmap

### Phase 1: Foundation (Complete)
- FastAPI backend structure
- 7 router modules
- Service layer organization
- SQLite task persistence

### Phase 2: Audio Engine (Complete)
- Unified AudioAnalyzer
- Whisper integration
- Demucs stem separation
- Section detection

### Phase 3: Mashup Pipeline (Complete)
- ChromaDB integration
- 8 mashup type implementations
- Curator matching logic
- Compatibility scoring

### Phase 4: Video Engine (Complete)
- Abstract backend interface
- AnimateDiff, WAN, SVD, SDXL backends
- Model router
- Beat synchronization

### Phase 5: Frontend (Complete)
- React + TypeScript + Vite
- 5-tab interface
- WebSocket progress
- Style selection

### Phase 6: Testing (Complete)
- 409 unit tests
- 53 integration tests
- 36 frontend tests
- 498 total

### Phase 7: Nova Fade (In Progress)
- Character constitution
- Identity LoRA training (awaiting Blender renders)
- DJ video generation
- Drift testing

### Phase 8: Polish (Planned)
- Cloud backend stability
- Cost optimization
- Performance tuning
- Documentation

---

## 10. Glossary

| Term | Definition |
|------|------------|
| Camelot Wheel | System for key compatibility (e.g., 8B = C major) |
| Kill-and-Revive | Unload current model before loading next (VRAM management) |
| LoRA | Low-Rank Adaptation — efficient fine-tuning technique |
| Nova Fade | AI DJ character with defined visual identity |
| RRF | Reciprocal Rank Fusion — method for combining search results |
| Stem | Isolated audio component (vocals, drums, bass, other) |
| Trigger Token | Special token that activates a LoRA (e.g., "novafade_char") |

---

## 11. Appendix: Cost Estimates

### Per-Video Costs (4-minute, 24 scenes)

| Path | Cost |
|------|------|
| Local only | $0.05-0.10 (LLM API) |
| Cloud video | $0.75-1.50 (RunPod) |
| Mixed (local gen, cloud fallback) | $0.20-0.50 |

### Comparison to Alternatives

| Solution | Cost | Quality |
|----------|------|---------|
| Beat_Studio (local) | ~$0.10 | Professional |
| Beat_Studio (cloud) | ~$1.00 | Professional |
| BeatCanvas (original) | $2-18 | Good |
| Freelance video editor | $500+ | Variable |
| Professional production | $5,000+ | Excellent |

---

## 12. References

- `ARCHITECTURE.md` — Technical architecture details
- `CLAUDE.md` — Development commands and patterns
- `plan.md` — Full implementation plan with phase details
- `research.md` — Codebase analysis and interconnections
- `thoughts/ledgers/CONTINUITY_CLAUDE-beat-studio.md` — Session continuity
