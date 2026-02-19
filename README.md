# Beat Studio

AI-powered music video production. Upload a song (or create a mashup), pick a visual style, and get a continuous professional-quality video — not a slideshow.

## Two Workflows

**Path A — Original song to video**
Upload audio → AI analysis (BPM, key, lyrics, mood, sections) → auto-generate scene prompts → pick animation style → generate continuous video synced to the music.

**Path B — Mashup to video**
Use the 8-type mashup engine (semantic matching, stem separation, ChromaDB) to blend two songs → generate a music video for the mashup, or...

**Path B2 — DJ video with Nova Fade**
Take any mashup → Nova Fade (your AI DJ character) performs it on camera, beat-synced, with programmatically enforced character identity.

---

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| CUDA GPU | Optional (cloud fallback available) |
| ffmpeg | Any recent version |
| VRAM | 6 GB min / 12 GB recommended |

---

## Quick Start

### 1. Clone and install backend

```bash
git clone <repo-url>
cd Beat_Studio

# Create Python environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or: venv\Scripts\activate  (Windows)

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.template .env
# Edit .env and add your API keys (optional but recommended):
#   ANTHROPIC_API_KEY=your_key   (Claude for semantic analysis)
#   OPENAI_API_KEY=your_key      (fallback LLM)
#   RUNPOD_API_KEY=your_key      (cloud video generation)
```

### 3. Run setup checker

```bash
python scripts/setup_check.py
```

This checks Python, system tools, GPU, installed models, and API keys.
On first run it will report that some models need downloading — that's expected.

### 4. Download required models

```bash
# Whisper (audio transcription)
python -c "import whisper; whisper.load_model('base')"

# Sentence Transformers (ChromaDB embeddings)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

AnimateDiff-Lightning and SDXL Base are likely already in your HuggingFace cache if you've used diffusers before. Check with:

```bash
python scripts/setup_check.py --models
```

### 5. Install frontend

```bash
cd frontend
npm install
cd ..
```

### 6. Start the application

```bash
# Terminal 1 — FastAPI backend
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — React frontend
cd frontend && npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Architecture

```
React Frontend (Vite + TypeScript)
  └── 5 tabs: Upload & Song | Mashup Workshop | Video Studio | LoRA Manager | System Status
      ├── WebSocket progress tracking
      └── REST API calls

FastAPI Backend (Python)
  ├── /api/audio     — Upload, analyze, transcribe
  ├── /api/mashup    — ChromaDB library, matching, 8 mashup types
  ├── /api/video     — Execution planning, video generation, scene editing
  ├── /api/lora      — Registry, training, HuggingFace/Civitai downloads
  ├── /api/nova-fade — Nova Fade DJ character pipeline
  ├── /api/tasks     — Background task queue + WebSocket progress
  └── /api/system    — GPU status, model inventory, health check

Services
  ├── Audio Engine     — librosa analysis, Whisper transcription, Demucs stems
  ├── Mashup Pipeline  — 8 mashup types, ChromaDB 0.4.22, semantic matching
  ├── Prompt Engine    — Narrative analysis, scene prompts, 24 animation styles
  ├── Video Engine     — 8 backends (AnimateDiff, WAN 2.6, SDXL, SVD, cloud)
  ├── LoRA Engine      — Registry, SDXL training, drift detection
  └── Nova Fade        — Character constitution, canonical images, DJ timeline
```

---

## Video Backends

| Backend | Type | VRAM | Quality | Cost |
|---------|------|------|---------|------|
| AnimateDiff-Lightning | Local | 5.6 GB | Good | Free |
| SDXL + ControlNet | Local | 8 GB | Good (rotoscope) | Free |
| SVD (image-to-video) | Local | 7.5 GB | Good | Free |
| WAN 2.6 (local) | Local | 12 GB | Excellent | Free |
| WAN 2.6 (RunPod) | Cloud | — | Excellent | ~$0.05/scene |
| SkyReels DF (RunPod) | Cloud | — | Seamless stitching | ~$0.03/scene |

The system selects the best available backend automatically and presents you with local vs cloud options before generating.

**Hard rules:**
- No Ken Burns effects — every frame is generated or interpolated
- No Blender — all video via AI generation models
- No slideshows — continuous video with smooth scene transitions

---

## Animation Styles (24 available)

| Category | Styles |
|----------|--------|
| Traditional | Cel Animation, Watercolor, Ink Wash, Pencil Sketch |
| Modern/Digital | Motion Graphics, Collage, Pixel Art, Low-Poly 3D, Isometric |
| Stylized Realism | Rotoscope, Oil Painting, Comic Book, Pop Art, Ukiyo-e |
| Abstract | Synthwave, Graffiti, Art Deco, Impressionist, Psychedelic |
| Photorealistic | Photorealistic, Cinematic |

The system recommends styles based on genre, mood, and energy. You can override.

---

## Mashup Types (8 available)

| Type | Description |
|------|-------------|
| Classic | Vocal from A + instrumental from B |
| Stem Swap | Mix stems from 3+ songs (drums/bass/vocals/other) |
| Energy Match | Align high-energy sections dynamically |
| Adaptive Harmony | Auto-fix key clashes via pitch-shifting |
| Theme Fusion | Filter sections by lyrical themes |
| Semantic-Aligned | Question→answer, narrative→reflection pairing |
| Role-Aware | Vocals shift between lead/harmony/call/response/texture |
| Conversational | Songs talk to each other like a dialogue |

---

## Nova Fade DJ Character

Nova Fade is an AI DJ character with a programmatically enforced identity constitution.

- **5 expressions**: neutral confident, mischievous grin, focused intensity, delighted joy, drop anticipation
- **5 gestures**: left/right deck scratch, crossfader tap, downbeat head nod, spotlight presentation
- **4 themes**: sponsor neon, award elegant, mashup chaos, chill lofi
- **Drift detection**: CLIP-based identity regression (S_id ≥ 0.75, S_face ≥ 0.70)

To set up Nova Fade, use the LoRA Manager tab to train the identity LoRA from canonical reference images.

---

## LoRA Management

Beat Studio has a full LoRA lifecycle:

1. **Registry** — `backend/config/loras.yaml` tracks all available LoRAs
2. **Recommender** — given a song analysis and style, suggests LoRAs that would help
3. **Downloader** — fetch from HuggingFace or Civitai directly from the UI
4. **Trainer** — train new SDXL LoRAs from your own image datasets (rank 16, adamw8bit)
5. **Drift testing** — CLIP-based regression to detect style/character drift

---

## Testing

```bash
# Backend unit tests (409 tests)
python -m pytest backend/tests/unit/ -q

# Backend integration tests (53 tests)
python -m pytest backend/tests/integration/ -q

# All backend tests
python -m pytest backend/tests/ -q

# Frontend tests (36 tests)
cd frontend && npm test -- --run
```

**Total test count: 498** (462 backend + 36 frontend)

---

## Project Structure

```
Beat_Studio/
├── backend/
│   ├── main.py                    # FastAPI app, 7 routers
│   ├── routers/                   # audio, mashup, video, lora, nova_fade, tasks, system
│   ├── services/
│   │   ├── audio/                 # AudioAnalyzer, stem detection, Whisper
│   │   ├── mashup/                # 8-type mashup pipeline (from AI_Mixer)
│   │   ├── prompt/                # Narrative analysis, scene prompts, style mapper
│   │   ├── video/
│   │   │   ├── backends/          # 8 video backends (AnimateDiff, WAN 2.6, SDXL, SVD...)
│   │   │   ├── beat_sync.py       # Beat-aligned scene planning
│   │   │   ├── transition_engine.py
│   │   │   ├── assembler.py
│   │   │   └── encoder.py
│   │   ├── lora/                  # Registry, trainer, downloader, recommender
│   │   ├── nova_fade/             # Character constitution, canonical prompts, DJ generator
│   │   └── shared/                # Config, VRAM manager, task manager, hardware detector
│   ├── config/
│   │   ├── settings.yaml          # Main config
│   │   ├── checkpoints.yaml       # Model inventory
│   │   ├── animation_styles.yaml  # 24 animation styles
│   │   ├── loras.yaml             # LoRA registry
│   │   └── nova_fade_constitution.yaml
│   └── tests/
│       ├── unit/                  # 409 unit tests
│       └── integration/           # 53 integration tests
├── frontend/
│   └── src/
│       ├── App.tsx                # 5-tab root
│       ├── components/            # 14 components
│       ├── hooks/                 # useWebSocket, useTaskPolling
│       └── types/                 # TypeScript types
├── output/
│   ├── videos/                   # Final rendered videos
│   ├── loras/                    # Trained LoRA files
│   └── nova_fade/                # Canonical images, drift runs, datasets
├── scripts/
│   └── setup_check.py            # Environment validation script
├── requirements.txt
└── pyproject.toml
```

---

## Configuration

`backend/config/settings.yaml` controls all tuneable parameters:

```yaml
audio:
  sample_rate: 44100
  default_format: wav

mashup:
  chroma_collection: tiki_library
  bpm_tolerance: 0.05
  weight_bpm: 0.35
  weight_key: 0.30
  weight_energy: 0.20
  weight_genre: 0.15

video:
  default_quality: professional    # basic | professional | cinematic
  local_preferred: true
  default_backend: animatediff

lora:
  registry_path: config/loras.yaml
```

---

## Constraints

- **ChromaDB pinned at 0.4.22** — do not upgrade (breaking changes)
- **One model in VRAM at a time** — kill-and-revive pattern
- **No Ken Burns** — explicitly forbidden
- **No Blender** — all video via AI generation
- **Nova Fade constitution** — enforced programmatically; changes require version increment

---

## Legal

For personal/educational use. Users are responsible for:
- Having rights to any audio processed
- Complying with HuggingFace and Civitai terms of service
- Not distributing copyrighted content without permission
