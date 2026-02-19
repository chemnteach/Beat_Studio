"""Beat Studio — FastAPI application entry point.

All routers are mounted here. No dead-code routers allowed — if a router
module exists, it must be mounted in this file.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import audio, lora, mashup, nova_fade, system, tasks, video

app = FastAPI(
    title="Beat Studio",
    version="1.0.0",
    description="AI-powered music video production — audio mashups + video generation.",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
# Critical: every imported router must be mounted. No orphan routers.
app.include_router(audio.router,      prefix="/api/audio",      tags=["Audio"])
app.include_router(mashup.router,     prefix="/api/mashup",     tags=["Mashup"])
app.include_router(video.router,      prefix="/api/video",      tags=["Video"])
app.include_router(lora.router,       prefix="/api/lora",       tags=["LoRA"])
app.include_router(nova_fade.router,  prefix="/api/nova-fade",  tags=["Nova Fade"])
app.include_router(tasks.router,      prefix="/api/tasks",      tags=["Tasks"])
app.include_router(system.router,     prefix="/api/system",     tags=["System"])
