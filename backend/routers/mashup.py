"""Mashup router — ingest, library, matching, and mashup creation."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, status
from pydantic import BaseModel

router = APIRouter()


class IngestRequest(BaseModel):
    source: str  # file path or YouTube URL


class MashupCreateRequest(BaseModel):
    song_a_id: str
    song_b_id: str
    mashup_type: str = "classic"


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest_song(request: IngestRequest) -> Dict[str, str]:
    """Ingest a song from a local file or YouTube URL."""
    return {"task_id": "stub", "status": "queued", "source": request.source}


@router.post("/ingest/batch", status_code=status.HTTP_202_ACCEPTED)
async def ingest_batch(sources: List[str]) -> Dict[str, Any]:
    """Batch ingest songs from a list of sources."""
    return {"task_id": "stub", "count": len(sources), "status": "queued"}


@router.get("/library")
async def list_library() -> Dict[str, Any]:
    """List all songs in the ChromaDB library."""
    return {"songs": [], "total": 0}


@router.get("/library/search")
async def search_library(q: str) -> Dict[str, Any]:
    """Search the library by semantic query."""
    return {"results": [], "query": q}


@router.post("/match")
async def match_songs(song_id: str, criteria: str = "hybrid", top: int = 5) -> Dict[str, Any]:
    """Find compatible matches for a song."""
    return {"song_id": song_id, "matches": [], "criteria": criteria}


@router.post("/create", status_code=status.HTTP_202_ACCEPTED)
async def create_mashup(request: MashupCreateRequest) -> Dict[str, str]:
    """Create a mashup from two songs."""
    return {"task_id": "stub", "status": "queued", "type": request.mashup_type}


@router.get("/types")
async def list_mashup_types() -> Dict[str, Any]:
    """List all 8 mashup types with descriptions."""
    types = [
        {"name": "classic", "description": "Vocal from A + instrumental from B"},
        {"name": "stem_swap", "description": "Mix stems from 3+ songs"},
        {"name": "energy_match", "description": "Align high-energy sections"},
        {"name": "adaptive_harmony", "description": "Auto-fix key clashes via pitch-shifting"},
        {"name": "theme_fusion", "description": "Filter sections by lyrical themes"},
        {"name": "semantic_aligned", "description": "Question→answer pairing"},
        {"name": "role_aware", "description": "Vocals shift between lead/harmony/call/response"},
        {"name": "conversational", "description": "Songs talk to each other like a dialogue"},
    ]
    return {"types": types}
