"""Mashup library backed by ChromaDB — ported from AI_Mixer mixer/memory/.

Single module combining client, schema, and query operations.
ChromaDB pinned at 0.4.22 (never upgrade without testing).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("beat_studio.mashup.memory")

try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMA = True
except ImportError:  # pragma: no cover
    _HAS_CHROMA = False
    chromadb = None  # type: ignore[assignment]


# ── exceptions ────────────────────────────────────────────────────────────────

class SchemaError(Exception):
    """Schema validation failure."""


class MemoryError(Exception):
    """ChromaDB operation failure."""


# ── schema helpers ────────────────────────────────────────────────────────────

_REQUIRED_FIELDS = ("source", "path", "artist", "title", "sample_rate")


def sanitize_id(artist: str, title: str) -> str:
    """Generate a ChromaDB-safe song ID from artist and title.

    Rules:
    - Lowercase, spaces → underscores
    - Remove all chars except a-z 0-9 _
    - Collapse consecutive underscores
    - Strip leading/trailing underscores
    - Max 128 characters
    """
    if not artist or not title:
        raise SchemaError("Artist and title cannot be empty")
    combined = f"{artist}_{title}".lower().replace(" ", "_")
    combined = re.sub(r"[^a-z0-9_]", "", combined)
    combined = re.sub(r"_+", "_", combined).strip("_")
    if not combined:
        raise SchemaError(
            f"Sanitization produced empty ID for artist='{artist}', title='{title}'"
        )
    return combined[:128]


def validate_metadata(metadata: Dict[str, Any]) -> None:
    """Validate metadata before ChromaDB insertion.

    Raises:
        SchemaError: If required fields are missing or values out of range.
    """
    for field in _REQUIRED_FIELDS:
        if field not in metadata:
            raise SchemaError(f"Missing required field: '{field}'")

    bpm = metadata.get("bpm")
    if bpm is not None and not (20 <= bpm <= 300):
        raise SchemaError(f"BPM {bpm} out of range [20, 300]")

    for field in ("irony_score", "energy_level", "valence"):
        val = metadata.get(field)
        if val is not None and not (0 <= val <= 10):
            raise SchemaError(f"{field} {val} out of range [0, 10]")


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metadata to ChromaDB-compatible flat dict (no lists/nested dicts)."""
    flat: Dict[str, Any] = {}
    for key, val in metadata.items():
        if isinstance(val, list):
            flat[key] = json.dumps(val)
        elif isinstance(val, dict):
            flat[key] = json.dumps(val)
        else:
            flat[key] = val
    return flat


def _deserialize_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Restore JSON-serialized list/dict fields back to Python objects."""
    result: Dict[str, Any] = {}
    for key, val in raw.items():
        if isinstance(val, str) and val.startswith(("[", "{")):
            try:
                result[key] = json.loads(val)
            except json.JSONDecodeError:
                result[key] = val
        else:
            result[key] = val
    return result


# ── MashupLibrary ─────────────────────────────────────────────────────────────

class MashupLibrary:
    """ChromaDB-backed music library for mashup candidate storage and retrieval.

    Usage::

        lib = MashupLibrary()
        song_id = lib.upsert_song("Taylor Swift", "Shake It Off", metadata, transcript)
        matches = lib.query_hybrid(target_song_id=song_id, max_results=5)
    """

    COLLECTION_NAME = "tiki_library"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
    ):
        if not _HAS_CHROMA:  # pragma: no cover
            raise MemoryError("chromadb not installed — run: pip install chromadb==0.4.22")
        if persist_directory is None:
            persist_directory = "backend/data/library_cache/chroma"
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._make_embedding_fn(),
        )
        logger.info("MashupLibrary ready: collection=%s dir=%s",
                    collection_name, persist_directory)

    # ── internal ──────────────────────────────────────────────────────────────

    def _make_embedding_fn(self):
        try:
            from chromadb.utils import embedding_functions
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.EMBEDDING_MODEL
            )
        except Exception as exc:
            logger.warning("SentenceTransformer unavailable: %s — falling back to hash embedding", exc)
            return _HashEmbeddingFunction()

    # ── public CRUD ───────────────────────────────────────────────────────────

    def upsert_song(
        self,
        artist: str,
        title: str,
        metadata: Dict[str, Any],
        transcript: str = "",
        force_id: Optional[str] = None,
    ) -> str:
        """Insert or update a song. Returns the song_id used."""
        validate_metadata(metadata)
        song_id = force_id or sanitize_id(artist, title)

        if "date_added" not in metadata:
            metadata = dict(metadata)
            metadata["date_added"] = datetime.utcnow().isoformat()

        mood = metadata.get("mood_summary", "")
        document = f"{transcript}\n\n[MOOD]: {mood}".strip()

        flat_meta = _serialize_metadata(metadata)

        self._collection.upsert(
            ids=[song_id],
            documents=[document],
            metadatas=[flat_meta],
        )
        logger.info("Upserted: %s (%s — %s)", song_id, artist, title)
        return song_id

    def get_song(self, song_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a song by ID. Returns None if not found."""
        try:
            result = self._collection.get(
                ids=[song_id],
                include=["metadatas", "documents"],
            )
            if not result["ids"]:
                return None
            return {
                "id": result["ids"][0],
                "metadata": _deserialize_metadata(result["metadatas"][0]),
                "document": result["documents"][0],
            }
        except Exception as exc:
            raise MemoryError(f"get_song failed: {exc}") from exc

    def delete_song(self, song_id: str) -> bool:
        """Delete a song. Returns True if deleted, False if not found."""
        existing = self._collection.get(ids=[song_id])
        if not existing["ids"]:
            return False
        self._collection.delete(ids=[song_id])
        return True

    def list_all(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Return all songs in the library."""
        result = self._collection.get(
            include=["metadatas"],
            limit=limit,
        )
        songs = []
        for song_id, meta in zip(result["ids"], result["metadatas"]):
            songs.append({"id": song_id, "metadata": _deserialize_metadata(meta)})
        return songs

    def count(self) -> int:
        """Return total number of songs in the library."""
        return self._collection.count()

    def close(self) -> None:
        """Release ChromaDB resources."""
        try:
            self._client = None  # type: ignore[assignment]
        except Exception:
            pass

    # ── query operations ──────────────────────────────────────────────────────

    def query_harmonic(
        self,
        target_bpm: float,
        target_camelot: str,
        bpm_tolerance: float = 0.05,
        max_results: int = 10,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find songs by BPM + Camelot key compatibility.

        Returns list of {id, metadata, compatibility_score} dicts.
        """
        exclude_ids = exclude_ids or []
        all_songs = self.list_all()
        compatible = []

        bpm_low = target_bpm * (1 - bpm_tolerance)
        bpm_high = target_bpm * (1 + bpm_tolerance)
        compatible_keys = _get_compatible_camelot_keys(target_camelot)

        for song in all_songs:
            if song["id"] in exclude_ids:
                continue
            meta = song["metadata"]
            bpm = meta.get("bpm", 0) or 0
            camelot = meta.get("camelot", "")
            if bpm_low <= bpm <= bpm_high and camelot in compatible_keys:
                dist = _camelot_distance(target_camelot, camelot)
                bpm_diff = abs(bpm - target_bpm) / target_bpm
                score = 1.0 - (dist * 0.1) - (bpm_diff * 0.5)
                compatible.append({**song, "compatibility_score": round(score, 3)})

        compatible.sort(key=lambda x: x["compatibility_score"], reverse=True)
        return compatible[:max_results]

    def query_semantic(
        self,
        mood_summary: str,
        max_results: int = 10,
        exclude_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find semantically similar songs using ChromaDB vector search."""
        exclude_ids = exclude_ids or []
        try:
            results = self._collection.query(
                query_texts=[mood_summary],
                n_results=max_results + len(exclude_ids),
                include=["metadatas", "distances"],
            )
            output = []
            for song_id, meta, dist in zip(
                results["ids"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                if song_id in exclude_ids:
                    continue
                output.append({
                    "id": song_id,
                    "metadata": _deserialize_metadata(meta),
                    "compatibility_score": round(1.0 - float(dist), 3),
                })
            return output[:max_results]
        except Exception as exc:
            logger.warning("query_semantic failed: %s", exc)
            return []

    def query_hybrid(
        self,
        target_song_id: str,
        max_results: int = 5,
        bpm_tolerance: float = 0.05,
        harmonic_weight: float = 0.6,
        semantic_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """Hybrid matching: 60% harmonic + 40% semantic via RRF.

        Args:
            target_song_id: The song we're matching against.
            max_results: Number of results to return.
        """
        target = self.get_song(target_song_id)
        if target is None:
            raise MemoryError(f"Song not found: {target_song_id}")

        meta = target["metadata"]
        bpm = meta.get("bpm", 120.0) or 120.0
        camelot = meta.get("camelot", "8B") or "8B"
        mood = meta.get("mood_summary", "") or ""

        harmonic = self.query_harmonic(
            target_bpm=bpm,
            target_camelot=camelot,
            bpm_tolerance=bpm_tolerance,
            max_results=max_results * 3,
            exclude_ids=[target_song_id],
        )
        semantic = self.query_semantic(
            mood_summary=mood,
            max_results=max_results * 3,
            exclude_ids=[target_song_id],
        )

        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        k = 60

        for rank, song in enumerate(harmonic):
            rrf_scores[song["id"]] = rrf_scores.get(song["id"], 0) + harmonic_weight / (k + rank + 1)
        for rank, song in enumerate(semantic):
            rrf_scores[song["id"]] = rrf_scores.get(song["id"], 0) + semantic_weight / (k + rank + 1)

        # Merge metadata
        id_to_song: Dict[str, Dict[str, Any]] = {}
        for song in harmonic + semantic:
            id_to_song[song["id"]] = song

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for song_id, score in ranked[:max_results]:
            if song_id in id_to_song:
                result = dict(id_to_song[song_id])
                result["compatibility_score"] = round(score, 4)
                results.append(result)
        return results


# ── Hash-based fallback embedding (used when sentence_transformers absent) ────

class _HashEmbeddingFunction:
    """Deterministic 384-dim embedding from SHA256 hash. Test-only quality."""

    DIM = 384

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        import hashlib
        import math
        import struct
        results = []
        needed = self.DIM * 4  # bytes needed
        for text in input:
            h = hashlib.sha256(text.encode()).digest()
            repeats = math.ceil(needed / len(h))
            raw = (h * repeats)[:needed]
            floats = list(struct.unpack(f"{self.DIM}f", raw))
            # Normalise to unit vector
            norm = sum(x * x for x in floats) ** 0.5 or 1.0
            results.append([f / norm for f in floats])
        return results


# ── Camelot wheel helpers ─────────────────────────────────────────────────────

def _get_compatible_camelot_keys(camelot: str) -> List[str]:
    """Return Camelot keys compatible with the given key (±1 + same + relative)."""
    if not camelot or len(camelot) < 2:
        return []
    try:
        num = int(camelot[:-1])
        mode = camelot[-1]  # "A" or "B"
    except (ValueError, IndexError):
        return [camelot]

    compatible = [camelot]
    for delta in (-1, 1):
        adjacent = ((num - 1 + delta) % 12) + 1
        compatible.append(f"{adjacent}{mode}")
    # Relative major/minor
    other_mode = "B" if mode == "A" else "A"
    compatible.append(f"{num}{other_mode}")
    return compatible


def _camelot_distance(key1: str, key2: str) -> int:
    """Return circular distance between two Camelot keys (0 = identical)."""
    if key1 == key2:
        return 0
    try:
        num1, mode1 = int(key1[:-1]), key1[-1]
        num2, mode2 = int(key2[:-1]), key2[-1]
    except (ValueError, IndexError):
        return 6  # maximum distance
    if mode1 != mode2:
        return 3  # mode switch penalty
    diff = abs(num1 - num2)
    return min(diff, 12 - diff)
