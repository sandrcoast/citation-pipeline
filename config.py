# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
config.py — Single source of truth for all project settings.

Every tunable value lives here. All other modules import from this file.
To customize for your deployment, edit THIS file or override via env vars.

Usage:
    from config import cfg

    # Access any setting:
    cfg.OLLAMA_MODEL          # "gemma3:1b"
    cfg.OLLAMA_URL            # "http://localhost:11434"
    cfg.RETENTION_DAYS        # 180

    # Override via environment variables:
    #   set OLLAMA_MODEL=gemma3:4b
    #   set OLLAMA_URL=http://gpu-server:11434
    #   set RETENTION_DAYS=365
"""

import os
from pathlib import Path

_HERE = Path(__file__).parent


def _env(key: str, default: str) -> str:
    """Read from environment or use default."""
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    """Read integer from environment or use default."""
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    """Read float from environment or use default."""
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    """Read boolean from environment or use default."""
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes")


def _env_list(key: str, default: list) -> list:
    """Read comma-separated list from environment or use default."""
    raw = os.environ.get(key)
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return default


class Config:
    """
    All project configuration in one place.
    Defaults work out of the box for a single-machine Ollama setup.
    Override any value via environment variable.
    """

    # ── Model ─────────────────────────────────────────────────────────
    # The base Ollama model tag. Use the exact tag from `ollama list`.
    # Common values: "gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b"
    OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "gemma3:1b")

    # The citation-aware model name (created from Modelfile).
    # Convention: base model tag + "-cite" suffix, colons replaced with dashes.
    OLLAMA_CITE_MODEL: str = _env(
        "OLLAMA_CITE_MODEL",
        _env("OLLAMA_MODEL", "gemma3:1b").replace(":", "-") + "-cite",
    )
    # Result: "gemma3:1b" → "gemma3-1b-cite"

    # ── Ollama Connection ─────────────────────────────────────────────
    OLLAMA_URL: str = _env("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT_S: int = _env_int("OLLAMA_TIMEOUT_S", 30)

    # ── Gemma 3 Context Window ────────────────────────────────────────
    # Max input chars to send per extraction call (~4 chars ≈ 1 token).
    # gemma3:1b context = 8192 tokens → ~25,000 safe input chars.
    # Increase if using a model with larger context (e.g., 32K).
    GEMMA_MAX_INPUT_CHARS: int = _env_int("GEMMA_MAX_INPUT_CHARS", 25000)
    MAX_REFS_PER_CHUNK: int = _env_int("MAX_REFS_PER_CHUNK", 30)

    # ── Extraction ────────────────────────────────────────────────────
    MAX_TEXT_HEAD_CHARS: int = _env_int("MAX_TEXT_HEAD_CHARS", 2000)
    MAX_TEXT_TAIL_CHARS: int = _env_int("MAX_TEXT_TAIL_CHARS", 3000)

    # Concurrency: how many Ollama calls at once (GPU-bound).
    # gemma3:1b on a single GPU → 5 is safe. Multi-GPU → increase.
    MAX_CONCURRENT_EXTRACTIONS: int = _env_int("MAX_CONCURRENT_EXTRACTIONS", 5)
    MAX_CONCURRENT_FETCHES: int = _env_int("MAX_CONCURRENT_FETCHES", 20)

    # PDF fallback when HTML is truncated or missing bibliography
    ENABLE_PDF_FALLBACK: bool = _env_bool("ENABLE_PDF_FALLBACK", True)
    PDF_FETCH_TIMEOUT_S: int = _env_int("PDF_FETCH_TIMEOUT_S", 15)
    FETCH_TIMEOUT_S: int = _env_int("FETCH_TIMEOUT_S", 8)

    # ── Search ────────────────────────────────────────────────────────
    SEARXNG_URL: str = _env("SEARXNG_URL", "http://localhost:8888")

    # Optional: restrict search to specific domains (comma-separated).
    # Example: SCOPED_DOMAINS=library.myorg.edu,arxiv.org
    SCOPED_DOMAINS: list = _env_list("SCOPED_DOMAINS", [])

    # ── Middleware / API ──────────────────────────────────────────────
    MIDDLEWARE_HOST: str = _env("MIDDLEWARE_HOST", "0.0.0.0")
    MIDDLEWARE_PORT: int = _env_int("MIDDLEWARE_PORT", 8000)
    MIDDLEWARE_WORKERS: int = _env_int("MIDDLEWARE_WORKERS", 4)

    # ── Storage — PostgreSQL ──────────────────────────────────────────
    PG_DSN: str = _env(
        "PG_DSN",
        "postgresql://citation_user:citation_pass@localhost:5432/citations",
    )

    # ── Storage — Vector DB ───────────────────────────────────────────
    # ChromaDB (default, embedded — zero config)
    CHROMA_PERSIST_DIR: str = _env("CHROMA_PERSIST_DIR", str(_HERE / "data" / "chromadb"))
    CHROMA_COLLECTION: str = _env("CHROMA_COLLECTION", "citations")

    # Qdrant (alternative — set USE_QDRANT=true to switch)
    USE_QDRANT: bool = _env_bool("USE_QDRANT", False)
    QDRANT_URL: str = _env("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = _env("QDRANT_COLLECTION", "citations")

    # ── Retention ─────────────────────────────────────────────────────
    RETENTION_DAYS: int = _env_int("RETENTION_DAYS", 180)  # 6 months
    CLEANUP_INTERVAL_HOURS: int = _env_int("CLEANUP_INTERVAL_HOURS", 24)
    SOFT_DELETE_GRACE_DAYS: int = _env_int("SOFT_DELETE_GRACE_DAYS", 30)

    # ── CrossRef Enrichment ───────────────────────────────────────────
    CROSSREF_MAILTO: str = _env("CROSSREF_MAILTO", "citation-pipeline@example.com")
    CROSSREF_MAX_CONCURRENT: int = _env_int("CROSSREF_MAX_CONCURRENT", 10)
    CROSSREF_TIMEOUT_S: int = _env_int("CROSSREF_TIMEOUT_S", 5)
    CROSSREF_MIN_MATCH_SCORE: float = _env_float("CROSSREF_MIN_MATCH_SCORE", 0.75)


# ── Singleton ─────────────────────────────────────────────────────────
# Import this everywhere:  from config import cfg
cfg = Config()
