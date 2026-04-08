# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
config.py — Single source of truth for all project settings.

    from config import cfg
    cfg.OLLAMA_MODEL, cfg.OLLAMA_URL, cfg.CHROMA_PERSIST_DIR
"""

import os
from pathlib import Path

_HERE = Path(__file__).parent


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


class Config:
    # ── Model ─────────────────────────────────────────────────────────
    OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "gemma3:1b")
    OLLAMA_CITE_MODEL: str = _env(
        "OLLAMA_CITE_MODEL",
        _env("OLLAMA_MODEL", "gemma3:1b").replace(":", "-") + "-cite",
    )

    # ── Ollama Connection ─────────────────────────────────────────────
    OLLAMA_URL: str = _env("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT_S: int = _env_int("OLLAMA_TIMEOUT_S", 120)

    # ── Extraction Concurrency ────────────────────────────────────────
    MAX_CONCURRENT_EXTRACTIONS: int = _env_int("MAX_CONCURRENT_EXTRACTIONS", 5)

    # ── Model Context ─────────────────────────────────────────────────
    OLLAMA_NUM_CTX: int = _env_int("OLLAMA_NUM_CTX", 32768)
    OLLAMA_NUM_PREDICT: int = _env_int("OLLAMA_NUM_PREDICT", 4096)

    # ── Middleware / API ──────────────────────────────────────────────
    MIDDLEWARE_HOST: str = _env("MIDDLEWARE_HOST", "0.0.0.0")
    MIDDLEWARE_PORT: int = _env_int("MIDDLEWARE_PORT", 8000)
    MIDDLEWARE_WORKERS: int = _env_int("MIDDLEWARE_WORKERS", 4)

    # ── Storage — ChromaDB ────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = _env("CHROMA_PERSIST_DIR", str(_HERE / "data" / "chromadb"))
    CHROMA_SOURCES_COLLECTION: str = _env("CHROMA_SOURCES_COLLECTION", "sources")
    CHROMA_CITATIONS_COLLECTION: str = _env("CHROMA_CITATIONS_COLLECTION", "citations")


cfg = Config()
