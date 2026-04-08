# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
middleware/proxy.py — FastAPI entry point.

Flow (citations=true):
  1. Single Ollama call → (answer_text, [CitationRecord])
  2. Reconcile sources against ChromaDB (global cache check + insert new)
  3. Upsert citations into ChromaDB (inline, before response)
  4. Return {response, citation_metadata (A2A), citation_user}

Flow (citations=false): transparent proxy — forwards to Ollama unchanged.

Run:
    uvicorn middleware.proxy:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

from config import cfg
from core.extractor import CitationExtractor, ExtractorConfig
from core.models import PromptCitationResult
from storage.store import CitationStore, StoreConfig

logger = logging.getLogger(__name__)


extractor: CitationExtractor = None  # type: ignore
store: CitationStore = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global extractor, store
    extractor = CitationExtractor(ExtractorConfig())
    store = CitationStore(StoreConfig())
    await store.initialize()
    logger.info("Citation middleware started")
    yield
    await store.cleanup()


app = FastAPI(
    title="Citation Extraction Middleware for Ollama",
    description="LLM-driven single-call citation extraction with ChromaDB cache",
    version="0.2.0",
    lifespan=lifespan,
)


def json_utf8(content: dict, status_code: int = 200) -> Response:
    return Response(
        content=json.dumps(content, ensure_ascii=False, default=str),
        media_type="application/json",
        status_code=status_code,
    )


# ── Request models ────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    model: str = cfg.OLLAMA_MODEL
    prompt: str
    stream: bool = False
    options: Optional[dict] = None
    citations: bool = Field(default=False)


class ChatRequest(BaseModel):
    model: str = cfg.OLLAMA_MODEL
    messages: list[dict]
    stream: bool = False
    options: Optional[dict] = None
    citations: bool = False


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/generate")
async def generate(request: GenerateRequest):
    prompt_id = str(uuid.uuid4())
    start = time.monotonic()

    # Transparent pass-through when citations disabled
    if not request.citations:
        forwarded = await _forward_to_ollama(
            "/api/generate", request.model_dump(exclude={"citations"})
        )
        if forwarded is None:
            return JSONResponse(status_code=502, content={"error": "Ollama unavailable"})
        forwarded["_prompt_id"] = prompt_id
        return json_utf8(forwarded)

    # Single-call extraction
    answer, records = await extractor.generate_with_citations(
        request.prompt, prompt_id
    )

    # Reconcile sources against global cache + insert new (inline)
    records = store.reconcile_sources(records, prompt_id)

    elapsed = int((time.monotonic() - start) * 1000)
    result = PromptCitationResult(
        prompt_id=prompt_id,
        user_query=request.prompt,
        model=request.model,
        citations=records,
        extraction_time_ms=elapsed,
    )

    # Store citations inline — response reflects what's in the DB
    store.store_prompt_result(result)

    return json_utf8({
        "model": request.model,
        "response": answer,
        "_prompt_id": prompt_id,
        "_total_ms": elapsed,
        "citation_metadata": result.to_a2a_envelope(),
        "citation_user": result.to_user_response(),
    })


@app.post("/api/chat")
async def chat(request: ChatRequest):
    prompt_id = str(uuid.uuid4())
    start = time.monotonic()

    if not request.citations:
        forwarded = await _forward_to_ollama(
            "/api/chat", request.model_dump(exclude={"citations"})
        )
        if forwarded is None:
            return JSONResponse(status_code=502, content={"error": "Ollama unavailable"})
        forwarded["_prompt_id"] = prompt_id
        return json_utf8(forwarded)

    user_msgs = [m for m in request.messages if m.get("role") == "user"]
    query = user_msgs[-1]["content"] if user_msgs else ""

    answer, records = await extractor.generate_with_citations(query, prompt_id)
    records = store.reconcile_sources(records, prompt_id)

    elapsed = int((time.monotonic() - start) * 1000)
    result = PromptCitationResult(
        prompt_id=prompt_id,
        user_query=query,
        model=request.model,
        citations=records,
        extraction_time_ms=elapsed,
    )
    store.store_prompt_result(result)

    return json_utf8({
        "model": request.model,
        "message": {"role": "assistant", "content": answer},
        "_prompt_id": prompt_id,
        "_total_ms": elapsed,
        "citation_metadata": result.to_a2a_envelope(),
        "citation_user": result.to_user_response(),
    })


@app.get("/api/citations/{prompt_id}")
async def get_citations(prompt_id: str):
    result = await store.get_by_prompt(prompt_id)
    if result is None:
        return json_utf8({"error": "Citations not found"}, status_code=404)
    return json_utf8(result)


@app.get("/api/citations/search/{query}")
async def search_citations(query: str, limit: int = 20):
    results = await store.semantic_search(query, limit=limit)
    return json_utf8({"results": results})


@app.get("/health")
async def health():
    ollama_ok = await _check_ollama()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": "connected" if ollama_ok else "unreachable",
        "store": store.status() if store else "not_initialized",
    }


# ── Ollama helpers (transparent proxy path) ───────────────────────────────

async def _forward_to_ollama(path: str, payload: dict) -> Optional[dict]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{cfg.OLLAMA_URL}{path}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=cfg.OLLAMA_TIMEOUT_S),
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
    except Exception as e:
        logger.error(f"Ollama forward failed: {e}")
        return None


async def _check_ollama() -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{cfg.OLLAMA_URL}/api/version",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                return resp.status == 200
    except Exception:
        return False
