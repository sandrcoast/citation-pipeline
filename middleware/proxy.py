# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
middleware/proxy.py — FastAPI proxy between users and Ollama.

This is the main entry point. It:
  1. Receives user prompts (Ollama-compatible API format)
  2. Forwards to Ollama/Gemma 3 for the main response
  3. If citations=true, runs citation extraction in background
  4. Returns the response with citation metadata attached
  5. Stores citations for later retrieval (Art. 5)

Run:
    uvicorn middleware.proxy:app --host 0.0.0.0 --port 8000 --workers 4

For high concurrency (dozens of simultaneous prompts):
    - 4 uvicorn workers handle HTTP connections
    - Each worker shares the same asyncio event loop
    - The extractor's semaphore limits GPU load per worker
    - Total GPU load = workers × max_concurrent_extractions

Architecture note:
    This proxy is 100% transparent to existing Ollama clients.
    Without citations=true, it just forwards requests unchanged.
    With citations=true, it adds a `citation_metadata` field to responses.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.extractor import CitationExtractor, ExtractorConfig, SourceText
from core.models import PromptCitationResult
from storage.store import CitationStore, StoreConfig

logger = logging.getLogger(__name__)

# ── Global instances (initialized at startup) ─────────────────────────────

extractor: CitationExtractor = None  # type: ignore
store: CitationStore = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize extractor and store on startup, cleanup on shutdown."""
    global extractor, store

    config = ExtractorConfig()
    extractor = CitationExtractor(config)

    store_config = StoreConfig()
    store = CitationStore(store_config)
    await store.initialize()

    logger.info("Citation middleware started")
    yield

    await store.cleanup()
    logger.info("Citation middleware stopped")


app = FastAPI(
    title="Citation Extraction Middleware for Ollama",
    description="Transparent proxy that adds citation metadata to Ollama responses",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request/Response Models ───────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """
    Ollama-compatible /api/generate request, extended with citation flag.
    Existing Ollama clients work unchanged — `citations` defaults to False.
    """
    model: str = "gemma3"
    prompt: str
    system: Optional[str] = None
    stream: bool = False
    options: Optional[dict] = None

    # Citation extension — the only non-standard field
    citations: bool = Field(
        default=False,
        description="Set true to enable citation extraction for this prompt."
    )


class ChatRequest(BaseModel):
    """
    Ollama-compatible /api/chat request with citation extension.
    """
    model: str = "gemma3"
    messages: list[dict]
    stream: bool = False
    options: Optional[dict] = None
    citations: bool = False


# ── Proxy Endpoints ───────────────────────────────────────────────────────

@app.post("/api/generate")
async def proxy_generate(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    """
    Drop-in replacement for Ollama's /api/generate.
    Adds citation_metadata to response when citations=true.
    """
    prompt_id = str(uuid.uuid4())
    start = time.monotonic()

    # Step 1: Forward to Ollama (main response)
    ollama_response = await _forward_to_ollama(
        "/api/generate",
        request.model_dump(exclude={"citations"}),
    )

    if ollama_response is None:
        return JSONResponse(
            status_code=502,
            content={"error": "Ollama backend unavailable"},
        )

    # Step 2: If citations requested, extract them
    if request.citations:
        citation_result = await _run_citation_extraction(
            query=request.prompt,
            response_text=ollama_response.get("response", ""),
            prompt_id=prompt_id,
            model=request.model,
        )

        # Attach metadata to response (Art. 4: A2A-compatible)
        ollama_response["citation_metadata"] = citation_result.to_a2a_envelope()
        ollama_response["citation_user"] = citation_result.to_user_response()

        # Store in background (Art. 5: don't block the response)
        background_tasks.add_task(
            _store_citations, citation_result
        )

    elapsed = int((time.monotonic() - start) * 1000)
    ollama_response["_prompt_id"] = prompt_id
    ollama_response["_total_ms"] = elapsed

    return JSONResponse(content=ollama_response)


@app.post("/api/chat")
async def proxy_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
):
    """
    Drop-in replacement for Ollama's /api/chat.
    Same citation logic as /api/generate.
    """
    prompt_id = str(uuid.uuid4())

    ollama_response = await _forward_to_ollama(
        "/api/chat",
        request.model_dump(exclude={"citations"}),
    )

    if ollama_response is None:
        return JSONResponse(
            status_code=502,
            content={"error": "Ollama backend unavailable"},
        )

    if request.citations:
        # Extract query from last user message
        user_msgs = [m for m in request.messages if m.get("role") == "user"]
        query = user_msgs[-1]["content"] if user_msgs else ""

        response_text = ""
        if "message" in ollama_response:
            response_text = ollama_response["message"].get("content", "")

        citation_result = await _run_citation_extraction(
            query=query,
            response_text=response_text,
            prompt_id=prompt_id,
            model=request.model,
        )

        ollama_response["citation_metadata"] = citation_result.to_a2a_envelope()
        ollama_response["citation_user"] = citation_result.to_user_response()

        background_tasks.add_task(_store_citations, citation_result)

    return JSONResponse(content=ollama_response)


# ── Citation Retrieval (for A2A or later lookup) ─────────────────────────

@app.get("/api/citations/{prompt_id}")
async def get_citations(prompt_id: str):
    """
    Retrieve citation metadata for a past prompt.
    Another agent (A2A) or the user can call this within the retention period.
    """
    result = await store.get_by_prompt(prompt_id)
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Citations not found or expired"},
        )
    return JSONResponse(content=result)


@app.get("/api/citations/search/{query}")
async def search_citations(query: str, limit: int = 20):
    """
    Semantic search across all stored citations.
    Useful for: "find all citations about attention mechanisms"
    """
    results = await store.semantic_search(query, limit=limit)
    return JSONResponse(content={"results": results})


@app.get("/health")
async def health():
    """Health check — also verifies Ollama connectivity."""
    ollama_ok = await _check_ollama()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": "connected" if ollama_ok else "unreachable",
        "store": store.status() if store else "not_initialized",
    }


# ── Internal Helpers ──────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434"


async def _forward_to_ollama(path: str, payload: dict) -> Optional[dict]:
    """Forward request to Ollama backend."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_URL}{path}",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Ollama returned {resp.status}")
                    return None
                return await resp.json()
    except Exception as e:
        logger.error(f"Ollama forward failed: {e}")
        return None


async def _check_ollama() -> bool:
    """Quick connectivity check."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{OLLAMA_URL}/api/version",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def _run_citation_extraction(
    query: str,
    response_text: str,
    prompt_id: str,
    model: str,
) -> PromptCitationResult:
    """
    Full citation extraction pipeline for one prompt.
    Runs search + fetch + extract concurrently.
    """
    start = time.monotonic()

    # Extract URLs already in the LLM response
    import re
    response_urls = re.findall(r'https?://\S+', response_text)
    response_sources = [
        SourceText(url=u, text="", discovery_method="llm_training_knowledge")
        for u in response_urls[:10]
    ]

    # Run web search + extraction in parallel
    try:
        search_citations, response_citations = await asyncio.gather(
            extractor.search_and_extract(query, prompt_id),
            extractor.extract_from_texts(response_sources, prompt_id)
            if response_sources else asyncio.coroutine(lambda: [])(),
            return_exceptions=True,
        )
    except Exception as e:
        logger.error(f"Citation extraction failed: {e}")
        search_citations = []
        response_citations = []

    # Flatten, handling exceptions from gather
    all_citations = []
    for batch in [search_citations, response_citations]:
        if isinstance(batch, list):
            all_citations.extend(batch)

    # Dedup by CID
    seen = set()
    unique = []
    for c in all_citations:
        if c.cid not in seen:
            seen.add(c.cid)
            unique.append(c)

    # Enrich: resolve missing DOIs via CrossRef (async, non-blocking)
    try:
        from core.enrichment import CrossRefEnricher
        enricher = CrossRefEnricher()
        unique = await enricher.enrich_batch(unique)
    except ImportError:
        logger.debug("CrossRef enrichment not available (missing aiohttp?)")
    except Exception as e:
        logger.warning(f"CrossRef enrichment failed (non-fatal): {e}")

    elapsed = int((time.monotonic() - start) * 1000)

    return PromptCitationResult(
        prompt_id=prompt_id,
        user_query=query,
        model=model,
        citations=unique,
        extraction_time_ms=elapsed,
    )


async def _store_citations(result: PromptCitationResult):
    """Background task: store citations. Doesn't block response delivery."""
    try:
        await store.store_prompt_result(result)
        logger.info(
            f"Stored {len(result.citations)} citations for prompt={result.prompt_id}"
        )
    except Exception as e:
        logger.error(f"Failed to store citations: {e}")
