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
from core.models import CitationRecord, DiscoveryMethod, SourceType
from core.web_fetch import FetchedPage, extract_urls, fetch_all
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

    answer, result, elapsed, fetched_sum = await _run_citation_flow(
        request.prompt, prompt_id, request.model, start
    )
    return json_utf8({
        "model": request.model,
        "response": answer,
        "_prompt_id": prompt_id,
        "_total_ms": elapsed,
        "_fetched_sources": fetched_sum,
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

    answer, result, elapsed, fetched_sum = await _run_citation_flow(
        query, prompt_id, request.model, start
    )
    return json_utf8({
        "model": request.model,
        "message": {"role": "assistant", "content": answer},
        "_prompt_id": prompt_id,
        "_total_ms": elapsed,
        "_fetched_sources": fetched_sum,
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


# ── Shared citation flow ──────────────────────────────────────────────────

async def _run_citation_flow(
    query: str,
    prompt_id: str,
    model: str,
    start: float,
) -> tuple[str, PromptCitationResult, int, list[dict]]:
    """
    Shared citation-extraction pipeline used by both /api/generate and /api/chat.
    Returns (answer, result, elapsed_ms, fetched_summary).
    """
    fetched = await _maybe_fetch(query)
    answer, records = await extractor.generate_with_citations(
        query, prompt_id, fetched_pages=fetched
    )
    fetched_records = _citations_from_fetched(fetched, prompt_id)
    existing_cids = {r.cid for r in records}
    records += [r for r in fetched_records if r.cid not in existing_cids]
    records = store.reconcile_sources(records, prompt_id)
    elapsed = int((time.monotonic() - start) * 1000)
    result = PromptCitationResult(
        prompt_id=prompt_id,
        user_query=query,
        model=model,
        citations=records,
        extraction_time_ms=elapsed,
    )
    store.store_prompt_result(result)
    return answer, result, elapsed, _fetched_summary(fetched)


# ── Web fetch helpers ─────────────────────────────────────────────────────

def _citations_from_fetched(pages: list[FetchedPage], prompt_id: str) -> list[CitationRecord]:
    """
    Build CitationRecord objects from structured page metadata.

    Two sources per page:
    1. Article-level meta (window.dataLayer / citation_* tags) → one record
       for the article itself.
    2. meta["references"] list → one record per entry in the article's own
       bibliography (raw citation text + DOI when available).
    """
    records: list[CitationRecord] = []

    for page in pages:
        meta = page.meta or {}

        # ── Article-level record ──────────────────────────────────────
        if meta.get("title"):
            authors = meta.get("authors") or []
            if isinstance(authors, str):
                authors = [authors]
            try:
                st = SourceType(meta["source_type"]) if meta.get("source_type") else SourceType.UNKNOWN
            except ValueError:
                st = SourceType.UNKNOWN
            records.append(CitationRecord(
                title=meta["title"],
                authors=authors,
                doi=meta.get("doi"),
                access_url=page.final_url or page.url,
                publisher=meta.get("publisher"),
                date_published=meta.get("date_published"),
                source_type=st,
                discovery_method=DiscoveryMethod.USER_PROVIDED,
                confidence=0.95,
                prompt_id=prompt_id,
            ))

        # ── Reference-list records ────────────────────────────────────
        for ref in meta.get("references", []):
            raw = ref.get("raw", "").strip()
            if not raw:
                continue
            doi = ref.get("doi") or None
            records.append(CitationRecord(
                title=raw[:300],
                authors=[],
                doi=doi,
                access_url=f"https://doi.org/{doi}" if doi else None,
                raw_citation_fragment=raw[:500],
                source_type=SourceType.JOURNAL_ARTICLE,
                discovery_method=DiscoveryMethod.USER_PROVIDED,
                confidence=0.85,
                prompt_id=prompt_id,
            ))

    return records


async def _maybe_fetch(text: str) -> list[FetchedPage]:
    if not cfg.WEB_FETCH_ENABLED:
        return []
    urls = extract_urls(text)
    if not urls:
        return []
    logger.info(f"Fetching {len(urls)} URL(s) from prompt")
    return await fetch_all(urls)


def _fetched_summary(pages: list[FetchedPage]) -> list[dict]:
    return [
        {
            "url": p.url,
            "final_url": p.final_url,
            "status": p.status,
            "title": p.title,
            "chars": len(p.text),
            "via_playwright": p.via_playwright,
            "refs_found": len((p.meta or {}).get("references", [])),
            "error": p.error,
        }
        for p in pages
    ]


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
