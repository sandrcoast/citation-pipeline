# Citation Extraction Pipeline for Ollama + Gemma 3

## Overview

This pipeline sits as a **middleware proxy** between users and an Ollama instance
running Gemma 3. It intercepts every prompt/response cycle and, when citations
are enabled, performs lightweight extraction on discovered online sources.

## Architecture

```
┌─────────┐     ┌──────────────────────┐     ┌────────────┐
│  User /  │────▶│  Citation Middleware  │────▶│   Ollama   │
│  Client  │◀────│  (Python / FastAPI)   │◀────│  Gemma 3   │
└─────────┘     └──────┬───────────────┘     └────────────┘
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
        ┌─────────┐ ┌──────┐ ┌──────────┐
        │ Search  │ │Vector│ │  RDBMS   │
        │ (SearX) │ │  DB  │ │(Postgres)│
        └─────────┘ └──────┘ └──────────┘
```

## Pipeline Steps (per prompt with citations=true)

### Phase 1: Query + Search (synchronous, blocking)
1. User sends prompt with `"citations": true` header/flag
2. Middleware forwards prompt to Ollama/Gemma 3
3. Gemma 3 generates response (may include URLs, claims)
4. Middleware also fires parallel web search via SearXNG
   to gather source URLs related to the prompt

### Phase 2: Source Fetching (async, parallel)
5. Collect all URLs from:
   - Gemma 3's response text (regex URL extraction)
   - SearXNG search results (top 10)
   - Optional domain-scoped search (configurable via `scoped_domains`)
6. Fetch each URL concurrently (aiohttp, 5s timeout, max 20 concurrent)
7. Extract text content (trafilatura for HTML, pdfplumber for PDFs)

### Phase 3: Citation Extraction (async, batched)
8. For each fetched page, send a LIGHTWEIGHT extraction prompt to Gemma 3:
   - System: "Extract all citations/references. Return JSON array."
   - Input: first 3000 chars of page text (enough for bibliographies)
   - Cost: ~200-400 tokens per source
9. Parse JSON responses, normalize fields
10. Compute content-hash CID for each citation

### Phase 4: Dedup + Match + Store (deterministic, no LLM)
11. Dedup by CID across all sources in this prompt
12. Optionally match against a local catalog (by DOI or fuzzy title)
13. Store in vector DB (ChromaDB/Qdrant) + RDBMS
14. Attach citation metadata to response

### Phase 5: Response Delivery
15. Return to user:
    - Original Gemma 3 response
    - `citation_metadata` block (A2A-compatible JSON)
    - `prompt_id` for later retrieval

## Key Design Choices

- **SearXNG** for search: self-hosted, no API keys, federated
- **ChromaDB** for vector storage: embedded, zero-config, Python-native
- **PostgreSQL** for RDBMS: reliable, widely available, excellent JSON support
- **FastAPI** for middleware: async-native, OpenAPI docs auto-generated
- **trafilatura** for HTML→text: fast, accurate, handles messy pages
- **Gemma 3** for extraction: runs on same Ollama instance, no external API

## Concurrency Model

```
User Prompt ──┬──▶ Ollama (main response)     ← sequential
              └──▶ SearXNG (parallel search)   ← async
                     │
                     ▼
              Fetch N URLs (asyncio.gather)     ← concurrent, semaphore=20
                     │
                     ▼
              Extract citations (batch to Ollama) ← concurrent, semaphore=5
                     │
                     ▼
              Dedup + Store (sync, fast)         ← deterministic
```

For high concurrency (many simultaneous users), the middleware runs
behind uvicorn with multiple workers. Citation extraction calls to Ollama
are rate-limited by a semaphore to avoid overwhelming the GPU.
