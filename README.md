# Citation Extraction Pipeline for Ollama + Gemma 3

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A lightweight, open-source middleware that intercepts Ollama prompts and
automatically extracts, normalizes, and stores academic citation metadata
from online sources. Works with any Ollama + Gemma 3 deployment — from a
single developer laptop to a multi-user server.

## Project Map (5 Articles)

| Art. | Deliverable | Files | What it does |
|------|-------------|-------|--------------|
| **1** | Pipeline Architecture | `ARCHITECTURE.md` | Conceptual flow: search → fetch → extract → store |
| **2** | Sample Article | `samples/sample_article.py` | Jane Doe article with APA, MLA, Chicago, and web citations |
| **3** | Extraction Engine | `core/extractor.py` | Async extraction with semaphore-based concurrency control |
| **4** | A2A Middleware | `middleware/proxy.py` + `core/models.py` | FastAPI proxy with A2A-compatible metadata output |
| **5** | Storage Layer | `storage/store.py` | ChromaDB + PostgreSQL with TTL-based retention (6–12 months) |

## Quick Start

### 1. Run the tests (no dependencies needed)
```bash
cd citation-pipeline
PYTHONPATH=. python tests/test_core_logic.py
```

### 2. Start the full stack
```bash
# Start Ollama, SearXNG, PostgreSQL
docker-compose up -d

# Pull and configure Gemma 3
docker exec -it citation-pipeline-ollama-1 ollama pull gemma3:1b
docker exec -it citation-pipeline-ollama-1 ollama create gemma3-1b-cite -f /modelfiles/Modelfile

# Start the middleware
pip install -r requirements.txt
uvicorn middleware.proxy:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Send a prompt with citations enabled
```bash
# Without citations (transparent proxy — same as Ollama)
curl http://localhost:8000/api/generate \
  -d '{"model":"gemma3-1b-cite","prompt":"What is attention in neural networks?"}'

# With citations (adds citation_metadata to response)
curl http://localhost:8000/api/generate \
  -d '{"model":"gemma3-1b-cite","prompt":"What is attention in neural networks?","citations":true}'
```

### 4. Retrieve citations later (A2A compatible)
```bash
# By prompt ID (returned in every response)
curl http://localhost:8000/api/citations/{prompt_id}

# Semantic search across all stored citations
curl http://localhost:8000/api/citations/search/attention%20mechanisms
```

## Architecture

```
┌─────────┐     ┌──────────────────────┐     ┌────────────┐
│  User /  │────▶│  Citation Middleware  │────▶│   Ollama   │
│  Client  │◀────│  (FastAPI :8000)      │◀────│  Gemma 3   │
└─────────┘     └──────┬───────────────┘     └────────────┘
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
        ┌─────────┐ ┌──────┐ ┌──────────┐
        │ SearXNG │ │Chroma│ │ Postgres │
        │ :8888   │ │  DB  │ │  :5432   │
        └─────────┘ └──────┘ └──────────┘
```

## Response Format (with citations=true)

```json
{
  "model": "gemma3-1b-cite",
  "response": "Attention mechanisms allow neural networks to...",
  "_prompt_id": "a1b2c3d4-...",

  "citation_metadata": {
    "schema": "citation_extraction",
    "version": "1.0",
    "prompt_id": "a1b2c3d4-...",
    "total_citations": 3,
    "citations": [
      {
        "type": "citation_record",
        "cid": "a7f3c9d1...",
        "title": "Attention Is All You Need",
        "source_type": "conference_paper",
        "authors": ["Vaswani, A.", "Shazeer, N."],
        "date_published": "2017",
        "citation_style_detected": "APA",
        "publisher": "NeurIPS",
        "doi": "10.48550/arXiv.1706.03762",
        "confidence": 0.95,
        "discovery_method": "in_page_bibliography"
      }
    ]
  },

  "citation_user": {
    "prompt_id": "a1b2c3d4-...",
    "citations": [
      {
        "title": "Attention Is All You Need",
        "type": "conference_paper",
        "authors": ["Vaswani, A.", "Shazeer, N."],
        "date": "2017",
        "publisher": "NeurIPS",
        "url": "https://doi.org/10.48550/arXiv.1706.03762"
      }
    ]
  }
}
```

## Key Design Decisions

- **Content-hash CID**: SHA-256 of (title + authors + date). Same paper = same ID everywhere. Enables dedup across prompts, users, and deployments — no org-specific IDs needed.
- **Semaphore concurrency**: GPU is the bottleneck, so Ollama calls are limited (default: 5 concurrent). URL fetching uses a larger semaphore (20) since it's I/O-bound.
- **Dual storage**: ChromaDB for semantic search (embedded, zero-config), PostgreSQL for relational queries and TTL management.
- **TTL retention**: Citations expire after 6 months (configurable up to 12 months). Soft-delete → 30-day grace → hard-delete.
- **A2A protocol**: The `citation_metadata` block follows Google's A2A spec. Another agent can parse it directly without documentation.
- **SearXNG**: Self-hosted search aggregator. No API keys, no rate limits, no vendor lock-in.
- **Transparent proxy**: Without `citations: true`, the middleware is invisible. Existing Ollama clients work unchanged.
- **Scoped domains** (optional): Pin search to specific domains (e.g., your organization's digital library). Disabled by default — searches the open web.
- **Bibliography-aware extraction**: Detects bibliography/references sections by header pattern, extracts the full section, and chunks it for Gemma 3 (30 refs per chunk by default). Handles systematic reviews with 200+ references.
- **PDF fallback**: When HTML is truncated or bibliographies are stripped by HTML extractors, automatically fetches the PDF version and extracts text via pdfplumber. Supports Springer, Elsevier, and arXiv URL patterns.
- **CrossRef enrichment**: After extraction, resolves missing DOIs via the CrossRef API (free, no API key). Most PDF bibliographies don't embed DOIs — this step fills them in.

## Tested Coverage

Validated on "Deep Learning and Neurology: A Systematic Review" (Valliani et al., 2019, Neurology and Therapy) — a systematic review with 83 Vancouver-style references:

```
Before fixes:  25/83 = 30.1%  (truncation lost 58 refs)
After fixes:   83/83 = 100.0%
```

## Swapping Vector DB

ChromaDB is the default (embedded, simplest). To switch to Qdrant:

```python
# In storage/store.py or via environment variable
StoreConfig(use_qdrant=True, qdrant_url="http://localhost:6333")
```

Both backends implement the same interface. The middleware doesn't know which one is running.

## File Structure

```
citation-pipeline/
├── LICENSE                  # Apache License 2.0
├── ARCHITECTURE.md          # Art.1: Pipeline design doc
├── README.md                # This file
├── docker-compose.yml       # Full stack orchestration
├── Dockerfile               # Middleware container
├── requirements.txt         # Python dependencies
├── core/
│   ├── models.py            # Art.4: Pydantic data models + A2A views
│   ├── extractor.py         # Art.3: Async extraction with bibliography detection + PDF fallback
│   └── enrichment.py        # CrossRef DOI enrichment (post-extraction)
├── middleware/
│   └── proxy.py             # Art.4: FastAPI proxy with A2A output
├── storage/
│   └── store.py             # Art.5: ChromaDB + PostgreSQL + TTL
├── ollama_patch/
│   └── Modelfile            # Gemma 3 citation-aware model config
├── samples/
│   └── sample_article.py    # Art.2: Test fixture with mixed citations
└── tests/
    ├── test_core_logic.py   # Stdlib-only tests (9/9 passing)
    ├── test_pipeline.py     # Full integration tests (needs dependencies)
    ├── test_springer_article.py  # Real article test (25-ref subset)
    └── test_full_83refs.py  # Full 83-ref coverage test
```
