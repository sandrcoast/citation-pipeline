# Quick Start — Windows 11 + Ollama + Gemma 3

You already have: **Windows 11, Git, GitHub, Claude Code, Ollama (running), Gemma 3 (pulled)**

## Step 1: Clone and enter the project

Open your terminal (PowerShell, CMD, or Git Bash):

```bash
git clone https://github.com/YOUR-ORG/citation-pipeline.git
cd citation-pipeline
```

Or if you downloaded the `.tar.gz` from our chat:

```bash
tar -xzf citation-pipeline.tar.gz
cd citation-pipeline
```

## Step 2: Choose your model (one-time setup)

Open `config.py` in any editor. The first setting is your model:

```python
OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "gemma3:1b")
```

This is the **single source of truth** — every other file reads from here.
Change `"gemma3:1b"` to match the model you have installed:

| Tag | Size | GPU RAM | Quality |
|-----|------|---------|---------|
| `gemma3:1b` | 1B params | ~2 GB | Good for testing, fast |
| `gemma3:4b` | 4B params | ~4 GB | Solid extraction quality |
| `gemma3:12b` | 12B params | ~10 GB | High accuracy |
| `gemma3:27b` | 27B params | ~20 GB | Best extraction quality |

The default `gemma3:1b` works on virtually any GPU.

## Step 3: Verify Ollama is running with your model

```bash
ollama list
```

You should see `gemma3:1b` in the list. If not:

```bash
ollama pull gemma3:1b
```

Verify it responds:

```bash
curl http://localhost:11434/api/generate -d "{\"model\":\"gemma3:1b\",\"prompt\":\"hello\",\"stream\":false}"
```

## Step 4: Create the citation-aware model

The Modelfile wraps your base model with a citation-extraction system prompt.
Before building, check that the `FROM` line in `ollama_patch/Modelfile` matches
your chosen model in `config.py`:

```
FROM gemma3:1b
Hint: Default folder for Ollama models in Windows 11:
C:\Users\<Username>\.ollama
```

Then build:

```bash
ollama create gemma3-1b-cite -f ollama_patch/Modelfile 

```

> **Note on naming:** The cite model name is auto-derived in `config.py` as
> `OLLAMA_CITE_MODEL`. For `gemma3:1b` it becomes `gemma3-1b-cite`.
> If you changed your base model, update both `config.py` and the `FROM` line
> in `ollama_patch/Modelfile`, then rebuild with the matching name.

## Step 5: Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, aiohttp, trafilatura, chromadb, pdfplumber, and friends.

> **Note:** PostgreSQL is optional. Without it, the pipeline auto-falls back
> to SQLite (zero config, works immediately). You can add Postgres later.

## Step 6: Run the tests (no Ollama needed)

```bash
set PYTHONPATH=.
python tests/test_core_logic.py
```

Expected: `9/9 passed — ALL PASSED ✓`

Full article coverage test:

```bash
python tests/test_full_83refs.py
```

Expected: `Coverage: 83/83 = 100.0%`

## Step 7: Start the middleware

```bash
set PYTHONPATH=.
uvicorn middleware.proxy:app --host 0.0.0.0 --port 8000
```

The middleware is now running at `http://localhost:8000`.
It proxies all requests to Ollama at `http://localhost:11434`.

## Step 8: Try it

### Normal prompt (no citations — transparent proxy):

```bash
curl http://localhost:8000/api/generate -d "{\"model\":\"gemma3-1b-cite\",\"prompt\":\"What is deep learning?\",\"stream\":false}"
```

This behaves exactly like calling Ollama directly.

### Prompt with citations enabled:

```bash
curl http://localhost:8000/api/generate -d "{\"model\":\"gemma3-1b-cite\",\"prompt\":\"What is deep learning?\",\"stream\":false,\"citations\":true}"
```

The response now includes two extra fields:
- `citation_metadata` — structured A2A envelope (for bots)
- `citation_user` — clean list (for humans)

### Retrieve citations later:

```bash
curl http://localhost:8000/api/citations/PROMPT_ID_FROM_RESPONSE
```

### Search across all stored citations:

```bash
curl http://localhost:8000/api/citations/search/attention%20mechanisms
```

### Health check:

```bash
curl http://localhost:8000/health
```

## What's happening under the hood

```
Your prompt
    |
    v
+-------------------------+
|  localhost:8000          |  <-- you talk to this
|  Citation Middleware     |
|  (FastAPI)              |
+--------+----------------+
         |
    +----+----+
    v         v
 Ollama    SearXNG <- optional, see below
 :11434    :8888
```

1. Your prompt goes to the middleware
2. Middleware forwards it to Ollama/Gemma 3 (normal response)
3. If `citations: true`, middleware also:
   - Searches the web (via SearXNG, if running)
   - Fetches source pages (HTML or PDF)
   - Finds the bibliography section
   - Chunks it for Gemma 3 extraction
   - Deduplicates by content hash
   - Enriches missing DOIs via CrossRef
   - Stores in ChromaDB + SQLite
4. Returns the response with citation metadata attached

## Configuration: config.py

All settings live in `config.py`. Every module reads from it.
You can override any value via environment variables:

```bash
set OLLAMA_MODEL=gemma3:4b
set OLLAMA_URL=http://gpu-server:11434
set RETENTION_DAYS=365
set SEARXNG_URL=http://localhost:8888
set PG_DSN=postgresql://user:pass@localhost:5432/citations
set USE_QDRANT=true
uvicorn middleware.proxy:app --host 0.0.0.0 --port 8000
```

Key settings at a glance:

| Variable | Default | What it controls |
|----------|---------|-----------------|
| `OLLAMA_MODEL` | `gemma3:1b` | Base model tag |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `MAX_CONCURRENT_EXTRACTIONS` | `5` | Parallel GPU calls (lower = safer) |
| `RETENTION_DAYS` | `180` | How long citations are kept |
| `ENABLE_PDF_FALLBACK` | `true` | Fetch PDF when HTML is truncated |
| `SEARXNG_URL` | `http://localhost:8888` | Web search (optional) |
| `PG_DSN` | `postgresql://...` | Postgres connection (fallback: SQLite) |
| `USE_QDRANT` | `false` | Use Qdrant instead of ChromaDB |
| `CROSSREF_MAILTO` | `citation-pipeline@example.com` | Your email for CrossRef polite pool |

## Optional: Add SearXNG for web search

Without SearXNG, the pipeline extracts citations from URLs that Gemma 3
mentions in its response. With SearXNG, it also actively searches the web.

Easiest way (Docker):

```bash
docker run -d -p 8888:8080 searxng/searxng
```

The middleware auto-detects SearXNG at `localhost:8888`.

## Optional: Add PostgreSQL for production storage

Without Postgres, citations are stored in SQLite at `./data/citations.db`.
This works fine for testing and single-user setups.

For multi-user / production:

```bash
docker run -d -p 5432:5432 -e POSTGRES_DB=citations -e POSTGRES_USER=citation_user -e POSTGRES_PASSWORD=citation_pass postgres:16-alpine
```

Then set the environment variable before starting the middleware:

```bash
set PG_DSN=postgresql://citation_user:citation_pass@localhost:5432/citations
uvicorn middleware.proxy:app --host 0.0.0.0 --port 8000
```

## Optional: Use Qdrant instead of ChromaDB

ChromaDB runs embedded (in-process, zero config). To switch to Qdrant:

```bash
docker run -d -p 6333:6333 qdrant/qdrant
pip install qdrant-client
```

Then:

```bash
set USE_QDRANT=true
set QDRANT_URL=http://localhost:6333
uvicorn middleware.proxy:app --host 0.0.0.0 --port 8000
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `connection refused :11434` | Start Ollama: `ollama serve` |
| `model not found` | Pull it: `ollama pull gemma3:1b` |
| `pip install fails` | Try: `pip install -r requirements.txt --break-system-packages` |
| `uvicorn not found` | `pip install uvicorn[standard]` |
| `No module named core` | Set PYTHONPATH: `set PYTHONPATH=.` (Windows) |
| Citations empty | SearXNG not running (optional) — Gemma 3 still extracts from its own response URLs |
| Slow first request | Ollama loading model into GPU memory — subsequent requests are fast |
| Wrong model name | Check `config.py` OLLAMA_MODEL matches your `ollama list` output |

## File quick reference

```
citation-pipeline/
├── config.py               <- EDIT THIS FIRST: all settings in one place
├── ollama_patch/Modelfile  <- match FROM tag to config.py, then: ollama create ...
├── middleware/proxy.py     <- the server: uvicorn middleware.proxy:app ...
├── core/extractor.py       <- extraction engine (bibliography detection, PDF, chunking)
├── core/models.py          <- data model (CitationRecord, views, A2A envelope)
├── core/enrichment.py      <- CrossRef DOI lookup (post-extraction)
├── storage/store.py        <- ChromaDB + Postgres/SQLite + TTL cleanup
└── tests/                  <- run these first to verify everything works
```
