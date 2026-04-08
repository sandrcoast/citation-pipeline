# Quick Start — Windows 11 + Ollama + Gemma 3

You already have: **Windows 11, Git, Ollama (running), Gemma 3 (pulled)**.

## Step 1: Enter the project

```bash
cd citation-pipeline
```

## Step 2: Choose your model

Edit [config.py](config.py) — the first setting is:

```python
OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "gemma3:1b")
```

Match this to `ollama list`. Options: `gemma3:1b`, `gemma3:4b`, `gemma3:12b`, `gemma3:27b`.

## Step 3: Verify Ollama

```bash
ollama list
curl http://localhost:11434/api/version
```

If the model is missing: `ollama pull gemma3:1b`.

## Step 4: Install Python dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

Core packages: fastapi, uvicorn, pydantic, aiohttp, chromadb, playwright.
The `playwright install chromium` step downloads the Chromium browser used as a
fallback for JS-rendered pages (e.g. Nature, Springer). Only needed once per environment.

## Step 5: Start the middleware

```bash
set PYTHONPATH=.
uvicorn middleware.proxy:app --host 0.0.0.0 --port 8000
```

ChromaDB persists to `./data/chromadb/` — created on first write.

## Step 6: Try it

### Transparent proxy (no citations)

```bash
curl http://localhost:8000/api/generate -H "Content-Type: application/json" -d "{\"model\":\"gemma3:1b\",\"prompt\":\"What is deep learning?\"}"
```

Same behavior as hitting Ollama directly.

### With citations

```bash
curl http://localhost:8000/api/generate -H "Content-Type: application/json" -d "{\"model\":\"gemma3:1b\",\"prompt\":\"What is attention in neural networks?\",\"citations\":true}"
```

The response includes `citation_metadata` (A2A envelope) and `citation_user`
(clean list). Each citation carries `source_id` and `source_cached` so you
can tell fresh extractions from cache hits.

### With a URL in the prompt (web fetch)

```bash
curl http://localhost:8000/api/generate -H "Content-Type: application/json" -d "{\"model\":\"gemma3:1b\",\"prompt\":\"Summarize https://arxiv.org/html/2504.00358v2 and cite it.\",\"citations\":true}"
```

The middleware fetches the URL before the LLM call. The response includes a
`_fetched_sources` debug key showing what was fetched (url, status, chars, refs_found).
Set `WEB_FETCH_ENABLED=false` to disable fetching and restore pass-through behaviour.

### Retrieve by prompt ID

```bash
curl http://localhost:8000/api/citations/<prompt_id_from_response>
```

### Semantic search

```bash
curl "http://localhost:8000/api/citations/search/attention%20mechanisms"
```

### Health check

```bash
curl http://localhost:8000/health
```

### Verify ChromaDB collections manually

After sending at least one request with `citations: true`, confirm data was stored:

```bash
# Count and preview sources (global dedup table)
python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/chromadb')
col = client.get_collection('sources')
print('sources count:', col.count())
result = col.get(limit=5, include=['metadatas'])
for i, meta in enumerate(result['metadatas']):
    print(f'  [{i}]', meta.get('title', '?'), '|', meta.get('source_type', '?'))
"

# Count and preview citations (per-prompt records)
python -c "
import chromadb
client = chromadb.PersistentClient(path='./data/chromadb')
col = client.get_collection('citations')
print('citations count:', col.count())
result = col.get(limit=5, include=['metadatas'])
for i, meta in enumerate(result['metadatas']):
    print(f'  [{i}]', meta.get('title', '?'), '| prompt:', str(meta.get('prompt_id', '?'))[:8])
"
```

Both commands run against the embedded ChromaDB at `./data/chromadb/` — no server needed.

## Step 7: Interactive REPL (optional)

If crafting curl commands by hand gets tedious, use the bundled REPL client:

```bash
python app.py
```

It prints a short banner, shows current settings, and waits for your prompt.
The client builds the JSON body, **prints the exact curl command** it is about
to execute, runs it, and prints the **model's prose answer** followed by a
**compact metadata summary** and up to **three citation records**. The complete
JSON response is saved to `results/<timestamp>_<slug>.json` so the terminal
stays readable.

```
Citation Pipeline REPL
Middleware: http://localhost:8000   Model: gemma3:1b   Citations: on
Type your prompt and press Enter. Press Ctrl+C to exit.
Type /help for shortcuts.

> what are references in https://arxiv.org/html/2504.00358v2

$ curl -s -X POST http://localhost:8000/api/generate -H ... -d '{...}'

--- LLM answer ---
<the model's prose answer, printed as a clean text block>

--- response metadata ---
{
  "model": "gemma3:1b",
  "_prompt_id": "...",
  "_total_ms": 4077,
  "citation_records_count": 12,
  "_fetched_sources": [ ... ]
}

--- first 3 citation records ---
[ {...}, {...}, {...} ]

Full collection of citation records is available in following file in results/260408_201500_arxiv.orghtml25.json
```

The slug in the filename is built from the first URL in the prompt (scheme
stripped, special chars removed, first 15 chars), or from the prompt text
itself if no URL is present.

Shortcuts (prefix `/`):

| Shortcut | Effect |
|---|---|
| `/help` | print shortcut + inline-flag reference |
| `/status` | show current url / model / citations |
| `/citations on\|off` | toggle the citations flag |
| `/model <name>` | change model |
| `/url <base>` | change middleware base URL |
| `/quit` | exit (aliases: `/q`, `/exit`) |

Inline per-request flags (appended to the prompt, stripped before sending):

- `--citations:y|n`, `--cit:y|n`, `-cit:y|n` — override citations for one request.

Example: `summarize https://example.com --cit:n` sends the prompt with citations
disabled regardless of the session default.

The REPL is a thin stateless client — it does **not** manage the venv, start
the uvicorn server, touch git, or install Playwright. Start the middleware
first (Step 5), then launch the REPL in another terminal.

## What happens under the hood (citations=true)

1. Middleware extracts any URLs from the prompt and fetches them (aiohttp fast
   path; Playwright fallback for JS-rendered pages like Nature/Springer).
2. Structured metadata (title, authors, DOI, reference list) is extracted from
   each fetched page and turned into `CitationRecord` objects directly.
3. Fetched page text is injected into the prompt as a `<fetched_sources>` block
   so the LLM can cite real content.
4. Middleware makes **one** call to Ollama with the enriched prompt, asking for
   `<answer>---REFERENCES---<JSON array>`.
5. Splits the response at the `---REFERENCES---` marker. If the model used a
   non-standard header, the parser tries common variants before falling back to
   URL extraction from the references section.
6. Parses the JSON array into `CitationRecord` objects; merges with records
   built from fetched page metadata (deduped by content hash).
7. For each record, computes `source_id` (doi → url → title+authors hash).
8. Looks up `source_id`s in the ChromaDB `sources` collection. Marks existing
   ones as `source_cached=true`; inserts new ones.
9. Upserts all records into the ChromaDB `citations` collection.
10. Returns the response with `citation_records_count`, `citation_metadata`,
    `citation_user`, and `_fetched_sources` attached.

All of that is inline — no background tasks. The response you receive is
guaranteed to match what's in ChromaDB.

## Configuration

Everything in [config.py](config.py), overridable via env vars:

| Variable | Default | What it controls |
|---|---|---|
| `OLLAMA_MODEL` | `gemma3:1b` | Base model tag |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_TIMEOUT_S` | `120` | Per-call timeout |
| `MAX_CONCURRENT_EXTRACTIONS` | `5` | Parallel Ollama calls |
| `OLLAMA_NUM_CTX` | `32768` | Model context window (tokens) |
| `OLLAMA_NUM_PREDICT` | `4096` | Max output tokens |
| `CHROMA_PERSIST_DIR` | `./data/chromadb` | ChromaDB location |
| `CHROMA_SOURCES_COLLECTION` | `sources` | Source dedup collection |
| `CHROMA_CITATIONS_COLLECTION` | `citations` | Per-prompt records collection |
| `WEB_FETCH_ENABLED` | `true` | Enable/disable URL pre-fetching |
| `WEB_FETCH_TIMEOUT_S` | `15` | Per-URL fetch timeout (seconds) |
| `WEB_FETCH_MAX_URLS` | `3` | Max URLs fetched per prompt |
| `WEB_FETCH_MAX_BYTES` | `2000000` | Max response size per URL (bytes) |
| `WEB_FETCH_MAX_CHARS_PER_PAGE` | `60000` | Max extracted text per page (chars) |
| `WEB_FETCH_CONCURRENCY` | `3` | Parallel fetch limit |

## Gemma 3:1b token limits

| Setting | Value | Notes |
|---|---|---|
| Context window (`num_ctx`) | 32 768 tokens | Full native capacity; configurable via `OLLAMA_NUM_CTX` |
| Output budget (`num_predict`) | 4 096 tokens | Max tokens the model will generate; configurable via `OLLAMA_NUM_PREDICT` |
| System prompt overhead | ~400 tokens | Reserved by the pipeline |
| **Effective user query budget** | **~28 000 tokens** | ≈ 21 000 words |

> The pipeline uses **text-only** capabilities of the Gemma API.
> No vision, no multimodal inputs.

## Troubleshooting

| Problem | Fix |
|---|---|
| `connection refused :11434` | Start Ollama: `ollama serve` |
| `model not found` | `ollama pull gemma3:1b` |
| `No module named core` | `set PYTHONPATH=.` |
| `citation_records_count: 0` | The parser handles common header variants and plain URL lists automatically. For persistently empty results, try a larger model (`gemma3:4b`+). |
| Slow first request | Ollama loading model into GPU — subsequent requests are fast |

## File map

```
citation-pipeline/
├── app.py                    ← interactive REPL client (optional)
├── config.py                 ← all settings
├── core/
│   ├── extractor.py          ← single Ollama call + resilient output parser
│   ├── models.py             ← CitationRecord, Source, A2A views
│   └── web_fetch.py          ← URL fetcher, HTML→text, Playwright fallback
├── middleware/proxy.py       ← FastAPI entry point + reconcile flow
├── results/                  ← REPL saves full JSON responses here
└── storage/store.py          ← ChromaDB two-collection store
```
