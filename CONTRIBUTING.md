# Contributing to Citation Pipeline

Apache 2.0 — contributions welcome.

## Getting Started

```bash
git clone https://github.com/sandrcoast/citation-pipeline
cd citation-pipeline
pip install -r requirements.txt
set PYTHONPATH=.
uvicorn middleware.proxy:app --host 0.0.0.0 --port 8000
```

## Project Structure

| File | Role |
|---|---|
| `app.py` | Interactive REPL client. Stateless; drives the middleware over curl and saves full JSON responses to `results/`. |
| `config.py` | All settings. Single source of truth for Ollama, ChromaDB, and web-fetch config. |
| `core/models.py` | `CitationRecord`, `Source`, `compute_source_id`, A2A envelope views. Change the schema here. |
| `core/extractor.py` | Single-call Ollama extractor. Builds the enriched prompt, calls Ollama, and parses the references block with graceful fallbacks for non-standard model output. |
| `core/web_fetch.py` | URL fetcher — aiohttp fast path, Playwright fallback for JS-rendered pages, HTML→text, structured metadata extraction. |
| `middleware/proxy.py` | FastAPI entry point. Orchestrates fetch → extract → reconcile and returns the A2A response envelope. |
| `storage/store.py` | ChromaDB store. `sources` collection for global dedup; `citations` collection for per-prompt records. |

## How to Contribute

### Adding a Citation Style

1. Add the style to the `CitationStyle` enum in `core/models.py`.
2. Update the extraction prompt in `core/extractor.py` if needed.

### Extending the A2A Metadata

The `to_a2a_meta()` method in `CitationRecord` defines the schema.
When adding fields:
- Bump the `version` in `PromptCitationResult.to_a2a_envelope()`
- New fields must have defaults (backward compatibility)

### Modifying the Storage Layer

`CitationStore` in `storage/store.py` exposes:
- `reconcile_sources(records, prompt_id)` — cache lookup + insert new sources
- `store_prompt_result(result)` — upsert citations for a prompt
- `get_by_prompt(prompt_id)` — retrieve by prompt
- `semantic_search(query, limit)` — full-text search over stored citations

## Architectural Invariants

- **The LLM is the sole producer of `citation_records`.** Middleware fetches,
  enriches the prompt, parses the LLM's output, and stores the result. It
  never synthesizes a `CitationRecord` from scraped HTML, page metadata, or
  any other non-LLM source. This is what makes the pipeline A2A-compatible:
  the next agent in a chain reproduces the same behaviour purely by being
  prompted the same way.
- **Single-pass LLM calls only** — `temperature=0.0`, no retry loops, no
  multi-pass extraction.

## Code Style

- Python 3.10+, type hints throughout
- Async-first — use `async def` for all I/O
- Pydantic models for data, dataclasses for configuration

## Reporting Issues

Please include:
- Ollama version and Gemma 3 model tag (`ollama list`)
- OS and Python version
- Relevant error output or the saved `results/*.json` file
