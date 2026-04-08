# Contributing to Citation Pipeline

Thank you for your interest in contributing! This project is licensed under
Apache 2.0 and welcomes contributions from anyone working with Ollama + Gemma 3.

## Getting Started

```bash
git clone https://github.com/sandrcoast/citation-pipeline
cd citation-pipeline
pip install -r requirements.txt
```

Start the middleware:

```bash
set PYTHONPATH=.
uvicorn middleware.proxy:app --host 0.0.0.0 --port 8000
```

## Project Structure

- `app.py` — Interactive REPL client. Stateless thin wrapper over curl; does not manage server/venv/git.
- `config.py` — All settings. Single source of truth for Ollama + ChromaDB + web-fetch config.
- `core/models.py` — Data models. Change the schema here; all views update automatically.
- `core/extractor.py` — Single-call Ollama extractor + output parser.
- `core/web_fetch.py` — URL fetcher (aiohttp + Playwright fallback), HTML→text, structured metadata extraction.
- `middleware/proxy.py` — FastAPI entry point + inline reconcile flow.
- `storage/store.py` — ChromaDB two-collection store (sources + citations).

## How to Contribute

### Adding a New Citation Style

1. Add the style to the `CitationStyle` enum in `core/models.py`
2. Update the extraction prompt in `core/extractor.py` if needed

### Extending the A2A Metadata

The `to_a2a_meta()` method in `CitationRecord` defines the schema.
When adding fields:
- Bump the `version` in `PromptCitationResult.to_a2a_envelope()`
- Ensure backward compatibility (new fields should have defaults)
- Document the change in a migration note

### Modifying the Storage Layer

`CitationStore` in `storage/store.py` exposes:
- `reconcile_sources(records, prompt_id)` — cache lookup + insert new sources
- `store_prompt_result(result)` — upsert citations for a prompt
- `get_by_prompt(prompt_id)` — retrieve by prompt
- `semantic_search(query, limit)` — full-text search over stored citations

## Code Style

- Python 3.10+, type hints everywhere
- Async-first (use `async def` for I/O operations)
- Pydantic models for data validation
- Dataclasses for configuration
- Descriptive comments over clever code

## Reporting Issues

When filing issues, please include:
- Your Ollama version and Gemma 3 model tag
- OS and Python version
- Relevant error output
