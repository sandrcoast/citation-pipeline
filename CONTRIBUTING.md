# Contributing to Citation Pipeline

Thank you for your interest in contributing! This project is licensed under
Apache 2.0 and welcomes contributions from anyone working with Ollama + Gemma 3.

## Getting Started

```bash
git clone https://github.com/sandrcoast/citation-pipeline
cd citation-pipeline

# Run tests (no dependencies needed)
PYTHONPATH=. python tests/test_core_logic.py

# Install dependencies for full stack
pip install -r requirements.txt
```

## Project Structure

- `core/models.py` — Data models. Change the schema here; all views update automatically.
- `core/extractor.py` — Extraction engine. Tune concurrency knobs in `ExtractorConfig`.
- `middleware/proxy.py` — FastAPI proxy. Transparent to Ollama clients unless `citations=true`.
- `storage/store.py` — Dual storage. Swap ChromaDB ↔ Qdrant via config flag.
- `samples/` — Test fixtures. Add more sample articles with diverse citation styles.
- `tests/` — `test_core_logic.py` runs with stdlib only. `test_pipeline.py` needs pip deps.

## How to Contribute

### Adding a New Vector DB Backend

1. Create a class implementing the same interface as `ChromaVectorStore`:
   - `async initialize()`
   - `async upsert(records: list[CitationRecord])`
   - `async search(query: str, limit: int) → list[dict]`
   - `async delete_by_cids(cids: list[str])`
   - `count() → int`
2. Add a config flag in `StoreConfig`
3. Wire it into `CitationStore.__init__`

### Adding a New Citation Style

1. Add the style to the `CitationStyle` enum in `core/models.py`
2. Add a sample article in `samples/` demonstrating the style
3. Update the extraction prompt in `core/extractor.py` if needed
4. Add expected extraction results to the sample article

### Extending the A2A Metadata

The `to_a2a_meta()` method in `CitationRecord` defines the schema.
When adding fields:
- Bump the `version` in `PromptCitationResult.to_a2a_envelope()`
- Ensure backward compatibility (new fields should have defaults)
- Document the change in a migration note

## Code Style

- Python 3.10+, type hints everywhere
- Async-first (use `async def` for I/O operations)
- Pydantic models for data validation
- Dataclasses for configuration
- Descriptive comments over clever code

## Testing

```bash
# Stdlib-only tests (always run these)
PYTHONPATH=. python tests/test_core_logic.py

# Full integration tests (needs pip dependencies)
PYTHONPATH=. pytest tests/test_pipeline.py -v
```

All PRs must pass the stdlib-only test suite at minimum.

## Reporting Issues

When filing issues, please include:
- Your Ollama version and Gemma 3 model tag
- OS
- Python version
- Whether you're using ChromaDB or Qdrant
- Relevant error output
