# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
storage/store.py — ChromaDB-only citation store.

Two collections:
  - sources:   global dedup table, keyed by source_id (O(1) cache lookup)
  - citations: per-prompt records, keyed by cid

Hot path is collection.get(ids=[...]) — no vector search for reconcile.
Vector embeddings are computed by ChromaDB automatically on upsert,
which keeps /api/citations/search/{query} working.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config import cfg
from core.models import (
    CitationRecord,
    PromptCitationResult,
    Source,
    SourceType,
    compute_source_id,
)

logger = logging.getLogger(__name__)


@dataclass
class StoreConfig:
    chroma_persist_dir: str = cfg.CHROMA_PERSIST_DIR
    sources_collection: str = cfg.CHROMA_SOURCES_COLLECTION
    citations_collection: str = cfg.CHROMA_CITATIONS_COLLECTION


class CitationStore:
    """
    Unified ChromaDB store. Coordinates sources + citations collections.
    Call reconcile_sources() before store_prompt_result() to populate
    source_id fields and set source_cached flags.
    """

    def __init__(self, config: Optional[StoreConfig] = None):
        self.config = config or StoreConfig()
        self._sources = None
        self._citations = None

    async def initialize(self):
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=self.config.chroma_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._sources = client.get_or_create_collection(
            name=self.config.sources_collection,
            metadata={"description": "Global source dedup table", "hnsw:space": "cosine"},
        )
        self._citations = client.get_or_create_collection(
            name=self.config.citations_collection,
            metadata={"description": "Per-prompt citation records", "hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB initialized: {self._sources.count()} sources, "
            f"{self._citations.count()} citations"
        )

    # ── Reconcile: the heart of the cache flow ────────────────────────

    def reconcile_sources(
        self, records: list[CitationRecord], prompt_id: str
    ) -> list[CitationRecord]:
        """
        For each record: compute source_id, check if it exists in the
        sources collection, set source_cached flag, insert new sources.

        Mutates records in-place (sets source_id, source_cached) and
        returns the same list.
        """
        if not records:
            return records

        # 1. Compute source_ids
        for r in records:
            r.source_id = compute_source_id(r.doi, r.access_url, r.title, r.authors)

        unique_ids = list({r.source_id for r in records})

        # 2. Global cache lookup
        existing = self._sources.get(ids=unique_ids, include=["metadatas"])
        existing_ids: set[str] = set(existing.get("ids") or [])

        # 3. Mark cached, build new sources to insert
        new_sources: dict[str, Source] = {}
        for r in records:
            r.source_cached = r.source_id in existing_ids
            if not r.source_cached and r.source_id not in new_sources:
                canonical = (
                    f"doi:{r.doi}" if r.doi
                    else (f"url:{r.access_url}" if r.access_url else f"title:{r.title}")
                )
                new_sources[r.source_id] = Source(
                    source_id=r.source_id,
                    title=r.title,
                    canonical_ref=canonical,
                    source_type=r.source_type.value,
                    first_prompt_id=prompt_id,
                )

        # 4. Insert new sources
        if new_sources:
            srcs = list(new_sources.values())
            self._sources.upsert(
                ids=[s.source_id for s in srcs],
                documents=[s.title for s in srcs],
                metadatas=[s.to_vector_meta() for s in srcs],
            )
            logger.info(
                f"Reconciled {len(records)} citations: "
                f"{len(existing_ids)} cache hits, {len(new_sources)} new sources"
            )

        return records

    # ── Citation storage ──────────────────────────────────────────────

    def store_prompt_result(self, result: PromptCitationResult):
        """Inline (synchronous) upsert of all citations for a prompt."""
        if not result.citations:
            return
        self._citations.upsert(
            ids=[r.cid for r in result.citations],
            documents=[
                f"{r.title} | {' '.join(r.authors)} | {r.publisher or ''}"
                for r in result.citations
            ],
            metadatas=[r.to_vector_meta() for r in result.citations],
        )

    # ── Retrieval ─────────────────────────────────────────────────────

    async def get_by_prompt(self, prompt_id: str) -> Optional[dict]:
        res = self._citations.get(
            where={"prompt_id": prompt_id}, include=["metadatas"]
        )
        ids = res.get("ids") or []
        if not ids:
            return None
        return {
            "prompt_id": prompt_id,
            "citations": res.get("metadatas") or [],
        }

    async def semantic_search(self, query: str, limit: int = 20) -> list[dict]:
        res = self._citations.query(
            query_texts=[query],
            n_results=limit,
            include=["metadatas", "distances"],
        )
        out = []
        ids = res.get("ids") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]
        for i, _cid in enumerate(ids[0]):
            meta = dict(metas[0][i])
            meta["_similarity"] = 1 - dists[0][i]
            out.append(meta)
        return out

    def status(self) -> str:
        if not self._citations or not self._sources:
            return "not_initialized"
        return f"ok ({self._sources.count()} sources, {self._citations.count()} citations)"

    async def cleanup(self):
        """No-op. ChromaDB PersistentClient flushes on upsert."""
        return
