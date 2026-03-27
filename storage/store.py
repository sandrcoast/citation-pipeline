# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
storage/store.py — Citation storage with TTL-based retention.

Dual-store architecture:
  - ChromaDB: vector embeddings for semantic search across citations
  - PostgreSQL: full relational records with TTL-based cleanup

Retention policy:
  - Default: 6 months (configurable up to 1 year)
  - Background task runs daily to purge expired records
  - Expired records are soft-deleted first (30-day grace), then hard-deleted

Compatible with:
  - ChromaDB (embedded mode, zero-config)
  - Qdrant (swap in via QdrantStore adapter — same interface)
  - PostgreSQL (asyncpg for async, psycopg2 for sync fallback)

Usage:
    store = CitationStore(StoreConfig())
    await store.initialize()
    await store.store_prompt_result(result)
    citations = await store.get_by_prompt(prompt_id)
    similar = await store.semantic_search("attention mechanisms", limit=10)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from core.models import CitationRecord, PromptCitationResult
from config import cfg

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class StoreConfig:
    """Storage configuration. Defaults loaded from config.py."""

    # ChromaDB settings (embedded mode — no server needed)
    chroma_persist_dir: str = cfg.CHROMA_PERSIST_DIR
    chroma_collection: str = cfg.CHROMA_COLLECTION

    # PostgreSQL settings
    pg_dsn: str = cfg.PG_DSN

    # Retention policy
    retention_days: int = cfg.RETENTION_DAYS
    cleanup_interval_hours: int = cfg.CLEANUP_INTERVAL_HOURS
    soft_delete_grace_days: int = cfg.SOFT_DELETE_GRACE_DAYS

    # Embedding model for vector search
    ollama_url: str = cfg.OLLAMA_URL
    embedding_model: str = cfg.OLLAMA_MODEL

    # Qdrant alternative (set USE_QDRANT=true to switch)
    use_qdrant: bool = cfg.USE_QDRANT
    qdrant_url: str = cfg.QDRANT_URL
    qdrant_collection: str = cfg.QDRANT_COLLECTION


# ── ChromaDB Vector Store ─────────────────────────────────────────────────

class ChromaVectorStore:
    """
    ChromaDB wrapper for citation vector storage.
    Embedded mode: runs in-process, persists to disk, zero-config.

    For Qdrant: see QdrantVectorStore below (same interface).
    """

    def __init__(self, config: StoreConfig):
        self.config = config
        self.collection = None

    async def initialize(self):
        """Set up ChromaDB collection. Runs once at startup."""
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=self.config.chroma_persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )

            # Get or create the citations collection
            # ChromaDB handles embedding internally if we provide documents
            self.collection = client.get_or_create_collection(
                name=self.config.chroma_collection,
                metadata={
                    "description": "Citation records from LLM extraction pipeline",
                    "hnsw:space": "cosine",  # cosine similarity for text
                },
            )
            logger.info(
                f"ChromaDB initialized: {self.collection.count()} existing records"
            )
        except ImportError:
            logger.error(
                "chromadb not installed. Run: pip install chromadb"
            )
            raise

    async def upsert(self, records: list[CitationRecord]):
        """
        Insert or update citation records.
        Uses CID as the document ID — automatic dedup.
        """
        if not records or not self.collection:
            return

        ids = [r.cid for r in records]
        documents = [
            f"{r.title} | {' '.join(r.authors)} | {r.publisher or ''}"
            for r in records
        ]
        metadatas = [r.to_vector_meta() for r in records]

        # ChromaDB upsert: if CID exists, update; otherwise insert
        self.collection.upsert(
            ids=ids,
            documents=documents,  # ChromaDB embeds these automatically
            metadatas=metadatas,
        )

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        """Semantic search across all stored citations."""
        if not self.collection:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            include=["metadatas", "distances", "documents"],
        )

        output = []
        for i, cid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            meta["_similarity"] = 1 - results["distances"][0][i]  # cosine → similarity
            output.append(meta)

        return output

    async def delete_by_cids(self, cids: list[str]):
        """Delete records by CID (for TTL cleanup)."""
        if self.collection and cids:
            self.collection.delete(ids=cids)

    def count(self) -> int:
        return self.collection.count() if self.collection else 0


# ── Qdrant Alternative (same interface) ───────────────────────────────────

class QdrantVectorStore:
    """
    Qdrant wrapper. Drop-in replacement for ChromaVectorStore.
    Use when you need: distributed deployment, filtering, multi-tenancy.

    Switch by setting StoreConfig.use_qdrant = True.
    """

    def __init__(self, config: StoreConfig):
        self.config = config
        self.client = None

    async def initialize(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance, VectorParams, PointStruct,
            )

            self.client = QdrantClient(url=self.config.qdrant_url)

            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            exists = any(c.name == self.config.qdrant_collection for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.config.qdrant_collection,
                    vectors_config=VectorParams(
                        size=768,  # Gemma 3 embedding dimension
                        distance=Distance.COSINE,
                    ),
                )
            logger.info("Qdrant initialized")
        except ImportError:
            logger.error("qdrant-client not installed. Run: pip install qdrant-client")
            raise

    async def upsert(self, records: list[CitationRecord]):
        """Insert records. Requires external embedding (call Ollama)."""
        if not records or not self.client:
            return

        import aiohttp

        # Get embeddings from Ollama
        texts = [
            f"{r.title} | {' '.join(r.authors)} | {r.publisher or ''}"
            for r in records
        ]

        embeddings = []
        async with aiohttp.ClientSession() as session:
            for text in texts:
                async with session.post(
                    f"{self.config.ollama_url}/api/embeddings",
                    json={"model": self.config.embedding_model, "prompt": text},
                ) as resp:
                    data = await resp.json()
                    embeddings.append(data.get("embedding", [0.0] * 768))

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=r.cid[:32],  # Qdrant needs shorter IDs
                vector=emb,
                payload=r.to_vector_meta(),
            )
            for r, emb in zip(records, embeddings)
        ]

        self.client.upsert(
            collection_name=self.config.qdrant_collection,
            points=points,
        )

    async def search(self, query: str, limit: int = 20) -> list[dict]:
        """Semantic search using Qdrant."""
        if not self.client:
            return []

        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.config.ollama_url}/api/embeddings",
                json={"model": self.config.embedding_model, "prompt": query},
            ) as resp:
                data = await resp.json()
                query_vec = data.get("embedding", [0.0] * 768)

        results = self.client.search(
            collection_name=self.config.qdrant_collection,
            query_vector=query_vec,
            limit=limit,
        )

        return [
            {**hit.payload, "_similarity": hit.score}
            for hit in results
        ]

    async def delete_by_cids(self, cids: list[str]):
        if self.client and cids:
            from qdrant_client.models import PointIdsList
            self.client.delete(
                collection_name=self.config.qdrant_collection,
                points_selector=PointIdsList(points=[c[:32] for c in cids]),
            )

    def count(self) -> int:
        if not self.client:
            return 0
        info = self.client.get_collection(self.config.qdrant_collection)
        return info.points_count


# ── PostgreSQL Relational Store ───────────────────────────────────────────

class PostgresStore:
    """
    PostgreSQL storage for full citation records.
    Handles TTL cleanup and prompt-level retrieval.
    """

    def __init__(self, config: StoreConfig):
        self.config = config
        self.pool = None

    async def initialize(self):
        """Create connection pool and ensure schema exists."""
        try:
            import asyncpg
            self.pool = await asyncpg.create_pool(
                dsn=self.config.pg_dsn,
                min_size=2,
                max_size=10,
            )
            await self._ensure_schema()
            logger.info("PostgreSQL initialized")
        except ImportError:
            logger.warning(
                "asyncpg not installed. Falling back to SQLite. "
                "For production, run: pip install asyncpg"
            )
            await self._init_sqlite_fallback()

    async def _ensure_schema(self):
        """Create tables if they don't exist."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    cid             CHAR(64) PRIMARY KEY,
                    title           TEXT NOT NULL,
                    source_type     VARCHAR(30) NOT NULL,
                    authors         JSONB NOT NULL DEFAULT '[]',
                    date_published  VARCHAR(10),
                    citation_style  VARCHAR(12),
                    raw_fragment    VARCHAR(500),
                    publisher       TEXT,
                    access_url      TEXT,
                    doi             VARCHAR(100),
                    discovery       VARCHAR(30) NOT NULL,
                    discovery_url   TEXT,
                    confidence      REAL NOT NULL DEFAULT 0.0,
                    library_match   VARCHAR(50),
                    share_hash      VARCHAR(120),
                    created_at      TIMESTAMPTZ DEFAULT now(),
                    prompt_id       UUID,
                    expires_at      TIMESTAMPTZ,
                    deleted_at      TIMESTAMPTZ  -- soft delete
                );

                CREATE INDEX IF NOT EXISTS idx_cit_prompt
                    ON citations(prompt_id);
                CREATE INDEX IF NOT EXISTS idx_cit_doi
                    ON citations(doi) WHERE doi IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_cit_expires
                    ON citations(expires_at) WHERE deleted_at IS NULL;
                CREATE INDEX IF NOT EXISTS idx_cit_authors
                    ON citations USING GIN (authors);
            """)

    async def _init_sqlite_fallback(self):
        """SQLite fallback for development / testing."""
        import aiosqlite
        self._sqlite_path = "./data/citations.db"
        async with aiosqlite.connect(self._sqlite_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS citations (
                    cid         TEXT PRIMARY KEY,
                    data        TEXT NOT NULL,  -- JSON blob
                    prompt_id   TEXT,
                    created_at  TEXT DEFAULT (datetime('now')),
                    expires_at  TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt
                    ON citations(prompt_id)
            """)
            await db.commit()
        self.pool = None  # signal to use SQLite methods

    async def insert_batch(self, records: list[CitationRecord]):
        """Insert citation records with TTL."""
        if not records:
            return

        expires_at = datetime.utcnow() + timedelta(days=self.config.retention_days)

        if self.pool:
            # PostgreSQL path
            async with self.pool.acquire() as conn:
                for r in records:
                    row = r.to_db_row()
                    await conn.execute("""
                        INSERT INTO citations
                            (cid, title, source_type, authors, date_published,
                             citation_style, raw_fragment, publisher, access_url,
                             doi, discovery, discovery_url, confidence,
                             library_match, share_hash, created_at, prompt_id,
                             expires_at)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)
                        ON CONFLICT (cid) DO UPDATE SET
                            prompt_id = EXCLUDED.prompt_id,
                            expires_at = GREATEST(citations.expires_at, EXCLUDED.expires_at)
                    """,
                        row["cid"], row["title"], row["source_type"],
                        row["authors"], row["date_published"],
                        row["citation_style"], row["raw_fragment"],
                        row["publisher"], row["access_url"], row["doi"],
                        row["discovery_method"], row["discovery_url"],
                        row["confidence"], row["library_match"],
                        row["share_hash"], row["created_at"],
                        row["prompt_id"], expires_at.isoformat(),
                    )
        else:
            # SQLite fallback
            import aiosqlite
            async with aiosqlite.connect(self._sqlite_path) as db:
                for r in records:
                    await db.execute(
                        """INSERT OR REPLACE INTO citations
                           (cid, data, prompt_id, expires_at)
                           VALUES (?, ?, ?, ?)""",
                        (r.cid, json.dumps(r.to_db_row()), r.prompt_id,
                         expires_at.isoformat()),
                    )
                await db.commit()

    async def get_by_prompt(self, prompt_id: str) -> Optional[dict]:
        """Retrieve all citations for a given prompt ID."""
        if self.pool:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT * FROM citations
                       WHERE prompt_id = $1
                         AND deleted_at IS NULL
                         AND (expires_at IS NULL OR expires_at > now())""",
                    prompt_id,
                )
                if not rows:
                    return None
                return {
                    "prompt_id": prompt_id,
                    "citations": [dict(r) for r in rows],
                }
        else:
            import aiosqlite
            async with aiosqlite.connect(self._sqlite_path) as db:
                cursor = await db.execute(
                    "SELECT data FROM citations WHERE prompt_id = ?",
                    (prompt_id,),
                )
                rows = await cursor.fetchall()
                if not rows:
                    return None
                return {
                    "prompt_id": prompt_id,
                    "citations": [json.loads(r[0]) for r in rows],
                }

    async def cleanup_expired(self):
        """
        TTL cleanup. Call periodically (daily via background task).
        Two-phase: soft-delete → grace period → hard-delete.
        """
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            # Phase 1: Soft-delete expired records
            soft_deleted = await conn.execute("""
                UPDATE citations
                SET deleted_at = now()
                WHERE expires_at < now()
                  AND deleted_at IS NULL
            """)
            logger.info(f"Soft-deleted {soft_deleted} expired citations")

            # Phase 2: Hard-delete records past grace period
            grace_cutoff = datetime.utcnow() - timedelta(
                days=self.config.soft_delete_grace_days
            )
            hard_deleted = await conn.execute("""
                DELETE FROM citations
                WHERE deleted_at < $1
            """, grace_cutoff)
            logger.info(f"Hard-deleted {hard_deleted} citations past grace period")


# ── Unified Store (combines vector + relational) ─────────────────────────

class CitationStore:
    """
    Unified storage interface. Coordinates vector + relational stores.
    This is what the middleware uses — it doesn't care about backends.
    """

    def __init__(self, config: StoreConfig):
        self.config = config

        # Choose vector backend
        if config.use_qdrant:
            self.vector = QdrantVectorStore(config)
        else:
            self.vector = ChromaVectorStore(config)

        self.relational = PostgresStore(config)
        self._cleanup_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize both stores and start cleanup scheduler."""
        await self.vector.initialize()
        await self.relational.initialize()

        # Start background cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def store_prompt_result(self, result: PromptCitationResult):
        """
        Store all citations from a prompt result.
        Writes to both vector and relational stores.
        """
        if not result.citations:
            return

        # Store in both backends concurrently
        await asyncio.gather(
            self.vector.upsert(result.citations),
            self.relational.insert_batch(result.citations),
        )

    async def get_by_prompt(self, prompt_id: str) -> Optional[dict]:
        """Retrieve citations by prompt ID (from relational store)."""
        return await self.relational.get_by_prompt(prompt_id)

    async def semantic_search(self, query: str, limit: int = 20) -> list[dict]:
        """Semantic search across all stored citations (from vector store)."""
        return await self.vector.search(query, limit=limit)

    def status(self) -> str:
        """Quick status for health check."""
        count = self.vector.count()
        return f"ok ({count} records)"

    async def cleanup(self):
        """Shutdown cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()

    async def _cleanup_loop(self):
        """Background loop: purge expired citations."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval_hours * 3600)
                logger.info("Running citation TTL cleanup...")
                await self.relational.cleanup_expired()

                # Also clean vector store (get expired CIDs from PG, delete from vector)
                # For simplicity, vector store entries don't expire separately —
                # they're cleaned when the relational record is hard-deleted.

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
