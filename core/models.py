# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
core/models.py — Shared data models for the citation pipeline.

These Pydantic models define the canonical schema. Every component
(extractor, middleware, storage) imports from here. Change once,
propagates everywhere.

Compatible with:
  - ChromaDB metadata (flat dict via .to_vector_meta())
  - Qdrant payload  (flat dict via .to_vector_meta())
  - PostgreSQL      (via .to_db_row())
  - A2A protocol    (via .to_a2a_meta())
  - User-facing     (via .to_user_view())
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE_PAPER = "conference_paper"
    THESIS = "thesis"
    PREPRINT = "preprint"
    ACADEMIC_URL = "academic_url"
    INSTITUTIONAL_REPORT = "institutional_report"
    BLOG_NONACADEMIC = "blog_nonacademic"
    NEWS_ARTICLE = "news_article"
    DATASET = "dataset"
    STANDARD_SPEC = "standard_spec"
    UNKNOWN = "unknown"


class CitationStyle(str, Enum):
    APA = "APA"
    MLA = "MLA"
    CHICAGO = "Chicago"
    IEEE = "IEEE"
    VANCOUVER = "Vancouver"
    HARVARD = "Harvard"
    UNKNOWN = "unknown"


class DiscoveryMethod(str, Enum):
    WEB_SEARCH = "web_search_result"
    BIBLIOGRAPHY = "in_page_bibliography"
    FOOTNOTE = "in_page_footnote"
    INLINE = "in_page_inline_citation"
    FETCHED_DOC = "fetched_document_refs"
    LLM_KNOWLEDGE = "llm_training_knowledge"
    USER_PROVIDED = "user_provided"


# ── Core Citation Record ──────────────────────────────────────────────────

class CitationRecord(BaseModel):
    """
    One extracted citation. Immutable after creation.
    The `cid` (content ID) is auto-computed from title + authors + date.
    """

    # Identity — auto-computed, not set by caller
    cid: str = Field(default="", description="SHA-256 content hash. Auto-computed.")

    # Core bibliographic fields
    title: str
    source_type: SourceType = SourceType.UNKNOWN
    authors: list[str] = Field(default_factory=list)
    date_published: Optional[str] = None  # ISO date or year-only
    citation_style_detected: Optional[CitationStyle] = None

    # Raw extraction data (debug / engineer only)
    raw_citation_fragment: Optional[str] = Field(
        default=None, max_length=500,
        description="Verbatim text flagged as citation. Max 500 chars."
    )

    # Publishing / access info
    publisher: Optional[str] = None
    access_url: Optional[str] = None
    doi: Optional[str] = None

    # Provenance
    discovery_method: DiscoveryMethod = DiscoveryMethod.WEB_SEARCH
    discovery_source_url: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Optional: local catalog match (set after extraction by matching logic)
    # Use this to link citations to your own catalog/database if you have one.
    library_match_id: Optional[str] = None

    # Sharing (computed on demand)
    share_hash: Optional[str] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    prompt_id: Optional[str] = None

    def model_post_init(self, __context) -> None:
        """Auto-compute cid after all fields are set."""
        if not self.cid:
            self.cid = self._compute_cid()

    def _compute_cid(self) -> str:
        """
        Deterministic content hash: SHA-256(normalized title + sorted authors + date).
        Same paper always gets the same CID regardless of who extracts it.
        """
        norm_title = self.title.strip().lower()
        norm_authors = "|".join(sorted(a.strip().lower() for a in self.authors))
        norm_date = (self.date_published or "").strip()
        payload = f"{norm_title}||{norm_authors}||{norm_date}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compute_share_hash(self, deployment_salt: str) -> str:
        """HMAC-based hash for cross-deployment sharing. No internal IDs exposed."""
        import hmac
        self.share_hash = hmac.new(
            deployment_salt.encode(), self.cid.encode(), hashlib.sha256
        ).hexdigest()
        return self.share_hash

    # ── View projections ──────────────────────────────────────────────

    def to_vector_meta(self) -> dict:
        """
        Flat dict for ChromaDB / Qdrant metadata.
        Only indexed, searchable fields. No large text blobs.
        """
        return {
            "cid": self.cid,
            "title": self.title,
            "source_type": self.source_type.value,
            "authors_json": json.dumps(self.authors),  # ChromaDB needs flat values
            "date_published": self.date_published or "",
            "citation_style": self.citation_style_detected.value if self.citation_style_detected else "",
            "publisher": self.publisher or "",
            "doi": self.doi or "",
            "discovery_method": self.discovery_method.value,
            "confidence": self.confidence,
            "prompt_id": self.prompt_id or "",
        }

    def to_user_view(self) -> dict:
        """
        Clean dict shown to humans. No hashes, no debug fields.
        """
        return {
            "title": self.title,
            "type": self.source_type.value,
            "authors": self.authors,
            "date": self.date_published,
            "style": self.citation_style_detected.value if self.citation_style_detected else None,
            "publisher": self.publisher,
            "url": self.access_url,
        }

    def to_a2a_meta(self) -> dict:
        """
        A2A (Agent-to-Agent) protocol compatible metadata block.
        Follows the Google A2A spec: structured, typed, JSON-serializable.
        Includes enough info for another agent to verify or re-fetch.
        """
        return {
            "type": "citation_record",
            "version": "1.0",
            "cid": self.cid,
            "title": self.title,
            "source_type": self.source_type.value,
            "authors": self.authors,
            "date_published": self.date_published,
            "citation_style_detected": (
                self.citation_style_detected.value if self.citation_style_detected else None
            ),
            "publisher": self.publisher,
            "access_url": self.access_url,
            "doi": self.doi,
            "discovery_method": self.discovery_method.value,
            "confidence": self.confidence,
            "prompt_id": self.prompt_id,
            "extracted_at": self.created_at.isoformat(),
        }

    def to_db_row(self) -> dict:
        """Full record for PostgreSQL INSERT. All fields."""
        return {
            "cid": self.cid,
            "title": self.title,
            "source_type": self.source_type.value,
            "authors": json.dumps(self.authors),
            "date_published": self.date_published,
            "citation_style": (
                self.citation_style_detected.value if self.citation_style_detected else None
            ),
            "raw_fragment": self.raw_citation_fragment,
            "publisher": self.publisher,
            "access_url": self.access_url,
            "doi": self.doi,
            "discovery_method": self.discovery_method.value,
            "discovery_url": self.discovery_source_url,
            "confidence": self.confidence,
            "library_match": self.library_match_id,
            "share_hash": self.share_hash,
            "created_at": self.created_at.isoformat(),
            "prompt_id": self.prompt_id,
        }


# ── Prompt-level wrapper ─────────────────────────────────────────────────

class PromptCitationResult(BaseModel):
    """
    Wraps all citations extracted for a single user prompt.
    This is what gets attached to the LLM response and stored.
    """
    prompt_id: str
    user_query: str
    model: str = "gemma3"
    citations: list[CitationRecord] = Field(default_factory=list)
    extraction_time_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_a2a_envelope(self) -> dict:
        """
        Full A2A-compatible envelope. Another bot can parse this directly.
        Attach this to the LLM response metadata.
        """
        return {
            "schema": "citation_extraction",
            "version": "1.0",
            "prompt_id": self.prompt_id,
            "model": self.model,
            "total_citations": len(self.citations),
            "extraction_time_ms": self.extraction_time_ms,
            "citations": [c.to_a2a_meta() for c in self.citations],
            "timestamp": self.created_at.isoformat(),
        }

    def to_user_response(self) -> dict:
        """Attached to user-facing API response."""
        return {
            "prompt_id": self.prompt_id,
            "citations": [c.to_user_view() for c in self.citations],
        }
