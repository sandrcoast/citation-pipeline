# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
core/models.py — Shared data models for the citation pipeline.

Two entities:
  - Source: a unique referenced work (deduplicated globally across all prompts)
  - CitationRecord: one citation instance tied to a prompt, linked to a Source
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from config import cfg


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
    LLM_KNOWLEDGE = "llm_training_knowledge"
    USER_PROVIDED = "user_provided"


# ── Source (global dedup table) ───────────────────────────────────────────

def compute_source_id(
    doi: Optional[str],
    url: Optional[str],
    title: str,
    authors: list[str],
) -> str:
    """
    Stable source_id: prefer DOI, fall back to normalized URL,
    fall back to title+authors hash. Deterministic across prompts.
    """
    if doi and doi.strip():
        canonical = f"doi:{doi.strip().lower()}"
    elif url and url.strip():
        norm = re.sub(r"^https?://(www\.)?", "", url.strip().lower()).rstrip("/")
        canonical = f"url:{norm}"
    else:
        norm_title = title.strip().lower()
        norm_authors = "|".join(sorted(a.strip().lower() for a in authors))
        canonical = f"title:{norm_title}||{norm_authors}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class Source(BaseModel):
    """
    A unique referenced work. Deduplicated globally — one row per
    real-world source no matter how many prompts cite it.
    """
    source_id: str
    title: str
    canonical_ref: str  # doi or url or title-based fallback
    source_type: str = SourceType.UNKNOWN.value
    first_prompt_id: Optional[str] = None
    first_seen_at: datetime = Field(default_factory=datetime.utcnow)

    def to_vector_meta(self) -> dict:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "canonical_ref": self.canonical_ref,
            "source_type": self.source_type,
            "first_prompt_id": self.first_prompt_id or "",
            "first_seen_at": self.first_seen_at.isoformat(),
        }


# ── Core Citation Record ──────────────────────────────────────────────────

class CitationRecord(BaseModel):
    """
    One citation instance tied to a specific prompt.
    `cid` is the content hash (auto-computed).
    `source_id` links to the global Source (populated by store.reconcile_sources).
    """

    cid: str = Field(default="", description="SHA-256 content hash. Auto-computed.")
    source_id: str = Field(default="", description="FK to Source. Populated during reconcile.")

    # Core bibliographic fields
    title: str
    source_type: SourceType = SourceType.UNKNOWN
    authors: list[str] = Field(default_factory=list)
    date_published: Optional[str] = None
    citation_style_detected: Optional[CitationStyle] = None

    raw_citation_fragment: Optional[str] = Field(default=None, max_length=500)

    publisher: Optional[str] = None
    access_url: Optional[str] = None
    doi: Optional[str] = None

    discovery_method: DiscoveryMethod = DiscoveryMethod.LLM_KNOWLEDGE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    source_cached: bool = Field(
        default=False,
        description="True if source_id existed in ChromaDB before this prompt.",
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    prompt_id: Optional[str] = None

    def model_post_init(self, __context) -> None:
        if not self.cid:
            self.cid = self._compute_cid()

    def _compute_cid(self) -> str:
        norm_title = self.title.strip().lower()
        norm_authors = "|".join(sorted(a.strip().lower() for a in self.authors))
        norm_date = (self.date_published or "").strip()
        payload = f"{norm_title}||{norm_authors}||{norm_date}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── View projections ──────────────────────────────────────────────

    def to_vector_meta(self) -> dict:
        """Flat dict for ChromaDB metadata."""
        return {
            "cid": self.cid,
            "source_id": self.source_id,
            "title": self.title,
            "source_type": self.source_type.value,
            "authors_json": json.dumps(self.authors, ensure_ascii=False),
            "date_published": self.date_published or "",
            "citation_style": self.citation_style_detected.value if self.citation_style_detected else "",
            "publisher": self.publisher or "",
            "access_url": self.access_url or "",
            "doi": self.doi or "",
            "discovery_method": self.discovery_method.value,
            "confidence": self.confidence,
            "prompt_id": self.prompt_id or "",
            "created_at": self.created_at.isoformat(),
        }

    def to_user_view(self) -> dict:
        return {
            "title": self.title,
            "type": self.source_type.value,
            "authors": self.authors,
            "date": self.date_published,
            "style": self.citation_style_detected.value if self.citation_style_detected else None,
            "publisher": self.publisher,
            "url": self.access_url,
            "doi": self.doi,
        }

    def to_a2a_meta(self) -> dict:
        """A2A-compatible metadata block."""
        return {
            "type": "citation_record",
            "version": "1.0",
            "cid": self.cid,
            "source_id": self.source_id,
            "source_cached": self.source_cached,
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


# ── Prompt-level wrapper ─────────────────────────────────────────────────

class PromptCitationResult(BaseModel):
    """Wraps all citations extracted for a single user prompt."""
    prompt_id: str
    user_query: str
    model: str = cfg.OLLAMA_MODEL
    citations: list[CitationRecord] = Field(default_factory=list)
    extraction_time_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_a2a_envelope(self) -> dict:
        return {
            "schema": "citation_extraction",
            "version": "1.0",
            "prompt_id": self.prompt_id,
            "user_query": self.user_query,
            "model": self.model,
            "total_citations": len(self.citations),
            "extraction_time_ms": self.extraction_time_ms,
            "citations": [c.to_a2a_meta() for c in self.citations],
            "timestamp": self.created_at.isoformat(),
        }

    def to_user_response(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "citations": [c.to_user_view() for c in self.citations],
        }
