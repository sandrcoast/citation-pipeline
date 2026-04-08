# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
core/extractor.py — Single-call citation extractor.

One Ollama call produces both the natural-language answer and a JSON
references block. No web search, no fetching, no PDF parsing — the LLM
is the source of truth for which references it used to compose the answer.

Output format (Option C — trailing JSON block):

    <free-form answer prose>
    ---REFERENCES---
    [{"title": ..., "authors": [...], "date": ..., ...}, ...]

If the marker is missing, references default to [] and the whole response
is returned as the answer (graceful degradation).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import aiohttp

from config import cfg
from core.models import (
    CitationRecord,
    CitationStyle,
    DiscoveryMethod,
    SourceType,
)

logger = logging.getLogger(__name__)


REFERENCES_MARKER = "---REFERENCES---"


SYSTEM_PROMPT = f"""You are an assistant that answers questions AND discloses every source you relied on.

Output format (STRICT — follow exactly):

1. First, write your complete natural-language answer to the user's question.
2. Then, on a new line, write the literal marker: {REFERENCES_MARKER}
3. Then, output a JSON array listing EVERY source that informed your answer.
   Include training-knowledge sources (papers, books, standards) even if you
   did not browse the web.

Each JSON item must have these fields (use null for unknown):
- "title": title of the referenced work
- "authors": array of author names (Last, First Initial format)
- "date": publication year or full date
- "source_type": one of: journal_article, book, book_chapter, conference_paper, thesis, preprint, academic_url, institutional_report, blog_nonacademic, news_article, dataset, standard_spec, unknown
- "citation_style": one of: APA, MLA, Chicago, IEEE, Vancouver, Harvard, unknown
- "publisher": journal, publisher, or conference
- "doi": DOI if known (without https://doi.org/ prefix)
- "url": access URL if known
- "raw_fragment": a short verbatim citation-style string, max 300 chars
- "confidence": 0.0-1.0 your confidence in this reference

Rules:
- The marker {REFERENCES_MARKER} must appear EXACTLY ONCE, after the full answer.
- The references block must be a JSON array. No prose, no markdown fences.
- If you used no external sources, output an empty array: []
- Do not invent sources. If unsure, omit or mark confidence low."""


@dataclass
class ExtractorConfig:
    ollama_url: str = cfg.OLLAMA_URL
    model: str = cfg.OLLAMA_MODEL
    max_concurrent: int = cfg.MAX_CONCURRENT_EXTRACTIONS
    timeout_s: int = cfg.OLLAMA_TIMEOUT_S


class CitationExtractor:
    """Single-call answer+references extractor."""

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        self._sem = asyncio.Semaphore(self.config.max_concurrent)

    async def generate_with_citations(
        self,
        user_prompt: str,
        prompt_id: str,
    ) -> tuple[str, list[CitationRecord]]:
        """
        Single Ollama call. Returns (answer_text, citation_records).
        Records have empty source_id — populated later by store.reconcile_sources.
        """
        raw = await self._call_ollama(user_prompt)
        if not raw:
            return ("", [])

        answer, refs_json = self._split_output(raw)
        records = self._parse_references(refs_json, prompt_id)
        return (answer, records)

    # ── Ollama call ───────────────────────────────────────────────────

    async def _call_ollama(self, user_prompt: str) -> Optional[str]:
        payload = {
            "model": self.config.model,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,
                "num_predict": 4096,
            },
        }
        async with self._sem:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.config.ollama_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_s),
                    ) as resp:
                        if resp.status != 200:
                            logger.error(f"Ollama returned {resp.status}")
                            return None
                        data = await resp.json()
                        return data.get("response", "") or None
            except Exception as e:
                logger.error(f"Ollama call failed: {e}")
                return None

    # ── Output splitting ──────────────────────────────────────────────

    @staticmethod
    def _split_output(raw: str) -> tuple[str, str]:
        """
        Split LLM output into (answer, references_json_text).
        Uses rsplit to tolerate the marker appearing inside the answer.
        Returns ("", "[]") on malformed output — graceful degradation.
        """
        if REFERENCES_MARKER not in raw:
            return (raw.strip(), "[]")
        parts = raw.rsplit(REFERENCES_MARKER, 1)
        answer = parts[0].strip()
        refs = parts[1].strip() if len(parts) == 2 else "[]"
        # Strip markdown code fences if present
        refs = re.sub(r"^```(?:json)?\s*", "", refs)
        refs = re.sub(r"\s*```$", "", refs).strip()
        return (answer, refs or "[]")

    # ── Reference parsing ─────────────────────────────────────────────

    def _parse_references(
        self, refs_json: str, prompt_id: str
    ) -> list[CitationRecord]:
        items: Optional[list] = None
        try:
            parsed = json.loads(refs_json)
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        items = v
                        break
                if items is None:
                    items = [parsed]
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", refs_json, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                except json.JSONDecodeError:
                    items = self._salvage_json_objects(refs_json)
            else:
                items = self._salvage_json_objects(refs_json)

        if not items or not isinstance(items, list):
            return []

        records: list[CitationRecord] = []
        seen_cids: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            rec = self._build_record(item, prompt_id)
            if rec and rec.cid not in seen_cids:
                seen_cids.add(rec.cid)
                records.append(rec)
        return records

    def _build_record(
        self, item: dict, prompt_id: str
    ) -> Optional[CitationRecord]:
        try:
            title = (item.get("title") or "").strip()
            authors_raw = item.get("authors") or []
            if not isinstance(authors_raw, list):
                authors_raw = [str(authors_raw)]
            authors = [str(a).strip() for a in authors_raw if str(a).strip()]

            raw_url = (item.get("url") or "").strip()
            raw_doi = (item.get("doi") or "").strip()
            if raw_doi.startswith("http"):
                if not raw_url:
                    raw_url = raw_doi
                raw_doi = ""
            url = raw_url or None
            doi = raw_doi or None

            publisher = (item.get("publisher") or "").strip() or None
            date_str = str(item.get("date") or "").strip() or None

            # Quality filter
            has_year = bool(date_str and re.search(r"\b(19|20)\d{2}\b", date_str))
            has_strong_signal = bool(doi or url or publisher or has_year)
            has_substantive_title = len(title) >= 10
            has_real_author = any(len(a) >= 3 for a in authors)
            if not (has_strong_signal or (has_substantive_title and has_real_author)):
                return None

            return CitationRecord(
                title=title or "Unknown",
                source_type=self._map_source_type(item.get("source_type")),
                authors=authors,
                date_published=date_str,
                citation_style_detected=self._map_citation_style(item.get("citation_style")),
                raw_citation_fragment=(item.get("raw_fragment") or "")[:500] or None,
                publisher=publisher,
                access_url=url,
                doi=doi,
                discovery_method=DiscoveryMethod.LLM_KNOWLEDGE,
                confidence=float(item.get("confidence", 0.5) or 0.5),
                prompt_id=prompt_id,
            )
        except Exception as e:
            logger.debug(f"Skipping malformed citation item: {e}")
            return None

    @staticmethod
    def _salvage_json_objects(text: str) -> list[dict]:
        """Recover {...} objects from malformed JSON."""
        out: list[dict] = []
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        try:
                            obj = json.loads(text[start:i + 1])
                            if isinstance(obj, dict):
                                out.append(obj)
                        except json.JSONDecodeError:
                            pass
                        start = -1
        return out

    @staticmethod
    def _map_source_type(raw) -> SourceType:
        try:
            return SourceType(raw) if raw else SourceType.UNKNOWN
        except ValueError:
            return SourceType.UNKNOWN

    @staticmethod
    def _map_citation_style(raw) -> Optional[CitationStyle]:
        if not raw or raw == "null":
            return None
        try:
            return CitationStyle(raw)
        except ValueError:
            return CitationStyle.UNKNOWN
