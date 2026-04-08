# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
core/extractor.py — Two-call citation extractor.

Call 1 — /api/generate, temperature=TEMPERATURE_USER_PROMPT (default 0.6):
    Prose answer only. System prompt instructs the model to answer without
    emitting any references block — that keeps call 1 focused and avoids
    spending output budget on structured JSON that would be produced at a
    higher temperature anyway.

Call 2 — /api/chat, temperature=TEMPERATURE_CITATION_SYSTEM_PROMPT (default 0.0):
    Citation extraction. Sends the full three-message history
    [user: enriched_prompt | assistant: answer | user: citation request]
    and receives the structured ---REFERENCES--- JSON block at temperature 0.
    Deterministic extraction over the model's own answer text.

No web search, no fetching, no PDF parsing — the LLM is the source of truth
for which references it used to compose the answer.
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
from core.web_fetch import FetchedPage

logger = logging.getLogger(__name__)


REFERENCES_MARKER = "---REFERENCES---"

# Matches plain URLs in a text reference list (e.g. markdown links, bare URLs).
# Used as a last resort when the model emits a URL list instead of JSON.
_BARE_URL_RE = re.compile(r"https?://[^\s<>()\[\]\"'`]+", re.IGNORECASE)

# Secondary header patterns tolerated when the primary marker is absent.
# Matches lines like "**REFERENCES:**", "## References", "Sources:", etc.
# Uses loose [#*]* prefix and [*:]* suffix to handle markdown bold/heading variants.
_SECONDARY_REF_RE = re.compile(
    r"(?:^|\n)[ \t]*[#*]*[ \t]*(?:REFERENCES?|Sources?|Bibliography)[*:]*[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)


# ── System prompt for call 1 (prose answer) ──────────────────────────────
ANSWER_SYSTEM_PROMPT = """You are a research assistant. Answer the user's question clearly and concisely.
If a <fetched_sources> block is provided, base your answer on the content of those pages — treat them as authoritative ground truth. Prefer fetched content over training knowledge when they conflict.
Do NOT output any references, citations, or a ---REFERENCES--- block. A separate citation step will handle that."""


# ── System prompt for call 2 (citation extraction) ───────────────────────
CITATION_SYSTEM_PROMPT = f"""You are a citation extraction assistant. Your sole job is to produce a structured JSON references array for the answer that was just given.

OUTPUT FORMAT (the parser depends on these — violations cause data loss):
1. On its own line, write the literal marker: {REFERENCES_MARKER}
2. Then a single JSON array `[...]`. No prose before or after. No markdown code fences. No trailing commentary.

Each JSON object must include ALL of the following fields (use null for unknowns — never omit a field):
- "title": title of the referenced work
- "authors": array of author names (Last, First Initial format)
- "date": publication year or full date
- "source_type": one of: journal_article, book, book_chapter, conference_paper, thesis, preprint, academic_url, institutional_report, blog_nonacademic, news_article, dataset, standard_spec, unknown
- "citation_style": one of: APA, MLA, Chicago, IEEE, Vancouver, Harvard, unknown
- "publisher": journal, publisher, or conference name
- "doi": DOI if known (without https://doi.org/ prefix), else null
- "url": access URL if known, else null
- "raw_fragment": short verbatim citation-style string, max 300 chars
- "confidence": 0.0–1.0 confidence in this reference

CITATION RULES:
- If the original question contained a <fetched_sources> block, include one entry per fetched URL that informed the answer, using the exact URL and page title from the block.
- For factual answers, the array MUST contain at least one item. An empty array is only valid for opinion, creative, or trivial restatement answers.
- DO NOT invent DOIs, URLs, or page numbers. Use null for any field you cannot confirm.
- Quality bar: populate as many fields as possible — title + authors + date + null doi is more valuable than a fabricated DOI.

WORKED EXAMPLE A — training-knowledge answer:
{REFERENCES_MARKER}
[
  {{"title": "Modular elliptic curves and Fermat's Last Theorem", "authors": ["Wiles, A."], "date": "1995", "source_type": "journal_article", "citation_style": "APA", "publisher": "Annals of Mathematics", "doi": "10.2307/2118559", "url": null, "raw_fragment": "Wiles, A. (1995). Annals of Mathematics, 141(3), 443-551.", "confidence": 0.95}}
]

WORKED EXAMPLE B — fetched-source answer:
{REFERENCES_MARKER}
[
  {{"title": "Nuclear Matrix Elements for Neutrinoless Double-Beta Decay", "authors": ["Grebe, A. V."], "date": "2025", "source_type": "preprint", "citation_style": "APA", "publisher": "arXiv", "doi": null, "url": "https://arxiv.org/html/2504.00358v2", "raw_fragment": "Grebe, A. V. (2025). arXiv:2504.00358v2.", "confidence": 1.0}}
]"""


# ── Citation request message (final user turn in call 2) ─────────────────
CITATION_REQUEST_MSG = (
    f"Based on your answer above, produce ONLY the structured citation references. "
    f"Start with the literal marker {REFERENCES_MARKER} on its own line, "
    f"then a single JSON array. No prose, no markdown fences."
)


@dataclass
class ExtractorConfig:
    ollama_url: str = cfg.OLLAMA_URL
    model: str = cfg.OLLAMA_MODEL
    max_concurrent: int = cfg.MAX_CONCURRENT_EXTRACTIONS
    timeout_s: int = cfg.OLLAMA_TIMEOUT_S
    num_ctx: int = cfg.OLLAMA_NUM_CTX
    num_predict: int = cfg.OLLAMA_NUM_PREDICT
    temperature_user: float = cfg.TEMPERATURE_USER_PROMPT
    temperature_citation: float = cfg.TEMPERATURE_CITATION_SYSTEM_PROMPT


class CitationExtractor:
    """Two-call answer+citations extractor."""

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        self._sem = asyncio.Semaphore(self.config.max_concurrent)

    async def generate_with_citations(
        self,
        user_prompt: str,
        prompt_id: str,
        fetched_pages: Optional[list[FetchedPage]] = None,
    ) -> tuple[str, list[CitationRecord]]:
        """
        Two Ollama calls. Returns (answer_text, citation_records).
        Call 1: prose answer at temperature_user via /api/generate.
        Call 2: citation JSON via /api/chat with full message history at temperature_citation.
        Records have empty source_id — populated later by store.reconcile_sources.
        """
        enriched = self._build_enriched_prompt(user_prompt, fetched_pages)

        answer = await self._call_answer(enriched)
        if not answer:
            return ("", [])

        refs_raw = await self._call_citations(user_prompt, answer, fetched_pages)
        if not refs_raw:
            return (answer, [])

        _, refs_json = self._split_output(refs_raw)
        records = self._parse_references(refs_json, prompt_id)
        return (answer, records)

    # ── Prompt enrichment ─────────────────────────────────────────────

    @staticmethod
    def _build_enriched_prompt(
        user_prompt: str, fetched_pages: Optional[list[FetchedPage]]
    ) -> str:
        """Inject fetched page text as <fetched_sources> context for call 1."""
        if not fetched_pages:
            return user_prompt
        lines: list[str] = ["<fetched_sources>"]
        for i, page in enumerate(fetched_pages, start=1):
            lines.append(f"[{i}] URL: {page.final_url or page.url}")
            if page.title:
                lines.append(f"    TITLE: {page.title}")
            if page.ok:
                lines.append("    TEXT:")
                for ln in page.text.splitlines():
                    lines.append(f"    {ln}")
            else:
                lines.append(f"    [fetch failed: {page.failure_reason}]")
            lines.append("")
        lines.append("</fetched_sources>")
        lines.append("")
        lines.append(user_prompt)
        return "\n".join(lines)

    # ── Prompt builders ───────────────────────────────────────────────

    @staticmethod
    def _build_lean_context(
        user_prompt: str, fetched_pages: Optional[list[FetchedPage]]
    ) -> str:
        """Compact context for call 2: original prompt + fetched URL/title pairs only.
        Avoids re-sending full page text so call 2 stays well within num_ctx."""
        if not fetched_pages:
            return user_prompt
        lines = [user_prompt, "", "Sources consulted:"]
        for i, page in enumerate(fetched_pages, start=1):
            url = page.final_url or page.url
            label = f"[{i}] {url}"
            if page.title:
                label += f" — {page.title}"
            lines.append(label)
        return "\n".join(lines)

    # ── Ollama calls ──────────────────────────────────────────────────

    async def _call_answer(self, enriched_prompt: str) -> Optional[str]:
        """Call 1 — prose answer only, /api/generate, temperature_user."""
        payload = {
            "model": self.config.model,
            "system": ANSWER_SYSTEM_PROMPT,
            "prompt": enriched_prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature_user,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
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
                            logger.error(f"Ollama answer call returned {resp.status}")
                            return None
                        data = await resp.json()
                        return data.get("response", "") or None
            except Exception as e:
                logger.error(f"Ollama answer call failed: {e}")
                return None

    async def _call_citations(
        self,
        user_prompt: str,
        answer: str,
        fetched_pages: Optional[list[FetchedPage]] = None,
    ) -> Optional[str]:
        """Call 2 — citation JSON, /api/chat with full history, temperature_citation.

        Uses a lean context for messages[0] (URL+title only, not full page text) to
        keep call 2 well within the model's context window.
        """
        lean_ctx = self._build_lean_context(user_prompt, fetched_pages)
        messages = [
            {"role": "user", "content": lean_ctx},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": CITATION_REQUEST_MSG},
        ]
        payload = {
            "model": self.config.model,
            "system": CITATION_SYSTEM_PROMPT,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature_citation,
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
            },
        }
        async with self._sem:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.config.ollama_url}/api/chat",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_s),
                    ) as resp:
                        if resp.status != 200:
                            logger.error(f"Ollama citations call returned {resp.status}")
                            return None
                        data = await resp.json()
                        return (data.get("message") or {}).get("content", "") or None
            except Exception as e:
                logger.error(f"Ollama citations call failed: {e}")
                return None

    # ── Output splitting ──────────────────────────────────────────────

    @staticmethod
    def _split_output(raw: str) -> tuple[str, str]:
        """
        Split LLM output into (answer, references_json_text).
        Requires the marker to appear at the start of a line — prevents false
        splits when a small model echoes the system prompt inline
        (e.g. "write the literal marker: ---REFERENCES---").

        Falls back to secondary markdown headers (e.g. "**REFERENCES:**",
        "## References") when the primary marker is absent.
        Returns (raw, "[]") on malformed output — graceful degradation.
        """
        # Match marker only at start of a line (or at start of string)
        pattern = re.compile(
            r"(?:^|\n)" + re.escape(REFERENCES_MARKER), re.MULTILINE
        )
        matches = list(pattern.finditer(raw))
        if not matches:
            # Secondary: tolerate common markdown reference-section headers
            sec_matches = list(_SECONDARY_REF_RE.finditer(raw))
            if sec_matches:
                logger.debug("Primary marker missing; split on secondary reference header")
                last = sec_matches[-1]
                answer = raw[: last.start()].strip()
                refs = raw[last.end() :].strip()
                refs = re.sub(r"^```(?:json)?\s*", "", refs)
                refs = re.sub(r"\s*```$", "", refs).strip()
                return (answer, refs or "[]")
            # Tertiary: if the whole output looks like a JSON array/object (call 2
            # sometimes emits bare JSON with no marker), treat it as the refs block.
            stripped = raw.strip()
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped).strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                logger.debug("No marker found but output looks like JSON; treating as refs block")
                return ("", stripped)
            return (raw.strip(), "[]")

        # Use last match — tolerates marker appearing in the answer prose
        last = matches[-1]
        answer = raw[: last.start()].strip()
        refs = raw[last.end() :].strip()
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
            return self._url_list_fallback(refs_json, prompt_id)

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
    def _url_list_fallback(text: str, prompt_id: str) -> list[CitationRecord]:
        """
        Last-resort extraction when the LLM emitted a plain-text or markdown URL
        list instead of a JSON array. Produces minimal CitationRecords (URL only,
        low confidence) so the sources are at least tracked in ChromaDB.
        """
        urls = list(dict.fromkeys(_BARE_URL_RE.findall(text)))  # dedup, order-preserving
        if not urls:
            return []
        records: list[CitationRecord] = []
        seen_cids: set[str] = set()
        for url in urls:
            try:
                rec = CitationRecord(
                    title=url,
                    source_type=SourceType.UNKNOWN,
                    authors=[],
                    access_url=url,
                    discovery_method=DiscoveryMethod.LLM_KNOWLEDGE,
                    confidence=0.3,
                    prompt_id=prompt_id,
                )
                if rec.cid not in seen_cids:
                    seen_cids.add(rec.cid)
                    records.append(rec)
            except Exception as e:
                logger.debug(f"URL fallback record failed for {url!r}: {e}")
        logger.debug(f"URL-list fallback produced {len(records)} citation record(s)")
        return records

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
