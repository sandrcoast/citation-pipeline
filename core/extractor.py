# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
core/extractor.py — Citation extraction engine.

Calls Ollama's /api/generate endpoint with a lightweight prompt
to extract citations from fetched web page text. Handles:
  - Concurrent extraction across multiple sources (asyncio + semaphore)
  - Rate-limiting to avoid overwhelming Ollama GPU
  - Batched processing for dozens of simultaneous user prompts
  - Graceful degradation if Ollama is overloaded

Usage:
    extractor = CitationExtractor(ollama_url="http://localhost:11434")
    citations = await extractor.extract_from_texts([
        SourceText(url="https://...", text="...page content..."),
        SourceText(url="https://...", text="...page content..."),
    ], prompt_id="abc-123")
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from core.models import (
    CitationRecord,
    CitationStyle,
    DiscoveryMethod,
    SourceType,
)

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────

@dataclass
class ExtractorConfig:
    """Tunable knobs. Adjust per-hardware."""

    ollama_url: str = "http://localhost:11434"
    model: str = "gemma3"  # Ollama model tag

    # Concurrency: how many extraction calls to Ollama at once.
    # Gemma 3 on a single GPU → keep this low (3-5).
    # Multi-GPU or quantized → can go higher.
    max_concurrent_extractions: int = 5

    # How many URLs to fetch concurrently (I/O bound, can be high)
    max_concurrent_fetches: int = 20

    # Text truncation: we send only this many chars to Gemma for extraction.
    # Bibliographies are usually near the end, so we take last N chars too.
    max_text_head_chars: int = 2000  # first N chars (abstract, intro citations)
    max_text_tail_chars: int = 3000  # last N chars — DEFAULT for typical articles

    # Dynamic tail sizing: if a bibliography section is detected,
    # we extract the FULL bibliography instead of a fixed tail.
    # For very long bibliographies (>200 refs), we chunk into segments
    # and send each chunk to Gemma 3 separately.
    max_refs_per_chunk: int = 30     # Gemma 3 handles ~30 refs per call reliably
    gemma3_max_input_chars: int = 25000  # ~6K tokens, safe within 8192 ctx

    # PDF fallback: when HTML extraction looks incomplete, try the PDF.
    # Works for publishers with predictable PDF URLs (Springer, Elsevier, etc.)
    enable_pdf_fallback: bool = True
    pdf_fetch_timeout_s: int = 15

    # Timeouts
    ollama_timeout_s: int = 30
    fetch_timeout_s: int = 8

    # Search
    searxng_url: str = "http://localhost:8888"  # self-hosted SearXNG

    # Optional: restrict search to specific domains (e.g., your org's digital library).
    # Leave empty to search the open web only.
    # Example: ["library.myorg.edu", "arxiv.org", "scholar.google.com"]
    scoped_domains: list = None  # type: ignore


# ── Extraction Prompt ─────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a citation extraction assistant. Given a text fragment from a web page or document, find ALL references, citations, and bibliographic entries.

For each citation found, return a JSON object with these fields:
- "title": title of the referenced work
- "authors": array of author names (Last, First Initial format)
- "date": publication year or full date
- "source_type": one of: journal_article, book, book_chapter, conference_paper, thesis, preprint, academic_url, institutional_report, blog_nonacademic, news_article, dataset, standard_spec, unknown
- "citation_style": one of: APA, MLA, Chicago, IEEE, Vancouver, Harvard, unknown (or null if no formal style)
- "publisher": journal name, publisher, or conference
- "doi": DOI if present (without https://doi.org/ prefix)
- "url": access URL if present
- "raw_fragment": the exact citation text, max 300 chars
- "confidence": 0.0-1.0 your confidence in extraction accuracy

Return ONLY a JSON array. No explanation, no markdown fences. If no citations found, return [].

Example output:
[{"title":"Example Paper","authors":["Smith, J.","Doe, A."],"date":"2023","source_type":"journal_article","citation_style":"APA","publisher":"Nature","doi":"10.1234/example","url":null,"raw_fragment":"Smith, J. & Doe, A. (2023). Example Paper. Nature.","confidence":0.95}]"""


# ── Source Text Container ─────────────────────────────────────────────────

@dataclass
class SourceText:
    """A fetched web page's content, ready for extraction."""
    url: str
    text: str
    discovery_method: DiscoveryMethod = DiscoveryMethod.WEB_SEARCH


# ── Extractor Engine ──────────────────────────────────────────────────────

class CitationExtractor:
    """
    Async citation extraction engine.

    Designed for high concurrency:
    - Semaphore limits concurrent Ollama calls (GPU is the bottleneck)
    - URL fetching uses a separate, larger semaphore (I/O bound)
    - Each user prompt gets its own extraction task
    - Multiple user prompts can run simultaneously
    """

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        # Semaphore: limits how many Ollama calls run in parallel
        self._ollama_sem = asyncio.Semaphore(self.config.max_concurrent_extractions)
        # Semaphore: limits how many URL fetches run in parallel
        self._fetch_sem = asyncio.Semaphore(self.config.max_concurrent_fetches)

    # ── Public API ────────────────────────────────────────────────────

    async def extract_from_texts(
        self,
        sources: list[SourceText],
        prompt_id: str,
    ) -> list[CitationRecord]:
        """
        Extract citations from multiple source texts concurrently.

        This is the main entry point. Call this once per user prompt.
        It handles parallelism, dedup, and normalization internally.

        Args:
            sources: List of fetched page texts to extract from
            prompt_id: Unique ID for this user prompt (for provenance)

        Returns:
            Deduplicated list of CitationRecord objects
        """
        start = time.monotonic()

        # Run extraction on all sources concurrently (semaphore-limited)
        tasks = [
            self._extract_single(source, prompt_id)
            for source in sources
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results, skip errors
        all_citations: list[CitationRecord] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Extraction failed for {sources[i].url}: {result}"
                )
                continue
            all_citations.extend(result)

        # Dedup by content hash (CID)
        seen_cids: set[str] = set()
        unique: list[CitationRecord] = []
        for c in all_citations:
            if c.cid not in seen_cids:
                seen_cids.add(c.cid)
                unique.append(c)

        elapsed = int((time.monotonic() - start) * 1000)
        logger.info(
            f"Extracted {len(unique)} unique citations from "
            f"{len(sources)} sources in {elapsed}ms (prompt={prompt_id})"
        )
        return unique

    async def search_and_extract(
        self,
        query: str,
        prompt_id: str,
    ) -> list[CitationRecord]:
        """
        Full pipeline: search → fetch → extract.

        1. Searches SearXNG for the query (open web)
        2. Optionally searches scoped domains (if configured)
        3. Fetches all result URLs
        4. Extracts citations from each
        """
        # Phase 1: Search
        search_urls = await self._search(query)
        logger.info(f"Search returned {len(search_urls)} URLs for prompt={prompt_id}")

        # Phase 2: Fetch all URLs concurrently
        sources = await self._fetch_urls(search_urls)
        logger.info(f"Fetched {len(sources)} pages for prompt={prompt_id}")

        # Phase 3: Extract
        return await self.extract_from_texts(sources, prompt_id)

    # ── Internal: Ollama Call ─────────────────────────────────────────

    async def _extract_single(
        self,
        source: SourceText,
        prompt_id: str,
    ) -> list[CitationRecord]:
        """
        Extract citations from a single source text.
        Uses bibliography detection + chunked extraction for long ref lists.
        Semaphore-limited to avoid overwhelming Ollama GPU.
        """
        if len(source.text.strip()) < 50:
            return []  # too short to contain citations

        # Prepare text chunks (1 for normal articles, N for long bibliographies)
        chunks = self._prepare_extraction_text(source.text)

        all_records: list[CitationRecord] = []
        for chunk in chunks:
            async with self._ollama_sem:  # ← rate limiting happens here
                raw_json = await self._call_ollama(chunk)

            if not raw_json:
                continue

            records = self._parse_extraction_response(
                raw_json, source.url, source.discovery_method, prompt_id
            )
            all_records.extend(records)

        return all_records

    async def _call_ollama(self, text: str) -> Optional[str]:
        """
        Call Ollama /api/generate with the extraction prompt.
        Returns raw response text (should be JSON array).
        """
        payload = {
            "model": self.config.model,
            "system": EXTRACTION_SYSTEM_PROMPT,
            "prompt": f"Extract all citations from this text:\n\n{text}",
            "stream": False,
            "options": {
                "temperature": 0.1,      # low temp for structured extraction
                "num_predict": 2048,      # enough for ~10 citations
            },
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.ollama_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.ollama_timeout_s),
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Ollama returned {resp.status}")
                        return None
                    data = await resp.json()
                    return data.get("response", "")
        except asyncio.TimeoutError:
            logger.warning("Ollama extraction timed out")
            return None
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return None

    # ── Internal: Search ──────────────────────────────────────────────

    async def _search(self, query: str) -> list[str]:
        """
        Search via SearXNG (self-hosted). Returns list of URLs.
        If scoped_domains are configured, also runs domain-specific searches.
        """
        urls: list[str] = []

        try:
            async with aiohttp.ClientSession() as session:
                # General web search
                params = {
                    "q": query,
                    "format": "json",
                    "engines": "google,bing,duckduckgo",
                    "pageno": 1,
                }
                async with session.get(
                    f"{self.config.searxng_url}/search",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        urls.extend(r["url"] for r in data.get("results", [])[:10])

                # Optional: scoped domain searches (e.g., org library, arxiv)
                for domain in (self.config.scoped_domains or []):
                    params["q"] = f"site:{domain} {query}"
                    async with session.get(
                        f"{self.config.searxng_url}/search",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            urls.extend(r["url"] for r in data.get("results", [])[:5])

        except Exception as e:
            logger.error(f"Search failed: {e}")

        return list(dict.fromkeys(urls))  # dedup preserving order

    # ── Internal: URL Fetching ────────────────────────────────────────

    async def _fetch_urls(self, urls: list[str]) -> list[SourceText]:
        """Fetch all URLs concurrently, return extracted text."""
        tasks = [self._fetch_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sources: list[SourceText] = []
        for result in results:
            if isinstance(result, SourceText):
                sources.append(result)
        return sources

    async def _fetch_single(self, url: str) -> Optional[SourceText]:
        """Fetch a single URL and extract readable text.
        Handles both HTML and PDF content types.
        If HTML bibliography looks incomplete, attempts PDF fallback."""
        async with self._fetch_sem:  # rate-limit concurrent fetches
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=self.config.fetch_timeout_s),
                        headers={"User-Agent": "CitationBot/1.0 (github.com/citation-pipeline)"},
                    ) as resp:
                        if resp.status != 200:
                            return None
                        content_type = resp.headers.get("Content-Type", "")

                        # ── PDF content ──────────────────────────────
                        if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
                            pdf_bytes = await resp.read()
                            text = self._extract_pdf_text(pdf_bytes)
                            if text:
                                return SourceText(
                                    url=url, text=text,
                                    discovery_method=DiscoveryMethod.WEB_SEARCH,
                                )
                            return None

                        # ── HTML content ─────────────────────────────
                        raw = await resp.text()

                # RC3 fix: trafilatura with favor_recall to keep bibliography sections
                try:
                    import trafilatura
                    text = trafilatura.extract(
                        raw,
                        include_links=True,
                        include_tables=True,
                        favor_recall=True,  # ← keeps more content incl. references
                    ) or ""
                except ImportError:
                    # Fallback: naive HTML tag stripping
                    text = re.sub(r"<[^>]+>", " ", raw)
                    text = re.sub(r"\s+", " ", text).strip()

                if len(text) < 50:
                    return None

                # ── PDF fallback: if bibliography looks truncated ────
                if self.config.enable_pdf_fallback:
                    bib = self._find_bibliography(text)
                    if bib is not None:
                        ref_count = self._estimate_ref_count(bib)
                        # Heuristic: if the last ref number in body is much higher
                        # than what we found in bibliography, it's truncated
                        max_inline = self._find_max_inline_ref_number(text)
                        if max_inline > 0 and ref_count < max_inline * 0.6:
                            logger.info(
                                f"Bibliography looks truncated ({ref_count} refs found, "
                                f"~{max_inline} expected). Trying PDF fallback for {url}"
                            )
                            pdf_text = await self._fetch_pdf_fallback(url, session=None)
                            if pdf_text and len(pdf_text) > len(text):
                                text = pdf_text

                return SourceText(
                    url=url, text=text,
                    discovery_method=DiscoveryMethod.WEB_SEARCH,
                )

            except Exception as e:
                logger.debug(f"Fetch failed for {url}: {e}")
                return None

    # ── Internal: Text Preparation ─────────────────────────────────────

    def _prepare_extraction_text(self, text: str) -> list[str]:
        """
        Smart text preparation for extraction. Returns a list of text chunks
        (usually 1, but multiple for very long bibliographies).

        Strategy:
          1. Try to find the bibliography/references section
          2. If found: extract it whole, chunk if too long for Gemma 3
          3. If not found: fall back to head+tail truncation
        """
        bib = self._find_bibliography(text)

        if bib is not None and len(bib.strip()) > 100:
            # We found a bibliography section — use it whole
            head = text[:self.config.max_text_head_chars]

            # If bibliography fits in one Gemma 3 call, send it all
            if len(bib) <= self.config.gemma3_max_input_chars:
                return [head + "\n\n" + bib]

            # Otherwise, chunk the bibliography
            return self._chunk_bibliography(head, bib)
        else:
            # No bibliography detected — fall back to head+tail
            return [self._truncate_text(text)]

    def _find_bibliography(self, text: str) -> Optional[str]:
        """
        Locate the references/bibliography section in article text.
        Looks for common section headers and extracts everything after.
        """
        # Common bibliography section headers (case-insensitive)
        patterns = [
            r'(?:^|\n)\s*References\s*\n',
            r'(?:^|\n)\s*REFERENCES\s*\n',
            r'(?:^|\n)\s*Bibliography\s*\n',
            r'(?:^|\n)\s*BIBLIOGRAPHY\s*\n',
            r'(?:^|\n)\s*Works Cited\s*\n',
            r'(?:^|\n)\s*Literature Cited\s*\n',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                bib_text = text[match.start():]
                # Trim trailing non-reference content (acknowledgments after refs, etc.)
                # Look for sections that commonly follow references
                end_patterns = [
                    r'\n\s*Acknowledg[e]?ments?\s*\n',
                    r'\n\s*Author [Ii]nformation\s*\n',
                    r'\n\s*Supplementary\s',
                    r'\n\s*Additional [Ii]nformation\s*\n',
                    r'\n\s*Rights and [Pp]ermissions\s*\n',
                    r'\n\s*About this article\s*\n',
                ]
                for ep in end_patterns:
                    end_match = re.search(ep, bib_text, re.IGNORECASE)
                    if end_match:
                        bib_text = bib_text[:end_match.start()]
                        break

                return bib_text

        return None

    def _estimate_ref_count(self, bibliography: str) -> int:
        """Estimate number of references in a bibliography section."""
        # Vancouver style: numbered refs like "1." "2." etc.
        numbered = re.findall(r'^\s*\d+\.?\s+\w', bibliography, re.MULTILINE)
        if numbered:
            return len(numbered)
        # APA/MLA: count lines that look like citations (Author, Year pattern)
        apa_like = re.findall(r'\n\s*[A-Z][a-z]+[\s,].*?\(\d{4}\)', bibliography)
        if apa_like:
            return len(apa_like)
        # Fallback: count double-newline-separated blocks
        blocks = [b for b in bibliography.split('\n\n') if len(b.strip()) > 30]
        return max(len(blocks), 1)

    def _find_max_inline_ref_number(self, text: str) -> int:
        """Find the highest reference number cited inline (e.g., [83] or [ref-CR83])."""
        # Bracketed numbers: [1], [83], [1,2,3]
        matches = re.findall(r'\[(\d+)\]', text)
        if matches:
            return max(int(m) for m in matches)
        # ref-CR style (Springer HTML)
        cr_matches = re.findall(r'ref-CR(\d+)', text)
        if cr_matches:
            return max(int(m) for m in cr_matches)
        return 0

    def _chunk_bibliography(self, head: str, bibliography: str) -> list[str]:
        """
        Split a long bibliography into chunks that each fit in Gemma 3's context.
        Each chunk gets the article head prepended for context.
        """
        # Split bibliography into individual references
        # Try numbered refs first (Vancouver): "1. ...", "2. ..."
        refs = re.split(r'\n(?=\s*\d+\.?\s+[A-Z])', bibliography)

        # If that didn't work well, try double-newline splitting
        if len(refs) < 3:
            refs = [r for r in bibliography.split('\n\n') if len(r.strip()) > 20]

        # Group refs into chunks of max_refs_per_chunk
        chunks = []
        chunk_size = self.config.max_refs_per_chunk
        for i in range(0, len(refs), chunk_size):
            chunk_refs = '\n'.join(refs[i:i + chunk_size])
            chunk_text = head + "\n\n[Bibliography chunk " \
                         f"{i // chunk_size + 1}]\n\n" + chunk_refs
            chunks.append(chunk_text)

        logger.info(
            f"Split bibliography into {len(chunks)} chunks "
            f"({len(refs)} refs, {chunk_size} per chunk)"
        )
        return chunks if chunks else [head + "\n\n" + bibliography]

    def _truncate_text(self, text: str) -> str:
        """
        Fallback truncation when no bibliography section is detected.
        Takes first N chars + last M chars.
        """
        head = self.config.max_text_head_chars
        tail = self.config.max_text_tail_chars

        if len(text) <= head + tail:
            return text

        return text[:head] + "\n\n[...truncated...]\n\n" + text[-tail:]

    # ── Internal: PDF Support ─────────────────────────────────────────

    def _extract_pdf_text(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using pdfplumber."""
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
                return '\n\n'.join(pages)
        except ImportError:
            logger.warning("pdfplumber not installed. Run: pip install pdfplumber")
            return None
        except Exception as e:
            logger.warning(f"PDF extraction failed: {e}")
            return None

    async def _fetch_pdf_fallback(
        self, html_url: str, session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[str]:
        """
        Attempt to fetch the PDF version of an article when HTML is truncated.
        Constructs PDF URLs for known publishers (Springer, Elsevier, etc.)
        """
        pdf_urls = self._construct_pdf_urls(html_url)
        if not pdf_urls:
            return None

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            for pdf_url in pdf_urls:
                try:
                    async with session.get(
                        pdf_url,
                        timeout=aiohttp.ClientTimeout(
                            total=self.config.pdf_fetch_timeout_s
                        ),
                        headers={"User-Agent": "CitationBot/1.0 (github.com/citation-pipeline)"},
                    ) as resp:
                        if resp.status == 200:
                            ct = resp.headers.get("Content-Type", "")
                            if "pdf" in ct.lower():
                                pdf_bytes = await resp.read()
                                text = self._extract_pdf_text(pdf_bytes)
                                if text and len(text) > 500:
                                    logger.info(
                                        f"PDF fallback succeeded: {pdf_url} "
                                        f"({len(text)} chars)"
                                    )
                                    return text
                except Exception as e:
                    logger.debug(f"PDF fallback failed for {pdf_url}: {e}")
                    continue
        finally:
            if close_session:
                await session.close()

        return None

    @staticmethod
    def _construct_pdf_urls(html_url: str) -> list[str]:
        """
        Construct probable PDF URLs from an HTML article URL.
        Supports major academic publishers.
        """
        urls = []

        # Springer: link.springer.com/article/DOI → .../content/pdf/DOI.pdf
        if "springer.com/article/" in html_url:
            doi_part = html_url.split("/article/")[-1].split("?")[0]
            urls.append(f"https://link.springer.com/content/pdf/{doi_part}.pdf")

        # Elsevier/ScienceDirect: extract DOI, construct PDF
        if "sciencedirect.com" in html_url:
            pii_match = re.search(r'/pii/(S\d+)', html_url)
            if pii_match:
                urls.append(
                    f"https://www.sciencedirect.com/science/article/pii/"
                    f"{pii_match.group(1)}/pdfft"
                )

        # arXiv: /abs/ID → /pdf/ID.pdf
        if "arxiv.org/abs/" in html_url:
            arxiv_id = html_url.split("/abs/")[-1].split("?")[0]
            urls.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")

        # Generic: try appending .pdf or /pdf to the URL
        if not urls:
            base = html_url.split("?")[0]
            urls.append(base + ".pdf")
            urls.append(base + "/pdf")

        return urls

    def _parse_extraction_response(
        self,
        raw: str,
        source_url: str,
        discovery_method: DiscoveryMethod,
        prompt_id: str,
    ) -> list[CitationRecord]:
        """
        Parse Gemma 3's JSON response into CitationRecord objects.
        Handles malformed JSON gracefully.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            items = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON array in the response
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse extraction JSON from {source_url}")
                    return []
            else:
                return []

        if not isinstance(items, list):
            return []

        records: list[CitationRecord] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                record = CitationRecord(
                    title=item.get("title", "Unknown"),
                    source_type=self._map_source_type(item.get("source_type", "unknown")),
                    authors=item.get("authors", []),
                    date_published=str(item.get("date", "")) or None,
                    citation_style_detected=self._map_citation_style(item.get("citation_style")),
                    raw_citation_fragment=(item.get("raw_fragment", "") or "")[:500] or None,
                    publisher=item.get("publisher"),
                    access_url=item.get("url"),
                    doi=item.get("doi"),
                    discovery_method=discovery_method,
                    discovery_source_url=source_url,
                    confidence=float(item.get("confidence", 0.5)),
                    prompt_id=prompt_id,
                )
                records.append(record)
            except Exception as e:
                logger.debug(f"Skipping malformed citation item: {e}")

        return records

    @staticmethod
    def _map_source_type(raw: str) -> SourceType:
        try:
            return SourceType(raw)
        except ValueError:
            return SourceType.UNKNOWN

    @staticmethod
    def _map_citation_style(raw: Optional[str]) -> Optional[CitationStyle]:
        if not raw or raw == "null":
            return None
        try:
            return CitationStyle(raw)
        except ValueError:
            return CitationStyle.UNKNOWN
