# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
core/web_fetch.py — URL fetcher + HTML-to-text extractor.

Strategy (per URL):
  1. aiohttp GET — fast, zero overhead.  Good for static HTML.
  2. Playwright (Chromium, headless) — fallback when aiohttp returns empty
     body text (JS-rendered pages like Nature, Springer, etc.).

Both paths:
  - Extract visible body text (html.parser, skips script/style/nav/footer).
  - Extract structured citation metadata: window.dataLayer (Nature/Springer)
    and <meta name="citation_*"> tags.
  - Extract the article's own reference list (Nature-style
    c-article-references__text + data-doi) and store in meta["references"].

Results land in FetchedPage, used by:
  - core/extractor.py  → enriches the LLM prompt with body text
  - middleware/proxy.py → builds CitationRecord objects from meta + references
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import unescape as html_unescape
from html.parser import HTMLParser
from typing import Optional

import aiohttp

from config import cfg

logger = logging.getLogger(__name__)


_URL_RE = re.compile(r"https?://[^\s<>\"'`\])}]+", re.IGNORECASE)
_SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "aside", "svg", "form"}
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class FetchedPage:
    url: str
    final_url: Optional[str] = None
    title: Optional[str] = None
    text: str = ""
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: int = 0
    error: Optional[str] = None
    via_playwright: bool = False
    js_rendered: bool = False   # True when aiohttp got empty body (needs Playwright)
    # Structured citation fields extracted from window.dataLayer / meta tags.
    # meta["references"] is a list[dict] of the article's own reference list.
    meta: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.text)

    @property
    def failure_reason(self) -> str:
        if self.error:
            return self.error
        if self.status != 200:
            return f"HTTP {self.status}"
        if not self.text:
            return "fetched but no text extracted (JS-rendered page?)"
        return "unknown"


# ── URL detection ─────────────────────────────────────────────────────────

def extract_urls(text: str) -> list[str]:
    """Extract deduplicated URLs from free text, capped at WEB_FETCH_MAX_URLS."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:)]}'\"")
        if url not in seen:
            seen.add(url)
            out.append(url)
        if len(out) >= cfg.WEB_FETCH_MAX_URLS:
            break
    return out


# ── HTML → plain text ─────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self.title: str = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title += data
            return
        self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


# ── Structured metadata extraction ───────────────────────────────────────

def _extract_citation_meta(html: str) -> Optional[dict]:
    """
    Extract article-level citation fields from inline scripts / meta tags.
    Returns dict with subset of: title, authors, doi, publisher,
    date_published, source_type, references (list of dicts).
    Returns None if nothing useful found.
    """
    result: dict = {}

    # ── 1. window.dataLayer (Nature / Springer) ───────────────────────
    m = re.search(r"window\.dataLayer\s*=\s*(\[.*?\]);", html, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            layer = data[0] if isinstance(data, list) and data else {}
            content = layer.get("content", {})
            ci = content.get("contentInfo", {})
            art = content.get("article", {})
            jrn = content.get("journal", {})

            if ci.get("title"):
                result["title"] = ci["title"]
            if ci.get("authors"):
                result["authors"] = ci["authors"]
            if ci.get("publishedAtString"):
                result["date_published"] = ci["publishedAtString"]
            if jrn.get("title"):
                pub = jrn["title"]
                if jrn.get("volume"):
                    pub += f", vol. {jrn['volume']}"
                result["publisher"] = pub
                result["source_type"] = "journal_article"
            if art.get("doi"):
                result["doi"] = art["doi"]
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # ── 2. <meta name="citation_*"> tags ─────────────────────────────
    if not result.get("title"):
        meta_fields: dict[str, list[str]] = {}
        for m2 in re.finditer(
            r'<meta\s+name=["\']citation_(\w+)["\']\s+content=["\']([^"\']*)["\']',
            html, re.IGNORECASE,
        ):
            key, val = m2.group(1).lower(), m2.group(2).strip()
            if val:
                meta_fields.setdefault(key, []).append(val)
        if meta_fields.get("title"):
            result["title"] = meta_fields["title"][0]
            result["authors"] = meta_fields.get("author", [])
            if meta_fields.get("doi"):
                result["doi"] = meta_fields["doi"][0]
            if meta_fields.get("publication_date"):
                result["date_published"] = meta_fields["publication_date"][0]
            if meta_fields.get("journal_title"):
                result["publisher"] = meta_fields["journal_title"][0]
                result["source_type"] = "journal_article"

    # ── 3. Article reference list (Nature-style) ──────────────────────
    references = _extract_article_references(html)
    if references:
        result["references"] = references

    return result if result.get("title") or result.get("references") else None


def _extract_article_references(html: str) -> list[dict]:
    """
    Parse the article's own bibliography from rendered HTML.

    Handles:
    - Nature / Scientific Reports: <li data-counter=...>
        <p class="c-article-references__text">...</p>
        <a data-doi="...">
    - Generic: <li> items containing recognisable citation text
    """
    refs: list[dict] = []

    # Nature pattern: <li data-counter="N">...<p class="c-article-references__text">
    items = re.findall(
        r'<li[^>]*data-counter[^>]*>(.*?)</li>',
        html, re.DOTALL | re.IGNORECASE,
    )
    if items:
        for item in items:
            text_m = re.search(
                r'class="c-article-references__text"[^>]*>(.*?)</p>',
                item, re.DOTALL | re.IGNORECASE,
            )
            raw = ""
            if text_m:
                raw = re.sub(r"<[^>]+>", " ", text_m.group(1))
                raw = html_unescape(re.sub(r"\s+", " ", raw).strip())
            else:
                raw = re.sub(r"<[^>]+>", " ", item)
                raw = html_unescape(re.sub(r"\s+", " ", raw).strip())

            doi_m = re.search(r'data-doi=["\']([^"\']+)["\']', item, re.IGNORECASE)
            doi = doi_m.group(1).strip() if doi_m else None

            # Also try href="https://doi.org/..."
            if not doi:
                href_m = re.search(r'href="https://doi\.org/([^"]+)"', item, re.IGNORECASE)
                if href_m:
                    doi = href_m.group(1).strip()

            if raw:
                refs.append({"raw": raw, "doi": doi})
        return refs

    # Fallback: generic <ol>/<ul> with likely citation lines (has year pattern)
    list_items = re.findall(r'<li[^>]*>(.*?)</li>', html, re.DOTALL | re.IGNORECASE)
    for item in list_items:
        text = re.sub(r"<[^>]+>", " ", item)
        text = html_unescape(re.sub(r"\s+", " ", text).strip())
        if re.search(r'\b(19|20)\d{2}\b', text) and len(text) > 30:
            doi_m = re.search(r'href="https://doi\.org/([^"]+)"', item, re.IGNORECASE)
            refs.append({"raw": text[:400], "doi": doi_m.group(1) if doi_m else None})

    return refs[:50]  # cap at 50 to avoid noise


def _meta_to_text(meta: dict) -> str:
    """Format citation meta dict as readable text for the LLM prompt context."""
    lines = []
    if meta.get("title"):
        lines.append(f"Title: {meta['title']}")
    if meta.get("authors"):
        authors = meta["authors"]
        lines.append(f"Authors: {', '.join(authors) if isinstance(authors, list) else authors}")
    if meta.get("date_published"):
        lines.append(f"Published: {meta['date_published']}")
    if meta.get("publisher"):
        lines.append(f"Journal/Publisher: {meta['publisher']}")
    if meta.get("doi"):
        lines.append(f"DOI: {meta['doi']}")
    refs = meta.get("references", [])
    if refs:
        lines.append(f"\nReferences ({len(refs)} total):")
        for i, r in enumerate(refs[:10], 1):
            doi_str = f" [DOI: {r['doi']}]" if r.get("doi") else ""
            lines.append(f"  {i}. {r['raw'][:200]}{doi_str}")
        if len(refs) > 10:
            lines.append(f"  ... and {len(refs) - 10} more")
    return "\n".join(lines)


def html_to_text(html: str) -> tuple[str, str, Optional[dict], bool]:
    """
    Return (title, plain_text, citation_meta, body_was_empty).
    body_was_empty=True when the HTML had no parseable body text — signals
    that Playwright may be needed for the full JS-rendered content.
    Falls back to formatted metadata text when body is empty.
    """
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as e:
        logger.debug(f"HTML parse error (partial result kept): {e}")
    text = parser.get_text()
    meta = _extract_citation_meta(html)
    body_was_empty = not text
    if body_was_empty and meta:
        text = _meta_to_text(meta)
        logger.debug("Body text empty — used structured metadata fallback")
    limit = cfg.WEB_FETCH_MAX_CHARS_PER_PAGE
    if len(text) > limit:
        text = text[:limit] + "\n[...truncated]"
    return (parser.title.strip(), text, meta, body_was_empty)


# ── aiohttp fetch (fast path) ─────────────────────────────────────────────

async def fetch_url(session: aiohttp.ClientSession, url: str) -> FetchedPage:
    page = FetchedPage(url=url)
    headers = {
        "User-Agent": cfg.WEB_FETCH_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=cfg.WEB_FETCH_TIMEOUT_S),
            allow_redirects=True,
        ) as resp:
            page.status = resp.status
            page.final_url = str(resp.url)
            ctype = resp.headers.get("Content-Type", "").lower()
            if resp.status != 200:
                page.error = f"HTTP {resp.status}"
                return page
            if "html" not in ctype and "xml" not in ctype:
                page.error = f"non-html content-type: {ctype or 'unknown'}"
                return page
            raw = await resp.content.read(cfg.WEB_FETCH_MAX_BYTES + 1)
            if len(raw) > cfg.WEB_FETCH_MAX_BYTES:
                raw = raw[: cfg.WEB_FETCH_MAX_BYTES]
            charset = resp.charset or "utf-8"
            try:
                html = raw.decode(charset, errors="replace")
            except LookupError:
                html = raw.decode("utf-8", errors="replace")
            title, text, meta, body_empty = html_to_text(html)
            page.title = title or None
            page.text = text
            page.meta = meta or None
            page.js_rendered = body_empty  # True → Playwright fallback needed
    except asyncio.TimeoutError:
        page.error = "timeout"
    except Exception as e:
        page.error = f"{type(e).__name__}: {e}"
    return page


# ── Playwright fetch (JS-rendered fallback) ───────────────────────────────

def _playwright_fetch_sync(url: str, timeout_ms: int) -> FetchedPage:
    """Synchronous Playwright fetch — runs in a thread via asyncio.to_thread."""
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    page_obj = FetchedPage(url=url, via_playwright=True)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
                )
                pg = ctx.new_page()
                response = pg.goto(url, wait_until="networkidle", timeout=timeout_ms)
                page_obj.status = response.status if response else 0
                page_obj.final_url = pg.url
                html = pg.content()
            finally:
                browser.close()

        title, text, meta, _ = html_to_text(html)
        page_obj.title = title or None
        page_obj.text = text
        page_obj.meta = meta or None
    except Exception as e:
        page_obj.error = f"playwright: {type(e).__name__}: {e}"
    return page_obj


async def fetch_url_playwright(url: str) -> FetchedPage:
    """Async wrapper — offloads Playwright to a thread pool."""
    timeout_ms = cfg.WEB_FETCH_TIMEOUT_S * 1000
    return await asyncio.to_thread(_playwright_fetch_sync, url, timeout_ms)


# ── Unified fetch (aiohttp → Playwright fallback) ─────────────────────────

async def fetch_all(urls: list[str]) -> list[FetchedPage]:
    if not urls:
        return []
    sem = asyncio.Semaphore(cfg.WEB_FETCH_CONCURRENCY)

    async def _one(u: str) -> FetchedPage:
        async with sem:
            async with aiohttp.ClientSession() as session:
                page = await fetch_url(session, u)
            # Fall back to Playwright when body text was JS-rendered
            if page.status == 200 and page.js_rendered:
                logger.info(f"aiohttp got JS-rendered page for {u} — retrying with Playwright")
                page = await fetch_url_playwright(u)
            return page

    return await asyncio.gather(*(_one(u) for u in urls))
