# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
core/web_fetch.py — URL fetcher + HTML-to-text extractor.

Strategy (per URL):
  1. aiohttp GET — fast, zero overhead. Good for static HTML.
  2. Playwright (Chromium, headless) — fallback when aiohttp returns empty
     or sparse body text (JS-rendered pages like arXiv, Nature, Springer).

The fetcher produces cleaned page text only. It does NOT synthesize any
CitationRecord — that is the LLM's job. The page text is injected into the
prompt as a <fetched_sources> block, and the LLM emits structured records
based on what it reads.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


# ── HTML → text ──────────────────────────────────────────────────────────

def html_to_text(html: str) -> tuple[str, str, bool]:
    """
    Return (title, plain_text, body_was_empty).
    body_was_empty=True when the HTML had no parseable body text — signals
    that Playwright may be needed for the full JS-rendered content.
    """
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception as e:
        logger.debug(f"HTML parse error (partial result kept): {e}")
    text = parser.get_text()
    body_was_empty = not text
    limit = cfg.WEB_FETCH_MAX_CHARS_PER_PAGE
    if len(text) > limit:
        text = text[:limit] + "\n[...truncated]"
    return (parser.title.strip(), text, body_was_empty)


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
            title, text, body_empty = html_to_text(html)
            page.title = title or None
            page.text = text
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

        title, text, _ = html_to_text(html)
        page_obj.title = title or None
        page_obj.text = text
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
            # Fall back to Playwright when body is empty OR suspiciously sparse
            # (arxiv/html, some publishers serve stub HTML to non-browser UAs)
            sparse = (
                page.status == 200
                and not page.js_rendered
                and len(page.text) < 3000
            )
            if page.status == 200 and (page.js_rendered or sparse):
                reason = "JS-rendered" if page.js_rendered else f"sparse body ({len(page.text)} chars)"
                logger.info(f"aiohttp {reason} for {u} — retrying with Playwright")
                page = await fetch_url_playwright(u)
            return page

    return await asyncio.gather(*(_one(u) for u in urls))
