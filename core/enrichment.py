# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
"""
core/enrichment.py — Post-extraction enrichment for citation records.

After Gemma 3 extracts citations from page text, many records will be
missing DOIs (especially from PDFs where DOIs aren't embedded in the
reference text). This module resolves missing DOIs via the CrossRef API.

CrossRef is free, no API key required (polite pool), and covers
>130 million DOIs. Rate limit: ~50 req/s with polite headers.

Usage:
    enricher = CrossRefEnricher()
    enriched = await enricher.enrich_batch(records)
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import CitationRecord

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class EnricherConfig:
    """Configuration for the CrossRef enrichment step."""

    # CrossRef polite pool: include a mailto for better rate limits
    # See: https://github.com/CrossRef/rest-api-doc#good-manners--more-reliable-service
    mailto: str = "citation-pipeline@example.com"  # ← set to your real email

    # Concurrency: CrossRef allows ~50 req/s for polite users
    max_concurrent: int = 10

    # Timeout per request
    timeout_s: int = 5

    # Minimum confidence to accept a CrossRef match
    # CrossRef returns a score; we normalize it to 0-1
    min_match_score: float = 0.75


class CrossRefEnricher:
    """
    Resolves missing DOIs by querying CrossRef's /works endpoint.
    Queries by title + first author for best match accuracy.
    """

    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, config: Optional[EnricherConfig] = None):
        self.config = config or EnricherConfig()
        self._sem = asyncio.Semaphore(self.config.max_concurrent)

    async def enrich_batch(self, records: list) -> list:
        """
        Enrich a batch of CitationRecords with missing DOIs.
        Only queries CrossRef for records that don't already have a DOI.
        Returns the same list with DOIs filled in where found.
        """
        tasks = []
        for record in records:
            if not record.doi:
                tasks.append(self._enrich_single(record))

        if not tasks:
            logger.info("All records already have DOIs, skipping enrichment")
            return records

        results = await asyncio.gather(*tasks, return_exceptions=True)
        enriched_count = sum(1 for r in results if isinstance(r, str))
        logger.info(
            f"CrossRef enrichment: {enriched_count}/{len(tasks)} DOIs resolved"
        )
        return records

    async def _enrich_single(self, record) -> Optional[str]:
        """Query CrossRef for a single record's DOI."""
        query = self._build_query(record)
        if not query:
            return None

        async with self._sem:
            try:
                async with aiohttp.ClientSession() as session:
                    params = {
                        "query.bibliographic": query,
                        "rows": 1,
                        "select": "DOI,title,author,score",
                        "mailto": self.config.mailto,
                    }
                    async with session.get(
                        self.BASE_URL,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_s),
                        headers={
                            "User-Agent": (
                                f"CitationPipeline/1.0 "
                                f"(mailto:{self.config.mailto})"
                            ),
                        },
                    ) as resp:
                        if resp.status != 200:
                            return None
                        data = await resp.json()

                items = data.get("message", {}).get("items", [])
                if not items:
                    return None

                best = items[0]
                score = best.get("score", 0)

                # CrossRef scores vary widely; normalize roughly
                # Scores >100 are typically strong matches
                normalized_score = min(score / 150, 1.0)

                if normalized_score >= self.config.min_match_score:
                    doi = best.get("DOI")
                    if doi:
                        record.doi = doi
                        # Update access_url if we got a DOI
                        if not record.access_url:
                            record.access_url = f"https://doi.org/{doi}"
                        logger.debug(
                            f"Resolved DOI for '{record.title[:40]}': {doi}"
                        )
                        return doi

            except asyncio.TimeoutError:
                logger.debug(f"CrossRef timeout for '{record.title[:40]}'")
            except Exception as e:
                logger.debug(f"CrossRef error for '{record.title[:40]}': {e}")

        return None

    @staticmethod
    def _build_query(record) -> Optional[str]:
        """
        Build a CrossRef bibliographic query from a CitationRecord.
        Format: "title + first author last name + year"
        """
        parts = []

        if record.title and len(record.title) > 5:
            # Use first 100 chars of title to avoid query length issues
            parts.append(record.title[:100])

        if record.authors:
            # Take first author's last name
            first_author = record.authors[0]
            last_name = first_author.split(",")[0].strip()
            if len(last_name) > 1:
                parts.append(last_name)

        if record.date_published:
            parts.append(record.date_published[:4])

        if len(parts) < 2:
            return None  # not enough info for a reliable query

        return " ".join(parts)
