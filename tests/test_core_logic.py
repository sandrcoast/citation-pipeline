"""
tests/test_core_logic.py — Stdlib-only tests for core pipeline logic.

Tests the fundamental algorithms (hashing, views, dedup, serialization)
without requiring pydantic or any external dependencies.
Run anywhere: python tests/test_core_logic.py
"""

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Minimal CitationRecord (stdlib only, mirrors core/models.py) ─────────

@dataclass
class CitationRecord:
    title: str
    source_type: str = "unknown"
    authors: list = field(default_factory=list)
    date_published: Optional[str] = None
    citation_style_detected: Optional[str] = None
    raw_citation_fragment: Optional[str] = None
    publisher: Optional[str] = None
    access_url: Optional[str] = None
    doi: Optional[str] = None
    discovery_method: str = "web_search_result"
    discovery_source_url: Optional[str] = None
    confidence: float = 0.0
    library_match_id: Optional[str] = None
    share_hash: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    prompt_id: Optional[str] = None

    def __post_init__(self):
        self.cid = self._compute_cid()

    def _compute_cid(self) -> str:
        norm_title = self.title.strip().lower()
        norm_authors = "|".join(sorted(a.strip().lower() for a in self.authors))
        norm_date = (self.date_published or "").strip()
        payload = f"{norm_title}||{norm_authors}||{norm_date}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compute_share_hash(self, salt: str) -> str:
        self.share_hash = hmac.new(
            salt.encode(), self.cid.encode(), hashlib.sha256
        ).hexdigest()
        return self.share_hash

    def to_vector_meta(self) -> dict:
        return {
            "cid": self.cid,
            "title": self.title,
            "source_type": self.source_type,
            "authors_json": json.dumps(self.authors),
            "date_published": self.date_published or "",
            "citation_style": self.citation_style_detected or "",
            "publisher": self.publisher or "",
            "doi": self.doi or "",
            "discovery_method": self.discovery_method,
            "confidence": self.confidence,
            "prompt_id": self.prompt_id or "",
        }

    def to_user_view(self) -> dict:
        return {
            "title": self.title,
            "type": self.source_type,
            "authors": self.authors,
            "date": self.date_published,
            "style": self.citation_style_detected,
            "publisher": self.publisher,
            "url": self.access_url,
        }

    def to_a2a_meta(self) -> dict:
        return {
            "type": "citation_record",
            "version": "1.0",
            "cid": self.cid,
            "title": self.title,
            "source_type": self.source_type,
            "authors": self.authors,
            "date_published": self.date_published,
            "citation_style_detected": self.citation_style_detected,
            "publisher": self.publisher,
            "access_url": self.access_url,
            "doi": self.doi,
            "discovery_method": self.discovery_method,
            "confidence": self.confidence,
            "prompt_id": self.prompt_id,
            "extracted_at": self.created_at,
        }

    def to_db_row(self) -> dict:
        return {
            "cid": self.cid,
            "title": self.title,
            "source_type": self.source_type,
            "authors": json.dumps(self.authors),
            "date_published": self.date_published,
            "citation_style": self.citation_style_detected,
            "raw_fragment": self.raw_citation_fragment,
            "publisher": self.publisher,
            "access_url": self.access_url,
            "doi": self.doi,
            "discovery_method": self.discovery_method,
            "discovery_url": self.discovery_source_url,
            "confidence": self.confidence,
            "library_match": self.library_match_id,
            "share_hash": self.share_hash,
            "created_at": self.created_at,
            "prompt_id": self.prompt_id,
        }


# ── Tests ─────────────────────────────────────────────────────────────────

def test_cid_computation():
    """CID is deterministic content hash for dedup."""
    r1 = CitationRecord(
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N."],
        date_published="2017",
    )
    r2 = CitationRecord(
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N."],
        date_published="2017",
    )
    r3 = CitationRecord(
        title="BERT",
        authors=["Devlin, J."],
        date_published="2019",
    )

    assert len(r1.cid) == 64, f"CID should be 64-char SHA-256 hex, got {len(r1.cid)}"
    assert r1.cid == r2.cid, "Same content → same CID"
    assert r1.cid != r3.cid, "Different content → different CID"

    # Author order shouldn't matter (sorted internally)
    r4 = CitationRecord(
        title="Attention Is All You Need",
        authors=["Shazeer, N.", "Vaswani, A."],  # reversed
        date_published="2017",
    )
    assert r1.cid == r4.cid, "Author order should not affect CID"
    print("  ✓ CID: deterministic, order-independent, 64-char hex")


def test_vector_view_flatness():
    """Vector metadata must be flat primitives for ChromaDB/Qdrant."""
    r = CitationRecord(
        title="Dense Passage Retrieval",
        source_type="conference_paper",
        authors=["Karpukhin, V.", "Oguz, B."],
        date_published="2020",
        citation_style_detected="APA",
        publisher="EMNLP",
        doi="10.18653/v1/2020.emnlp-main.550",
        confidence=0.92,
        prompt_id="p-001",
    )

    vec = r.to_vector_meta()
    for key, val in vec.items():
        assert isinstance(val, (str, int, float)), (
            f"Field '{key}' is {type(val).__name__}, must be str/int/float"
        )

    assert "cid" in vec
    assert "confidence" in vec
    assert vec["source_type"] == "conference_paper"
    print(f"  ✓ Vector view: {len(vec)} flat fields, all primitives")


def test_user_view_privacy():
    """User view must not expose internal/debug fields."""
    r = CitationRecord(
        title="Test Paper",
        source_type="journal_article",
        authors=["Author, A."],
        date_published="2024",
        raw_citation_fragment="Author, A. (2024). Test. Nature.",
        confidence=0.9,
        prompt_id="p-001",
    )

    user = r.to_user_view()
    forbidden = ["cid", "raw_citation_fragment", "confidence",
                  "prompt_id", "discovery_method", "share_hash"]
    for f in forbidden:
        assert f not in user, f"User view should not contain '{f}'"

    assert "title" in user
    assert "authors" in user
    print(f"  ✓ User view: {len(user)} clean fields, no internal data exposed")


def test_a2a_envelope():
    """A2A metadata must be JSON-serializable and typed."""
    citations = [
        CitationRecord(
            title=f"Paper {i}",
            authors=[f"Author{i}, A."],
            date_published=str(2020 + i),
            confidence=0.9,
            prompt_id="p-001",
        )
        for i in range(3)
    ]

    envelope = {
        "schema": "citation_extraction",
        "version": "1.0",
        "prompt_id": "p-001",
        "model": "gemma3",
        "total_citations": len(citations),
        "extraction_time_ms": 1200,
        "citations": [c.to_a2a_meta() for c in citations],
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Must be JSON-serializable
    json_str = json.dumps(envelope)
    reparsed = json.loads(json_str)
    assert reparsed["total_citations"] == 3
    assert reparsed["schema"] == "citation_extraction"
    assert all(c["type"] == "citation_record" for c in reparsed["citations"])
    print(f"  ✓ A2A envelope: valid JSON, {len(json_str)} bytes, typed records")


def test_share_hash_isolation():
    """Different deployments get different hashes for same content."""
    r = CitationRecord(title="Test", authors=["A, B."], date_published="2024")

    h1 = r.compute_share_hash("deployment-alpha-salt-2026")
    h2 = r.compute_share_hash("deployment-beta-salt-2026")

    assert h1 != h2, "Different salts → different hashes"
    assert len(h1) == 64, "Share hash should be 64-char HMAC hex"
    print(f"  ✓ Share hashes: deployment-isolated, 64-char HMAC")


def test_dedup_simulation():
    """Simulate dedup across multiple sources finding the same paper."""
    # Same paper found in 3 different sources
    sources = [
        CitationRecord(
            title="Attention Is All You Need",
            authors=["Vaswani, A."],
            date_published="2017",
            discovery_source_url=f"https://source{i}.example.com",
            confidence=0.8 + (i * 0.05),
        )
        for i in range(3)
    ]

    # Dedup by CID (keep highest confidence)
    seen = {}
    for c in sources:
        if c.cid not in seen or c.confidence > seen[c.cid].confidence:
            seen[c.cid] = c

    assert len(seen) == 1, f"Should dedup to 1 record, got {len(seen)}"
    best = list(seen.values())[0]
    assert best.confidence == 0.9, "Should keep highest confidence"
    print(f"  ✓ Dedup: 3 sources → 1 record, kept confidence={best.confidence}")


def test_concurrent_prompts():
    """Simulate concurrent prompt processing."""

    async def process_prompt(pid: str, n: int):
        await asyncio.sleep(0.005)
        return [
            CitationRecord(
                title=f"Paper-{pid}-{i}",
                authors=[f"Author{i}"],
                date_published="2024",
                prompt_id=pid,
            )
            for i in range(n)
        ]

    async def run():
        start = time.monotonic()
        tasks = [process_prompt(f"p-{i:03d}", 5) for i in range(50)]
        results = await asyncio.gather(*tasks)
        elapsed = (time.monotonic() - start) * 1000

        total = sum(len(batch) for batch in results)
        all_cids = [c.cid for batch in results for c in batch]
        unique_cids = set(all_cids)

        assert total == 250
        assert len(unique_cids) == 250, "All CIDs should be unique across prompts"
        return total, elapsed

    total, ms = asyncio.run(run())
    print(f"  ✓ Concurrent: 50 prompts × 5 citations = {total} records in {ms:.0f}ms")


def test_db_row_serialization():
    """DB row must have JSON-string authors for JSONB column."""
    r = CitationRecord(
        title="Test",
        authors=["Smith, J.", "Doe, A."],
        date_published="2024",
        source_type="journal_article",
        doi="10.1234/test",
        confidence=0.88,
    )

    row = r.to_db_row()
    assert isinstance(row["authors"], str)
    assert json.loads(row["authors"]) == ["Smith, J.", "Doe, A."]
    assert row["cid"] == r.cid
    print(f"  ✓ DB row: {len(row)} columns, authors as JSON string")


def test_sample_article_extraction():
    """Test extraction expectations from the sample article."""
    # Import the sample article's expected results
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from samples.sample_article import SAMPLE_ARTICLE

    expected = SAMPLE_ARTICLE["expected_citations"]

    # Verify each expected citation has required fields
    required = ["title", "source_type", "authors", "discovery_method"]
    for cit in expected:
        for field in required:
            assert field in cit, f"Missing '{field}' in expected citation"

    # Verify style distribution (Art.2: 2-3 standard styles + 1 web source)
    styles = [c.get("citation_style_detected") for c in expected]
    assert "APA" in styles, "Should detect APA citations"
    assert "MLA" in styles, "Should detect MLA citations"
    assert "Chicago" in styles, "Should detect Chicago citations"

    types = [c["source_type"] for c in expected]
    assert "blog_nonacademic" in types, "Should include web/blog source"

    print(f"  ✓ Sample article: {len(expected)} expected citations")
    print(f"    Styles: {[s for s in styles if s]}")
    print(f"    Types: {types}")


# ── Runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("CID Computation (dedup key)", test_cid_computation),
        ("Vector View (ChromaDB/Qdrant)", test_vector_view_flatness),
        ("User View (privacy)", test_user_view_privacy),
        ("A2A Envelope (agent protocol)", test_a2a_envelope),
        ("Share Hash (cross-deployment)", test_share_hash_isolation),
        ("Dedup Across Sources", test_dedup_simulation),
        ("Concurrent Prompts (50x)", test_concurrent_prompts),
        ("DB Row Serialization", test_db_row_serialization),
        ("Sample Article Expectations", test_sample_article_extraction),
    ]

    print("\n" + "═" * 60)
    print("  CITATION PIPELINE — CORE LOGIC TESTS (stdlib only)")
    print("═" * 60)

    passed = failed = 0
    for name, fn in tests:
        print(f"\n▸ {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'═' * 60}")
    status = "ALL PASSED ✓" if failed == 0 else f"{failed} FAILED"
    print(f"  {passed}/{len(tests)} passed — {status}")
    print(f"{'═' * 60}\n")
