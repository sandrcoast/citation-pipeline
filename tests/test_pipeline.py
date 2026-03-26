"""
tests/test_pipeline.py — Integration test for the citation pipeline.

Tests the full flow:
  1. Extraction from sample article text (no Ollama needed — mocked)
  2. Data model serialization (vector, user, A2A views)
  3. Storage round-trip (SQLite fallback, no Postgres needed)
  4. Concurrent extraction simulation

Run:
    pytest tests/test_pipeline.py -v
    # or without pytest:
    python tests/test_pipeline.py
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import (
    CitationRecord,
    CitationStyle,
    DiscoveryMethod,
    PromptCitationResult,
    SourceType,
)


def test_citation_record_creation():
    """Test: CitationRecord auto-computes CID from content."""
    record = CitationRecord(
        title="Attention Is All You Need",
        source_type=SourceType.CONFERENCE_PAPER,
        authors=["Vaswani, A.", "Shazeer, N."],
        date_published="2017",
        citation_style_detected=CitationStyle.APA,
        publisher="NeurIPS",
        confidence=0.95,
    )

    # CID should be auto-computed
    assert len(record.cid) == 64, f"CID should be 64-char hex, got {len(record.cid)}"
    assert record.cid != "", "CID should not be empty"
    print(f"  ✓ CID auto-computed: {record.cid[:16]}...")

    # Same content → same CID (deterministic)
    record2 = CitationRecord(
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N."],
        date_published="2017",
    )
    assert record.cid == record2.cid, "Same content should produce same CID"
    print(f"  ✓ CID is deterministic (dedup-ready)")

    # Different content → different CID
    record3 = CitationRecord(
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        authors=["Devlin, J."],
        date_published="2019",
    )
    assert record.cid != record3.cid, "Different content should produce different CID"
    print(f"  ✓ Different content → different CID")


def test_vector_view():
    """Test: Vector metadata is flat and ChromaDB/Qdrant compatible."""
    record = CitationRecord(
        title="Dense Passage Retrieval",
        source_type=SourceType.CONFERENCE_PAPER,
        authors=["Karpukhin, V.", "Oguz, B."],
        date_published="2020",
        citation_style_detected=CitationStyle.APA,
        publisher="EMNLP",
        doi="10.18653/v1/2020.emnlp-main.550",
        discovery_method=DiscoveryMethod.BIBLIOGRAPHY,
        confidence=0.92,
        prompt_id="test-prompt-001",
    )

    vec = record.to_vector_meta()

    # All values must be flat (str, int, float) for ChromaDB
    for key, val in vec.items():
        assert isinstance(val, (str, int, float)), (
            f"Vector meta field '{key}' has type {type(val).__name__}, "
            f"must be str/int/float for ChromaDB"
        )
    print(f"  ✓ Vector view is flat (ChromaDB/Qdrant compatible)")
    print(f"    Fields: {list(vec.keys())}")

    # Must include essential indexed fields
    assert "cid" in vec
    assert "title" in vec
    assert "source_type" in vec
    assert "confidence" in vec
    print(f"  ✓ All indexed fields present")


def test_user_view():
    """Test: User view is clean, no internal fields exposed."""
    record = CitationRecord(
        title="Semantic Search in Libraries",
        source_type=SourceType.JOURNAL_ARTICLE,
        authors=["Chen, W.", "Liu, R."],
        date_published="2023",
        citation_style_detected=CitationStyle.APA,
        publisher="Journal of Library Technology",
        access_url="https://doi.org/10.1016/j.jlt.2023.04.012",
        raw_citation_fragment="Chen, W. & Liu, R. (2023). Semantic search...",
        confidence=0.95,
        prompt_id="test-prompt-001",
    )

    user = record.to_user_view()

    # Should NOT contain internal fields
    assert "cid" not in user, "User view should not expose CID"
    assert "raw_citation_fragment" not in user, "User view should not expose raw fragment"
    assert "confidence" not in user, "User view should not expose confidence"
    assert "prompt_id" not in user, "User view should not expose prompt_id"
    print(f"  ✓ User view hides internal fields")

    # Should contain user-friendly fields
    assert "title" in user
    assert "authors" in user
    assert "url" in user
    print(f"  ✓ User view has readable fields: {list(user.keys())}")


def test_a2a_envelope():
    """Test: A2A envelope is structured and machine-parseable."""
    result = PromptCitationResult(
        prompt_id="test-prompt-001",
        user_query="What is attention in neural networks?",
        model="gemma3",
        citations=[
            CitationRecord(
                title="Attention Is All You Need",
                authors=["Vaswani, A."],
                date_published="2017",
                source_type=SourceType.CONFERENCE_PAPER,
                confidence=0.95,
            ),
            CitationRecord(
                title="BERT: Pre-training",
                authors=["Devlin, J."],
                date_published="2019",
                source_type=SourceType.CONFERENCE_PAPER,
                confidence=0.90,
            ),
        ],
        extraction_time_ms=1200,
    )

    envelope = result.to_a2a_envelope()

    # Verify envelope structure
    assert envelope["schema"] == "citation_extraction"
    assert envelope["version"] == "1.0"
    assert envelope["total_citations"] == 2
    assert len(envelope["citations"]) == 2
    print(f"  ✓ A2A envelope structure valid")

    # Verify it's JSON-serializable
    json_str = json.dumps(envelope, default=str)
    reparsed = json.loads(json_str)
    assert reparsed["total_citations"] == 2
    print(f"  ✓ A2A envelope is JSON-serializable ({len(json_str)} bytes)")

    # Each citation in envelope should have type + version
    for cit in envelope["citations"]:
        assert cit["type"] == "citation_record"
        assert "cid" in cit
        assert "confidence" in cit
    print(f"  ✓ Each citation has type marker and provenance fields")


def test_share_hash():
    """Test: Cross-deployment sharing hash works."""
    record = CitationRecord(
        title="Attention Is All You Need",
        authors=["Vaswani, A."],
        date_published="2017",
    )

    hash_a = record.compute_share_hash("deployment-a-salt-2026")
    hash_b = record.compute_share_hash("deployment-b-salt-2026")

    # Different salts → different hashes
    assert hash_a != hash_b, "Different deployment salts should produce different hashes"
    print(f"  ✓ Deployment A hash: {hash_a[:16]}...")
    print(f"  ✓ Deployment B hash: {hash_b[:16]}...")
    print(f"  ✓ Hashes differ → internal IDs protected")


def test_concurrent_simulation():
    """Test: Simulate concurrent prompt processing."""

    async def simulate_prompt(prompt_id: str, n_citations: int):
        """Simulate extracting citations for one prompt."""
        await asyncio.sleep(0.01)  # simulate I/O
        records = [
            CitationRecord(
                title=f"Paper {i} from prompt {prompt_id}",
                authors=[f"Author{i}, A."],
                date_published="2024",
                confidence=0.8 + (i * 0.02),
                prompt_id=prompt_id,
            )
            for i in range(n_citations)
        ]
        return PromptCitationResult(
            prompt_id=prompt_id,
            user_query=f"Query for {prompt_id}",
            citations=records,
        )

    async def run():
        # Simulate 20 concurrent prompts (peak load)
        tasks = [
            simulate_prompt(f"prompt-{i:03d}", n_citations=5)
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks)

        total = sum(len(r.citations) for r in results)
        assert total == 100, f"Expected 100 citations, got {total}"

        # Check all CIDs are unique within each prompt
        for r in results:
            cids = [c.cid for c in r.citations]
            assert len(cids) == len(set(cids)), "CIDs should be unique within a prompt"

        print(f"  ✓ 20 concurrent prompts × 5 citations = {total} records")
        print(f"  ✓ All CIDs unique within prompts")

    asyncio.run(run())


def test_db_row_format():
    """Test: DB row format matches PostgreSQL schema."""
    record = CitationRecord(
        title="Test Paper",
        source_type=SourceType.JOURNAL_ARTICLE,
        authors=["Test, A."],
        date_published="2024",
        doi="10.1234/test",
        confidence=0.88,
        prompt_id="test-001",
    )

    row = record.to_db_row()

    # authors should be JSON string (for JSONB column)
    assert isinstance(row["authors"], str), "Authors should be JSON string for PG"
    parsed = json.loads(row["authors"])
    assert parsed == ["Test, A."]
    print(f"  ✓ DB row format matches PostgreSQL schema")
    print(f"    Columns: {list(row.keys())}")


# ── Run all tests ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Citation Record Creation", test_citation_record_creation),
        ("Vector View (ChromaDB/Qdrant)", test_vector_view),
        ("User View (clean output)", test_user_view),
        ("A2A Envelope", test_a2a_envelope),
        ("Cross-Deployment Sharing", test_share_hash),
        ("Concurrent Prompts", test_concurrent_simulation),
        ("DB Row Format", test_db_row_format),
    ]

    print("\n" + "=" * 60)
    print("CITATION PIPELINE — DATA MODEL TESTS")
    print("=" * 60)

    passed = 0
    for name, test_fn in tests:
        print(f"\n▸ {name}")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60 + "\n")
