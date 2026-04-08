"""
tests/test_pipeline.py — Unit tests for the refactored pipeline.

Covers: model dedup, output splitter, reference parser, source_id
computation, A2A envelope. No Ollama, no ChromaDB required.

Run:
    pytest tests/test_pipeline.py -v
    # or:
    python tests/test_pipeline.py
"""

import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.extractor import CitationExtractor, REFERENCES_MARKER
from core.models import (
    CitationRecord,
    CitationStyle,
    PromptCitationResult,
    SourceType,
    compute_source_id,
)


def test_cid_deterministic():
    a = CitationRecord(title="Attention Is All You Need",
                       authors=["Vaswani, A.", "Shazeer, N."], date_published="2017")
    b = CitationRecord(title="Attention Is All You Need",
                       authors=["Shazeer, N.", "Vaswani, A."], date_published="2017")
    assert a.cid == b.cid
    assert len(a.cid) == 64
    print("  ✓ CID stable across author order")


def test_source_id_prefers_doi():
    sid_doi = compute_source_id(doi="10.1234/x", url="https://a.com", title="T", authors=["A"])
    sid_url = compute_source_id(doi=None, url="https://a.com", title="T", authors=["A"])
    sid_title = compute_source_id(doi=None, url=None, title="T", authors=["A"])
    assert sid_doi != sid_url != sid_title
    # Same DOI → same source_id regardless of URL differences
    sid_doi2 = compute_source_id(doi="10.1234/x", url="https://other.com", title="Z", authors=[])
    assert sid_doi == sid_doi2
    print("  ✓ source_id prefers DOI, stable under URL/title variation")


def test_split_output_with_marker():
    raw = f"The answer is 42.\n\n{REFERENCES_MARKER}\n[{{\"title\":\"X\"}}]"
    answer, refs = CitationExtractor._split_output(raw)
    assert answer == "The answer is 42."
    assert refs.startswith("[")
    print("  ✓ Output splitter extracts answer + refs JSON")


def test_split_output_missing_marker():
    raw = "Just an answer, no references."
    answer, refs = CitationExtractor._split_output(raw)
    assert answer == raw
    assert refs == "[]"
    print("  ✓ Missing marker → graceful degradation (empty refs)")


def test_split_output_strips_fences():
    raw = f"Answer.\n{REFERENCES_MARKER}\n```json\n[{{\"title\":\"X\"}}]\n```"
    _, refs = CitationExtractor._split_output(raw)
    assert refs.startswith("[") and refs.endswith("]")
    print("  ✓ Markdown fences stripped from refs block")


def test_parse_references_quality_filter():
    ex = CitationExtractor.__new__(CitationExtractor)
    # Stub with title+authors but no strong signal → dropped
    weak = ex._parse_references('[{"title":"X","authors":["A"]}]', "p1")
    assert len(weak) == 0
    # Real-looking ref with year → kept
    real = ex._parse_references(
        '[{"title":"Attention Is All You Need","authors":["Vaswani, A."],'
        '"date":"2017","publisher":"NeurIPS","doi":"10.48550/arXiv.1706.03762"}]',
        "p1",
    )
    assert len(real) == 1
    assert real[0].doi == "10.48550/arXiv.1706.03762"
    assert real[0].prompt_id == "p1"
    print("  ✓ Quality filter drops stubs, keeps real citations")


def test_parse_references_salvage_malformed():
    ex = CitationExtractor.__new__(CitationExtractor)
    # Truncated JSON — missing closing ]
    malformed = '[{"title":"BERT Pre-training","authors":["Devlin, J."],"date":"2019","publisher":"NAACL"}'
    records = ex._parse_references(malformed, "p1")
    assert len(records) == 1
    print("  ✓ Salvage recovers citations from truncated JSON")


def test_a2a_envelope_shape():
    result = PromptCitationResult(
        prompt_id="p1",
        user_query="q",
        citations=[
            CitationRecord(
                title="Attention Is All You Need",
                authors=["Vaswani, A."],
                date_published="2017",
                source_type=SourceType.CONFERENCE_PAPER,
                citation_style_detected=CitationStyle.APA,
                publisher="NeurIPS",
                doi="10.48550/arXiv.1706.03762",
                confidence=0.95,
                prompt_id="p1",
            ),
        ],
    )
    env = result.to_a2a_envelope()
    assert env["schema"] == "citation_extraction"
    assert env["total_citations"] == 1
    cit = env["citations"][0]
    assert cit["type"] == "citation_record"
    assert "source_id" in cit  # new field
    assert "source_cached" in cit
    # Round-trip JSON
    json.dumps(env, default=str)
    print("  ✓ A2A envelope has source_id + source_cached, JSON-serializable")


def test_user_view_hides_internals():
    r = CitationRecord(
        title="X", authors=["A, B."], date_published="2020",
        publisher="P", doi="10.1/x", confidence=0.9, prompt_id="p1",
    )
    u = r.to_user_view()
    assert "cid" not in u
    assert "source_id" not in u
    assert "confidence" not in u
    assert u["title"] == "X"
    assert u["doi"] == "10.1/x"
    print("  ✓ User view clean, no internal fields")


if __name__ == "__main__":
    tests = [
        ("CID deterministic", test_cid_deterministic),
        ("source_id computation", test_source_id_prefers_doi),
        ("Output splitter (with marker)", test_split_output_with_marker),
        ("Output splitter (missing marker)", test_split_output_missing_marker),
        ("Output splitter (fenced)", test_split_output_strips_fences),
        ("Reference quality filter", test_parse_references_quality_filter),
        ("Reference salvage", test_parse_references_salvage_malformed),
        ("A2A envelope", test_a2a_envelope_shape),
        ("User view", test_user_view_hides_internals),
    ]
    print("\n" + "=" * 60)
    print("CITATION PIPELINE — REFACTORED TESTS")
    print("=" * 60)
    passed = 0
    for name, fn in tests:
        print(f"\n▸ {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 60}\nResults: {passed}/{len(tests)} passed\n{'=' * 60}\n")
    sys.exit(0 if passed == len(tests) else 1)
