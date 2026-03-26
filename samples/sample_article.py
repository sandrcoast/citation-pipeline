# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Citation Pipeline Contributors
# Sample Article: Placeholder for Citation Extraction Testing
# This file simulates a digitized article with mixed citation styles.
# The article content is fictional — used as a test fixture.

SAMPLE_ARTICLE = {
    "title": "Advances in Neural Retrieval-Augmented Generation for Academic Libraries",
    "author": "Doe, Jane",
    "affiliation": "Department of Computer Science, State University",
    "date": "2025-09-15",
    "doi": "10.1234/sample.2025.0042",
    "url": "https://library.stateuniversity.edu/articles/2025/doe-neural-rag",

    # The full text of the article, containing inline citations
    # and a bibliography section at the end.
    "full_text": """
Advances in Neural Retrieval-Augmented Generation for Academic Libraries

Jane Doe
Department of Computer Science, State University
Published: September 15, 2025

Abstract

This paper examines the application of retrieval-augmented generation (RAG)
techniques to academic library search systems. We build on foundational work
in attention mechanisms (Vaswani et al., 2017) and recent advances in dense
retrieval (Karpukhin et al., 2020) to propose a hybrid system that serves
both students and faculty researchers.

1. Introduction

The landscape of academic information retrieval has shifted dramatically since
the introduction of transformer architectures. As noted by Chen and Liu (2023),
"traditional keyword-based search fails to capture the semantic nuance required
for interdisciplinary research" (p. 45). Our approach leverages dense passage
retrieval combined with generative summarization to bridge this gap.

Modern library systems must contend with heterogeneous document formats.
According to a comprehensive survey by the Association of Research Libraries
(2024), over 78% of university libraries now maintain fully digitized
collections, yet only 12% have implemented semantic search capabilities.

2. Related Work

The foundational transformer architecture was introduced by Vaswani et al.
in their landmark paper, enabling parallel processing of sequence data that
fundamentally changed NLP. Dense passage retrieval, as formalized by
Karpukhin et al. (2020), demonstrated that learned representations
outperform BM25 on open-domain question answering benchmarks.

Smith and Patel (2024) extended these ideas to domain-specific academic
search, reporting a 34% improvement in retrieval accuracy on the TREC-LIB
benchmark. Their work, published in the Journal of Information Science,
provides the closest baseline to our approach.

3. Methodology

[... methodology section omitted for brevity ...]

4. Results

[... results section omitted for brevity ...]

5. Conclusion

Our hybrid RAG system demonstrates significant improvements over existing
academic library search tools. Future work will focus on multilingual
retrieval and integration with institutional repositories.

References

[APA Style Citations]
Chen, W., & Liu, R. (2023). Semantic search in academic libraries: Challenges
    and opportunities. Journal of Library Technology, 58(3), 42-61.
    https://doi.org/10.1016/j.jlt.2023.04.012

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D.,
    & Yih, W. (2020). Dense passage retrieval for open-domain question
    answering. In Proceedings of the 2020 Conference on Empirical Methods
    in Natural Language Processing (EMNLP) (pp. 6769-6781).
    https://doi.org/10.18653/v1/2020.emnlp-main.550

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
    Kaiser, L., & Polosukhin, I. (2017). Attention is all you need.
    In Advances in Neural Information Processing Systems (NeurIPS), 30.

[MLA Style Citation — mixed in from a co-author's section]
Smith, Robert, and Priya Patel. "Domain-Specific Dense Retrieval for Academic
    Search." Journal of Information Science, vol. 49, no. 2, 2024, pp. 178-195.

[Chicago Style Citation — from a footnote in section 1]
Association of Research Libraries. "2024 Survey of Digital Library
    Infrastructure." Report. Washington, DC: ARL, 2024.
    https://www.arl.org/reports/digital-infrastructure-2024.

[Web Source — non-academic blog post referenced in introduction]
Martinez, Carlos. "Why University Search Still Sucks." TechEd Blog,
    March 12, 2025. https://teched-blog.example.com/university-search-2025.
""",

    # ── EXPECTED EXTRACTION RESULTS ──────────────────────────────────
    # This is what our pipeline should produce from the article above.
    "expected_citations": [
        {
            "cid": "PLACEHOLDER_HASH_1",
            "title": "Semantic search in academic libraries: Challenges and opportunities",
            "source_type": "journal_article",
            "authors": ["Chen, W.", "Liu, R."],
            "date_published": "2023",
            "citation_style_detected": "APA",
            "raw_citation_fragment": (
                'Chen, W., & Liu, R. (2023). Semantic search in academic libraries: '
                'Challenges and opportunities. Journal of Library Technology, 58(3), 42-61.'
            ),
            "publisher": "Journal of Library Technology",
            "access_url": "https://doi.org/10.1016/j.jlt.2023.04.012",
            "doi": "10.1016/j.jlt.2023.04.012",
            "discovery_method": "in_page_bibliography",
            "confidence": 0.95,
        },
        {
            "cid": "PLACEHOLDER_HASH_2",
            "title": "Dense passage retrieval for open-domain question answering",
            "source_type": "conference_paper",
            "authors": ["Karpukhin, V.", "Oguz, B.", "Min, S.", "Lewis, P.",
                        "Wu, L.", "Edunov, S.", "Chen, D.", "Yih, W."],
            "date_published": "2020",
            "citation_style_detected": "APA",
            "raw_citation_fragment": (
                'Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., '
                'Chen, D., & Yih, W. (2020). Dense passage retrieval for open-domain '
                'question answering. In Proceedings of EMNLP (pp. 6769-6781).'
            ),
            "publisher": "EMNLP",
            "access_url": "https://doi.org/10.18653/v1/2020.emnlp-main.550",
            "doi": "10.18653/v1/2020.emnlp-main.550",
            "discovery_method": "in_page_bibliography",
            "confidence": 0.97,
        },
        {
            "cid": "PLACEHOLDER_HASH_3",
            "title": "Attention is all you need",
            "source_type": "conference_paper",
            "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N.",
                        "Uszkoreit, J.", "Jones, L.", "Gomez, A. N.",
                        "Kaiser, L.", "Polosukhin, I."],
            "date_published": "2017",
            "citation_style_detected": "APA",
            "raw_citation_fragment": (
                'Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is '
                'all you need. In Advances in Neural Information Processing Systems.'
            ),
            "publisher": "NeurIPS",
            "access_url": None,
            "doi": None,
            "discovery_method": "in_page_bibliography",
            "confidence": 0.96,
        },
        {
            "cid": "PLACEHOLDER_HASH_4",
            "title": "Domain-Specific Dense Retrieval for Academic Search",
            "source_type": "journal_article",
            "authors": ["Smith, R.", "Patel, P."],
            "date_published": "2024",
            "citation_style_detected": "MLA",
            "raw_citation_fragment": (
                'Smith, Robert, and Priya Patel. "Domain-Specific Dense Retrieval '
                'for Academic Search." Journal of Information Science, vol. 49, no. 2, '
                '2024, pp. 178-195.'
            ),
            "publisher": "Journal of Information Science",
            "access_url": None,
            "doi": None,
            "discovery_method": "in_page_bibliography",
            "confidence": 0.93,
        },
        {
            "cid": "PLACEHOLDER_HASH_5",
            "title": "2024 Survey of Digital Library Infrastructure",
            "source_type": "institutional_report",
            "authors": ["Association of Research Libraries"],
            "date_published": "2024",
            "citation_style_detected": "Chicago",
            "raw_citation_fragment": (
                'Association of Research Libraries. "2024 Survey of Digital Library '
                'Infrastructure." Report. Washington, DC: ARL, 2024.'
            ),
            "publisher": "ARL",
            "access_url": "https://www.arl.org/reports/digital-infrastructure-2024",
            "doi": None,
            "discovery_method": "in_page_bibliography",
            "confidence": 0.88,
        },
        {
            "cid": "PLACEHOLDER_HASH_6",
            "title": "Why University Search Still Sucks",
            "source_type": "blog_nonacademic",
            "authors": ["Martinez, C."],
            "date_published": "2025",
            "citation_style_detected": None,  # blog, no formal style
            "raw_citation_fragment": (
                'Martinez, Carlos. "Why University Search Still Sucks." TechEd Blog, '
                'March 12, 2025.'
            ),
            "publisher": "TechEd Blog",
            "access_url": "https://teched-blog.example.com/university-search-2025",
            "doi": None,
            "discovery_method": "in_page_bibliography",
            "confidence": 0.80,  # lower: non-academic, no DOI
        },
    ],
}


if __name__ == "__main__":
    import json
    print(json.dumps(SAMPLE_ARTICLE["expected_citations"], indent=2, default=str))
    print(f"\nTotal expected citations: {len(SAMPLE_ARTICLE['expected_citations'])}")
