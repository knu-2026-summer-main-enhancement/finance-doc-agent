from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from datastore.scope import document_scope
from rag.vector import (
    _build_vector_evidence,
    _ensure_multi_document_coverage,
    _is_numeric_eligibility_question,
    _numeric_evidence_documents,
    _numeric_llm_answer_is_complete,
    _required_evidence_missing,
    _retrieve_verified_documents,
)


class _LowScoreHeadingStore:
    """A short Korean section title below the semantic cutoff but in the text."""

    def __init__(self, *documents: Document):
        self.documents = documents

    def similarity_search_with_relevance_scores(self, *_args, **_kwargs):
        return [(document, 0.1552) for document in self.documents]

    def max_marginal_relevance_search(self, *_args, **_kwargs):
        return list(self.documents)


class _ScoredStore(_LowScoreHeadingStore):
    def __init__(self, scored, siblings=None):
        self.scored = scored
        self.documents = tuple(document for document, _score in scored)
        self.siblings = siblings or []

    def similarity_search_with_relevance_scores(self, *_args, **_kwargs):
        return self.scored

    def get(self, *_args, **_kwargs):
        return {
            "documents": [doc.page_content for doc in self.siblings],
            "metadatas": [doc.metadata for doc in self.siblings],
        }


class VectorRetrievalTests(unittest.TestCase):
    def test_exact_section_heading_is_kept_below_semantic_cutoff(self):
        source = "scholarship-plan.pdf"
        document = Document(
            page_content="I. 추진배경\n학생 지원이 필요한 이유를 설명합니다.",
            metadata={"source": source, "page": 1},
        )
        with (
            document_scope([source]),
            patch("rag.vector.get_vectorstore", return_value=_LowScoreHeadingStore(document)),
        ):
            docs = asyncio.run(_retrieve_verified_documents(["추진배경 뭐야?"]))

        self.assertEqual(docs, [document])

    def test_legacy_heading_does_not_expand_unrelated_page_chunks(self):
        source = "scholarship-plan.pdf"
        heading = Document(
            page_content="PDF 문서의 섹션 제목(검색 키워드): I 추진배경",
            metadata={"source": source, "page": 1},
        )
        prose = Document(
            page_content="장학 제도의 사각지대에 놓인 학생을 지원합니다.",
            metadata={"source": source, "page": 1},
        )
        other_page = Document(
            page_content="신청 서류와 접수 방법입니다.",
            metadata={"source": source, "page": 3},
        )
        with (
            document_scope([source]),
            patch("rag.vector.get_vectorstore", return_value=_LowScoreHeadingStore(heading, prose, other_page)),
        ):
            docs = asyncio.run(_retrieve_verified_documents(["추진배경 뭐야?"]))

        self.assertEqual(docs, [heading])

    def test_section_chunk_does_not_pull_the_next_section_from_same_page(self):
        source = "scholarship-plan.pdf"
        section = Document(
            page_content="PDF 문서 섹션:\nI 추진배경\n첫 번째와 두 번째 사유",
            metadata={
                "source": source,
                "page": 1,
                "content_type": "pdf_section_child",
                "section_id": "p1:s0",
            },
        )
        next_section = Document(
            page_content="II 관련근거\n규정 내용",
            metadata={"source": source, "page": 1},
        )
        with (
            document_scope([source]),
            patch("rag.vector.get_vectorstore", return_value=_LowScoreHeadingStore(section, next_section)),
        ):
            docs = asyncio.run(_retrieve_verified_documents(["추진배경 뭐야?"]))

        self.assertEqual(docs, [section])

    def test_best_candidate_is_kept_without_an_absolute_score_cutoff(self):
        source = "scholarship-plan.pdf"
        document = Document(
            page_content="III. 신청 절차\n제출 서류와 접수 방법입니다.",
            metadata={"source": source, "page": 3},
        )
        with (
            document_scope([source]),
            patch("rag.vector.get_vectorstore", return_value=_LowScoreHeadingStore(document)),
        ):
            docs = asyncio.run(_retrieve_verified_documents(["추진배경 뭐야?"]))

        self.assertEqual(docs, [document])

    def test_relative_score_margin_drops_far_weaker_candidate(self):
        source = "scholarship-plan.pdf"
        best = Document(
            page_content="지원 목적과 추진 배경",
            metadata={"source": source, "page": 1},
        )
        weak = Document(
            page_content="신청서 서식",
            metadata={"source": source, "page": 4},
        )
        store = _ScoredStore([(best, 0.22), (weak, -0.05)])
        with (
            document_scope([source]),
            patch("rag.vector.get_vectorstore", return_value=store),
        ):
            docs = asyncio.run(_retrieve_verified_documents(["지원 취지가 뭐야?"]))

        self.assertEqual(docs, [best])

    def test_selected_child_expands_to_complete_parent_section(self):
        source = "scholarship-plan.pdf"
        first = Document(
            page_content="추진배경\n첫 번째 이유",
            metadata={
                "source": source,
                "page": 1,
                "content_type": "pdf_section_child",
                "section_id": "p1:s0",
                "section_title": "Ⅰ 추진배경",
                "child_index": 0,
            },
        )
        second = Document(
            page_content="추진배경\n두 번째와 세 번째 이유",
            metadata={
                "source": source,
                "page": 1,
                "content_type": "pdf_section_child",
                "section_id": "p1:s0",
                "section_title": "Ⅰ 추진배경",
                "child_index": 1,
            },
        )
        store = _ScoredStore([(second, 0.12)], siblings=[second, first])
        with (
            document_scope([source]),
            patch("rag.vector.get_vectorstore", return_value=store),
        ):
            docs = asyncio.run(_retrieve_verified_documents(["추진배경 뭐야?"]))

        self.assertEqual(docs, [first, second])

    def test_large_parent_expands_only_around_best_child(self):
        source = "plan.pdf"
        siblings = [
            Document(
                page_content=f"subsection {index}",
                metadata={
                    "source": source,
                    "page": index + 1,
                    "content_type": "pdf_section_child",
                    "section_id": "p1:s0",
                    "section_title": "Ⅲ 선발계획",
                    "child_index": index,
                    "child_count": 7,
                },
            )
            for index in range(7)
        ]
        selected = siblings[1]
        store = _ScoredStore([(selected, 0.2)], siblings=siblings)
        with (
            document_scope([source]),
            patch("rag.vector.get_vectorstore", return_value=store),
        ):
            docs = asyncio.run(_retrieve_verified_documents(["선발 대상 뭐야"]))

        self.assertEqual(docs, siblings[0:3])

    def test_vector_evidence_groups_parent_children(self):
        documents = [
            Document(
                page_content="first",
                metadata={
                    "source": "plan.pdf",
                    "page": 3,
                    "section_id": "p3:s0",
                    "section_title": "Ⅳ 추진일정",
                    "child_count": 2,
                },
            ),
            Document(
                page_content="second",
                metadata={
                    "source": "plan.pdf",
                    "page": 3,
                    "section_id": "p3:s0",
                    "section_title": "Ⅳ 추진일정",
                    "child_count": 2,
                },
            ),
        ]

        evidence = _build_vector_evidence(documents)

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["section_title"], "Ⅳ 추진일정")
        self.assertEqual(evidence[0]["chunk_count"], 2)

    def test_explicit_comparison_covers_each_selected_document(self):
        first_source = "first.pdf"
        second_source = "second.pdf"
        first = Document(
            page_content="첫 문서 지원 목적",
            metadata={"source": first_source, "page": 1},
        )
        second = Document(
            page_content="둘째 문서 지원 목적",
            metadata={"source": second_source, "page": 1},
        )

        class _PerSourceStore:
            def similarity_search_with_relevance_scores(self, *_args, **kwargs):
                source = kwargs["filter"]["source"]
                return [(second, 0.1)] if source == second_source else [(first, 0.1)]

        with document_scope([first_source, second_source]):
            docs = asyncio.run(_ensure_multi_document_coverage(
                _PerSourceStore(),
                ["두 문서의 지원 목적을 비교해줘"],
                [first],
            ))

        self.assertEqual(docs, [first, second])

    def test_numeric_eligibility_question_uses_dedicated_llm_path(self):
        self.assertTrue(_is_numeric_eligibility_question(
            "학부생이 직전학기 평점평균 1.9면 장학금 받을 수 있음?"
        ))
        self.assertFalse(_is_numeric_eligibility_question(
            "학부생 장학금 선발 기준 알려줘"
        ))

    def test_numeric_eligibility_context_keeps_measurement_chunk(self):
        relevant = Document(
            page_content="(학부) 직전학기 평점평균 2.0 이상",
            metadata={"source": "plan.pdf"},
        )
        unrelated = Document(
            page_content="장학생 추천 절차와 제출 서류",
            metadata={"source": "plan.pdf"},
        )

        docs = _numeric_evidence_documents(
            "학부생 평점평균 1.9면 받을 수 있어?",
            [unrelated, relevant],
        )

        self.assertEqual(docs, [relevant])

    def test_numeric_llm_answer_requires_conclusion_and_two_values(self):
        self.assertFalse(_numeric_llm_answer_is_complete(
            "충족하지 않습니다."
        ))
        self.assertTrue(_numeric_llm_answer_is_complete(
            "충족하지 않습니다. 1.9점은 문서 기준 2.0 이상보다 낮습니다."
        ))

    def test_schedule_question_with_how_does_not_become_procedure_question(self):
        docs = [
            Document(
                page_content=(
                    "Ⅳ 향후 추진일정\n"
                    "장학생 추천: 2026. 4. 28.까지\n"
                    "장학생 선발 확정: 2026. 5. 1.까지\n"
                    "장학금 지급: 2026. 5월 중"
                )
            )
        ]

        self.assertEqual(
            _required_evidence_missing("향후 추진일정은 어떻게 돼?", docs),
            "",
        )

    def test_application_question_still_checks_procedure_evidence(self):
        docs = [Document(page_content="장학금 지원 규모는 1인당 400만원입니다.")]

        self.assertEqual(
            _required_evidence_missing("장학금은 어떻게 신청해?", docs),
            "질문한 절차나 방법은 문서에서 확인할 수 없습니다.",
        )
