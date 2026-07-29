from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from datastore.scope import document_scope
from rag.vector import (
    _DOCUMENT_REASONING_RE,
    _DOC_EXPLAIN_RE,
    _build_vector_evidence,
    _conditional_evidence_documents,
    _condition_alignment_notes,
    _ensure_multi_document_coverage,
    _ensure_section_completeness,
    _explicit_exclusion_answer,
    _explicit_scope_or_support_answer,
    _explicit_mandatory_condition_answer,
    _explicit_preference_answer,
    _explicit_priority_answer,
    _explicit_scalar_field_answer,
    _explicit_table_comparison_answer,
    _repair_explicit_difference_answer,
    _ensure_exception_evidence,
    _is_numeric_eligibility_question,
    _numeric_evidence_documents,
    _numeric_llm_answer_is_complete,
    _numeric_answer_issue,
    _numeric_relation_contradiction,
    _explicit_numeric_criteria_answer,
    _document_reasoning_issue,
    _required_evidence_missing,
    _rerank_candidates,
    _retrieve_verified_documents,
    _table_alignment_notes,
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
    def test_colloquial_origin_question_uses_document_explanation_search(self):
        for question in (
            "이 장학금은 왜 만들어졌어?",
            "이 제도가 생긴 이유가 뭐야?",
        ):
            with self.subTest(question=question):
                self.assertIsNotNone(_DOC_EXPLAIN_RE.search(question))

    def test_specific_body_match_survives_generic_title_match(self):
        generic = Document(
            page_content="추천서와 명단을 제출",
            metadata={"source": "plan.pdf", "section_title": "장학생 추천"},
        )
        specific = Document(
            page_content="외부재단 장학금 수혜자는 타장학금 수혜 가능 여부 확인",
            metadata={"source": "plan.pdf", "section_title": "선발 기준"},
        )
        candidates = {
            ("generic", "", ""): {
                "doc": generic,
                "score": 0.5,
                "queries": {"외부재단 장학생 추천"},
            },
            ("specific", "", ""): {
                "doc": specific,
                "score": 0.5,
                "queries": {"외부재단 장학생 추천"},
            },
        }

        ranked = _rerank_candidates(
            ["외부재단 장학생 추천"],
            candidates,
            [],
        )

        self.assertIn(specific, ranked)

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

    def test_process_answer_appends_omitted_source_item(self):
        docs = [
            Document(
                page_content=(
                    "1. 선발 절차\n"
                    "◦ (학생과) 대상인원 배정\n"
                    "◦ (단과대학) 심사 및 추천\n"
                    "◦ (학생과) 최종 선발 및 지급"
                ),
                metadata={"section_id": "p1:s0", "child_count": 1},
            )
        ]

        answer = _ensure_section_completeness(
            "단과대학이 심사 및 추천합니다.",
            docs,
        )

        self.assertIn("대상인원 배정", answer)
        self.assertIn("최종 선발 및 지급", answer)

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
        self.assertTrue(_is_numeric_eligibility_question(
            "직전학기 4.0이고 총평점 3.4면 추천할 수 있어?"
        ))
        self.assertFalse(_is_numeric_eligibility_question(
            "학부생 장학금 선발 기준 알려줘"
        ))
        self.assertFalse(_is_numeric_eligibility_question(
            "2026년 8월 졸업 예정자는 추천 가능해?"
        ))

    def test_document_reasoning_detects_table_and_conditional_questions(self):
        questions = (
            "직전학기 평점이 3.55면 반영점수가 얼마야?",
            "기초생활수급자와 3구간 학생의 점수 차이는?",
            "성적이 높으면 소득구간과 관계없이 무조건 선발돼?",
            "규정에 해당하면 추천할 수 있나?",
        )

        for question in questions:
            with self.subTest(question=question):
                self.assertIsNotNone(_DOCUMENT_REASONING_RE.search(question))

    def test_pdf_table_alignment_notes_preserve_positional_mapping(self):
        docs = [
            Document(page_content=(
                "학자금지원\n"
                "기초 차상위 1구간 2구간 3구간\n"
                "구간\n"
                "반영점수 50 48 46 44 42"
            ))
        ]

        note = _table_alignment_notes(docs)

        self.assertIn("기초 = 50점", note)
        self.assertIn("3구간 = 42점", note)

    def test_pdf_range_table_notes_preserve_boundaries_and_scores(self):
        docs = [
            Document(page_content=(
                "성적기준 반영점수(50점)\n"
                "직전학기 4.50 4.40 4.30\n"
                "~ ~ ~ 4.1\n"
                "평점평균 4.41 4.31 4.21\n"
                "반영점수 50 48 46 44"
            ))
        ]

        note = _table_alignment_notes(docs)

        self.assertIn("4.41~4.50 = 50점", note)
        self.assertIn("4.1 = 44점", note)

    def test_reasoning_repair_detects_unconditional_contradiction(self):
        issue = _document_reasoning_issue(
            "성적이 높으면 소득구간과 관계없이 무조건 선발돼?",
            "소득 구간에 상관없이 무조건 선발됩니다.",
            "자격 요건: 성적 및 소득 기준 모두 충족 시 추천 가능",
        )

        self.assertIn("모두 충족", issue)

    def test_reasoning_repair_detects_exclusion_article_contradiction(self):
        issue = _document_reasoning_issue(
            "장학금규정 제12조에 해당하면 추천할 수 있나?",
            "제12조에 해당하는 자는 추천할 수 있습니다.",
            "부산대학교 장학금규정 제12조에 해당하는 경우 제외",
        )

        self.assertIn("제외 사유", issue)

    def test_reasoning_repair_detects_omitted_score_input(self):
        issue = _document_reasoning_issue(
            "직전학기 평점 4.15의 반영점수는?",
            "4.10 이상이면 42점입니다.",
            "[PDF 범위표 열 정렬 보조] 4.11~4.20 = 44점",
        )

        self.assertIn("4.15", issue)

    def test_reasoning_repair_detects_wrong_range_score(self):
        issue = _document_reasoning_issue(
            "직전학기 평점 4.15의 반영점수는?",
            "평점 4.15의 반영점수는 42점입니다.",
            "[PDF 범위표 열 정렬 보조] 4.11~4.20 = 44점",
        )

        self.assertIn("44점", issue)

    def test_reasoning_repair_detects_unsupported_graduate_scope(self):
        issue = _document_reasoning_issue(
            "대학원생도 장학금을 받을 수 있어?",
            "대학원생도 장학금을 받을 수 있습니다.",
            "선발인원: 학부 재학생 27명",
        )

        self.assertIn("범위를 확대", issue)

    def test_reasoning_repair_requires_explicit_support_type(self):
        issue = _document_reasoning_issue(
            "등록금 감면 장학금이야?",
            "등록금 감면 장학금이 아닙니다.",
            "지원내용: 학업장려금(생활비)",
        )

        self.assertIn("학업장려금", issue)

    def test_exact_article_exclusion_can_fall_back_to_source_clause(self):
        answer = _explicit_exclusion_answer(
            "장학금규정 제12조에 해당하면 추천할 수 있나?",
            [Document(page_content=(
                "- 부산대학교 장학금규정 제12조에 해당하는 경우 제외"
            ))],
        )

        self.assertIn("제12조", answer)
        self.assertIn("추천 제외 대상", answer)

    def test_qualified_answer_appends_explicit_source_exception(self):
        answer = _ensure_exception_evidence(
            "공대 졸업 예정자도 무조건 제외돼?",
            "공대 졸업 예정자가 무조건 제외되는 것은 아닙니다.",
            [Document(page_content=(
                "* 공과대학은 수업연한 이내이면 졸업 예정자도 추천 가능"
            ))],
        )

        self.assertIn("수업연한", answer)
        self.assertIn("추천 가능", answer)

    def test_explicit_scope_fallback_does_not_expand_to_graduate_students(self):
        answer = _explicit_scope_or_support_answer(
            "대학원생도 장학금을 받을 수 있어?",
            [Document(page_content="(선발인원) 학부 재학생 27명")],
        )

        self.assertIn("학부 재학생 대상", answer)
        self.assertIn("대학원생이 포함된다는 근거는", answer)

    def test_explicit_support_fallback_uses_document_support_type(self):
        answer = _explicit_scope_or_support_answer(
            "등록금 감면 장학금이야?",
            [Document(page_content=(
                "(지원내용) 학업장려금(생활비) 1인당 4,000,000원"
            ))],
        )

        self.assertIn("학업장려금(생활비)", answer)
        self.assertIn("등록금 감면", answer)

    def test_explicit_and_clause_rejects_one_failed_condition(self):
        answer = _explicit_mandatory_condition_answer(
            "성적 기준만 맞고 소득 기준은 안 맞아도 추천할 수 있어?",
            [Document(page_content=(
                "자격 요건: 성적 및 소득 기준 모두 충족 시 단과대학 추천가능"
            ))],
        )

        self.assertIn("모두 충족", answer)
        self.assertIn("추천할 수 없습니다", answer)

    def test_explicit_scalar_fields_read_count_and_per_person_amount(self):
        docs = [Document(page_content=(
            "(선발인원) 학부 재학생 27명\n"
            "(지원내용) 학업장려금(생활비) 1인당 4,000,000원"
        ))]

        self.assertIn(
            "학부 재학생 27명",
            _explicit_scalar_field_answer("장학생은 총 몇 명이야?", docs),
        )
        self.assertIn(
            "4,000,000원",
            _explicit_scalar_field_answer("한 명당 지원금은?", docs),
        )

    def test_explicit_scalar_fields_keep_multi_field_request_together(self):
        docs = [Document(page_content=(
            "(소요예산) 금 108,000,000원\n"
            "(선발인원) 학부 재학생 27명\n"
            "(지원내용) 학업장려금(생활비) 1인당 4,000,000원"
        ))]

        answer = _explicit_scalar_field_answer(
            "총 소요예산, 선발인원, 1인당 지원액을 한 번에 정리해줘.",
            docs,
        )

        self.assertIn("108,000,000원", answer)
        self.assertIn("27명", answer)
        self.assertIn("4,000,000원", answer)

    def test_explicit_scalar_table_uses_matching_rows_last_total(self):
        docs = [Document(page_content=(
            "공과대학 72,853,035 2,603,083 75,456,118\n"
            "공과대학 외 단과대학 36,426,518 1,301,541 37,728,059"
        ))]

        answer = _explicit_scalar_field_answer(
            "공과대학에 사용할 수 있는 금액은 총 얼마야?",
            docs,
        )

        self.assertIn("75,456,118원", answer)

    def test_preference_clause_does_not_become_automatic_exclusion(self):
        answer = _explicit_preference_answer(
            "다른 생활비 장학금을 받으면 무조건 탈락이야?",
            [Document(page_content=(
                "생활비 성격의 기타장학금을 수혜 받지 않는 자 우선 선발"
            ))],
        )

        self.assertIn("우선", answer)
        self.assertIn("자동 탈락", answer)
        self.assertIn("명시한 것은 아닙니다", answer)

    def test_preference_clause_ignores_an_unrelated_priority_topic(self):
        answer = _explicit_preference_answer(
            "성적이 높으면 소득과 관계없이 무조건 선발돼?",
            [Document(page_content=(
                "생활비 성격의 기타장학금을 수혜 받지 않는 자 우선 선발"
            ))],
        )

        self.assertEqual("", answer)

    def test_priority_rule_keeps_existing_before_new_candidates(self):
        answer = _explicit_priority_answer(
            "기존 장학생과 신규 지원자 중 누구를 먼저 추천해?",
            [Document(page_content=(
                "기존 선발자를 우선 추천\n"
                "잔여 인원에 대하여 신규 추천"
            ))],
        )

        self.assertIn("기존 선발자를 우선", answer)
        self.assertIn("잔여 인원", answer)
        self.assertIn("신규", answer)

    def test_table_comparison_uses_each_rows_last_total(self):
        docs = [Document(page_content=(
            "공과대학 72,853,035 2,603,083 75,456,118\n"
            "공과대학 외 단과대학 36,426,518 1,301,541 37,728,059"
        ))]

        answer = _explicit_table_comparison_answer(
            "공과대학과 공과대학 외 단과대학 금액을 비교해줘.",
            docs,
        )

        self.assertIn("75,456,118원", answer)
        self.assertIn("37,728,059원", answer)
        self.assertIn("차이", answer)

    def test_score_difference_is_recomputed_from_stated_operands(self):
        answer = _repair_explicit_difference_answer(
            "기초와 3구간의 점수 차이는?",
            "차이는 20점입니다. 기초는 50점, 3구간은 42점이므로 8점입니다.",
        )

        self.assertIn("50점", answer)
        self.assertIn("42점", answer)
        self.assertIn("차이는 8점", answer)
        self.assertNotIn("20점", answer)

    def test_score_difference_is_not_rewritten_when_operands_are_ambiguous(self):
        answer = (
            "두 점수의 차이는 8점입니다. "
            "기초는 50점 만점에서 50점, 3구간은 42점입니다."
        )

        repaired = _repair_explicit_difference_answer(
            "기초와 3구간의 점수 차이는?",
            answer,
        )

        self.assertEqual(answer, repaired)

    def test_unconditional_question_prefers_mandatory_condition_clause(self):
        mandatory = Document(page_content="자격 요건: 성적 및 소득 기준 모두 충족")
        ranking = Document(page_content="성적 반영점수가 높은 자 순으로 선발")

        docs = _conditional_evidence_documents(
            "성적이 높으면 소득과 관계없이 무조건 선발돼?",
            [ranking, mandatory],
        )

        self.assertEqual(docs, [mandatory])

    def test_condition_notes_align_question_values_and_source_thresholds(self):
        docs = [Document(page_content=(
            "자격 요건: 성적 및 소득 기준 모두 충족\n"
            "직전학기 평점평균 3.5 이상\n"
            "총평점평균 3.5 이상"
        ))]

        note = _condition_alignment_notes(
            "직전학기는 4.0인데 총평점이 3.4면 추천 가능해?",
            docs,
        )

        self.assertIn("직전학기는 4.0", note)
        self.assertIn("총평점이 3.4", note)
        self.assertIn("3.5 이상", note)

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

    def test_numeric_eligibility_prefers_explicit_threshold_over_score_table(self):
        threshold = Document(
            page_content="학자금지원구간 3구간 이하",
            metadata={"source": "plan.pdf"},
        )
        score_table = Document(
            page_content="학자금지원구간 반영점수 1구간 46 2구간 44 3구간 42",
            metadata={"source": "plan.pdf"},
        )

        docs = _numeric_evidence_documents(
            "학자금지원구간이 4구간이면 통과해?",
            [score_table, threshold],
        )

        self.assertEqual(docs, [threshold])

    def test_numeric_llm_answer_requires_conclusion_and_two_values(self):
        self.assertFalse(_numeric_llm_answer_is_complete(
            "충족하지 않습니다."
        ))
        self.assertTrue(_numeric_llm_answer_is_complete(
            "충족하지 않습니다. 1.9점은 문서 기준 2.0 이상보다 낮습니다."
        ))
        self.assertFalse(_numeric_llm_answer_is_complete(
            "충족합니다. 문서 기준은 3.5 이상입니다.",
            "평점 3.4면 충족해?",
        ))
        self.assertIn(
            "3.4",
            _numeric_answer_issue(
                "충족합니다. 문서 기준은 3.5 이상입니다.",
                "평점 3.4면 충족해?",
            ),
        )

    def test_reversed_numeric_relation_is_detected(self):
        self.assertTrue(_numeric_relation_contradiction(
            "질문의 값 4.0이 기준값 3.5보다 낮아 미달합니다."
        ))
        self.assertFalse(_numeric_relation_contradiction(
            "질문의 값 4.0은 기준값 3.5 이상을 충족합니다."
        ))

    def test_explicit_numeric_fallback_compares_each_labeled_criterion(self):
        answer = _explicit_numeric_criteria_answer(
            "직전학기는 4.0인데 총평점이 3.4면 추천할 수 있어?",
            [Document(page_content=(
                "직전학기 평점평균 3.5 이상이며, "
                "총평점 평균이 3.5 이상인 자"
            ))],
        )

        self.assertIn("직전학기 평점평균 4", answer)
        self.assertIn("충족합니다", answer)
        self.assertIn("총평점 평균 3.4", answer)
        self.assertIn("미달합니다", answer)
        self.assertIn("추천할 수 없습니다", answer)

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
