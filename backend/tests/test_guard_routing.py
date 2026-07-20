from __future__ import annotations

import unittest
from typing import get_args
from unittest.mock import patch

from fastapi import BackgroundTasks

from main import (
    ChatRequest,
    ChatResponse,
    _route_with_guard,
    _schedule_shadow_question_engine,
)
from rag.guard import check_question, check_question_decision
from rag.question_analyzer import analyze_question
from rag.question_decision import QuestionDecision, QuestionOperation
from rag.router import (
    _ENGINE_BY_OPERATION,
    _route,
    engines_for_operations,
    pandas_strategy_for_operations,
    required_engines,
    route_analysis,
    route_operations,
)


class GuardRoutingTest(unittest.TestCase):
    def test_shadow_engine_is_opt_in_and_queued_without_changing_route(self):
        tasks = BackgroundTasks()
        with patch("main.QUESTION_ENGINE_MODE", "legacy"):
            self.assertFalse(
                _schedule_shadow_question_engine(
                    tasks,
                    "질문",
                    "PANDAS",
                    ["structured_query"],
                )
            )
        self.assertEqual(len(tasks.tasks), 0)

        with patch("main.QUESTION_ENGINE_MODE", "shadow"), patch(
            "main._get_df_schema",
            return_value='컬럼: "금액"',
        ):
            self.assertTrue(
                _schedule_shadow_question_engine(
                    tasks,
                    "질문",
                    "PANDAS",
                    ["structured_query"],
                )
            )
        self.assertEqual(len(tasks.tasks), 1)

    def test_every_llm_operation_has_an_engine_mapping(self):
        self.assertEqual(
            set(get_args(QuestionOperation)),
            set(_ENGINE_BY_OPERATION),
        )

    def test_operation_route_and_pandas_strategy_are_deterministic(self):
        self.assertEqual(route_operations(["sum_amount"]), "PANDAS")
        self.assertEqual(
            pandas_strategy_for_operations(["sum_amount"]),
            "DIRECT",
        )
        self.assertEqual(
            route_operations(["structured_query"]),
            "PANDAS",
        )
        self.assertEqual(
            pandas_strategy_for_operations(["structured_query"]),
            "QUERY_PLAN",
        )
        self.assertEqual(
            route_operations(["document_criteria"]),
            "VECTOR",
        )
        self.assertIsNone(
            pandas_strategy_for_operations(["document_criteria"])
        )

    def test_mixed_or_multiple_operations_require_guide(self):
        operations = ["sum_amount", "document_criteria"]
        self.assertEqual(
            engines_for_operations(operations),
            ["PANDAS", "VECTOR"],
        )
        self.assertEqual(route_operations(operations), "GUIDE")
        self.assertEqual(
            route_operations(["sum_amount", "average_amount"]),
            "GUIDE",
        )

    def test_llm_operation_guard_blocks_mixed_engines(self):
        decision = QuestionDecision(
            status="ready",
            operations=["sum_amount", "document_criteria"],
            reason="계산과 기준 검색의 혼합 요청",
            retrieval_query="지급 기준",
        )
        result = check_question_decision(decision)
        self.assertEqual(result.status, "GUIDE")
        self.assertEqual(result.reason_code, "CROSS_ENGINE_QUERY")
        self.assertEqual(
            result.domains,
            ["structured_data", "document_evidence"],
        )

    def test_llm_structured_query_passes_guard(self):
        decision = QuestionDecision(
            status="ready",
            operations=["structured_query"],
            reason="범용 표 조회",
        )
        result = check_question_decision(decision)
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.operations, ["structured_query"])
        self.assertEqual(result.domains, ["structured_data"])

    def test_spaced_and_unspaced_vague_references_are_guided(self):
        for question in ("그 사람 금액 알려줘", "그사람 금액 알려줘", "아까 그 사람 알려줘"):
            with self.subTest(question=question):
                result = check_question(question)
                self.assertEqual(result.status, "GUIDE")
                self.assertEqual(result.reason_code, "VAGUE_REFERENCE")

    def test_document_criteria_is_not_misclassified_as_amount_lookup(self):
        result = check_question("장학금 지급 기준 알려줘")
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.operations, ["document_criteria"])
        self.assertEqual(_route_with_guard("장학금 지급 기준 알려줘", result), "VECTOR")

    def test_natural_mode_forces_vector_without_changing_default_route(self):
        question = "가장 돈을 많이 낸 사람은?"
        result = check_question(question)
        self.assertEqual(_route_with_guard(question, result), "PANDAS")
        self.assertEqual(_route_with_guard(question, result, "natural"), "VECTOR")

        self.assertEqual(ChatRequest(question=question).mode, "auto")
        self.assertEqual(ChatRequest(question=question, mode="natural").mode, "natural")

    def test_result_how_and_procedure_how_are_distinguished(self):
        for question, operation in (
            ("누적 출연금액이 어떻게 돼?", "sum_amount"),
            ("평균 출연금액이 어떻게 돼?", "average_amount"),
        ):
            with self.subTest(question=question):
                result = check_question(question)
                self.assertEqual(result.status, "PASS")
                self.assertEqual(result.domains, ["structured_data"])
                self.assertIn(operation, result.operations)
                self.assertNotIn("document_procedure", result.operations)

        procedure = check_question("장학금은 어떻게 신청해?")
        self.assertEqual(procedure.status, "PASS")
        self.assertEqual(procedure.operations, ["document_procedure"])
        self.assertEqual(procedure.domains, ["document_evidence"])

    def test_explicit_amount_and_criteria_remains_cross_engine(self):
        result = check_question("장학금액과 지급 기준을 같이 알려줘")
        self.assertEqual(result.status, "GUIDE")
        self.assertEqual(result.reason_code, "CROSS_ENGINE_QUERY")
        self.assertEqual(result.domains, ["structured_data", "document_evidence"])

    def test_median_and_mode_have_pandas_engine_hints(self):
        for question, operation in (
            ("출연금액 중앙값은?", "median_amount"),
            ("가장 흔한 출연금액은?", "mode_amount"),
        ):
            with self.subTest(question=question):
                result = check_question(question)
                self.assertEqual(result.status, "PASS")
                self.assertIn(operation, result.operations)
                self.assertEqual(result.domains, ["structured_data"])

    def test_flexible_extreme_word_order_uses_same_guard_and_engine(self):
        cases = (
            ("가장 돈을 적게 낸 사람 누구야?", "min_person_by_amount"),
            ("제일 출연금을 적게 낸 사람은?", "min_person_by_amount"),
            ("돈을 가장 적게 낸 사람은?", "min_person_by_amount"),
            ("가장 돈을 많이 낸 사람은?", "max_person_by_amount"),
            ("제일 출연금을 많이 낸 사람은?", "max_person_by_amount"),
            ("돈을 가장 많이 낸 사람은?", "max_person_by_amount"),
        )
        for question, operation in cases:
            with self.subTest(question=question):
                result = check_question(question)
                self.assertEqual(result.status, "PASS")
                self.assertIn(operation, result.operations)
                self.assertEqual(_route_with_guard(question, result), "PANDAS")

    def test_multiple_aggregations_and_cross_engine_questions_are_guided(self):
        result = check_question("가장 많이 낸 사람과 가장 적게 낸 사람 알려줘")
        self.assertEqual(result.status, "GUIDE")
        self.assertEqual(result.reason_code, "MULTIPLE_AGGREGATIONS")

        result = check_question("가장 돈을 적게 낸 사람과 그 이유 알려줘")
        self.assertEqual(result.status, "GUIDE")
        self.assertEqual(result.reason_code, "CROSS_ENGINE_QUERY")

    def test_multiple_aggregations_without_connector_are_guided(self):
        result = check_question("출연금액 합계 평균 알려줘")
        self.assertEqual(result.status, "GUIDE")
        self.assertEqual(result.reason_code, "MULTIPLE_AGGREGATIONS")

    def test_all_comparisons_are_safely_guided(self):
        for question in (
            "연도별 출연금액 합계를 비교해줘",
            "49기와 58기의 평균 출연금액 비교해줘",
            "49기와 58기 출연금액 비교해줘",
            "연도별 출연금액을 비교해줘",
        ):
            with self.subTest(question=question):
                result = check_question(question)
                self.assertEqual(result.status, "GUIDE")
                self.assertEqual(result.reason_code, "COMPARISON_NOT_SUPPORTED")

    def test_entity_nouns_in_ranking_questions_are_not_list_requests(self):
        for question, operation in (
            ("기계과 출연자 중 가장 많이 낸 사람은?", "max_person_by_amount"),
            ("학과별 수혜자 중 가장 적게 받은 사람은?", "min_person_by_amount"),
            ("58기 기부자 중 가장 많이 낸 사람은?", "max_person_by_amount"),
        ):
            with self.subTest(question=question):
                result = check_question(question)
                self.assertEqual(result.status, "PASS")
                self.assertEqual(result.operations, [operation])
                self.assertEqual(result.domains, ["structured_data"])

    def test_entity_noun_still_supports_plain_and_explicit_list_requests(self):
        plain = check_question("기부자 알려줘")
        self.assertEqual(plain.status, "PASS")
        self.assertEqual(plain.operations, ["list_records"])

        combined = check_question("기부자 명단과 총액 알려줘")
        self.assertEqual(combined.status, "GUIDE")
        self.assertEqual(combined.reason_code, "MULTI_OPERATION")

    def test_chat_response_source_lists_are_independent(self):
        first = ChatResponse(answer="a", source="pandas")
        second = ChatResponse(answer="b", source="vector")
        first.sources.append("one.xlsx")
        self.assertEqual(second.sources, [])

    def test_question_is_analyzed_once_and_router_uses_the_shared_result(self):
        analysis = analyze_question("가장 돈을 적게 낸 사람 누구야?")
        self.assertEqual(analysis.operations, ["min_person_by_amount"])
        self.assertEqual(required_engines(analysis), ["PANDAS"])
        self.assertEqual(route_analysis(analysis), "PANDAS")

        result = check_question(analysis.question)
        self.assertIsNotNone(result.analysis)
        self.assertEqual(_route(analysis.question, result.analysis), "PANDAS")

    def test_document_inventory_is_not_treated_as_table_rows(self):
        for question in (
            "전체 문서 보여줘",
            "적재된 파일 목록 알려줘",
            "현재 문서 리스트 조회",
        ):
            with self.subTest(question=question):
                result = check_question(question)
                self.assertEqual(result.status, "PASS")
                self.assertEqual(result.operations, ["list_documents"])
                self.assertEqual(result.domains, ["document_inventory"])
                self.assertEqual(_route_with_guard(question, result), "DOCUMENTS")

    def test_compare_routes_to_pandas_but_guard_blocks_execution(self):
        analysis = analyze_question("연도별 출연금액을 비교해줘")
        self.assertEqual(analysis.operations, ["compare"])
        self.assertEqual(required_engines(analysis), ["PANDAS"])
        self.assertEqual(route_analysis(analysis), "PANDAS")

        result = check_question(analysis.question)
        self.assertEqual(result.status, "GUIDE")
        self.assertEqual(result.reason_code, "COMPARISON_NOT_SUPPORTED")


if __name__ == "__main__":
    unittest.main()
