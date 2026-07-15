from __future__ import annotations

import unittest

from main import ChatResponse, _route_with_guard
from rag.guard import check_question
from rag.question_analyzer import analyze_question
from rag.router import _route, required_engines, route_analysis


class GuardRoutingTest(unittest.TestCase):
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
