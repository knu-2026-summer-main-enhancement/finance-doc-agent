from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from datastore.query import _query_pandas_direct
from datastore.state import _df_labels, _df_namespace, _df_sources
from pandas_engine.aggregation import (
    amount_column_clarification,
    detect_aggregation_intent,
    detect_aggregation_intents,
    display_column_label,
)
from pandas_engine.formatter import _format_scalar_result
from rag.router import _route
from rag.question_analyzer import analyze_question
from utils.table_parser import _clean_dataframe


class AggregationQueryTest(unittest.TestCase):
    def setUp(self):
        raw = pd.DataFrame([
            {"발행번호": "2025-008", "출연일자": "2025-02-24", "기수": "58", "이름": "이*규", "출연금액": "1,000,000"},
            {"발행번호": "2025-008", "출연일자": "2025-08-05", "기수": "58", "이름": "이*규", "출연금액": "9,000,000"},
            {"발행번호": "2025-108", "출연일자": "2025-03-20", "기수": "49", "이름": "이*규", "출연금액": "1,000,000"},
            {"발행번호": "2025-010", "출연일자": "2025-03-21", "기수": "49", "이름": "김*호", "출연금액": "2,000,000"},
            {"발행번호": "2025-009", "출연일자": "2025-01-26", "기수": "", "이름": "현대중공업대공동문회", "출연금액": "1,000,000"},
        ])
        df = _clean_dataframe(raw, source_file="test2025.png", context_prefix="img_table0")
        self.assertIsNotNone(df)
        _df_namespace.clear()
        _df_sources.clear()
        _df_labels.clear()
        _df_namespace["df0"] = df
        _df_sources["df0"] = "test2025.png"
        _df_labels["df0"] = "2025년 기부금 후원 내역"

    def tearDown(self):
        _df_namespace.clear()
        _df_sources.clear()
        _df_labels.clear()

    def _payload(self, question: str) -> dict:
        result, sources = _query_pandas_direct(question)
        self.assertIsInstance(result, dict, question)
        self.assertEqual(sources, ["test2025.png"], question)
        return result

    def test_sum_accepts_varied_natural_phrases(self):
        questions = (
            "출연금액 총 얼만지 알려줘",
            "출연금액을 전부 합하면 얼마니?",
            "기부금 전체 금액은?",
            "모든 출연금액 합계 알려주세요",
            "출연금액 다 합하면 얼마인가요?",
            "기부한 돈을 전부 더하면 얼마야?",
            "모든 사람이 낸 출연금을 다 더한 값은?",
            "누적 출연금액이 어떻게 돼?",
        )
        for question in questions:
            with self.subTest(question=question):
                payload = self._payload(question)
                self.assertEqual(payload["operation"], "sum")
                self.assertEqual(payload["value"], 14_000_000)

    def test_mean_median_and_mode_accept_varied_phrases(self):
        cases = {
            "평균 출연금액이 얼마야?": ("mean", 2_800_000),
            "평균적으로 얼마씩 냈어?": ("mean", 2_800_000),
            "사람들이 낸 돈은 평균적으로 얼마야?": ("mean", 2_800_000),
            "출연금액의 중앙값을 구해줘": ("median", 1_000_000),
            "중간값은 얼마인가요?": ("median", 1_000_000),
            "돈을 크기순으로 놓았을 때 정중앙 값은?": ("median", 1_000_000),
            "가장 흔한 출연금액은?": ("mode", [1_000_000]),
            "출연금액 최빈값 알려줘": ("mode", [1_000_000]),
            "출연금액 중 제일 많이 등장한 값은?": ("mode", [1_000_000]),
        }
        for question, (operation, expected) in cases.items():
            with self.subTest(question=question):
                payload = self._payload(question)
                self.assertEqual(payload["operation"], operation)
                key = "values" if operation == "mode" else "value"
                self.assertEqual(payload[key], expected)

    def test_max_min_value_phrases(self):
        cases = {
            "가장 큰 출연금액은?": ("max", 9_000_000),
            "최대 금액이 얼마인지 알려줘": ("max", 9_000_000),
            "제일 높은 기부금은?": ("max", 9_000_000),
            "가장 돈을 많이 낸 금액은?": ("max", 9_000_000),
            "가장 작은 출연금액은?": ("min", 1_000_000),
            "최소 금액 알려줘": ("min", 1_000_000),
            "제일 적은 기부금은 얼마야?": ("min", 1_000_000),
            "가장 돈을 적게 낸 금액은?": ("min", 1_000_000),
        }
        for question, (operation, value) in cases.items():
            with self.subTest(question=question):
                payload = self._payload(question)
                self.assertEqual(payload["operation"], operation)
                self.assertEqual(payload["value"], value)

    def test_person_total_row_max_and_top_n_are_distinct(self):
        person = self._payload("가장 많은 돈을 기부한 사람 누구야?")
        self.assertEqual(person["scope"], "person_total")
        self.assertEqual(person["subjects"], [{"name": "이*규 (58기)", "value": 10_000_000.0}])

        min_person = self._payload("가장 돈을 적게 낸 사람 누구야?")
        self.assertEqual(min_person["scope"], "person_total")
        self.assertEqual(min_person["subjects"], [{"name": "이*규 (49기)", "value": 1_000_000.0}])

        row = self._payload("한 번에 가장 큰 금액을 낸 사람은?")
        self.assertEqual(row["scope"], "row")
        self.assertEqual(row["subjects"][0]["name"], "이*규")
        self.assertEqual(row["subjects"][0]["value"], 9_000_000)

        top = self._payload("누적 출연금액 상위 2명 보여줘")
        self.assertEqual([item["name"] for item in top["subjects"]], ["이*규 (58기)", "김*호"])
        self.assertEqual([item["value"] for item in top["subjects"]], [10_000_000, 2_000_000])

    def test_identifier_filter_is_applied_before_aggregation(self):
        cases = {
            "2025-008 출연금액 합계는?": ("sum", 10_000_000),
            "2025-008 평균 출연금액은?": ("mean", 5_000_000),
            "2025-008 중 가장 큰 출연금액은?": ("max", 9_000_000),
        }
        for question, (operation, value) in cases.items():
            with self.subTest(question=question):
                payload = self._payload(question)
                self.assertEqual(payload["operation"], operation)
                self.assertEqual(payload["value"], value)

    def test_people_count_row_count_and_per_capita_are_distinct(self):
        people = self._payload("기부한 사람은 모두 몇 명이야?")
        self.assertEqual((people["value"], people["unit"]), (3, "명"))

        rows = self._payload("출연 기록은 총 몇 건이야?")
        self.assertEqual((rows["value"], rows["unit"]), (5, "건"))

        per_capita = self._payload("기부자 한 사람당 평균 얼마를 냈어?")
        self.assertEqual(per_capita["operation"], "per_capita")
        self.assertAlmostEqual(per_capita["value"], 13_000_000 / 3)
        self.assertEqual(per_capita["people_count"], 3)

    def test_aggregation_result_is_formatted_without_llm(self):
        question = "가장 많이 기부한 사람은 누구야?"
        payload = self._payload(question)
        answer = _format_scalar_result(payload, question)
        self.assertIn("이*규", answer)
        self.assertIn("10,000,000원", answer)
        self.assertIn("계산 근거:", answer)
        self.assertIn("- 문서: test2025.png", answer)
        self.assertIn("- 계산 컬럼: 출연금액", answer)
        self.assertIn("- 계산 방식: 최댓값", answer)
        self.assertIn("- 조회 행: 4개", answer)
        self.assertIn("- 계산 사용 행: 4개", answer)
        self.assertIn("- 제외 행: 0개", answer)

    def test_count_result_includes_traceable_evidence(self):
        question = "기부한 사람은 모두 몇 명이야?"
        answer = _format_scalar_result(self._payload(question), question)
        self.assertIn("총 3명입니다.", answer)
        self.assertIn("- 문서: test2025.png", answer)
        self.assertIn("- 계산 방식: 개수 계산", answer)
        self.assertIn("- 조회 행: 4개", answer)

    def test_generic_dict_result_hides_internal_identity_columns(self):
        raw_result = {
            "이름": {0: "이*규"},
            "출연금액": {0: "1,000,000"},
            "성명_마스킹패턴": {0: "이*규"},
            "성명_마스킹여부": {0: True},
            "person_candidate_key": {0: "internal-key"},
            "ocr_row_index": {0: 17},
        }
        answer = _format_scalar_result(raw_result, "명단 보여줘")
        self.assertIn("이*규", answer)
        self.assertIn("출연금액", answer)
        self.assertNotIn("성명_마스킹패턴", answer)
        self.assertNotIn("성명_마스킹여부", answer)
        self.assertNotIn("person_candidate_key", answer)
        self.assertNotIn("internal-key", answer)
        self.assertNotIn("ocr_row_index", answer)

    def test_shared_analysis_skips_query_layer_redetection(self):
        question = "출연금액 총 얼만지 알려줘"
        analysis = analyze_question(question)

        with patch(
            "datastore.query.detect_aggregation_intents",
            side_effect=AssertionError("집계 의도를 다시 분석하면 안 됩니다."),
        ):
            result, sources = _query_pandas_direct(
                question,
                aggregation_intents=analysis.aggregation_intents,
            )

        self.assertEqual(result["operation"], "sum")
        self.assertEqual(result["value"], 14_000_000)
        self.assertEqual(sources, ["test2025.png"])

    def test_direct_query_rejects_multiple_aggregations(self):
        result, sources = _query_pandas_direct("출연금액 합계 평균 알려줘")
        self.assertEqual(result["type"], "aggregation_notice")
        self.assertEqual(result["kind"], "clarification")
        self.assertEqual(sources, [])

    def test_multiple_sources_require_document_selection(self):
        _df_namespace["df1"] = _df_namespace["df0"].copy()
        _df_sources["df1"] = "other2024.xlsx"
        _df_labels["df1"] = "2024년 기부금 후원 내역"

        result, sources = _query_pandas_direct("평균 출연금액 알려줘")
        self.assertEqual(result["type"], "aggregation_notice")
        self.assertEqual(result["kind"], "clarification")
        self.assertEqual(sources, [])

    def test_organization_only_rows_are_not_counted_as_people(self):
        only_org = _df_namespace["df0"]
        only_org = only_org[only_org["entity_type"] == "organization"].copy()
        _df_namespace["df0"] = only_org

        count, _ = _query_pandas_direct("기부한 사람은 몇 명이야?")
        self.assertEqual((count["value"], count["unit"]), (0, "명"))

        maximum, _ = _query_pandas_direct("가장 많이 기부한 사람은 누구야?")
        self.assertEqual(maximum["type"], "aggregation_notice")

    def test_router_sends_all_basic_aggregations_to_pandas(self):
        questions = (
            "출연금액 총 얼만지 알려줘",
            "평균 출연금액은?",
            "출연금액 중앙값은?",
            "가장 흔한 출연금액은?",
            "가장 큰 금액은?",
            "최소 금액은?",
            "총 몇 명이야?",
            "1인당 얼마야?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertIsNotNone(detect_aggregation_intent(question))
                self.assertEqual(_route(question), "PANDAS")

    def test_multiple_aggregation_operations_are_detected_together(self):
        intents = detect_aggregation_intents("출연금액 합계와 평균을 같이 알려줘")
        self.assertEqual([intent.operation for intent in intents], ["mean", "sum"])

        intents = detect_aggregation_intents("가장 많이 낸 사람과 가장 적게 낸 사람 알려줘")
        self.assertEqual([intent.operation for intent in intents], ["max", "min"])

    def test_multiple_amount_columns_are_selected_from_runtime_headers(self):
        raw = pd.DataFrame([
            {
                "이름": "김하늘",
                "융합인재_1단계_장학금액": "100,000",
                "col_2단계_목표달성_장학금_지급_예정_금액": "300,000",
            },
            {
                "이름": "이바다",
                "융합인재_1단계_장학금액": "200,000",
                "col_2단계_목표달성_장학금_지급_예정_금액": "400,000",
            },
        ])
        df = _clean_dataframe(raw, source_file="multi_amount.xlsx", context_prefix="s0")
        self.assertIsNotNone(df)
        _df_namespace.clear()
        _df_namespace["df0"] = df
        _df_sources["df0"] = "multi_amount.xlsx"
        _df_labels["df0"] = "다중 금액 테스트"

        first, _ = _query_pandas_direct("1단계 장학금 총액 알려줘")
        second, _ = _query_pandas_direct("2단계 목표달성 장학금 총액 알려줘")

        self.assertEqual(first["label"], "융합인재_1단계_장학금액")
        self.assertEqual(first["value"], 300_000)
        self.assertEqual(second["label"], "col_2단계_목표달성_장학금_지급_예정_금액")
        self.assertEqual(second["value"], 700_000)

    def test_ambiguous_multiple_amount_columns_request_clarification(self):
        raw = pd.DataFrame([
            {"이름": "김하늘", "기준_장학금액": "100,000", "지급_장학금액": "200,000"},
        ])
        df = _clean_dataframe(raw, source_file="ambiguous.xlsx", context_prefix="s0")
        self.assertIsNotNone(df)
        _df_namespace.clear()
        _df_namespace["df0"] = df
        _df_sources["df0"] = "ambiguous.xlsx"
        _df_labels["df0"] = "모호한 금액 테스트"

        result, sources = _query_pandas_direct("장학금 총액 알려줘")

        self.assertEqual(result["type"], "aggregation_notice")
        self.assertEqual(result["kind"], "clarification")
        self.assertIn("기준 장학금액", result["message"])
        self.assertIn("지급 장학금액", result["message"])
        self.assertNotIn("_", result["message"])
        self.assertEqual(sources, ["ambiguous.xlsx"])

    def test_internal_column_prefix_is_hidden_in_user_facing_amount_labels(self):
        raw_column = "col_2단계_목표달성_장학금_지급_예정_금액"
        self.assertEqual(
            display_column_label(raw_column),
            "2단계 목표달성 장학금 지급 예정 금액",
        )
        message = amount_column_clarification([
            "융합인재_1단계_장학금액",
            raw_column,
        ])
        self.assertIn("융합인재 1단계 장학금액", message)
        self.assertIn("2단계 목표달성 장학금 지급 예정 금액", message)
        self.assertNotIn("col_", message)


if __name__ == "__main__":
    unittest.main()
