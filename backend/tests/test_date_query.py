from __future__ import annotations

import unittest

import pandas as pd

from datastore.query import _query_pandas_direct
from datastore.state import _df_labels, _df_namespace, _df_sources
from pandas_engine.formatter import _format_scalar_result
from rag.question_analyzer import analyze_question
from rag.router import route_analysis
from utils.table_parser import _clean_dataframe


class DateQueryTest(unittest.TestCase):
    def setUp(self):
        raw = pd.DataFrame([
            {"출연일자": "2025-03-01", "이름": "김하나", "출연금액": "1,000,000"},
            {"출연일자": "2025-03-20", "이름": "김하나", "출연금액": "2,000,000"},
            {"출연일자": "2025-03-15", "이름": "이두리", "출연금액": "5,000,000"},
            {"출연일자": "2025-04-01", "이름": "김하나", "출연금액": "7,000,000"},
            {"출연일자": "2025-04-10", "이름": "이두리", "출연금액": "4,000,000"},
        ])
        df = _clean_dataframe(raw, source_file="donations.xlsx", context_prefix="sheet0")
        self.assertIsNotNone(df)
        _df_namespace.clear()
        _df_sources.clear()
        _df_labels.clear()
        _df_namespace["df0"] = df
        _df_sources["df0"] = "donations.xlsx"
        _df_labels["df0"] = "후원 내역"

    def tearDown(self):
        _df_namespace.clear()
        _df_sources.clear()
        _df_labels.clear()

    def _query(self, question: str):
        analysis = analyze_question(question)
        return analysis, _query_pandas_direct(
            question,
            aggregation_intents=analysis.aggregation_intents,
            date_filter=analysis.date_filter,
        )

    def _replace_dataframe(self, raw: pd.DataFrame) -> None:
        df = _clean_dataframe(
            raw,
            source_file="payments.xlsx",
            context_prefix="sheet0",
        )
        self.assertIsNotNone(df)
        _df_namespace.clear()
        _df_sources.clear()
        _df_labels.clear()
        _df_namespace["df0"] = df
        _df_sources["df0"] = "payments.xlsx"
        _df_labels["df0"] = "지급 내역"

    def test_month_range_list_filters_rows_without_llm(self):
        analysis, (result, sources) = self._query("3~4월에 낸 사람 리스트 알려줘")
        self.assertEqual(route_analysis(analysis), "PANDAS")
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["이름"]), {"김하나", "이두리"})
        self.assertNotIn("출연금액", result.columns)
        self.assertEqual(sources, ["donations.xlsx"])
        self.assertEqual(result.attrs["date_filter_evidence"]["period"], "2025년 3~4월")

    def test_single_month_maximum_amount(self):
        analysis, (result, _) = self._query("3월에 가장 많이 낸 돈은?")
        self.assertEqual(result["operation"], "max")
        self.assertEqual(result["scope"], "value")
        self.assertEqual(result["value"], 5_000_000)
        answer = _format_scalar_result(result, analysis.question)
        self.assertIn("조회 기간: 2025년 3월", answer)
        self.assertIn("날짜 컬럼: 출연일자", answer)

    def test_single_month_maximum_person_uses_period_total(self):
        _, (result, _) = self._query("4월에 가장 많이 낸 사람 누구야?")
        self.assertEqual(result["scope"], "person_total")
        self.assertEqual(result["subjects"], [{"name": "김하나", "value": 7_000_000.0}])

    def test_multiple_years_require_explicit_scope(self):
        extra = _df_namespace["df0"].iloc[[0]].copy()
        extra.loc[:, "출연일자"] = "2024-03-01"
        _df_namespace["df0"] = pd.concat([_df_namespace["df0"], extra], ignore_index=True)
        _, (result, _) = self._query("3월 출연금액 합계 알려줘")
        self.assertEqual(result["type"], "aggregation_notice")
        self.assertIn("여러 연도", result["message"])

    def test_date_filter_combines_with_existing_value_filter(self):
        _, (result, _) = self._query("3월 김하나 출연금액 합계 알려줘")
        self.assertEqual(result["value"], 3_000_000)
        self.assertEqual(result["matched_rows"], 2)

    def test_document_without_matching_month_is_not_reported_as_source(self):
        other = _clean_dataframe(
            pd.DataFrame([{
                "출연일자": "2025-05-01",
                "이름": "박다른",
                "출연금액": "9,000,000",
            }]),
            source_file="other.xlsx",
            context_prefix="sheet0",
        )
        self.assertIsNotNone(other)
        _df_namespace["df1"] = other
        _df_sources["df1"] = "other.xlsx"
        _df_labels["df1"] = "다른 후원 내역"

        _, (result, sources) = self._query("3월 출연금액 합계 알려줘")

        self.assertEqual(result["value"], 8_000_000)
        self.assertEqual(sources, ["donations.xlsx"])

    def test_month_only_column_is_a_valid_date_filter(self):
        self._replace_dataframe(pd.DataFrame([
            {"지급월": "11월", "이름": "김하나", "장학금액": "1,000,000"},
            {"지급월": "12월", "이름": "이두리", "장학금액": "2,000,000"},
        ]))

        _, (result, _) = self._query("지급월이 12월인 사람 알려줘")

        self.assertEqual(result["이름"].tolist(), ["이두리"])
        self.assertEqual(result.attrs["date_filter_evidence"]["column"], "지급월")
        self.assertEqual(result.attrs["date_filter_evidence"]["period"], "12월")

    def test_separate_year_and_month_columns_are_combined(self):
        self._replace_dataframe(pd.DataFrame([
            {"년": 2024, "월": 12, "이름": "김하나"},
            {"년": 2025, "월": 12, "이름": "이두리"},
        ]))

        _, (result, _) = self._query("2025년 12월인 사람 알려줘")

        self.assertEqual(result["이름"].tolist(), ["이두리"])
        evidence = result.attrs["date_filter_evidence"]
        self.assertEqual(evidence["column"], "월")
        self.assertEqual(evidence["year_column"], "년")

    def test_separate_year_and_month_support_cross_year_ranges(self):
        self._replace_dataframe(pd.DataFrame([
            {"년": 2024, "월": 12, "이름": "김하나"},
            {"년": 2025, "월": 1, "이름": "이두리"},
            {"년": 2025, "월": 3, "이름": "박세나"},
        ]))

        _, (result, _) = self._query("2024년 12월부터 2025년 2월까지인 사람 알려줘")

        self.assertEqual(result["이름"].tolist(), ["김하나", "이두리"])

    def test_cross_year_month_range_prefers_unambiguous_complete_date(self):
        self._replace_dataframe(pd.DataFrame([
            {"년": 2025, "월": 5, "결제등록날짜": "2025-05-31", "이름": "김하나"},
            {"년": 2025, "월": 6, "결제등록날짜": "2025-06-01", "이름": "이두리"},
            {"년": 2025, "월": 12, "결제등록날짜": "2025-12-31", "이름": "박세나"},
            {"년": 2026, "월": 1, "결제등록날짜": "2026-01-01", "이름": "최도윤"},
            {"년": 2026, "월": 2, "결제등록날짜": "2026-02-01", "이름": "한지우"},
        ]))

        _, (result, _) = self._query("2025년 6월부터 2026년 1월까지 목록 알려줘")

        self.assertEqual(result["이름"].tolist(), ["이두리", "박세나", "최도윤"])
        evidence = result.attrs["date_filter_evidence"]
        self.assertEqual(evidence["column"], "결제등록날짜")
        self.assertEqual(evidence["period"], "2025년 6월~2026년 1월")

    def test_cross_year_range_uses_year_month_components_when_date_column_is_stale(self):
        self._replace_dataframe(pd.DataFrame([
            {"년": 2025, "월": 6, "결제등록날짜": "2021-06-01", "이름": "김하나"},
            {"년": 2025, "월": 12, "결제등록날짜": None, "이름": "이두리"},
            {"년": 2026, "월": 1, "결제등록날짜": None, "이름": "박세나"},
            {"년": 2026, "월": 2, "결제등록날짜": None, "이름": "최도윤"},
        ]))

        _, (result, _) = self._query("2025년 6월부터 2026년 1월까지 목록 알려줘")

        self.assertEqual(result["이름"].tolist(), ["김하나", "이두리", "박세나"])
        evidence = result.attrs["date_filter_evidence"]
        self.assertEqual(evidence["column"], "월")
        self.assertEqual(evidence["year_column"], "년")

    def test_year_condition_is_rejected_when_only_month_exists(self):
        self._replace_dataframe(pd.DataFrame([
            {"지급월": 12, "이름": "김하나"},
        ]))

        _, (result, _) = self._query("2025년 12월 지급자 알려줘")

        self.assertEqual(result["type"], "aggregation_notice")
        self.assertIn("월 정보만", result["message"])

    def test_explicit_month_column_wins_over_an_unrelated_full_date(self):
        self._replace_dataframe(pd.DataFrame([
            {"신청일자": "2025-12-01", "지급월": 1, "이름": "김하나"},
            {"신청일자": "2025-01-01", "지급월": 12, "이름": "이두리"},
        ]))

        _, (result, _) = self._query("지급월이 12월인 사람 알려줘")

        self.assertEqual(result["이름"].tolist(), ["이두리"])
        self.assertEqual(result.attrs["date_filter_evidence"]["column"], "지급월")

    def test_ambiguous_temporal_columns_require_a_specific_basis(self):
        self._replace_dataframe(pd.DataFrame([
            {"신청일자": "2025-12-01", "지급월": 1, "이름": "김하나"},
            {"신청일자": "2025-01-01", "지급월": 12, "이름": "이두리"},
        ]))

        _, (result, _) = self._query("12월인 사람 알려줘")

        self.assertEqual(result["type"], "aggregation_notice")
        self.assertIn("날짜 기준을 하나로 결정할 수 없습니다", result["message"])


if __name__ == "__main__":
    unittest.main()
