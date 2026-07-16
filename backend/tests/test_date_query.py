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


if __name__ == "__main__":
    unittest.main()
