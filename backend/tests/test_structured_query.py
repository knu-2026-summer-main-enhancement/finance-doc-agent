from __future__ import annotations

import unittest

import pandas as pd

from datastore.query import _query_pandas_direct, _search_name_pandas
from datastore.state import _df_labels, _df_namespace, _df_sources
from pandas_engine.formatter import _format_dataframe_result_for_question
from utils.table_parser import _clean_dataframe, classify_name_entity


class StructuredQueryTest(unittest.TestCase):
    def setUp(self):
        raw = pd.DataFrame([
            {
                "발행번호": "2025-008",
                "출연일자": "2025-02-24",
                "기수": "58",
                "이름": "이*규",
                "출연금액": "1,000,000",
            },
            {
                "발행번호": "2025-008",
                "출연일자": "2025-08-05",
                "기수": "58",
                "이름": "이*규",
                "출연금액": "9,000,000",
            },
            {
                "발행번호": "2025-108",
                "출연일자": "2025-03-20",
                "기수": "49",
                "이름": "이*규",
                "출연금액": "1,000,000",
            },
            {
                "발행번호": "2025-009",
                "출연일자": "2025-01-26",
                "기수": "",
                "이름": "현대중공업대공동문회",
                "출연금액": "1,000,000",
            },
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

    def test_classifies_people_and_organizations(self):
        self.assertEqual(classify_name_entity("장*율")["entity_type"], "person_masked")
        self.assertEqual(
            classify_name_entity("현대중공업대공동문회")["entity_type"],
            "organization",
        )
        self.assertEqual(classify_name_entity("49기 동기회 기계과")["cohort_from_name"], "49회")

    def test_classifies_organizations_by_structure_without_specific_names(self):
        cases = {
            "56회 건축과 축쟁이": "organization",
            "56회 건축과 사우회": "organization",
            "푸른나무 모임": "organization",
            "한빛 사우회": "organization",
            "(주*산": "organization_masked",
        }
        for value, expected_type in cases.items():
            with self.subTest(value=value):
                result = classify_name_entity(value)
                self.assertEqual(result["entity_type"], expected_type)
                self.assertEqual(result["organization_name"], value)

        # 사람 이름의 마지막 글자가 '회'여도 단체로 오인하면 안 된다.
        self.assertEqual(classify_name_entity("김정회")["entity_type"], "person_real")

    def test_masked_name_search_filters_both_cohort_words(self):
        for cohort_word in ("58회", "58기"):
            with self.subTest(cohort_word=cohort_word):
                rows, sources, searched = _search_name_pandas(
                    f"{cohort_word} 이*규 출연금액 알려줘"
                )
                self.assertTrue(searched)
                self.assertIsNotNone(rows)
                self.assertEqual(set(rows["발행번호"]), {"2025-008"})
                self.assertEqual(set(rows["기수"]), {"58"})
                self.assertEqual(sources, ["test2025.png"])

    def test_identifier_search_returns_all_rows_for_issue_number(self):
        result, sources = _query_pandas_direct("2025-008 출연금액 알려줘")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["발행번호"]), {"2025-008"})
        self.assertEqual(sources, ["test2025.png"])

    def test_organization_search_uses_entity_columns(self):
        result, sources = _query_pandas_direct("현대중공업대공동문회 출연금액 알려줘")

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["entity_type"], "organization")
        self.assertEqual(sources, ["test2025.png"])

    def test_formatter_keeps_issue_number_group_and_sums_installments(self):
        result, _ = _query_pandas_direct("2025-008 출연금액 알려줘")
        answer = _format_dataframe_result_for_question(result, "2025-008 출연금액 알려줘")

        self.assertIn("2025-008", answer)
        self.assertIn("10,000,000원", answer)
        self.assertIn("총 2회", answer)

    def test_formatter_does_not_silently_choose_first_amount_column(self):
        rows = pd.DataFrame([
            {"이름": "김하늘", "기준_지원금액": "100,000", "실제_지원금액": "200,000"},
        ])

        ambiguous = _format_dataframe_result_for_question(rows, "김하늘 지원금액 알려줘")
        selected = _format_dataframe_result_for_question(rows, "김하늘 실제 지원금액 알려줘")

        self.assertIn("기준 지원금액", ambiguous)
        self.assertIn("실제 지원금액", ambiguous)
        self.assertNotIn("_", ambiguous)
        self.assertIn("200,000", selected)
        self.assertNotIn("100,000", selected)

    def test_plain_date_list_shows_every_row_as_names_only(self):
        rows = pd.DataFrame([
            {
                "년": 2025,
                "월": 1,
                "이름": "김하나" if index % 2 == 0 else "이두리",
                "학과": "경제학과",
                "결제금액": 10_000,
            }
            for index in range(60)
        ])
        rows.attrs["date_filter_evidence"] = {
            "period": "2025년 1월~2026년 1월",
            "column": "년·월",
        }

        answer = _format_dataframe_result_for_question(
            rows,
            "2025년 1월부터 2026년 1월까지 목록",
        )

        self.assertIn("총 60건", answer)
        self.assertNotIn("처음 50건", answer)
        self.assertEqual(answer.count("- 김하나") + answer.count("- 이두리"), 60)
        self.assertNotIn("경제학과", answer)
        self.assertNotIn("10,000", answer)

    def test_large_plain_list_has_bounded_name_preview(self):
        rows = pd.DataFrame({
            "이름": [f"회원{index:03d}" for index in range(250)],
            "결제금액": [10_000] * 250,
        })

        answer = _format_dataframe_result_for_question(rows, "전체 목록 보여줘")

        self.assertIn("총 250건", answer)
        self.assertIn("전체 250명 중 처음 200명 표시", answer)
        self.assertIn("회원199", answer)
        self.assertNotIn("회원200", answer)
        self.assertEqual(answer.count("\n- 회원"), 200)


if __name__ == "__main__":
    unittest.main()
