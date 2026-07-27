from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from rag.pandas_rag import (
    _answer_group_payment_comparison,
    _answer_period_payment_comparison,
    _answer_year_amount_comparison,
    _answer_year_nonpayer_comparison,
    _answer_year_person_comparison,
)


class YearAmountComparisonTests(unittest.TestCase):
    def test_compares_months_and_halves(self) -> None:
        frame = pd.DataFrame({
            "월": [1, 1, 2, 2, 7, 8],
            "이름": ["김하나", "이둘", "김하나", "박셋", "김하나", "이둘"],
            "결제_금액": [100_000, 100_000, 150_000, 250_000, 200_000, 300_000],
        })
        with patch("rag.pandas_rag._df_sources", {"df_fee": "회비.xlsx"}):
            month_answer, _, _ = _answer_period_payment_comparison(
                "1월과 2월 납부 현황 비교해줘", {"df_fee": frame}
            )
            half_answer, _, _ = _answer_period_payment_comparison(
                "상반기와 하반기 납부 금액·인원 비교해줘", {"df_fee": frame}
            )

        self.assertIn("1월: 200,000원, 2명", month_answer)
        self.assertIn("2월: 400,000원, 2명", month_answer)
        self.assertIn("상반기: 600,000원", half_answer)
        self.assertIn("하반기: 500,000원", half_answer)

    def test_summarizes_department_payment_rate(self) -> None:
        frame = pd.DataFrame({
            "학과": ["경영", "경영", "컴퓨터", "컴퓨터"],
            "이름": ["김하나", "이둘", "박셋", "최넷"],
            "결제_금액": [100_000, 0, 200_000, 200_000],
        })
        with patch("rag.pandas_rag._df_sources", {"df_fee": "회비.xlsx"}):
            answer, _, _ = _answer_group_payment_comparison(
                "학과별 납부 인원과 금액 비교해줘", {"df_fee": frame}
            )

        self.assertIn("경영: 납부 1명 / 전체 2명 (50.0%), 100,000원", answer)
        self.assertIn("컴퓨터: 납부 2명 / 전체 2명 (100.0%), 400,000원", answer)

    def test_compares_people_sets_and_person_amount_changes(self) -> None:
        frame = pd.DataFrame({
            "연도": [2025, 2025, 2025, 2026, 2026, 2026],
            "이름": ["김하나", "이둘", "박셋", "김하나", "이둘", "최넷"],
            "결제_금액": [100_000, 200_000, 300_000, 150_000, 100_000, 100_000],
        })
        with patch("rag.pandas_rag._df_sources", {"df_fee": "회비.xlsx"}):
            count_answer, _, _ = _answer_year_person_comparison(
                "2025년과 2026년 납부 인원 비교해줘", {"df_fee": frame}
            )
            both_answer, _, _ = _answer_year_person_comparison(
                "2025년과 2026년에 모두 낸 사람 누구야?", {"df_fee": frame}
            )
            later_answer, _, _ = _answer_year_person_comparison(
                "2025년에 안 냈다가 2026년에 낸 사람 누구야?", {"df_fee": frame}
            )
            increase_answer, _, _ = _answer_year_person_comparison(
                "2025년보다 2026년에 더 많이 낸 사람 보여줘", {"df_fee": frame}
            )
            decrease_answer, _, _ = _answer_year_person_comparison(
                "2025년보다 2026년에 금액이 줄어든 사람 보여줘", {"df_fee": frame}
            )

        self.assertIn("- 변화: +0명", count_answer)
        self.assertIn("김하나", both_answer)
        self.assertIn("이둘", both_answer)
        self.assertIn("최넷", later_answer)
        self.assertIn("김하나", increase_answer)
        self.assertIn("이둘", decrease_answer)

    def test_lists_people_missing_from_later_year(self) -> None:
        frame = pd.DataFrame({
            "연도": [2025, 2025, 2026, 2026],
            "이름": ["김하나", "이둘", "이둘", "박셋"],
            "결제_금액": [100_000, 100_000, 100_000, 100_000],
        })
        with patch("rag.pandas_rag._df_sources", {"df_fee": "회비.xlsx"}):
            answer, sources, route = _answer_year_nonpayer_comparison(
                "2025년에 낸 사람 중에 2026년도에 안낸 사람 누구야?",
                {"df_fee": frame},
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, ["회비.xlsx"])
        self.assertIn("김하나", answer)
        self.assertNotIn("| 이둘 |", answer)
        self.assertIn("1명", answer)

    def test_compares_two_explicit_years_in_one_document(self) -> None:
        frame = pd.DataFrame({
            "연도": [2025, 2025, 2026, 2026],
            "결제_금액": [100_000, 200_000, 250_000, 350_000],
        })
        with patch("rag.pandas_rag._df_sources", {"df_fee": "회비.xlsx"}):
            answer, sources, route = _answer_year_amount_comparison(
                "2025년과 2026년 결제 금액을 비교해줘",
                {"df_fee": frame},
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, ["회비.xlsx"])
        self.assertIn("300,000원", answer)
        self.assertIn("600,000원", answer)
        self.assertIn("+300,000원 (+100.0%)", answer)

    def test_requires_one_selected_document(self) -> None:
        frame = pd.DataFrame({"연도": [2025, 2026], "결제_금액": [1, 2]})
        with patch("rag.pandas_rag._df_sources", {
            "first": "첫문서.xlsx",
            "second": "둘째문서.xlsx",
        }):
            answer, _, _ = _answer_year_amount_comparison(
                "2025년과 2026년 결제 금액 비교",
                {"first": frame, "second": frame},
            )

        self.assertIn("하나 선택", answer)


if __name__ == "__main__":
    unittest.main()
