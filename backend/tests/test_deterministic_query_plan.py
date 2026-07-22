from __future__ import annotations

import unittest

import pandas as pd

from rag.deterministic_query_plan import build_schema_grounded_plan
from pandas_engine.plan_validator import validate_query_plan


class DeterministicQueryPlanTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "\ud68c\uc6d0\uba85": ["\ubc15\uc7ac\ud64d(\uc6a9\uc9c4)", "\uae40\ub098\ub2e4", "\uae40\ub098\ub2e4"],
                "\uacb0\uc81c_\uae08\uc561": [20_000, 30_000, 10_000],
                "\ub144": [2025, 2026, 2026],
                "\uc804\uacf5": ["\ud1a0\ubaa9\uacfc", "\uae30\uacc4\uacfc", "\uae30\uacc4\uacfc"],
                "\uc774\uba54\uc77c": ["first@example.com", "second@example.com", "second@example.com"],
                "\uc804\ud654\ubc88\ud638": ["01011112222", "01033334444", "01033334444"],
            }
        )

    def _plan(self, question: str, hint: str):
        return build_schema_grounded_plan(
            question,
            dataframes={"df0": self.df},
            operation_hint=hint,
        )

    def test_unambiguous_display_name_base_becomes_exact_person_filter(self):
        plan = self._plan("\ubc15\uc7ac\ud64d \uc5bc\ub9c8\uc57c?", "sum_amount")

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "sum")
        self.assertEqual(plan.filters[0].column, "\ud68c\uc6d0\uba85")
        self.assertEqual(plan.filters[0].value, "\ubc15\uc7ac\ud64d(\uc6a9\uc9c4)")
        validation = validate_query_plan(
            plan,
            question="\ubc15\uc7ac\ud64d \uc5bc\ub9c8\uc57c?",
            dataframes={"df0": self.df},
            source_by_alias={"df0": "test.xlsx"},
            operation_hint="sum_amount",
        )
        self.assertTrue(validation.is_executable, validation.issues)

    def test_payment_existence_question_becomes_person_amount_sum(self):
        plan = self._plan("\uae40\ub098\ub2e4 \ub3c8 \ub0c8\uc5b4?", "lookup_amount")

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "sum")
        self.assertEqual(plan.target, "\uacb0\uc81c_\uae08\uc561")
        self.assertEqual(plan.filters[0].value, "\uae40\ub098\ub2e4")

    def test_period_top_person_becomes_ranked_group_sum(self):
        plan = self._plan(
            "2025\ub144\ubd80\ud130 2026\ub144\uae4c\uc9c0 \uac00\uc7a5 \ub3c8\uc744 \ub9ce\uc774 \ub0b8 \uc0ac\ub78c",
            "structured_query",
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "group_sum")
        self.assertEqual(plan.group_by, ("\ud68c\uc6d0\uba85",))
        self.assertEqual(plan.filters[0].operator, "between")
        self.assertEqual(plan.filters[0].value, (2025, 2026))

    def test_contact_value_reverse_lookup_returns_person(self):
        plan = self._plan("second@example.com 이메일을 가진 사람 누구야?", "lookup_field")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.filters[0].column, "이메일")
        self.assertEqual(plan.select, ("회원명",))
        validation = validate_query_plan(
            plan,
            question="second@example.com 이메일을 가진 사람 누구야?",
            dataframes={"df0": self.df},
            source_by_alias={"df0": "test.xlsx"},
            operation_hint="lookup_field",
        )
        self.assertTrue(validation.is_executable, validation.issues)

    def test_phone_identifier_is_not_reused_as_money_filter(self):
        plan = self._plan("01033334444 전화번호를 가진 사람 누구야", "lookup_field")
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.filters), 1)
        self.assertEqual(plan.filters[0].column, "전화번호")

    def test_missing_alias_builds_null_filter_without_model(self):
        plan = self._plan("이메일이 공백인 사람 리스트 알려줘", "structured_query")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.filters[0].column, "이메일")
        self.assertEqual(plan.filters[0].operator, "is_null")

    def test_contact_filtered_amount_lookup_returns_sum(self):
        plan = self._plan("second@example.com 이메일을 가진 사람 얼마냈어?", "lookup_amount")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "sum")
        self.assertEqual(plan.target, "결제_금액")
        self.assertEqual(plan.filters[0].column, "이메일")

    def test_explicit_average_overrides_incorrect_sum_hint(self):
        plan = self._plan("기계과 평균 얼마냈어?", "sum_amount")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "mean")
        self.assertEqual(plan.target, "결제_금액")
        self.assertEqual(plan.filters[0].value, "기계과")

    def test_payment_frequency_counts_matching_rows(self):
        plan = self._plan("김나다 몇 번 돈 냈어?", "lookup_amount")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "count")
        self.assertFalse(plan.distinct_by)

    def test_mode_amount_is_not_total_sum(self):
        plan = self._plan("돈 최빈값 얼마야?", "sum_amount")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "mode")

    def test_lowest_total_person_is_ascending_group_sum(self):
        plan = self._plan("가장 돈 적게 낸 사람 누구야?", "structured_query")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "group_sum")
        self.assertEqual(plan.group_order, "asc")

    def test_department_alias_is_selected_for_person_lookup(self):
        plan = self._plan("김나다 무슨 과야?", "lookup_field")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.select, ("회원명", "전공"))


if __name__ == "__main__":
    unittest.main()
