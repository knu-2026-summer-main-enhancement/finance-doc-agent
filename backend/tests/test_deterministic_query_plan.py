from __future__ import annotations

import unittest

import pandas as pd

from rag.deterministic_query_plan import (
    ambiguous_person_lookup_candidates,
    build_auto_schema_grounded_plan,
    build_schema_grounded_plan,
    has_unmatched_person_amount_reference,
    has_unmatched_person_field_reference,
    is_grounded_person_amount_lookup_question,
    is_grounded_person_payment_existence_question,
)
from pandas_engine.plan_validator import validate_query_plan


class DeterministicQueryPlanTest(unittest.TestCase):
    def test_auto_plan_routes_cross_month_range_to_date_executor(self):
        operation, plan = build_auto_schema_grounded_plan(
            "2025년 6월부터 2026년 1월까지 목록",
            dataframes={},
        )

        self.assertEqual(operation, "structured_query")
        self.assertIsNone(plan)

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "\ud68c\uc6d0\uba85": ["\ubc15\uc7ac\ud64d(\uc6a9\uc9c4)", "\uae40\ub098\ub2e4", "\uae40\ub098\ub2e4"],
                "\uacb0\uc81c_\uae08\uc561": [20_000, 30_000, 10_000],
                "\ub144": [2025, 2026, 2026],
                "\uc804\uacf5": ["\ud1a0\ubaa9\uacfc", "\uae30\uacc4\uacfc", "\uae30\uacc4\uacfc"],
                "\uc774\uba54\uc77c": ["first@example.com", "second@example.com", "second@example.com"],
                "\uc804\ud654\ubc88\ud638": ["01011112222", "01033334444", "01033334444"],
                "\uacb0\uc81c_\ub4f1\ub85d_\ub0a0\uc9dc": ["2025-01-01", None, "2026-01-01"],
            }
        )

    def test_grounded_person_amount_lookup_is_not_a_global_sum(self):
        dataframes = {"payments": self.df}

        self.assertTrue(
            is_grounded_person_amount_lookup_question(
                "김나다 얼마야",
                dataframes=dataframes,
            )
        )
        self.assertFalse(
            is_grounded_person_amount_lookup_question(
                "없는이름 얼마야",
                dataframes=dataframes,
            )
        )

    def test_person_lookup_uses_only_canonical_person_filter(self):
        dataframe = pd.DataFrame(
            {
                "회원명": ["김현수", "김현민"],
                "성명_원문": ["김현수", "김현민"],
                "성명_검색키": ["김현수", "김현민"],
                "성명_마스킹패턴": ["김*수", "김*현"],
                "결제_금액": [10_000, 20_000],
            }
        )

        plan = build_schema_grounded_plan(
            "김현수 얼마야?",
            dataframes={"payments": dataframe},
            operation_hint="lookup_amount",
        )

        self.assertIsNotNone(plan)
        self.assertEqual(
            [(item.column, item.value) for item in plan.filters],
            [("회원명", "김현수")],
        )

    def test_reordered_variant_headers_support_person_amount_and_date_queries(self):
        dataframe = pd.DataFrame({
            "비고": ["첫 행", "둘째 행"],
            "납부자": ["김현수", "이서연"],
            "결제액(원)": ["10,000", "20,000"],
            "결제 일시": ["2026-01-03 10:30:00", "2026-02-01 09:00:00"],
        })

        amount_plan = build_schema_grounded_plan(
            "김현수 얼마야?",
            dataframes={"payments": dataframe},
            operation_hint="lookup_amount",
        )
        maximum_plan = build_schema_grounded_plan(
            "가장 큰 금액 뭐야?",
            dataframes={"payments": dataframe},
            operation_hint="max_amount",
        )

        self.assertIsNotNone(amount_plan)
        self.assertEqual(amount_plan.target, "결제액(원)")
        self.assertEqual(
            [(item.column, item.value) for item in amount_plan.filters],
            [("납부자", "김현수")],
        )
        self.assertIsNotNone(maximum_plan)
        self.assertEqual(maximum_plan.target, "결제액(원)")

    def test_ambiguous_amount_query_is_independent_of_column_order(self):
        values = {
            "성명": ["김현수", "이서연"],
            "장학금액": [100_000, 200_000],
            "납부금액": [10_000, 20_000],
        }
        orders = (
            ("성명", "장학금액", "납부금액"),
            ("성명", "납부금액", "장학금액"),
        )

        for columns in orders:
            with self.subTest(columns=columns):
                dataframe = pd.DataFrame({column: values[column] for column in columns})
                ambiguous = build_schema_grounded_plan(
                    "총 금액 얼마야?",
                    dataframes={"payments": dataframe},
                    operation_hint="sum_amount",
                )
                explicit = build_schema_grounded_plan(
                    "납부금액 총합 얼마야?",
                    dataframes={"payments": dataframe},
                    operation_hint="sum_amount",
                )

                self.assertIsNone(ambiguous)
                self.assertIsNotNone(explicit)
                self.assertEqual(explicit.target, "납부금액")

    def test_category_lookup_does_not_filter_on_derived_name_mask(self):
        dataframe = pd.DataFrame(
            {
                "회원명": ["임종식", "전기수"],
                "전공": ["전기과", "기계과"],
                "성명_마스킹패턴": ["임*식", "전*기"],
                "결제_금액": [10_000, 20_000],
            }
        )
        dataframe.attrs["source_columns"] = ["회원명", "전공", "결제_금액"]

        plan = build_schema_grounded_plan(
            "전기과 전체목록 보여줘",
            dataframes={"payments": dataframe},
            operation_hint="structured_query",
        )

        self.assertIsNotNone(plan)
        self.assertEqual(
            [(item.column, item.value) for item in plan.filters],
            [("전공", "전기과")],
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

    def test_display_name_base_is_reused_for_field_lookup(self):
        plan = self._plan("박재홍 학과 뭐야?", "lookup_field")

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.filters[0].column, "회원명")
        self.assertEqual(plan.filters[0].value, "박재홍(용진)")
        self.assertEqual(plan.select, ("회원명", "전공"))

    def test_ambiguous_partial_person_name_is_not_grounded(self):
        plan = self._plan("김나 전화번호 뭐야?", "lookup_field")

        self.assertIsNone(plan)

    def test_ambiguous_person_lookup_returns_candidates_not_absence(self):
        self.df.loc[1, "회원명"] = "김철한"
        self.df.loc[2, "회원명"] = "김철수"

        candidates = ambiguous_person_lookup_candidates(
            "김철 전화번호 뭐야?", dataframes={"df0": self.df}
        )

        self.assertEqual(candidates, ("김철한", "김철수"))

    def test_payment_existence_question_becomes_person_amount_sum(self):
        plan = self._plan("\uae40\ub098\ub2e4 \ub3c8 \ub0c8\uc5b4?", "lookup_amount")

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "sum")
        self.assertEqual(plan.target, "\uacb0\uc81c_\uae08\uc561")
        self.assertEqual(plan.filters[0].value, "\uae40\ub098\ub2e4")

    def test_amount_order_and_ordinal_are_explicit_list_rank(self):
        plan = self._plan("결제 금액 큰 순으로 2번째 기록 보여줘", "list_records")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.sort[0].column, "결제_금액")
        self.assertEqual(plan.sort[0].direction, "desc")
        self.assertEqual(plan.rank_position, 2)
        self.assertEqual(plan.tie_policy, "dense")

    def test_person_total_ordinal_uses_group_sum_not_payment_row(self):
        plan = self._plan("두 번째로 많이 낸 사람은 누구야?", "structured_query")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "group_sum")
        self.assertEqual(plan.group_by, ("회원명",))
        self.assertEqual(plan.rank_position, 2)

    def test_common_second_ordinal_typo_still_uses_person_total_rank(self):
        plan = self._plan("두번쨰로 돈 많이 낸 사람 누구야?", "structured_query")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "group_sum")
        self.assertEqual(plan.group_by, ("회원명",))
        self.assertEqual(plan.rank_position, 2)
        self.assertEqual(plan.group_order, "desc")

    def test_largest_amount_overrides_incorrect_sum_hint(self):
        for question in (
            "가장 큰 금액 뭐야?",
            "제일 많은 돈 얼마야?",
            "가장 많은 금액 알려줘",
            "돈 중에서 제일 많은 값 뭐야?",
        ):
            with self.subTest(question=question):
                plan = self._plan(question, "sum_amount")
                self.assertIsNotNone(plan)
                self.assertEqual(plan.operation, "max")
                self.assertEqual(plan.target, "결제_금액")

    def test_most_money_person_wording_remains_person_total_rank(self):
        plan = self._plan("제일 많은 돈 낸 사람 누구야?", "structured_query")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "group_sum")
        self.assertEqual(plan.group_order, "desc")
        self.assertEqual(plan.group_by, ("회원명",))

    def test_natural_order_wording_and_explicit_limit_are_grounded(self):
        plan = self._plan("2026년 결제 금액을 큰 순서대로 3건 보여줘", "structured_query")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.sort[0].direction, "desc")
        self.assertEqual(plan.limit, 3)

    def test_ordinal_large_record_is_dense_row_rank(self):
        plan = self._plan("2026년 두 번째로 큰 결제 내역 보여줘", "structured_query")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.sort[0].direction, "desc")
        self.assertEqual(plan.rank_position, 2)
        self.assertEqual(plan.tie_policy, "dense")

    def test_ordinal_large_person_total_is_group_rank(self):
        plan = self._plan("2026년 누적 결제 금액이 두 번째로 큰 사람 알려줘", "structured_query")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "group_sum")
        self.assertEqual(plan.group_order, "desc")
        self.assertEqual(plan.rank_position, 2)

    def test_short_person_money_terms_become_person_amount_sum(self):
        for question in (
            "\uae40\ub098\ub2e4 \uc5bc\ub9c8", "\uae40\ub098\ub2e4 \uae08\uc561",
            "\uae40\ub098\ub2e4 \ub3c8", "\uae40\ub098\ub2e4 \uae08\uc561 \uc870\ud68c\ud574\uc918",
        ):
            with self.subTest(question=question):
                plan = self._plan(question, "sum_amount")
                self.assertIsNotNone(plan)
                self.assertEqual(plan.operation, "sum")
                self.assertEqual(plan.target, "\uacb0\uc81c_\uae08\uc561")
                self.assertEqual(plan.filters[0].column, "\ud68c\uc6d0\uba85")
                self.assertEqual(plan.filters[0].value, "\uae40\ub098\ub2e4")

    def test_grounded_payment_existence_is_recognized_before_fallback(self):
        question = "\uae40\ub098\ub2e4 " + chr(0xB0C8) + chr(0xC74C) + "?"
        self.assertTrue(is_grounded_person_payment_existence_question(
            question, dataframes={"df0": self.df}
        ))
        self.assertFalse(is_grounded_person_payment_existence_question(
            "\uc5c6\ub294\uc0ac\ub78c " + chr(0xB0C8) + chr(0xC74C) + "?",
            dataframes={"df0": self.df},
        ))

    def test_absent_leading_person_amount_lookup_is_detected_without_name_hardcoding(self):
        self.assertTrue(has_unmatched_person_amount_reference(
            "\uc815\uacbd\ucc44 \uc5bc\ub9c8", dataframes={"df0": self.df}
        ))
        self.assertFalse(has_unmatched_person_amount_reference(
            "\uae40\ub098\ub2e4 \uc5bc\ub9c8", dataframes={"df0": self.df}
        ))
        self.assertFalse(has_unmatched_person_amount_reference(
            "\uae30\uacc4\uacfc \uc5bc\ub9c8", dataframes={"df0": self.df}
        ))

    def test_real_name_matches_only_compatible_masked_name(self):
        masked = self.df.iloc[[0]].copy()
        masked["회원명"] = ["추*진"]

        plan = build_schema_grounded_plan(
            "추교진 얼마", dataframes={"df0": masked}, operation_hint="sum_amount",
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.filters[0].value, "추*진")
        self.assertFalse(has_unmatched_person_amount_reference(
            "추교진 얼마", dataframes={"df0": masked},
        ))
        self.assertTrue(has_unmatched_person_amount_reference(
            "김교진 얼마", dataframes={"df0": masked},
        ))

    def test_absent_leading_person_field_lookup_is_detected_without_name_hardcoding(self):
        self.assertTrue(has_unmatched_person_field_reference(
            "\uc815\uacbd\ucc44 \uc804\ud654\ubc88\ud638 \ubb50\uc57c?", dataframes={"df0": self.df}
        ))
        self.assertFalse(has_unmatched_person_field_reference(
            "\uae40\ub098\ub2e4 \uc804\ud654\ubc88\ud638 \ubb50\uc57c?", dataframes={"df0": self.df}
        ))
        self.assertFalse(has_unmatched_person_field_reference(
            "\uae30\uacc4\uacfc \uc804\ud654\ubc88\ud638 \ubb50\uc57c?", dataframes={"df0": self.df}
        ))

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

    def test_cohort_list_has_exact_filter_and_person_projection(self):
        df = self.df.copy()
        df["기수"] = [49, 49, 58]

        plan = build_schema_grounded_plan(
            "49기 목록", dataframes={"df0": df}, operation_hint="list_records",
        )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.filters[0].column, "기수")
        self.assertEqual(plan.filters[0].value, 49)
        self.assertEqual(plan.select, ("회원명",))

    def test_year_and_month_amount_query_uses_component_date_columns(self):
        self.df[chr(0xC6D4)] = [1, 1, 2]
        question = (
            "2026" + chr(0xB144) + " 1" + chr(0xC6D4) + chr(0xC5D0) + " "
            + chr(0xB0B8) + " " + chr(0xB3C8) + " " + chr(0xCD1D) + chr(0xD569)
            + " " + chr(0xC5BC) + chr(0xB9C8) + chr(0xC57C) + "?"
        )
        plan = self._plan(question, "sum_amount")

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "sum")
        self.assertEqual(plan.target, "결제_금액")
        self.assertEqual(
            [(condition.column, condition.value) for condition in plan.filters],
            [("년", 2026), ("월", 1)],
        )

    def test_month_range_defers_to_date_filter_instead_of_using_first_month(self):
        plan = self._plan(
            "2025년 6월부터 2026년 1월까지 목록",
            "structured_query",
        )

        self.assertIsNone(plan)

    def test_person_count_uses_distinct_person_column(self):
        plan = self._plan("기계과 사람 수", "count_records")

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "count")
        self.assertEqual(plan.distinct_by, ("회원명",))

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

    def test_missing_fee_type_and_registered_date_do_not_expand_filters(self):
        self.df["회비_구분"] = ["정기", None, "정기"]

        fee_plan = self._plan("회비 종류가 미등록인 내역을 보여줘", "structured_query")
        date_plan = self._plan("결제 등록일 미등록 기록만 보여줘", "structured_query")

        self.assertIsNotNone(fee_plan)
        self.assertEqual(
            [(item.column, item.operator) for item in fee_plan.filters],
            [("회비_구분", "is_null")],
        )
        self.assertIsNotNone(date_plan)
        self.assertEqual(
            [(item.column, item.operator) for item in date_plan.filters],
            [("결제_등록_날짜", "is_null")],
        )

    def test_year_and_month_group_aliases_build_complete_group_plans(self):
        self.df["월"] = [1, 1, 2]
        year_plan = self._plan("해마다 납부액을 모아서 보여줘", "structured_query")
        month_plan = self._plan("달마다 납부액을 보여줘", "structured_query")

        self.assertIsNotNone(year_plan)
        self.assertEqual(year_plan.group_by, ("년",))
        self.assertIsNotNone(month_plan)
        self.assertEqual(month_plan.group_by, ("월",))

    def test_comparison_filter_and_projection_are_built_together(self):
        plan = self._plan("10만원 이상 낸 사람의 전공 알려줘", "structured_query")

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "list")
        self.assertEqual(
            [(item.column, item.operator, item.value) for item in plan.filters],
            [("결제_금액", "gte", "10만원")],
        )
        self.assertEqual(plan.select, ("회원명", "전공"))

    def test_money_alias_is_a_projection_but_null_filter_is_not(self):
        amount_plan = self._plan("기계과 회원 이름과 납부액 보여줘", "structured_query")
        null_plan = self._plan("2025년 가입자 중 이메일 없는 사람 목록", "structured_query")

        self.assertIsNotNone(amount_plan)
        self.assertEqual(amount_plan.select, ("회원명", "결제_금액"))
        self.assertIsNotNone(null_plan)
        self.assertEqual(null_plan.select, ("회원명",))

    def test_cohort_comparison_keeps_operator_and_person_list(self):
        df = self.df.copy()
        df["기수"] = [49, 50, 58]
        plan = build_schema_grounded_plan(
            "50기 이상 회원 목록", dataframes={"df0": df}, operation_hint="structured_query",
        )

        self.assertIsNotNone(plan)
        self.assertEqual(
            [(item.column, item.operator, item.value) for item in plan.filters],
            [("기수", "gte", 50)],
        )
        self.assertEqual(plan.select, ("회원명",))

    def test_numeric_rank_word_uses_dense_person_total_ranking(self):
        plan = self._plan("돈을 많이 낸 사람 2위는 누구야?", "structured_query")

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "group_sum")
        self.assertEqual(plan.group_by, ("회원명",))
        self.assertEqual(plan.group_order, "desc")
        self.assertEqual(plan.rank_position, 2)
        self.assertEqual(plan.tie_policy, "dense")

    def test_group_top_n_keeps_grouping_not_person_ranking(self):
        plan = self._plan("전공별 납부액 상위 3개 보여줘", "structured_query")

        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "group_sum")
        self.assertEqual(plan.group_by, ("전공",))
        self.assertEqual(plan.group_order, "desc")
        self.assertEqual(plan.top_n, 3)

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

    def test_person_field_terms_are_lookup_projections(self):
        for question, expected in (
            ("\uae40\ub098\ub2e4 \ud559\uacfc", ("회원명", "전공")),
            ("\uae40\ub098\ub2e4 \uc804\ud654\ubc88\ud638", ("회원명", "전화번호")),
        ):
            with self.subTest(question=question):
                plan = self._plan(question, "lookup_field")
                self.assertIsNotNone(plan)
                self.assertEqual(plan.operation, "list")
                self.assertEqual(plan.filters[0].column, "회원명")
                self.assertEqual(plan.filters[0].value, "\uae40\ub098\ub2e4")
                self.assertEqual(plan.select, expected)

    def test_person_payment_time_lookup_returns_all_temporal_evidence(self):
        plan = self._plan("\uae40\ub098\ub2e4 \uc5b8\uc81c \ub3c8 \ub0c8\uc5b4?", "lookup_field")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.filters[0].value, "\uae40\ub098\ub2e4")
        self.assertEqual(
            plan.select,
            ("회원명", "년", "결제_등록_날짜"),
        )


if __name__ == "__main__":
    unittest.main()
