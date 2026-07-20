from __future__ import annotations

import unittest

import pandas as pd

from datastore.scope import document_scope
from pandas_engine.plan_validator import validate_query_plan
from pandas_engine.query_plan import QueryPlan


class PlanValidatorTest(unittest.TestCase):
    def setUp(self):
        self.df0 = pd.DataFrame(
            {
                "이름": ["김철수", "이영희"],
                "출연금액": ["1,000,000", "500,000"],
                "출연일자": ["2025-03-01", "2025-04-01"],
                "기수": [59, 58],
                "__row_id": ["row-1", "row-2"],
            }
        )
        self.df0.attrs["semantic_schema"] = {
            "columns": {
                "이름": {"data_type": "string"},
                "출연금액": {"data_type": "money"},
                "출연일자": {"data_type": "date"},
                "기수": {"data_type": "number"},
            }
        }
        self.df1 = pd.DataFrame({"이름": ["박민수"], "점검상태": ["완료"]})
        self.dataframes = {"df0": self.df0, "df1": self.df1}
        self.sources = {"df0": "후원대장.xlsx", "df1": "점검표.xlsx"}

    def _plan(self, **overrides) -> QueryPlan:
        payload = {
            "status": "ready",
            "dataframe": "df0",
            "operation": "list",
            "select": ["이름", "출연금액"],
        }
        payload.update(overrides)
        return QueryPlan.model_validate(payload)

    def _validate(self, plan: QueryPlan, question: str | None = None):
        return validate_query_plan(
            plan,
            question=question,
            dataframes=self.dataframes,
            source_by_alias=self.sources,
            explicit_dataframe_aliases=(
                {str(plan.dataframe)}
                if str(plan.dataframe) in self.dataframes
                else None
            ),
        )

    def test_valid_plan_resolves_dataframe_and_source(self):
        result = self._validate(
            self._plan(
                filters=[
                    {"column": "기수", "operator": "gte", "value": 58},
                    {"column": "이름", "operator": "contains", "value": "김"},
                ]
            )
        )

        self.assertTrue(result.is_valid)
        self.assertTrue(result.is_executable)
        self.assertIs(result.dataframe, self.df0)
        self.assertEqual(result.source_file, "후원대장.xlsx")
        self.assertEqual(result.issues, ())

    def test_unknown_dataframe_is_rejected(self):
        result = self._validate(self._plan(dataframe="df99"))

        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "unknown_dataframe")

    def test_dataframe_outside_selected_document_scope_is_rejected(self):
        with document_scope(["점검표.xlsx"]):
            result = self._validate(self._plan(dataframe="df0"))

        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "dataframe_out_of_scope")

    def test_multiple_unselected_documents_require_clarification(self):
        plan = self._plan(dataframe="df0")
        result = validate_query_plan(
            plan,
            dataframes=self.dataframes,
            source_by_alias=self.sources,
        )

        self.assertEqual(result.status, "clarification")
        self.assertFalse(result.is_valid)
        self.assertFalse(result.is_executable)
        self.assertEqual(result.issues[0].code, "ambiguous_document_scope")

    def test_multiple_tables_from_one_source_are_not_document_ambiguity(self):
        plan = self._plan(dataframe="df0")
        result = validate_query_plan(
            plan,
            dataframes={"df0": self.df0, "df1": self.df1},
            source_by_alias={"df0": "같은문서.xlsx", "df1": "같은문서.xlsx"},
        )

        self.assertTrue(result.is_executable)

    def test_unknown_columns_are_reported_for_every_execution_field(self):
        result = self._validate(
            self._plan(
                select=["없는선택컬럼"],
                filters=[{"column": "없는필터컬럼", "operator": "eq", "value": "값"}],
                sort=[{"column": "없는정렬컬럼", "direction": "asc"}],
                distinct_by=["없는중복컬럼"],
            )
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(
            {issue.column for issue in result.issues},
            {"없는선택컬럼", "없는필터컬럼", "없는정렬컬럼", "없는중복컬럼"},
        )
        self.assertTrue(all(issue.code == "unknown_column" for issue in result.issues))

    def test_internal_identity_column_is_rejected(self):
        result = self._validate(self._plan(select=["이름", "__row_id"]))

        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "internal_column")

    def test_money_aggregation_and_date_range_are_accepted(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "sum",
                "target": "출연금액",
                "filters": [
                    {
                        "column": "출연일자",
                        "operator": "between",
                        "value": ["2025-03-01", "2025-04-30"],
                    }
                ],
            }
        )
        result = self._validate(plan)

        self.assertTrue(result.is_executable)

    def test_sum_over_text_column_is_rejected(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df1",
                "operation": "sum",
                "target": "점검상태",
            }
        )
        result = self._validate(plan)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "incompatible_target")

    def test_contains_over_number_column_is_rejected(self):
        result = self._validate(
            self._plan(
                filters=[
                    {"column": "기수", "operator": "contains", "value": "59"}
                ]
            )
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "incompatible_operator")

    def test_invalid_numeric_filter_value_is_rejected(self):
        result = self._validate(
            self._plan(
                filters=[
                    {"column": "기수", "operator": "gte", "value": "오십구"}
                ]
            )
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "incompatible_value")

    def test_question_literals_and_comparison_operators_are_preserved(self):
        result = self._validate(
            self._plan(
                filters=[
                    {"column": "기수", "operator": "gte", "value": 49},
                    {"column": "출연금액", "operator": "gte", "value": "200만원"},
                ]
            ),
            "전체 중 49기 이상에서 200만원 이상 낸 사람 알려줘",
        )

        self.assertTrue(result.is_executable)

    def test_wrong_money_conversion_is_safely_recovered(self):
        result = self._validate(
            self._plan(
                filters=[
                    {"column": "기수", "operator": "gte", "value": 49},
                    {"column": "출연금액", "operator": "gte", "value": 200_000},
                ]
            ),
            "전체 중 49기 이상에서 200만원 이상 낸 사람 알려줘",
        )

        self.assertTrue(result.is_executable)
        self.assertEqual(result.plan.filters[1].value, "200만원")
        self.assertEqual(result.plan.filters[1].source_text, "200만원 이상")

    def test_wrong_comparison_operator_is_safely_recovered(self):
        result = self._validate(
            self._plan(
                filters=[
                    {"column": "기수", "operator": "gte", "value": 49},
                    {"column": "출연금액", "operator": "gt", "value": "200만원"},
                ]
            ),
            "전체 중 49기 이상에서 200만원 이상 낸 사람 알려줘",
        )

        self.assertTrue(result.is_executable)
        self.assertEqual(result.plan.filters[1].operator, "gte")

    def test_missing_question_condition_is_rejected(self):
        result = self._validate(
            self._plan(
                filters=[
                    {"column": "기수", "operator": "gte", "value": 49},
                ]
            ),
            "전체 중 49기 이상에서 200만원 이상 낸 사람 알려줘",
        )

        self.assertFalse(result.is_valid)
        self.assertIn("literal_mismatch", {issue.code for issue in result.issues})

    def test_numeric_plan_filter_not_present_in_question_is_rejected(self):
        result = self._validate(
            self._plan(
                filters=[
                    {"column": "기수", "operator": "gte", "value": 49},
                    {"column": "출연금액", "operator": "gte", "value": "200만원"},
                    {"column": "출연금액", "operator": "lt", "value": "500만원"},
                ]
            ),
            "전체 중 49기 이상에서 200만원 이상 낸 사람 알려줘",
        )

        self.assertFalse(result.is_valid)
        self.assertIn(
            "ungrounded_numeric_filter",
            {issue.code for issue in result.issues},
        )

    def test_invalid_source_text_cannot_override_unique_question_evidence(self):
        result = self._validate(
            self._plan(
                filters=[
                    {
                        "column": "출연금액",
                        "operator": "gte",
                        "value": "200만원",
                        "source_text": "금액 200만원 이상",
                    }
                ]
            ),
            "200만원 이상 낸 사람 알려줘",
        )

        self.assertTrue(result.is_executable)
        self.assertEqual(result.plan.filters[0].source_text, "200만원 이상")

    def test_mixed_source_text_does_not_hide_a_missing_condition(self):
        result = self._validate(
            self._plan(
                filters=[
                    {
                        "column": "출연금액",
                        "operator": "gte",
                        "value": "200만원",
                        "source_text": "49기 이상에서 200만원 이상",
                    }
                ]
            ),
            "49기 이상에서 200만원 이상 낸 사람 알려줘",
        )

        self.assertFalse(result.is_valid)
        self.assertIn(
            "literal_mismatch",
            {issue.code for issue in result.issues},
        )

    def test_non_executable_statuses_pass_without_dataframe_lookup(self):
        for status in ("clarification", "not_applicable"):
            with self.subTest(status=status):
                plan = QueryPlan.model_validate(
                    {
                        "status": status,
                        "message": "질문의 조회 대상을 확인해 주세요.",
                    }
                )
                result = self._validate(plan)

                self.assertTrue(result.is_accepted)
                self.assertFalse(result.is_valid)
                self.assertFalse(result.is_executable)
                self.assertEqual(result.status, status)


if __name__ == "__main__":
    unittest.main()
