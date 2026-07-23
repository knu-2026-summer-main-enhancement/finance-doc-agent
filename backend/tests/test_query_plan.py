from __future__ import annotations

import unittest

from pydantic import ValidationError

from pandas_engine.query_plan import QueryPlan


class QueryPlanSchemaTest(unittest.TestCase):
    def test_list_plan_applies_safe_defaults(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "filters": [
                    {"column": "기수", "operator": "eq", "value": 59},
                    {
                        "column": "이름",
                        "operator": "contains",
                        "value": "김",
                    },
                ],
                "select": ["이름", "출연금액"],
            }
        )

        self.assertEqual(plan.effective_result_mode, "records")
        self.assertIsNone(plan.effective_limit)
        self.assertEqual(plan.filter_logic, "all")
        self.assertEqual(plan.filters[0].value, 59)

    def test_max_can_return_matching_records(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "max",
                "target": "출연금액",
                "result_mode": "records",
                "select": ["이름", "출연금액"],
                "top_n": 3,
            }
        )

        self.assertEqual(plan.effective_result_mode, "records")
        self.assertEqual(plan.effective_top_n, 3)

    def test_scalar_aggregation_requires_target(self):
        with self.assertRaises(ValidationError):
            QueryPlan.model_validate(
                {
                    "status": "ready",
                    "dataframe": "df0",
                    "operation": "sum",
                }
            )

    def test_scalar_aggregation_rejects_record_fields(self):
        with self.assertRaises(ValidationError):
            QueryPlan.model_validate(
                {
                    "status": "ready",
                    "dataframe": "df0",
                    "operation": "sum",
                    "target": "출연금액",
                    "select": ["이름"],
                }
            )

    def test_between_requires_exactly_two_values(self):
        with self.assertRaises(ValidationError):
            QueryPlan.model_validate(
                {
                    "status": "ready",
                    "dataframe": "df0",
                    "operation": "list",
                    "filters": [
                        {
                            "column": "출연일자",
                            "operator": "between",
                            "value": ["2025-03-01"],
                        }
                    ],
                }
            )

    def test_null_operator_rejects_value(self):
        with self.assertRaises(ValidationError):
            QueryPlan.model_validate(
                {
                    "status": "ready",
                    "dataframe": "df0",
                    "operation": "count",
                    "filters": [
                        {
                            "column": "비고",
                            "operator": "is_null",
                            "value": "없음",
                        }
                    ],
                }
            )

    def test_clarification_has_no_executable_fields(self):
        plan = QueryPlan.model_validate(
            {
                "status": "clarification",
                "message": "계산할 금액 항목을 지정해 주세요.",
                "candidates": ["1단계 장학금액", "2단계 장학금액"],
            }
        )

        self.assertIsNone(plan.operation)
        self.assertEqual(len(plan.candidates), 2)

        with self.assertRaises(ValidationError):
            QueryPlan.model_validate(
                {
                    "status": "clarification",
                    "message": "조회 대상을 지정해 주세요.",
                    "dataframe": "df0",
                    "operation": "list",
                }
            )

    def test_not_applicable_requires_message(self):
        with self.assertRaises(ValidationError):
            QueryPlan.model_validate(
                {
                    "status": "not_applicable",
                }
            )

    def test_unknown_fields_and_operations_are_rejected(self):
        with self.assertRaises(ValidationError):
            QueryPlan.model_validate(
                {
                    "status": "ready",
                    "dataframe": "df0",
                    "operation": "execute_python",
                    "python": "result = df0.iloc[0]",
                }
            )

    def test_json_schema_lists_only_supported_operations(self):
        schema = QueryPlan.model_json_schema()
        operation_schema = schema["properties"]["operation"]["anyOf"][0]

        self.assertEqual(
            operation_schema["enum"],
            ["list", "count", "sum", "mean", "median", "mode", "min", "max", "group_sum"],
        )

    def test_plan_and_nested_collections_are_immutable(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "select": ["이름"],
            }
        )

        with self.assertRaises(ValidationError):
            plan.operation = "count"
        with self.assertRaises(ValidationError):
            plan.select = ("출연금액",)
        self.assertIsInstance(plan.select, tuple)
        self.assertIsInstance(plan.filters, tuple)

    def test_llm_self_confidence_is_not_part_of_execution_contract(self):
        with self.assertRaises(ValidationError):
            QueryPlan.model_validate(
                {
                    "status": "ready",
                    "dataframe": "df0",
                    "operation": "list",
                    "confidence": 0.99,
                }
            )


if __name__ == "__main__":
    unittest.main()
