from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from pandas_engine.query_plan import QueryPlan

import main


class MainQuestionPreRoutingTest(unittest.IsolatedAsyncioTestCase):
    async def test_whole_table_projection_skips_question_llm(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "select": ["회원명", "전공"],
            }
        )
        decide = AsyncMock()
        with (
            patch.object(main, "scoped_mapping", return_value={}),
            patch.object(main, "build_schema_grounded_plan", return_value=plan) as build,
            patch.object(main, "decide_question", decide),
        ):
            guard, route, strategy = await main._resolve_llm_question(
                "전체 기록에서 회원명과 전공만 보여줘"
            )

        self.assertEqual(route, "PANDAS")
        self.assertEqual(strategy, "QUERY_PLAN")
        self.assertEqual(tuple(guard.operations), ("structured_query",))
        decide.assert_not_awaited()
        self.assertEqual(
            build.call_args.kwargs["operation_hint"],
            "structured_query",
        )

    async def test_group_sum_skips_question_llm_when_dp_can_plan_it(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "group_sum",
                "target": "결제_금액",
                "group_by": ["전공"],
            }
        )
        decide = AsyncMock()
        with (
            patch.object(main, "scoped_mapping", return_value={}),
            patch.object(main, "build_schema_grounded_plan", return_value=plan),
            patch.object(main, "decide_question", decide),
        ):
            _, route, strategy = await main._resolve_llm_question(
                "전공별 납부액을 묶어서 보여줘"
            )

        self.assertEqual(route, "PANDAS")
        self.assertEqual(strategy, "QUERY_PLAN")
        decide.assert_not_awaited()

    async def test_narrow_keyword_branch_falls_back_to_general_dp(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "group_sum",
                "target": "결제_금액",
                "group_by": ["전공"],
            }
        )

        def build_plan(*args, **kwargs):
            if kwargs["operation_hint"] == "lookup_field":
                return None
            if kwargs["operation_hint"] == "structured_query":
                return plan
            return None

        decide = AsyncMock()
        with (
            patch.object(main, "scoped_mapping", return_value={}),
            patch.object(main, "build_schema_grounded_plan", side_effect=build_plan) as build,
            patch.object(main, "decide_question", decide),
        ):
            guard, route, strategy = await main._resolve_llm_question(
                "전공별 납부액을 묶어서 보여줘"
            )

        self.assertEqual(route, "PANDAS")
        self.assertEqual(strategy, "QUERY_PLAN")
        self.assertEqual(tuple(guard.operations), ("structured_query",))
        self.assertEqual(
            [call.kwargs["operation_hint"] for call in build.call_args_list],
            ["lookup_field", "structured_query"],
        )
        decide.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
