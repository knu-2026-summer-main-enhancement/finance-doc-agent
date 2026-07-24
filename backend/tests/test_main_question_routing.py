from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from pandas_engine.query_plan import QueryPlan
from rag.question_decision import QuestionDecision

import main


class MainQuestionPreRoutingTest(unittest.IsolatedAsyncioTestCase):
    async def test_dp_auto_failure_delegates_to_llm_question_analysis(self):
        llm_decision = QuestionDecision.model_validate(
            {
                "status": "ready",
                "requests": [{"source_text": "복합 조건으로 조회해줘", "operation": "structured_query"}],
                "reason": "복합 조건 해석이 필요합니다.",
            }
        )
        decide = AsyncMock(return_value=llm_decision)
        with (
            patch.object(main, "scoped_mapping", return_value={}),
            patch.object(
                main,
                "build_auto_schema_grounded_plan",
                return_value=(None, None),
            ) as auto,
            patch.object(main, "decide_question", decide),
        ):
            resolution = await main._resolve_llm_question("복합 조건으로 조회해줘")

        self.assertEqual(resolution.route, "PANDAS")
        self.assertEqual(tuple(resolution.guard_result.operations), ("structured_query",))
        self.assertIsNone(resolution.plan)
        auto.assert_called_once()
        decide.assert_awaited_once()

    async def test_unknown_person_field_request_stops_before_dp_or_llm(self):
        decide = AsyncMock()
        with (
            patch.object(main, "scoped_mapping", return_value={}),
            patch.object(main, "ambiguous_person_lookup_candidates", return_value=()),
            patch.object(main, "has_unmatched_person_field_reference", return_value=True),
            patch.object(
                main,
                "build_auto_schema_grounded_plan",
                side_effect=AssertionError("D-P must not run after missing person"),
            ),
            patch.object(main, "decide_question", decide),
        ):
            resolution = await main._resolve_llm_question("없는 사람 전공 알려줘")

        self.assertEqual(resolution.route, "PANDAS")
        self.assertIsNotNone(resolution.answer)
        decide.assert_not_awaited()

    async def test_unknown_person_amount_request_stops_before_dp_or_llm(self):
        decide = AsyncMock()
        with (
            patch.object(main, "scoped_mapping", return_value={}),
            patch.object(main, "ambiguous_person_lookup_candidates", return_value=()),
            patch.object(main, "has_unmatched_person_field_reference", return_value=False),
            patch.object(main, "has_unmatched_person_amount_reference", return_value=True),
            patch.object(
                main,
                "build_auto_schema_grounded_plan",
                side_effect=AssertionError("D-P must not run after missing person"),
            ),
            patch.object(main, "decide_question", decide),
        ):
            resolution = await main._resolve_llm_question("없는 사람 얼마야")

        self.assertEqual(resolution.route, "PANDAS")
        self.assertIsNotNone(resolution.answer)
        decide.assert_not_awaited()

    async def test_cross_month_range_is_resolved_by_dp_auto_without_llm(self):
        decide = AsyncMock()
        with (
            patch.object(main, "scoped_mapping", return_value={}),
            patch.object(
                main,
                "build_auto_schema_grounded_plan",
                return_value=("structured_query", None),
            ) as auto,
            patch.object(main, "decide_question", decide),
        ):
            resolution = await main._resolve_llm_question(
                "2025년 6월부터 2026년 1월까지 목록"
            )

        self.assertEqual(resolution.route, "PANDAS")
        self.assertEqual(tuple(resolution.guard_result.operations), ("structured_query",))
        self.assertIsNone(resolution.plan)
        auto.assert_called_once()
        decide.assert_not_awaited()

    async def test_person_precheck_is_resolved_before_dp_or_llm(self):
        decide = AsyncMock()
        with (
            patch.object(main, "scoped_mapping", return_value={}),
            patch.object(
                main,
                "ambiguous_person_lookup_candidates",
                return_value=("김가나", "김가다"),
            ),
            patch.object(
                main,
                "build_auto_schema_grounded_plan",
                side_effect=AssertionError("D-P must not run after ambiguity"),
            ),
            patch.object(main, "decide_question", decide),
        ):
            resolution = await main._resolve_llm_question("김가 전공 알려줘")

        self.assertEqual(resolution.route, "PANDAS")
        self.assertIn("후보: 김가나, 김가다", resolution.answer or "")
        decide.assert_not_awaited()

    async def test_dp_auto_owns_fast_operation_selection(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "sum",
                "target": "결제_금액",
            }
        )
        decide = AsyncMock()
        with (
            patch.object(main, "scoped_mapping", return_value={}),
            patch.object(
                main,
                "build_auto_schema_grounded_plan",
                return_value=("lookup_amount", plan),
            ) as auto,
            patch.object(main, "decide_question", decide),
        ):
            resolution = await main._resolve_llm_question("김나다 얼마야?")

        self.assertEqual(resolution.route, "PANDAS")
        self.assertEqual(tuple(resolution.guard_result.operations), ("lookup_amount",))
        self.assertIs(resolution.plan, plan)
        auto.assert_called_once()
        decide.assert_not_awaited()

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
            patch.object(
                main, "build_auto_schema_grounded_plan",
                return_value=("structured_query", plan),
            ) as build,
            patch.object(main, "decide_question", decide),
        ):
            resolution = await main._resolve_llm_question(
                "전체 기록에서 회원명과 전공만 보여줘"
            )
            guard = resolution.guard_result
            route = resolution.route
            strategy = resolution.pandas_strategy

        self.assertEqual(route, "PANDAS")
        self.assertEqual(strategy, "QUERY_PLAN")
        self.assertEqual(tuple(guard.operations), ("structured_query",))
        self.assertIs(resolution.plan, plan)
        decide.assert_not_awaited()
        self.assertEqual(
            build.call_args.args[0],
            "전체 기록에서 회원명과 전공만 보여줘",
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
            patch.object(
                main, "build_auto_schema_grounded_plan",
                return_value=("structured_query", plan),
            ),
            patch.object(main, "decide_question", decide),
        ):
            resolution = await main._resolve_llm_question(
                "전공별 납부액을 묶어서 보여줘"
            )
            route = resolution.route
            strategy = resolution.pandas_strategy

        self.assertEqual(route, "PANDAS")
        self.assertEqual(strategy, "QUERY_PLAN")
        self.assertIs(resolution.plan, plan)
        decide.assert_not_awaited()

    async def test_dp_auto_returns_general_plan_after_narrow_interpretation(self):
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
            patch.object(
                main, "build_auto_schema_grounded_plan",
                return_value=("structured_query", plan),
            ) as auto,
            patch.object(main, "decide_question", decide),
        ):
            resolution = await main._resolve_llm_question(
                "전공별 납부액을 묶어서 보여줘"
            )
            guard = resolution.guard_result
            route = resolution.route
            strategy = resolution.pandas_strategy

        self.assertEqual(route, "PANDAS")
        self.assertEqual(strategy, "QUERY_PLAN")
        self.assertEqual(tuple(guard.operations), ("structured_query",))
        auto.assert_called_once()
        decide.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
