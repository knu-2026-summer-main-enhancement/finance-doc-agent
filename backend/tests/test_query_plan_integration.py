from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd

from datastore.state import _df_namespace, _df_sources
from pandas_engine.plan_validator import validate_query_plan
from pandas_engine.query_plan import QueryPlan
from rag.pandas_rag import _answer_pandas
from rag.query_planner import QueryPlannerError
from rag.question_analyzer import QuestionAnalysis, analyze_question


class QueryPlanPandasIntegrationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _df_namespace.clear()
        _df_sources.clear()
        self.df = pd.DataFrame(
            {
                "항목": ["A", "B", "C"],
                "상태": ["완료", "대기", "완료"],
                "점수": [10, 20, 30],
            }
        )
        self.df.attrs["semantic_schema"] = {
            "columns": {
                "항목": {"data_type": "string"},
                "상태": {"data_type": "string"},
                "점수": {"data_type": "number"},
            }
        }
        _df_namespace["df0"] = self.df
        _df_sources["df0"] = "업무목록.xlsx"
        self.analysis = QuestionAnalysis(question="복합 조건 조회")

    def tearDown(self):
        _df_namespace.clear()
        _df_sources.clear()

    def _validation(self, payload: dict):
        plan = QueryPlan.model_validate(payload)
        return validate_query_plan(plan)

    def _direct_misses(self):
        return patch.multiple(
            "rag.pandas_rag",
            _search_name_pandas=lambda question: (None, [], False),
            _query_pandas_direct=lambda *args, **kwargs: (None, []),
            _has_explicit_structured_filter=lambda question: False,
        )

    async def test_ready_plan_executes_and_formats_evidence(self):
        validation = self._validation(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "filters": [
                    {"column": "상태", "operator": "eq", "value": "완료"}
                ],
                "select": ["항목", "상태"],
            }
        )

        with self._direct_misses(), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=AsyncMock(return_value=validation),
        ):
            answer, sources, route = await _answer_pandas(
                "복합 조건 조회",
                analysis=self.analysis,
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, ["업무목록.xlsx"])
        self.assertIn("총 2건", answer)
        self.assertIn("조회 근거:", answer)
        self.assertIn("상태 = 완료", answer)
        self.assertIn("원본 3개", answer)

    async def test_lookup_field_hint_reaches_query_planner(self):
        validation = self._validation(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "filters": [
                    {"column": "항목", "operator": "eq", "value": "A"}
                ],
                "select": ["항목", "점수"],
            }
        )
        planner = AsyncMock(return_value=validation)

        with self._direct_misses(), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=planner,
        ):
            answer, _, _ = await _answer_pandas(
                "A 점수",
                strategy="QUERY_PLAN",
                operation_hint="lookup_field",
            )

        planner.assert_awaited_once_with(
            "A 점수",
            operation_hint="lookup_field",
        )
        self.assertIn("10", answer)

    async def test_query_plan_strategy_executes_cross_year_date_range_directly(self):
        date_df = pd.DataFrame([
            {"년": 2025, "월": 5, "결제등록날짜": None, "이름": "김하나"},
            {"년": 2025, "월": 6, "결제등록날짜": "2021-06-01", "이름": "이두리"},
            {"년": 2025, "월": 12, "결제등록날짜": None, "이름": "박세나"},
            {"년": 2026, "월": 1, "결제등록날짜": None, "이름": "최도윤"},
            {"년": 2026, "월": 2, "결제등록날짜": None, "이름": "한지우"},
        ])
        _df_namespace["df0"] = date_df
        _df_sources["df0"] = "결제내역.xlsx"
        question = "2025년 6월부터 2026년 1월까지 목록"
        planner = AsyncMock(side_effect=AssertionError("date range must not use P.JSON"))

        with patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=planner,
        ):
            answer, sources, route = await _answer_pandas(
                question,
                analysis=analyze_question(question),
                strategy="QUERY_PLAN",
                operation_hint="structured_query",
            )

        planner.assert_not_awaited()
        self.assertEqual(route, "pandas")
        self.assertEqual(sources, ["결제내역.xlsx"])
        self.assertIn("총 3건", answer)
        self.assertIn("2025년 6월~2026년 1월", answer)

    async def test_numeric_comparison_skips_masked_name_guessing(self):
        validation = self._validation(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "filters": [
                    {"column": "점수", "operator": "gte", "value": 20}
                ],
                "select": ["항목", "점수"],
            }
        )
        name_search = Mock(
            side_effect=AssertionError("숫자 비교 질문에서 이름 유사 검색이 호출됨")
        )

        with patch(
            "rag.pandas_rag._search_name_pandas",
            new=name_search,
        ), patch(
            "rag.pandas_rag._query_pandas_direct",
            new=lambda *args, **kwargs: (None, []),
        ), patch(
            "rag.pandas_rag._has_explicit_structured_filter",
            new=lambda question: False,
        ), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=AsyncMock(return_value=validation),
        ):
            answer, sources, route = await _answer_pandas(
                "점수가 20점 이상인 항목",
                analysis=self.analysis,
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, ["업무목록.xlsx"])
        self.assertIn("총 2건", answer)
        name_search.assert_not_called()

    async def test_query_plan_strategy_skips_every_direct_handler(self):
        validation = self._validation(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "filters": [
                    {"column": "점수", "operator": "gte", "value": 20}
                ],
                "select": ["항목", "점수"],
            }
        )

        with patch(
            "rag.pandas_rag._search_name_pandas",
            side_effect=AssertionError("이름 직접 검색이 호출되면 안 됩니다."),
        ), patch(
            "rag.pandas_rag._query_pandas_direct",
            side_effect=AssertionError("직접 조회가 호출되면 안 됩니다."),
        ), patch(
            "rag.pandas_rag._query_all_records",
            side_effect=AssertionError("전체 목록 조회가 호출되면 안 됩니다."),
        ), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=AsyncMock(return_value=validation),
        ):
            answer, sources, route = await _answer_pandas(
                "점수가 20 이상인 항목",
                strategy="QUERY_PLAN",
                allow_vector_fallback=False,
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, ["업무목록.xlsx"])
        self.assertIn("총 2건", answer)

    async def test_masked_name_lookup_recovers_from_query_plan_misclassification(self):
        matched = pd.DataFrame(
            {
                "이름": ["이*규", "이*규"],
                "출연금액": ["1,000,000", "9,000,000"],
                "_매칭유형": ["masked_direct_match", "masked_direct_match"],
            }
        )
        planner = AsyncMock(
            side_effect=AssertionError("검증된 마스킹 이름 조회가 QueryPlan으로 넘어감")
        )

        with patch(
            "rag.pandas_rag._search_name_pandas",
            return_value=(matched, ["test.png"], True),
        ), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=planner,
        ):
            answer, sources, route = await _answer_pandas(
                "이*규 얼마야",
                strategy="QUERY_PLAN",
                allow_vector_fallback=False,
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, ["test.png"])
        self.assertIn("10,000,000원", answer)
        self.assertIn("총 2회", answer)
        planner.assert_not_awaited()

    async def test_masked_name_with_numeric_condition_still_uses_query_plan(self):
        validation = self._validation(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "filters": [
                    {"column": "점수", "operator": "gte", "value": 20}
                ],
                "select": ["항목", "점수"],
            }
        )
        name_search = Mock(
            side_effect=AssertionError("복합 조건에서 이름 직접 검색이 호출됨")
        )

        with patch(
            "rag.pandas_rag._search_name_pandas",
            new=name_search,
        ), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=AsyncMock(return_value=validation),
        ):
            answer, sources, route = await _answer_pandas(
                "이*규 점수가 20점 이상인 항목",
                strategy="QUERY_PLAN",
                allow_vector_fallback=False,
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, ["업무목록.xlsx"])
        self.assertIn("총 2건", answer)
        name_search.assert_not_called()

    async def test_valid_empty_result_does_not_fall_back_to_vector(self):
        validation = self._validation(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "filters": [
                    {"column": "상태", "operator": "eq", "value": "없음"}
                ],
                "select": ["항목"],
            }
        )
        vector = AsyncMock(return_value=("잘못된 폴백", [], "vector"))

        with self._direct_misses(), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=AsyncMock(return_value=validation),
        ), patch("rag.vector._answer_vector", new=vector):
            answer, sources, route = await _answer_pandas(
                "복합 조건 조회",
                analysis=self.analysis,
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, ["업무목록.xlsx"])
        self.assertIn("조회된 데이터가 없습니다.", answer)
        vector.assert_not_awaited()

    async def test_invalid_plan_fails_closed_without_vector(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "select": ["없는컬럼"],
            }
        )
        validation = validate_query_plan(plan)
        vector = AsyncMock(return_value=("잘못된 폴백", [], "vector"))

        with self._direct_misses(), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=AsyncMock(return_value=validation),
        ), patch("rag.vector._answer_vector", new=vector):
            answer, sources, route = await _answer_pandas(
                "복합 조건 조회",
                analysis=self.analysis,
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, [])
        self.assertIn("안전하게 연결하지 못했습니다", answer)
        vector.assert_not_awaited()

    async def test_clarification_returns_guide_message_without_execution(self):
        validation = self._validation(
            {
                "status": "clarification",
                "message": "조회할 상태를 선택해 주세요.",
                "candidates": ["완료", "대기"],
            }
        )

        with self._direct_misses(), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=AsyncMock(return_value=validation),
        ):
            answer, sources, route = await _answer_pandas(
                "복합 조건 조회",
                analysis=self.analysis,
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, [])
        self.assertIn("조회할 상태를 선택", answer)
        self.assertIn("완료, 대기", answer)

    async def test_not_applicable_is_the_only_plan_status_using_vector(self):
        validation = self._validation(
            {
                "status": "not_applicable",
                "message": "문서 본문의 설명 검색이 필요합니다.",
            }
        )
        vector = AsyncMock(return_value=("문서 근거 답변", ["업무목록.xlsx"], "vector"))

        with self._direct_misses(), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=AsyncMock(return_value=validation),
        ), patch("rag.vector._answer_vector", new=vector):
            answer, sources, route = await _answer_pandas(
                "복합 조건 조회",
                analysis=self.analysis,
            )

        self.assertEqual((answer, sources, route), (
            "문서 근거 답변",
            ["업무목록.xlsx"],
            "vector",
        ))
        vector.assert_awaited_once()

    async def test_planner_failure_returns_safe_message(self):
        planner = AsyncMock(side_effect=QueryPlannerError("잘못된 JSON"))

        with self._direct_misses(), patch(
            "rag.pandas_rag.generate_validated_query_plan",
            new=planner,
        ):
            answer, sources, route = await _answer_pandas(
                "복합 조건 조회",
                analysis=self.analysis,
            )

        self.assertEqual(route, "pandas")
        self.assertEqual(sources, [])
        self.assertIn("안전한 표 조회 계획으로 변환하지 못했습니다", answer)

class QueryPlanSemanticContractTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "회원명": ["가나다", "가나다", "라마바"],
                "전화번호": ["010-1111-1111", "010-1111-1111", "010-2222-2222"],
                "회비구분": ["년회비", "년회비", "평생회비"],
            }
        )
        self.df.attrs["semantic_schema"] = {
            "columns": {
                "회원명": {
                    "concept": "entity", "role": "entity_name", "qualifier": "person",
                    "data_type": "string",
                },
                "전화번호": {"data_type": "string"},
                "회비구분": {"data_type": "string"},
            }
        }

    def test_person_count_is_normalized_to_distinct_person_identifier(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready", "dataframe": "df0", "operation": "count",
                "filters": [{"column": "회비구분", "operator": "eq", "value": "년회비"}],
            }
        )
        validation = validate_query_plan(
            plan,
            question="회비구분이 년회비인 사람 몇 명이야?",
            dataframes={"df0": self.df},
            source_by_alias={"df0": "test.xlsx"},
        )
        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.plan.distinct_by, ("회원명",))

    def test_lookup_field_rejects_requested_field_as_subject_filter(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready", "dataframe": "df0", "operation": "list",
                "filters": [{"column": "전화번호", "operator": "eq", "value": "가나다"}],
                "select": ["전화번호"],
            }
        )
        validation = validate_query_plan(
            plan,
            question="가나다 전화번호 뭐야?",
            dataframes={"df0": self.df},
            source_by_alias={"df0": "test.xlsx"},
            operation_hint="lookup_field",
        )
        self.assertEqual(validation.status, "invalid")
        self.assertIn(
            "return_column_used_as_filter",
            {issue.code for issue in validation.issues},
        )


if __name__ == "__main__":
    unittest.main()
