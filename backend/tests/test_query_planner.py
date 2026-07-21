from __future__ import annotations

import json
import unittest

import pandas as pd

from rag.query_planner import (
    QueryPlannerError,
    generate_query_plan,
    generate_validated_query_plan,
    parse_query_plan_response,
)


class FakeLLM:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("준비된 LLM 응답이 없습니다.")
        return self.responses.pop(0)


def _ready_payload(**overrides) -> dict:
    payload = {
        "status": "ready",
        "dataframe": "df0",
        "operation": "list",
        "select": ["항목명"],
    }
    payload.update(overrides)
    return payload


class QueryPlannerParsingTest(unittest.TestCase):
    def test_parses_plain_json(self):
        plan = parse_query_plan_response(
            json.dumps(_ready_payload(), ensure_ascii=False)
        )

        self.assertEqual(plan.status, "ready")
        self.assertEqual(plan.operation, "list")

    def test_extracts_json_from_code_fence_or_surrounding_text(self):
        for response in (
            "```json\n" + json.dumps(_ready_payload(), ensure_ascii=False) + "\n```",
            "계획입니다.\n" + json.dumps(_ready_payload(), ensure_ascii=False) + "\n끝",
        ):
            with self.subTest(response=response[:10]):
                plan = parse_query_plan_response(response)
                self.assertEqual(plan.dataframe, "df0")

    def test_python_or_unknown_operation_is_not_accepted(self):
        with self.assertRaises(ValueError):
            parse_query_plan_response("result = df0.iloc[0]")

        with self.assertRaises(ValueError):
            parse_query_plan_response(
                json.dumps(
                    _ready_payload(
                        operation="execute_python",
                        python="result = df0.iloc[0]",
                    ),
                    ensure_ascii=False,
                )
            )

    def test_normalizes_single_collection_items_without_changing_meaning(self):
        plan = parse_query_plan_response(
            json.dumps(
                _ready_payload(
                    select="이름",
                    sort={"column": "출연금액", "direction": "desc"},
                    limit=5,
                ),
                ensure_ascii=False,
            )
        )

        self.assertEqual(plan.select, ("이름",))
        self.assertEqual(len(plan.sort), 1)
        self.assertEqual(plan.sort[0].column, "출연금액")
        self.assertEqual(plan.limit, 5)


class QueryPlannerAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_lookup_field_completes_one_exact_schema_column(self):
        llm = FakeLLM(
            json.dumps(
                _ready_payload(
                    filters=[
                        {
                            "column": "성명",
                            "operator": "eq",
                            "value": "백서연",
                            "source_text": "백서연",
                        }
                    ],
                    select=[],
                ),
                ensure_ascii=False,
            )
        )

        plan = await generate_query_plan(
            "백서연 학과 알려줘",
            schema='데이터프레임: df0\n컬럼: "성명", "학과", "학년"',
            llm=llm,
            operation_hint="lookup_field",
        )

        self.assertEqual(plan.select, ("학과",))
        self.assertEqual(len(llm.prompts), 1)

    async def test_lookup_field_corrects_wrong_selected_column(self):
        llm = FakeLLM(
            json.dumps(
                _ready_payload(
                    filters=[
                        {
                            "column": "성명",
                            "operator": "eq",
                            "value": "백서연",
                            "source_text": "백서연",
                        }
                    ],
                    select=["학년"],
                ),
                ensure_ascii=False,
            )
        )

        plan = await generate_query_plan(
            "백서연 학과 알려줘",
            schema='데이터프레임: df0\n컬럼: "성명", "학과", "학년"',
            llm=llm,
            operation_hint="lookup_field",
        )

        self.assertEqual(plan.select, ("학과",))

    async def test_lookup_field_hint_is_preserved_and_validated(self):
        invalid = json.dumps(
            _ready_payload(operation="count", select=[]),
            ensure_ascii=False,
        )
        repaired = json.dumps(
            _ready_payload(
                filters=[
                    {
                        "column": "성명",
                        "operator": "eq",
                        "value": "정유진",
                        "source_text": "정유진",
                    }
                ],
                select=["성명", "학년"],
            ),
            ensure_ascii=False,
        )
        llm = FakeLLM(invalid, repaired)

        plan = await generate_query_plan(
            "정유진 몇학년",
            schema='데이터프레임: df0\n컬럼: "성명", "학년"',
            llm=llm,
            operation_hint="lookup_field",
        )

        self.assertEqual(plan.operation, "list")
        self.assertEqual(plan.select, ("성명", "학년"))
        self.assertEqual(plan.filters[0].value, "정유진")
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("lookup_field", llm.prompts[0])
        self.assertIn("lookup_field", llm.prompts[1])

    async def test_ranked_list_drops_filter_not_requested_by_user(self):
        llm = FakeLLM(
            json.dumps(
                _ready_payload(
                    filters=[
                        {"column": "출연금액", "operator": "gt", "value": 1_000_000}
                    ],
                    sort={"column": "출연금액", "direction": "desc"},
                    limit=5,
                ),
                ensure_ascii=False,
            )
        )

        plan = await generate_query_plan(
            "출연금액이 큰 순서대로 5개 알려줘",
            schema='데이터프레임: df0\n컬럼: "출연금액"',
            llm=llm,
        )

        self.assertEqual(plan.filters, ())
        self.assertEqual(plan.limit, 5)

    async def test_ranked_list_keeps_explicit_user_filter(self):
        llm = FakeLLM(
            json.dumps(
                _ready_payload(
                    filters=[
                        {"column": "출연금액", "operator": "gte", "value": 1_000_000}
                    ],
                    sort={"column": "출연금액", "direction": "desc"},
                    limit=5,
                ),
                ensure_ascii=False,
            )
        )

        plan = await generate_query_plan(
            "100만원 이상 중 출연금액이 큰 순서대로 5개 알려줘",
            schema='데이터프레임: df0\n컬럼: "출연금액"',
            llm=llm,
        )

        self.assertEqual(len(plan.filters), 1)

    async def test_multiple_filters_default_to_and_without_explicit_or(self):
        llm = FakeLLM(
            json.dumps(
                _ready_payload(
                    filters=[
                        {"column": "기수", "operator": "gte", "value": 50},
                        {"column": "출연금액", "operator": "gte", "value": 1_000_000},
                    ],
                    filter_logic="any",
                ),
                ensure_ascii=False,
            )
        )

        plan = await generate_query_plan(
            "기수가 50 이상이고 출연금액이 100만원 이상인 항목",
            schema='데이터프레임: df0\n컬럼: "기수", "출연금액"',
            llm=llm,
        )

        self.assertEqual(plan.filter_logic, "all")

    async def test_numeric_filter_is_corrected_only_from_exact_source_text(self):
        llm = FakeLLM(
            json.dumps(
                _ready_payload(
                    filters=[
                        {
                            "column": "기수",
                            "operator": "gte",
                            "value": 49,
                            "source_text": "49기 이상",
                        },
                        {
                            "column": "출연금액",
                            "operator": "gt",
                            "value": 200_000,
                            "source_text": "200만원 이상",
                        },
                    ],
                ),
                ensure_ascii=False,
            )
        )

        plan = await generate_query_plan(
            "전체 중 49기 이상에서 200만원 이상 낸 사람 알려줘",
            schema='데이터프레임: df0\n컬럼: "기수", "출연금액"',
            llm=llm,
        )

        self.assertEqual(plan.filters[0].operator, "gte")
        self.assertEqual(plan.filters[0].value, 49)
        self.assertEqual(plan.filters[1].operator, "gte")
        self.assertEqual(plan.filters[1].value, "200만원")
        self.assertEqual(plan.filters[1].source_text, "200만원 이상")

    async def test_filter_is_not_corrected_from_text_absent_in_question(self):
        llm = FakeLLM(
            json.dumps(
                _ready_payload(
                    filters=[
                        {
                            "column": "출연금액",
                            "operator": "gt",
                            "value": 200_000,
                            "source_text": "20만원 초과",
                        }
                    ],
                ),
                ensure_ascii=False,
            )
        )

        plan = await generate_query_plan(
            "200만원 이상 낸 사람 알려줘",
            schema='데이터프레임: df0\n컬럼: "출연금액"',
            llm=llm,
        )

        self.assertEqual(plan.filters[0].operator, "gt")
        self.assertEqual(plan.filters[0].value, 200_000)

    async def test_explicit_or_preserves_any_filter_logic(self):
        llm = FakeLLM(
            json.dumps(
                _ready_payload(
                    filters=[
                        {"column": "기수", "operator": "gte", "value": 50},
                        {"column": "출연금액", "operator": "gte", "value": 1_000_000},
                    ],
                    filter_logic="all",
                ),
                ensure_ascii=False,
            )
        )

        plan = await generate_query_plan(
            "기수가 50 이상이거나 출연금액이 100만원 이상인 항목",
            schema='데이터프레임: df0\n컬럼: "기수", "출연금액"',
            llm=llm,
        )

        self.assertEqual(plan.filter_logic, "any")

    async def test_prompt_contains_question_and_runtime_schema(self):
        llm = FakeLLM(json.dumps(_ready_payload(), ensure_ascii=False))

        await generate_query_plan(
            "상태가 완료인 항목",
            schema='데이터프레임: df0\n컬럼: "항목명", "상태"',
            llm=llm,
        )

        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("상태가 완료인 항목", llm.prompts[0])
        self.assertIn('"항목명", "상태"', llm.prompts[0])
        self.assertIn("Python 코드", llm.prompts[0])
        self.assertIn("오직 개수나 몇 개인지를 요청할 때만 count", llm.prompts[0])
        self.assertIn("특정 대상의 특정 컬럼값", llm.prompts[0])
        self.assertIn("필요한 필드만 추가", llm.prompts[0])
        self.assertIn("contains는 문자열 컬럼에만", llm.prompts[0])
        self.assertIn("직접 환산하거나 자릿수를 바꾸지 말고", llm.prompts[0])
        self.assertIn("모든 필터의 source_text", llm.prompts[0])
        self.assertIn("가장 짧은 원문 구절", llm.prompts[0])
        self.assertIn('"이상"은 gte', llm.prompts[0])
        self.assertIn("임의로 연도를 만들지 말고 clarification", llm.prompts[0])
        self.assertNotIn('"confidence"', llm.prompts[0])

    async def test_invalid_first_response_is_repaired_once(self):
        repaired = json.dumps(_ready_payload(), ensure_ascii=False)
        llm = FakeLLM('{"status": "ready"', repaired)

        plan = await generate_query_plan(
            "항목 목록",
            schema='데이터프레임: df0\n컬럼: "항목명"',
            llm=llm,
        )

        self.assertEqual(plan.operation, "list")
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("조건을 추가·삭제·완화하지 말고", llm.prompts[1])
        self.assertIn("응답에서 완전한 JSON 객체", llm.prompts[1])

    async def test_schema_validation_error_is_repaired_once(self):
        invalid = json.dumps(_ready_payload(operation="run"), ensure_ascii=False)
        repaired = json.dumps(_ready_payload(operation="count", select=[]), ensure_ascii=False)
        llm = FakeLLM(invalid, repaired)

        plan = await generate_query_plan(
            "항목 개수",
            schema='데이터프레임: df0\n컬럼: "항목명"',
            llm=llm,
        )

        self.assertEqual(plan.operation, "count")
        self.assertEqual(len(llm.prompts), 2)

    async def test_repair_prompt_does_not_repeat_large_dataframe_schema(self):
        repaired = json.dumps(_ready_payload(), ensure_ascii=False)
        llm = FakeLLM('{"status": "ready"', repaired)
        large_schema = "SCHEMA_SENTINEL\n" + ("긴스키마" * 3000)

        await generate_query_plan(
            "항목 목록",
            schema=large_schema,
            llm=llm,
        )

        self.assertIn("SCHEMA_SENTINEL", llm.prompts[0])
        self.assertNotIn("SCHEMA_SENTINEL", llm.prompts[1])
        self.assertLess(len(llm.prompts[1]), 5000)

    async def test_two_invalid_responses_fail_closed(self):
        llm = FakeLLM("JSON 아님", "여전히 JSON 아님")

        with self.assertRaises(QueryPlannerError) as caught:
            await generate_query_plan(
                "항목 목록",
                schema='데이터프레임: df0\n컬럼: "항목명"',
                llm=llm,
            )

        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(len(caught.exception.responses), 2)

    async def test_clarification_and_not_applicable_are_preserved(self):
        for status in ("clarification", "not_applicable"):
            with self.subTest(status=status):
                llm = FakeLLM(
                    json.dumps(
                        {
                            "status": status,
                            "message": "조회 대상을 확인할 수 없습니다.",
                        },
                        ensure_ascii=False,
                    )
                )
                plan = await generate_query_plan(
                    "질문",
                    schema="스키마",
                    llm=llm,
                )
                self.assertEqual(plan.status, status)

    async def test_generated_plan_is_immediately_checked_against_dataframe(self):
        df = pd.DataFrame({"항목명": ["A"], "상태": ["완료"]})
        schema = '데이터프레임: df0\n컬럼: "항목명", "상태"'
        llm = FakeLLM(
            json.dumps(
                _ready_payload(select=["존재하지않는컬럼"]),
                ensure_ascii=False,
            )
        )

        result = await generate_validated_query_plan(
            "항목 목록",
            schema=schema,
            llm=llm,
            dataframes={"df0": df},
            source_by_alias={"df0": "목록.xlsx"},
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(result.issues[0].code, "unknown_column")

    async def test_empty_question_fails_without_calling_llm(self):
        llm = FakeLLM(json.dumps(_ready_payload(), ensure_ascii=False))

        with self.assertRaises(QueryPlannerError):
            await generate_query_plan("  ", schema="스키마", llm=llm)

        self.assertEqual(llm.prompts, [])


if __name__ == "__main__":
    unittest.main()
