from __future__ import annotations

import json
import unittest

from rag.question_engine import (
    QuestionEngineError,
    compact_question_schema,
    compare_shadow_decision,
    decide_question,
    parse_question_decision,
)


class FakeLLM:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("준비된 LLM 응답이 없습니다.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _ready_payload(operation: str = "structured_query") -> dict:
    payload = {
        "status": "ready",
        "operations": [operation],
        "reason": "질문의 전체 의미에 따른 operation",
    }
    if operation.startswith("document_"):
        payload["retrieval_query"] = "장학금 지급 기준"
    return payload


class QuestionEngineParsingTest(unittest.TestCase):
    def test_requests_are_parsed_and_operations_are_derived(self):
        decision = parse_question_decision(
            json.dumps(
                {
                    "status": "ready",
                    "requests": [
                        {
                            "source_text": "장학금 규정",
                            "operation": "document_explain",
                        },
                        {
                            "source_text": "전체 목록",
                            "operation": "list_records",
                        },
                    ],
                    "reason": "규정과 전체 목록의 복합 요청",
                    "retrieval_query": "장학금 규정",
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(len(decision.requests), 2)
        self.assertEqual(
            decision.operations,
            ("document_explain", "list_records"),
        )

    def test_compact_schema_keeps_headers_and_drops_samples(self):
        compact = compact_question_schema(
            "파일: 명단.xlsx\n"
            "  데이터프레임: df0\n"
            '  컬럼(이 이름만 사용): "이름", "취득학점"\n'
            "  예시(값): 이름=[민감정보], 취득학점=144\n"
            '  검증된 컬럼 의미: "취득학점"=measure/value'
        )
        self.assertIn("파일: 명단.xlsx", compact)
        self.assertIn('"취득학점"', compact)
        self.assertNotIn("예시(값)", compact)
        self.assertNotIn("144", compact)

    def test_normalizes_singular_operation_and_discards_obsolete_fields(self):
        decision = parse_question_decision(
            json.dumps(
                {
                    "status": "ready",
                    "operation": "structured_query",
                    "reason": "복수 조건 표 조회",
                    "route": "PANDAS",
                    "intent": "structured_query",
                    "query": "실행하면 안 되는 조회식",
                },
                ensure_ascii=False,
            )
        )

        self.assertEqual(decision.operations, ("structured_query",))
        self.assertIsNone(decision.retrieval_query)

    def test_parses_json_inside_surrounding_text(self):
        payload = json.dumps(_ready_payload(), ensure_ascii=False)
        decision = parse_question_decision(f"결정:\n{payload}\n끝")
        self.assertEqual(decision.operations, ("structured_query",))

    def test_normalizes_one_candidate_string(self):
        decision = parse_question_decision(
            json.dumps(
                {
                    "status": "clarification",
                    "reason": "대상 부족",
                    "message": "항목을 지정해 주세요.",
                    "candidates": "출연금액",
                },
                ensure_ascii=False,
            )
        )
        self.assertEqual(decision.candidates, ("출연금액",))

    def test_original_question_is_safe_document_query_fallback(self):
        decision = parse_question_decision(
            json.dumps(
                {
                    "status": "ready",
                    "operations": ["sum_amount", "document_criteria"],
                    "reason": "계산과 기준 검색의 혼합 요청",
                },
                ensure_ascii=False,
            ),
            fallback_retrieval_query="총액과 지급 기준 알려줘",
        )
        self.assertEqual(
            decision.retrieval_query,
            "총액과 지급 기준 알려줘",
        )


class QuestionEngineAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def test_table_list_phrases_cannot_become_document_inventory(self):
        for question in (
            "전체목록",
            "전체 리스트",
            "표의 전체 목록 보여줘",
        ):
            with self.subTest(question=question):
                payload = {
                    "status": "ready",
                    "requests": [
                        {
                            "source_text": question,
                            "operation": "list_documents",
                        }
                    ],
                    "reason": "LLM의 잘못된 문서 목록 분류",
                }
                decision = await decide_question(
                    question,
                    schema="표 스키마",
                    llm=FakeLLM(json.dumps(payload, ensure_ascii=False)),
                )

                self.assertEqual(decision.operations, ("list_records",))
                self.assertEqual(decision.request_count, 1)

    async def test_duplicate_list_interpretations_collapse_to_table_records(self):
        payload = {
            "status": "ready",
            "requests": [
                {
                    "source_text": "전체목록",
                    "operation": "list_documents",
                },
                {
                    "source_text": "전체목록",
                    "operation": "list_records",
                },
            ],
            "reason": "두 목록 의미를 중복 생성",
        }
        decision = await decide_question(
            "전체목록",
            schema="표 스키마",
            llm=FakeLLM(json.dumps(payload, ensure_ascii=False)),
        )

        self.assertEqual(decision.operations, ("list_records",))
        self.assertEqual(decision.request_count, 1)

    async def test_explicit_file_inventory_remains_list_documents(self):
        payload = {
            "status": "ready",
            "requests": [
                {
                    "source_text": "현재 적재된 파일 목록",
                    "operation": "list_documents",
                }
            ],
            "reason": "명시적인 파일 보관 목록",
        }
        decision = await decide_question(
            "현재 적재된 파일 목록",
            schema="표 스키마",
            llm=FakeLLM(json.dumps(payload, ensure_ascii=False)),
        )

        self.assertEqual(decision.operations, ("list_documents",))

    async def test_prompt_contains_question_schema_and_operation_meanings(self):
        llm = FakeLLM(json.dumps(_ready_payload(), ensure_ascii=False))
        decision = await decide_question(
            "학점이 144점인 사람",
            schema='파일: 명단.xlsx\n컬럼: "이름", "취득학점"',
            llm=llm,
        )

        self.assertEqual(decision.operations, ("structured_query",))
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("학점이 144점인 사람", llm.prompts[0])
        self.assertIn('"취득학점"', llm.prompts[0])
        self.assertIn("structured_query", llm.prompts[0])
        self.assertIn("lookup_field", llm.prompts[0])
        self.assertIn("document_criteria", llm.prompts[0])
        self.assertIn('"requests"', llm.prompts[0])
        self.assertIn("source_text", llm.prompts[0])
        self.assertNotIn('"route":', llm.prompts[0])

    async def test_lookup_field_decision_uses_query_plan_strategy(self):
        payload = {
            "status": "ready",
            "requests": [
                {
                    "source_text": "홍길동 취득학점",
                    "operation": "lookup_field",
                }
            ],
            "reason": "특정 대상의 일반 컬럼값 조회",
        }
        decision = await decide_question(
            "홍길동 취득학점",
            schema='컬럼: "이름", "취득학점"',
            llm=FakeLLM(json.dumps(payload, ensure_ascii=False)),
        )

        self.assertEqual(decision.operations, ("lookup_field",))

    async def test_request_source_text_must_exist_in_original_question(self):
        invalid = {
            "status": "ready",
            "requests": [
                {
                    "source_text": "질문에 없는 요청",
                    "operation": "list_records",
                }
            ],
            "reason": "잘못된 원문 근거",
        }
        repaired = {
            "status": "ready",
            "requests": [
                {
                    "source_text": "전체 목록",
                    "operation": "list_records",
                }
            ],
            "reason": "전체 목록 요청",
        }
        llm = FakeLLM(
            json.dumps(invalid, ensure_ascii=False),
            json.dumps(repaired, ensure_ascii=False),
        )

        decision = await decide_question(
            "전체 목록 알려줘",
            schema="표 스키마",
            llm=llm,
        )

        self.assertEqual(decision.operations, ("list_records",))
        self.assertEqual(decision.requests[0].source_text, "전체 목록")
        self.assertEqual(len(llm.prompts), 2)

    async def test_invalid_first_response_is_repaired_once(self):
        repaired = json.dumps(
            _ready_payload("document_criteria"),
            ensure_ascii=False,
        )
        llm = FakeLLM('{"status": "ready"', repaired)

        decision = await decide_question(
            "지급 기준을 설명해줘",
            schema="표 없음",
            llm=llm,
        )

        self.assertEqual(decision.operations, ("document_criteria",))
        self.assertEqual(len(llm.prompts), 2)

    async def test_two_invalid_responses_fail_closed(self):
        llm = FakeLLM("JSON 아님", "여전히 JSON 아님")

        with self.assertRaises(QuestionEngineError):
            await decide_question("질문", schema="표 없음", llm=llm)

    async def test_shadow_comparison_reports_query_plan_strategy(self):
        llm = FakeLLM(json.dumps(_ready_payload(), ensure_ascii=False))
        comparison = await compare_shadow_decision(
            "학점이 144점인 사람",
            "PANDAS",
            ["lookup_amount"],
            '컬럼: "취득학점"',
            llm=llm,
        )

        self.assertIsNotNone(comparison)
        self.assertTrue(comparison.engine_matched)
        self.assertFalse(comparison.operation_matched)
        self.assertEqual(comparison.llm_route, "PANDAS")
        self.assertEqual(comparison.llm_strategy, "QUERY_PLAN")

    async def test_mixed_engines_are_shadowed_as_guide(self):
        payload = {
            "status": "ready",
            "operations": ["sum_amount", "document_criteria"],
            "reason": "계산과 기준 검색의 혼합 요청",
            "retrieval_query": "지급 기준",
        }
        llm = FakeLLM(json.dumps(payload, ensure_ascii=False))
        comparison = await compare_shadow_decision(
            "총액과 지급 기준 알려줘",
            "GUIDE",
            ["sum_amount", "document_criteria"],
            "표 스키마",
            llm=llm,
        )

        self.assertIsNotNone(comparison)
        self.assertEqual(comparison.llm_route, "GUIDE")
        self.assertTrue(comparison.engine_matched)
        self.assertTrue(comparison.operation_matched)

    async def test_shadow_failure_is_swallowed(self):
        llm = FakeLLM(RuntimeError("모델 중단"))
        comparison = await compare_shadow_decision(
            "질문",
            "VECTOR",
            ["document_explain"],
            "표 없음",
            llm=llm,
        )
        self.assertIsNone(comparison)


if __name__ == "__main__":
    unittest.main()
