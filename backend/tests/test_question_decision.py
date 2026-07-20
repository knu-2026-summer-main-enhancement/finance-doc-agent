from __future__ import annotations

import unittest

from pydantic import ValidationError

from rag.question_decision import QuestionDecision


class QuestionDecisionTest(unittest.TestCase):
    def test_requests_derive_unique_operations_and_preserve_request_count(self):
        decision = QuestionDecision(
            status="ready",
            requests=[
                {
                    "source_text": "장학금 규정",
                    "operation": "document_explain",
                },
                {
                    "source_text": "전체 목록",
                    "operation": "list_records",
                },
            ],
            reason="서로 다른 두 요청",
            retrieval_query="장학금 규정",
        )

        self.assertEqual(
            decision.operations,
            ("document_explain", "list_records"),
        )
        self.assertEqual(decision.request_count, 2)

    def test_repeated_operation_requests_remain_independent(self):
        decision = QuestionDecision(
            status="ready",
            requests=[
                {"source_text": "지급 기준", "operation": "document_criteria"},
                {"source_text": "신청 기준", "operation": "document_criteria"},
            ],
            reason="두 기준에 대한 별도 요청",
            retrieval_query="지급 기준 신청 기준",
        )

        self.assertEqual(decision.operations, ("document_criteria",))
        self.assertEqual(decision.request_count, 2)

    def test_ready_accepts_one_or_more_known_operations(self):
        decision = QuestionDecision(
            status="ready",
            operations=["sum_amount"],
            reason="금액 합계 요청",
        )
        self.assertEqual(decision.operations, ("sum_amount",))

        mixed = QuestionDecision(
            status="ready",
            operations=["sum_amount", "document_criteria"],
            reason="계산과 기준 검색의 혼합 요청",
            retrieval_query="지급 기준",
        )
        self.assertEqual(
            mixed.operations,
            ("sum_amount", "document_criteria"),
        )

    def test_document_operation_requires_retrieval_query(self):
        with self.assertRaises(ValidationError):
            QuestionDecision(
                status="ready",
                operations=["document_criteria"],
                reason="본문 기준 검색",
            )

    def test_non_document_operation_rejects_retrieval_query(self):
        with self.assertRaises(ValidationError):
            QuestionDecision(
                status="ready",
                operations=["structured_query"],
                reason="범용 표 조회",
                retrieval_query="사용하면 안 됨",
            )

    def test_duplicate_or_unknown_operations_are_rejected(self):
        with self.assertRaises(ValidationError):
            QuestionDecision(
                status="ready",
                operations=["sum_amount", "sum_amount"],
                reason="중복",
            )
        with self.assertRaises(ValidationError):
            QuestionDecision(
                status="ready",
                operations=["execute_python"],
                reason="허용되지 않은 작업",
            )

    def test_clarification_has_no_operations(self):
        decision = QuestionDecision(
            status="clarification",
            reason="조회 대상이 부족함",
            message="조회할 문서나 항목을 지정해 주세요.",
            candidates=["문서명", "컬럼명"],
        )
        self.assertEqual(decision.operations, ())
        self.assertEqual(decision.requests, ())
        self.assertEqual(decision.candidates, ("문서명", "컬럼명"))

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            QuestionDecision(
                status="ready",
                operations=["structured_query"],
                reason="표 조회",
                python="df.iloc[0]",
            )


if __name__ == "__main__":
    unittest.main()
