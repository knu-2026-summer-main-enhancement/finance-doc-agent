from __future__ import annotations

import unittest

from tests.eval import score_structured_facts


class EvaluationMoneyTest(unittest.TestCase):
    def test_amount_accepts_korean_money_shorthand(self):
        ok, checks = score_structured_facts("조건: 결제 금액=2만원", {"amount": 20_000})

        self.assertTrue(ok)
        self.assertTrue(checks[0]["ok"])

    def test_null_field_accepts_explicit_missing_value_answer(self):
        ok, checks = score_structured_facts("\uc804\ud654\ubc88\ud638\uac00 \uc5c6\uc74c", {"phone": None})

        self.assertTrue(ok)
        self.assertTrue(checks[0]["ok"])

    def test_clarification_requires_ambiguous_name_guidance(self):
        ok, checks = score_structured_facts(
            "유사한 이름의 회원이 여러 명입니다. 전체 이름을 알려 주세요.",
            {"clarification": True},
        )

        self.assertTrue(ok)
        self.assertTrue(checks[0]["ok"])

    def test_same_name_people_require_separate_totals(self):
        answer = (
            "같은 이름을 가진 서로 다른 인물이 있어 각각 계산했습니다.\n"
            "김현수 1: 40,000원 (2건)\n"
            "김현수 2: 40,000원 (2건)"
        )

        ok, checks = score_structured_facts(
            answer,
            {
                "person_totals": [
                    {"amount": 40_000, "payments": 2},
                    {"amount": 40_000, "payments": 2},
                ]
            },
        )

        self.assertTrue(ok)
        self.assertTrue(checks[0]["ok"])


if __name__ == "__main__":
    unittest.main()
