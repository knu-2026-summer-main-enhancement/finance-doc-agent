import unittest
from unittest.mock import AsyncMock, patch

from fastapi import BackgroundTasks, HTTPException

import main
from core.privacy import question_log_metadata


class PrivacyTest(unittest.IsolatedAsyncioTestCase):
    async def test_chat_hides_internal_exception_and_question_text(self):
        question = "김현수 이메일 secret@example.com 알려줘"
        request = main.ChatRequest(question=question, mode="auto")
        with patch.object(main, "QUESTION_ENGINE_MODE", "legacy"), patch.object(
            main, "_route_with_guard", return_value="PANDAS"
        ), patch.object(
            main, "_answer_pandas", new=AsyncMock(side_effect=RuntimeError("C:/private/data.xlsx"))
        ), patch.object(main.logger, "error") as logged:
            with self.assertRaises(HTTPException) as raised:
                await main.chat(request, BackgroundTasks(), None)

        self.assertEqual(raised.exception.status_code, 500)
        self.assertNotIn("C:/private", str(raised.exception.detail))
        self.assertNotIn(question, str(logged.call_args))
        self.assertNotIn("secret@example.com", str(logged.call_args))

    async def test_stream_hides_internal_exception_and_question_text(self):
        question = "010-1234-5678 결제 기록 알려줘"
        request = main.ChatRequest(question=question, mode="auto")
        with patch.object(main, "QUESTION_ENGINE_MODE", "legacy"), patch.object(
            main, "_route_with_guard", return_value="PANDAS"
        ), patch.object(
            main, "_answer_pandas", new=AsyncMock(side_effect=RuntimeError("database-password"))
        ), patch.object(main.logger, "error") as logged:
            response = await main.chat_stream(request, None)
            chunks = [chunk async for chunk in response.body_iterator]

        body = b"".join(
            chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
            for chunk in chunks
        ).decode("utf-8")
        self.assertNotIn("database-password", body)
        self.assertNotIn(question, str(logged.call_args))
        self.assertNotIn("010-1234-5678", str(logged.call_args))

    def test_question_metadata_is_stable_and_contains_no_plaintext(self):
        question = "private@example.com 금액 알려줘"

        first = question_log_metadata(question)
        second = question_log_metadata(question)

        self.assertEqual(first, second)
        self.assertEqual(first[1], len(question))
        self.assertNotIn("private@example.com", str(first))


if __name__ == "__main__":
    unittest.main()
