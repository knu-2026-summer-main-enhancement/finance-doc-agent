import asyncio
from threading import Event
import unittest

from rag.cancellation import (
    RequestCancelled,
    await_cancellable,
    reset_cancel_event,
    set_cancel_event,
)


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_inflight_awaitable_is_cancelled_promptly(self):
        event = Event()
        token = set_cancel_event(event)
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def slow_operation():
            started.set()
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                stopped.set()
                raise

        async def cancel_soon():
            await started.wait()
            await asyncio.sleep(0.01)
            event.set()

        try:
            cancel_task = asyncio.create_task(cancel_soon())
            with self.assertRaises(RequestCancelled):
                await await_cancellable(slow_operation())
            await cancel_task
            self.assertTrue(stopped.is_set())
        finally:
            reset_cancel_event(token)

    async def test_completed_awaitable_returns_its_value(self):
        event = Event()
        token = set_cancel_event(event)
        try:
            self.assertEqual(await await_cancellable(asyncio.sleep(0, result="ok")), "ok")
        finally:
            reset_cancel_event(token)
