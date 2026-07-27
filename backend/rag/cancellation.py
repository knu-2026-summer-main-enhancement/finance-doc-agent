"""Cooperative cancellation shared by chat execution paths."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from contextvars import ContextVar, Token
from threading import Event
from typing import TypeVar


class RequestCancelled(Exception):
    """Raised when the user has cancelled the active chat request."""


_cancel_event: ContextVar[Event | None] = ContextVar("chat_cancel_event", default=None)
T = TypeVar("T")


async def await_cancellable(awaitable: Awaitable[T]) -> T:
    """Stop an in-flight async operation as soon as the request is cancelled."""
    raise_if_cancelled()
    event = _cancel_event.get()
    if event is None:
        return await awaitable

    work = asyncio.ensure_future(awaitable)
    try:
        while not work.done():
            await asyncio.wait({work}, timeout=0.05)
            if event.is_set():
                break
        if event.is_set():
            if not work.done():
                work.cancel()
                try:
                    await work
                except asyncio.CancelledError:
                    pass
            raise RequestCancelled("Request cancelled")
        return await work
    finally:
        if not work.done():
            work.cancel()


async def next_cancellable(iterator: AsyncIterator[T]) -> T:
    """Read the next streaming chunk without waiting for another token to cancel."""
    return await await_cancellable(anext(iterator))


def set_cancel_event(event: Event | None) -> Token:
    return _cancel_event.set(event)


def reset_cancel_event(token: Token) -> None:
    _cancel_event.reset(token)


def raise_if_cancelled() -> None:
    event = _cancel_event.get()
    if event is not None and event.is_set():
        raise RequestCancelled("답변 생성을 중단했습니다.")
