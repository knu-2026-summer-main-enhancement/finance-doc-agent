"""Cooperative cancellation shared by chat execution paths."""

from __future__ import annotations

from contextvars import ContextVar, Token
from threading import Event


class RequestCancelled(Exception):
    """Raised when the user has cancelled the active chat request."""


_cancel_event: ContextVar[Event | None] = ContextVar("chat_cancel_event", default=None)


def set_cancel_event(event: Event | None) -> Token:
    return _cancel_event.set(event)


def reset_cancel_event(token: Token) -> None:
    _cancel_event.reset(token)


def raise_if_cancelled() -> None:
    event = _cancel_event.get()
    if event is not None and event.is_set():
        raise RequestCancelled("답변 생성을 중단했습니다.")
