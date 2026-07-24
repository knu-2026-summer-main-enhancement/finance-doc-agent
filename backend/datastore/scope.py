from __future__ import annotations

# 한 요청에서 선택한 문서 범위를 ContextVar로 격리한다.
# 전역 DataFrame을 직접 잘라 쓰지 말고 이 모듈의 범위 조회 함수를 사용한다.

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Mapping, TypeVar


_T = TypeVar("_T")
_selected_sources: ContextVar[tuple[str, ...]] = ContextVar(
    "selected_document_sources",
    default=(),
)


def _normalize_source(source: object) -> str:
    return os.path.basename(str(source or "").strip())


def selected_sources() -> tuple[str, ...]:
    return _selected_sources.get()


def source_scope_active() -> bool:
    return bool(selected_sources())


def source_is_selected(source: object) -> bool:
    selected = selected_sources()
    if not selected:
        return True
    return _normalize_source(source) in selected


def scoped_mapping(
    namespace: Mapping[str, _T],
    source_by_alias: Mapping[str, str],
) -> dict[str, _T]:
    return {
        alias: value
        for alias, value in namespace.items()
        if source_is_selected(source_by_alias.get(alias, alias))
    }


@contextmanager
def document_scope(sources: list[str] | tuple[str, ...] | None) -> Iterator[tuple[str, ...]]:
    normalized = tuple(dict.fromkeys(
        source
        for source in (_normalize_source(item) for item in (sources or []))
        if source
    ))
    token = _selected_sources.set(normalized)
    try:
        yield normalized
    finally:
        _selected_sources.reset(token)
