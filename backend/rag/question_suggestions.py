from __future__ import annotations

import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from rag.deterministic_query_plan import build_schema_grounded_plan
from utils.semantic_schema import infer_column_meaning
from utils.table_parser import is_masked_name, normalize_person_name


@dataclass(frozen=True)
class _Template:
    text: str
    operation: str
    label: str
    operation_hint: str | None = None
    path: str = "classified"
    requires_filter: bool = False


# Canonical examples cover every operation in QuestionOperation. They contain
# no document answer, person, year, amount, or test-specific literal.
_TEMPLATES = (
    _Template("현재 적재된 문서 목록 보여줘", "list_documents", "문서 목록"),
    _Template("전체 목록 보여줘", "list_records", "전체 목록", "list_records", "fast"),
    _Template("전체 기록은 몇 건이야?", "count_records", "건수", "count_records", "fast"),
    _Template("전체 인원 몇 명이야?", "count_records", "인원", "count_records", "fast"),
    _Template("총 금액 얼마야?", "sum_amount", "합계", "sum_amount", "fast"),
    _Template("평균 금액 얼마야?", "average_amount", "평균", "structured_query", "fast"),
    _Template("금액 중앙값 얼마야?", "median_amount", "중앙값"),
    _Template("금액 최빈값 얼마야?", "mode_amount", "최빈값", "structured_query", "fast"),
    _Template("가장 큰 금액 뭐야?", "max_amount", "최댓값", "max_amount", "fast"),
    _Template("가장 작은 금액 뭐야?", "min_amount", "최솟값", "min_amount", "fast"),
    _Template("돈을 가장 많이 낸 사람 누구야?", "max_person_by_amount", "최고 순위", "structured_query", "fast"),
    _Template("돈을 가장 적게 낸 사람 누구야?", "min_person_by_amount", "최저 순위", "structured_query", "fast"),
    _Template("가장 큰 금액과 가장 작은 금액 비교해줘", "compare", "비교"),
    _Template("금액을 큰 순서대로 보여줘", "structured_query", "정렬·조건", "structured_query", "fast"),
    _Template("문서에 기록된 이유를 알려줘", "document_reason", "이유", path="vector"),
    _Template("이 문서의 목적을 알려줘", "document_purpose", "목적", path="vector"),
    _Template("선정 및 지급 기준을 알려줘", "document_criteria", "기준", path="vector"),
    _Template("신청 및 지급 절차를 알려줘", "document_procedure", "절차", path="vector"),
    _Template("문서 내용을 설명해줘", "document_explain", "내용 설명", path="vector"),
)

_CACHE_LIMIT = 32
_base_cache: OrderedDict[tuple, tuple[dict[str, str], ...]] = OrderedDict()
_person_cache: OrderedDict[tuple, tuple[str, ...]] = OrderedDict()
_cache_lock = threading.Lock()


def _normalize(value: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", value).casefold())


def _scope_key(dataframes: Mapping[str, pd.DataFrame]) -> tuple:
    return tuple(
        (
            alias,
            id(df),
            len(df),
            tuple((str(column), str(dtype)) for column, dtype in df.dtypes.items()),
        )
        for alias, df in dataframes.items()
    )


def _result(template: _Template) -> dict[str, str]:
    path_labels = {
        "fast": "빠른 조회",
        "vector": "AI 문서 검색",
        "classified": "분류 후 조회",
    }
    return {
        "text": template.text,
        "label": template.label,
        "operation": template.operation,
        "path": template.path,
        "path_label": path_labels[template.path],
    }


def _compile_template(
    template: _Template,
    dataframes: Mapping[str, pd.DataFrame],
) -> dict[str, str] | None:
    if template.path != "fast":
        return _result(template)
    plan = build_schema_grounded_plan(
        template.text,
        dataframes=dataframes,
        operation_hint=template.operation_hint,
    )
    if plan is None or (template.requires_filter and not plan.filters):
        return None
    return _result(template)


def _base_suggestions(dataframes: Mapping[str, pd.DataFrame]) -> tuple[dict[str, str], ...]:
    key = _scope_key(dataframes)
    with _cache_lock:
        cached = _base_cache.get(key)
        if cached is not None:
            _base_cache.move_to_end(key)
            return cached

    compiled = tuple(
        result
        for template in _TEMPLATES
        if (result := _compile_template(template, dataframes)) is not None
    )
    with _cache_lock:
        _base_cache[key] = compiled
        _base_cache.move_to_end(key)
        while len(_base_cache) > _CACHE_LIMIT:
            _base_cache.popitem(last=False)
    return compiled


def _person_values(dataframes: Mapping[str, pd.DataFrame]) -> tuple[str, ...]:
    key = _scope_key(dataframes)
    with _cache_lock:
        cached = _person_cache.get(key)
        if cached is not None:
            _person_cache.move_to_end(key)
            return cached

    values: list[str] = []
    if len(dataframes) == 1:
        df = next(iter(dataframes.values()))
        for column in df.columns:
            meaning = infer_column_meaning(str(column), df[column])
            if meaning.concept == "entity" and meaning.role == "entity_name" and meaning.qualifier == "person":
                values = [
                    normalize_person_name(value)
                    for value in df[column].dropna().astype(str).unique()
                    if normalize_person_name(value)
                ]
                break
    result = tuple(values)
    with _cache_lock:
        _person_cache[key] = result
        _person_cache.move_to_end(key)
        while len(_person_cache) > _CACHE_LIMIT:
            _person_cache.popitem(last=False)
    return result


def _is_grounded_person_input(query: str, dataframes: Mapping[str, pd.DataFrame]) -> bool:
    entered = normalize_person_name(query)
    for stored in _person_values(dataframes):
        if entered == stored:
            return True
        if (
            is_masked_name(stored)
            and len(entered) == len(stored)
            and all(expected == "*" or expected == actual for expected, actual in zip(stored, entered))
        ):
            return True
    return False


def _dynamic_templates(
    query: str,
    dataframes: Mapping[str, pd.DataFrame],
) -> tuple[_Template, ...]:
    stripped = query.strip()
    candidates: list[_Template] = []

    # Echo only a name-shaped value already supplied by the user. D-P must
    # ground it to a row before the suggestion can leave the backend.
    if re.fullmatch(r"[가-힣*]{2,5}", stripped) and _is_grounded_person_input(stripped, dataframes):
        candidates.extend((
            _Template(f"{stripped} 금액 알려줘", "lookup_amount", "인물 금액", "lookup_amount", "fast", True),
            _Template(f"{stripped} 전체 기록 보여줘", "filter_records", "인물 기록", "filter_records", "fast", True),
            _Template(f"{stripped} 전화번호 알려줘", "lookup_field", "전화번호", "lookup_field", "fast", True),
            _Template(f"{stripped} 이메일 알려줘", "lookup_field", "이메일", "lookup_field", "fast", True),
            _Template(f"{stripped} 학과 알려줘", "lookup_field", "학과", "lookup_field", "fast", True),
        ))

    # A condition literal is supplied by the user, never invented from source
    # rows. This adds filter_records without exposing metadata values.
    if re.fullmatch(r"(?:\d{1,3}\s*기|(?:19|20)\d{2}\s*년)", stripped):
        candidates.append(_Template(
            f"{stripped} 기록 알려줘",
            "filter_records",
            "조건 목록",
            "filter_records",
            "fast",
            True,
        ))
    return tuple(candidates)


def _relevance(text: str, query: str) -> tuple[int, int]:
    if not query.strip():
        return (1, 0)
    normalized_text = _normalize(text)
    normalized_query = _normalize(query)
    if normalized_text.startswith(normalized_query):
        return (5, -len(text))
    if normalized_query in normalized_text:
        return (4, -len(text))
    tokens = [_normalize(token) for token in re.findall(r"[가-힣A-Za-z0-9*]+", query)]
    tokens = [token for token in tokens if token]
    matched = sum(token in normalized_text for token in tokens)
    return ((3 if tokens and matched == len(tokens) else 2 if matched else 0), matched)


def build_question_suggestions(
    query: str,
    *,
    dataframes: Mapping[str, pd.DataFrame],
    limit: int = 6,
) -> list[dict[str, str]]:
    """Return E-O autocomplete candidates for the current document scope."""

    if limit <= 0:
        return []

    candidates = list(_base_suggestions(dataframes))
    candidates.extend(
        result
        for template in _dynamic_templates(query, dataframes)
        if (result := _compile_template(template, dataframes)) is not None
    )

    ranked: list[tuple[tuple[int, int], int, dict[str, str]]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        key = _normalize(candidate["text"])
        if key in seen:
            continue
        seen.add(key)
        relevance = _relevance(candidate["text"], query)
        if query.strip() and relevance[0] == 0:
            continue
        ranked.append((relevance, -index, candidate))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked[:limit]]
