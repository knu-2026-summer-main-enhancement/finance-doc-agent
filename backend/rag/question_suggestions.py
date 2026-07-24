from __future__ import annotations

import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from typing import Mapping

import pandas as pd

from rag.deterministic_query_plan import build_schema_grounded_plan
from pandas_engine.date_filter import date_column_candidates
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
_LOCAL_PERSON_CATALOG_LIMIT = 500
_base_cache: OrderedDict[tuple, tuple[dict[str, str], ...]] = OrderedDict()
_person_cache: OrderedDict[tuple, tuple[str, ...]] = OrderedDict()
_cache_lock = threading.Lock()
_PERSON_PARTICLE = re.compile(r"(?:에게|한테|께|은|는|이|가|을|를|의|도|만)$")


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


def build_person_autocomplete_catalog(
    dataframes: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    """Return locally usable, schema-grounded person completion metadata.

    This is fetched once when the document scope changes.  Keystrokes must not
    cause another request or an LLM invocation.
    """

    names = _person_values(dataframes)
    if not names:
        return {"names": [], "actions": [], "mode": "local", "total": 0}

    # Validate the action wording against this schema once.  All returned
    # names originate from the same person column, so the valid shapes apply
    # consistently to each of them.
    sample_name = names[0]
    actions: list[dict[str, str]] = []
    for template in _dynamic_templates(sample_name, dataframes):
        result = _compile_template(template, dataframes)
        if result is None:
            continue
        suffix = result["text"].removeprefix(sample_name).strip()
        if suffix:
            actions.append({
                "suffix": suffix,
                "operation": result["operation"],
                "label": result["label"],
                "path": result["path"],
                "path_label": result["path_label"],
            })
    # Avoid exposing or iterating an entire large member list in the browser.
    # The UI switches to the prefix endpoint when this limit is exceeded.
    local_names = list(names) if len(names) <= _LOCAL_PERSON_CATALOG_LIMIT else []
    return {
        "names": local_names,
        "actions": actions,
        "mode": "local" if local_names else "remote",
        "total": len(names),
    }


def build_person_prefix_matches(
    prefix: str,
    *,
    dataframes: Mapping[str, pd.DataFrame],
    limit: int = 10,
) -> list[str]:
    """Return at most ``limit`` stored names for a two-or-more-character prefix.

    This is deliberately a narrow lookup endpoint: it receives no question
    intent, returns no contact fields, and never logs the supplied text.
    """

    entered = normalize_person_name(prefix)
    entered = _PERSON_PARTICLE.sub("", entered)
    if len(entered) < 2 or limit <= 0:
        return []
    matches: list[str] = []
    for stored in _person_values(dataframes):
        if stored.startswith(entered):
            matches.append(stored)
        elif is_masked_name(stored) and len(entered) <= len(stored) and all(
            expected == "*" or expected == actual
            for expected, actual in zip(stored, entered)
        ):
            matches.append(stored)
        if len(matches) >= limit:
            break
    return matches


def build_date_autocomplete_catalog(
    dataframes: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    """Return date completion actions only when the selected schema supports them."""

    if len(dataframes) != 1:
        return {"actions": []}
    df = next(iter(dataframes.values()))
    temporal_columns = date_column_candidates(df)
    year_columns: list[str] = []
    month_columns: list[str] = []
    complete_columns: list[str] = []
    for column in df.columns:
        if str(column).startswith("_"):
            continue
        meaning = infer_column_meaning(str(column), df[column])
        if meaning.concept != "temporal":
            continue
        if meaning.role == "year":
            year_columns.append(str(column))
        elif meaning.role == "month":
            month_columns.append(str(column))
        elif meaning.role in {"date", "year_month"}:
            complete_columns.append(str(column))

    # A year/month expression is safe without a qualifier only with one
    # component pair or one complete calendar column. Multiple business dates
    # remain available, but every suggestion names its chosen source column.
    supports_year_month = (
        len(year_columns) == 1 and len(month_columns) == 1
    ) or (
        len(complete_columns) == 1
        and len(temporal_columns) == 1
    )
    explicit_columns = list(dict.fromkeys(
        column for column in temporal_columns
        if column in complete_columns or (column in month_columns and len(year_columns) == 1)
    ))
    if not supports_year_month and not explicit_columns:
        return {"actions": []}

    prefix = f"{date.today().year}년 1월"
    actions: list[dict[str, str]] = []
    leads = [""] if supports_year_month else [f"{column} 기준" for column in explicit_columns]
    for lead in leads:
        stem = f"{lead} {prefix}".strip()
        templates = (
            _Template(f"{stem} 목록 보여줘", "filter_records", "날짜 목록", "list_records", "fast"),
            _Template(f"{stem} 금액 총합 알려줘", "sum_amount", "날짜 합계", "sum_amount", "fast"),
            _Template(f"{stem} 인원 몇 명이야?", "count_records", "날짜 인원", "count_records", "fast"),
        )
        for template in templates:
            result = _compile_template(template, dataframes)
            if result is None:
                continue
            actions.append({
                "lead": lead,
                "suffix": result["text"].removeprefix(stem).strip(),
                "operation": result["operation"],
                "label": result["label"],
                "path": result["path"],
                "path_label": result["path_label"],
            })
    return {"actions": actions}


def _all_dataframes_support(
    dataframes: Mapping[str, pd.DataFrame],
    predicate,
) -> bool:
    return bool(dataframes) and all(
        any(
            not str(column).startswith("_") and predicate(infer_column_meaning(str(column), df[column]))
            for column in df.columns
        )
        for df in dataframes.values()
    )


def _multi_document_suggestions(dataframes: Mapping[str, pd.DataFrame]) -> tuple[dict[str, str], ...]:
    """Return only cross-document prompts whose semantic prerequisites are shared.

    These are classified prompts, not fast QueryPlans: the final executor still
    decides whether selected documents can be combined for the exact question.
    """

    if len(dataframes) < 2:
        return ()
    candidates: list[dict[str, str]] = []
    if _all_dataframes_support(dataframes, lambda meaning: meaning.concept == "entity" and meaning.role == "entity_name" and meaning.qualifier == "person"):
        candidates.append({
            "text": "선택한 문서 전체 인원 몇 명이야?", "operation": "count_records",
            "label": "여러 문서 인원", "path": "classified", "path_label": "공통 스키마", "scope": "multi",
        })
    if _all_dataframes_support(dataframes, lambda meaning: meaning.role == "amount" or meaning.data_type == "money"):
        candidates.append({
            "text": "선택한 문서 전체 금액 알려줘", "operation": "sum_amount",
            "label": "여러 문서 합계", "path": "classified", "path_label": "공통 스키마", "scope": "multi",
        })
    if _all_dataframes_support(dataframes, lambda meaning: meaning.concept == "temporal" and meaning.role in {"date", "year_month", "month"}):
        candidates.append({
            "text": "선택한 문서를 날짜순으로 보여줘", "operation": "structured_query",
            "label": "여러 문서 날짜", "path": "classified", "path_label": "공통 스키마", "scope": "multi",
        })
    return tuple(candidates)


def _person_value_matches(entered: str, stored: str) -> bool:
    if entered == stored:
        return True
    return (
        is_masked_name(stored)
        and len(entered) == len(stored)
        and all(expected == "*" or expected == actual for expected, actual in zip(stored, entered))
    )


def _grounded_person_name(query: str, dataframes: Mapping[str, pd.DataFrame]) -> str | None:
    """Find a stored person name anywhere in a user phrase without guessing.

    Korean particles are removed only from the entered phrase.  The returned
    value always comes from the source column, so a suggestion never invents a
    person name and masked-name comparison keeps its existing strict policy.
    """

    compact = normalize_person_name(query)
    if not compact:
        return None
    candidates = [compact, _PERSON_PARTICLE.sub("", compact)]
    candidates.extend(
        _PERSON_PARTICLE.sub("", token)
        for token in re.findall(r"[가-힣*]{2,8}", compact)
    )
    for stored in _person_values(dataframes):
        if stored in compact:
            return stored
        for candidate in candidates:
            if _person_value_matches(candidate, stored):
                return stored
        if is_masked_name(stored):
            for start in range(max(0, len(compact) - len(stored) + 1)):
                if _person_value_matches(compact[start:start + len(stored)], stored):
                    return stored
    return None


def _dynamic_templates(
    query: str,
    dataframes: Mapping[str, pd.DataFrame],
) -> tuple[_Template, ...]:
    stripped = query.strip()
    candidates: list[_Template] = []

    # Echo only a stored person value already present in the user's phrase.
    # D-P still grounds the final question to a row before it can be suggested.
    if person_name := _grounded_person_name(stripped, dataframes):
        candidates.extend((
            _Template(f"{person_name} 금액 알려줘", "lookup_amount", "인물 금액", "lookup_amount", "fast", True),
            _Template(f"{person_name} 전체 기록 보여줘", "filter_records", "인물 기록", "filter_records", "fast", True),
            _Template(f"{person_name} 전화번호 알려줘", "lookup_field", "전화번호", "lookup_field", "fast", True),
            _Template(f"{person_name} 이메일 알려줘", "lookup_field", "이메일", "lookup_field", "fast", True),
            _Template(f"{person_name} 학과 알려줘", "lookup_field", "학과", "lookup_field", "fast", True),
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
    tokens = [
        normalized
        for token in tokens
        for normalized in (token, _PERSON_PARTICLE.sub("", token))
        if normalized
    ]
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
    candidates.extend(_multi_document_suggestions(dataframes))
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
