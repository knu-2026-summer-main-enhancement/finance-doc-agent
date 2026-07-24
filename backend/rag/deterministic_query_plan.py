from __future__ import annotations

# 자주 검증된 질문을 정규식과 실제 스키마로 해석해 QueryPlan을 빠르게 만든다.
# 특정 데이터의 사람명·연도·컬럼명을 넣지 말고 semantic schema에 근거해 계획한다.

import re
import unicodedata
from typing import Hashable, Mapping

import pandas as pd

from pandas_engine.money import parse_money_value
from pandas_engine.aggregation import resolve_amount_column
from pandas_engine.date_filter import parse_date_filter
from pandas_engine.plan_validator import column_data_type
from pandas_engine.query_grounding import parse_grounded_comparisons
from pandas_engine.query_plan import FilterCondition, QueryPlan
from utils.semantic_schema import infer_column_meaning, is_source_column
from utils.table_parser import is_masked_name, normalize_person_name


_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*년")
_YEAR_RANGE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*년?\s*(?:부터|에서|~|〜|-)\s*"
    r"((?:19|20)\d{2})\s*년?\s*(?:까지)?"
)
_MONTH = re.compile(r"(?<!\d)(1[0-2]|[1-9])\s*월")
_COHORT = re.compile(r"(?<!\d)(\d{1,3})\s*(?:기|회)(?!\d)")
_MONEY = re.compile(r"(?<!\d)(\d[\d,]*(?:\s*(?:만원|천원|원))?)(?!\d)")
_PERSON_COUNT = re.compile(r"(?:사람|인원|회원).*?(?:몇\s*명|수)|몇\s*명")
_SUM = re.compile(r"(?:총합|합계|총액|얼마(?:야|예|지|냈))")
_MEAN = re.compile(r"(?:평균|평균값|평균액)")
_MODE = re.compile(r"(?:최빈값|최빈액|가장\s*(?:흔한|많이\s*나온)\s*(?:값|금액)?)")
_MAX_VALUE = re.compile(
    r"(?:최댓값|최대(?:값|액|\s*금액)?|최고(?:값|액|\s*금액)?|"
    r"(?:가장|제일)\s*(?:큰|높은|많은)\s*(?:값|금액|돈|액)|"
    r"(?:값|금액|돈|액).{0,8}?(?:가장|제일)\s*(?:큰|높은|많은))"
)
_MIN_VALUE = re.compile(
    r"(?:최솟값|최소(?:값|액|\s*금액)?|최저(?:값|액|\s*금액)?|"
    r"(?:가장|제일)\s*(?:작은|낮은)\s*(?:값|금액|돈|액)|"
    r"(?:값|금액|돈|액).{0,8}?(?:가장|제일)\s*(?:작은|낮은))"
)
_ROW_COUNT = re.compile(r"(?:몇\s*번|몇\s*회|횟수)")
_MISSING = re.compile(
    r"(?:비어\s*있|안\s*적|미입력|미등록|누락|공백|"
    r"등록\s*되지\s*않|등록되지\s*않|없(?:는|어|어?))"
)
_PAYMENT_EXISTENCE = re.compile(
    r"(?:돈|금액|회비|결제|납부|후원|기부).{0,8}?(?:냈|내었|냈어|냈나요|했어|했나|했나요)"
)
_PERSON_LOOKUP_SUBJECT = r"([가-힣]{2,5}?(?:\s*[\(\[\{][^\]\)\}]*[\]\)\}])?)"
_LEADING_PERSON_AMOUNT_LOOKUP = re.compile(
    rf"^\s*{_PERSON_LOOKUP_SUBJECT}(?:에|은|는|이|가|의)?\s*(?:얼마|금액|돈)"
)
_LEADING_PERSON_FIELD_LOOKUP = re.compile(
    rf"^\s*{_PERSON_LOOKUP_SUBJECT}(?:에|은|는|이|가|의)?\s*"
    r"(?:전화번호|이메일|학과|전공|회비\s*구분)"
)
_GROUP_SUM_EXTREME = re.compile(
    r"(?:가장|제일|최고).{0,16}?(?:많이|많은|큰|높은).{0,12}?(?:사람|회원|인원)|"
    r"(?:사람|회원|인원).{0,12}?(?:가장|제일|최고).{0,16}?(?:많이|많은|큰|높은)"
)
_GROUP_SUM_MINIMUM = re.compile(
    r"(?:가장|제일)\s*(?:돈|금액|회비|결제)?.{0,8}?(?:적게|작게|낮게|최소로).{0,8}?(?:낸|지급한|결제한)?\s*(?:사람|회원|인원|누구(?:야|인지|인가)?)|"
    r"(?:돈|금액|회비|결제).{0,8}?(?:가장|제일)\s*(?:적게|작게|낮게).{0,8}?(?:낸|지급한|결제한)?\s*(?:사람|회원|인원|누구(?:야|인지|인가)?)"
)
_ORDINAL = re.compile(r"(?<!\d)(\d+)\s*(?:번째|째|위|등)")
_KOREAN_ORDINAL = re.compile(
    r"(첫|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*"
    r"(?:번째|번\s*[째쨰])"
)
_KOREAN_ORDINAL_VALUES = {"첫": 1, "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
_LIMIT = re.compile(r"(?<!\d)(\d+)\s*(?:건|개|명)(?!\d)")
_RANK_DESC = re.compile(r"(?:상위|top\s*\d*|높은\s*순|많은\s*순|큰\s*순)", re.IGNORECASE)
_RANK_ASC = re.compile(r"(?:하위|bottom\s*\d*|낮은\s*순|적은\s*순|작은\s*순)", re.IGNORECASE)
_GROUPING_REQUEST = re.compile(
    r"(?:별|마다|기준(?:으로|별)?|묶어서|나눠서|모아서|조합)"
)
_DESC_ORDER = re.compile(r"(?:내림차순|큰\s*순(?:서(?:대로)?)?|많은\s*순(?:서(?:대로)?)?|높은\s*순(?:서(?:대로)?)?|최신|늦은|가장\s*(?:큰|많은|높은))")
_ASC_ORDER = re.compile(r"(?:오름차순|작은\s*순(?:서(?:대로)?)?|적은\s*순(?:서(?:대로)?)?|낮은\s*순(?:서(?:대로)?)?|빠른|이른|가장\s*(?:작은|적은|낮은))")
_PERSON_DISPLAY_SUFFIX = re.compile(r"\s*[\(\[\{][^\]\)\}]*[\]\)\}]\s*$")
_NON_PERSON_LEADING_TOKENS = {
    "가장", "제일", "최고", "최저", "사람", "회원", "회원별",
    "전체", "전체기록", "모든회원", "각회원", "누구",
}

# These are user-facing Korean equivalents for a semantic category, not names
# from any one document. The selected column still has to be inferred from the
# dataframe schema before it can be used in a plan.
_SEMANTIC_REQUEST_TERMS = {
    "department": ("학과", "학부", "전공", "계열", "무슨 과"),
    "amount": ("금액", "납부액", "결제액", "회비", "후원금", "기부금"),
    "year": ("연도", "해마다", "각 연도", "연도 기준"),
    "month": ("월별", "달마다", "각 월", "월 기준"),
    "fee_type": ("회비 구분", "회비 종류", "회비 유형"),
    "registered_date": ("결제 등록 날짜", "결제 등록일", "등록 날짜", "등록일"),
    # "언제" has no column-name overlap.  Return all available temporal
    # evidence (components and full dates) so a sparse registered-date column
    # cannot hide otherwise valid payment timing information.
    "payment_time": ("언제", "날짜", "일자", "시기"),
}


def _norm(value: object) -> str:
    """Normalize column references across spaces, underscores and punctuation."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(value or "")).casefold())


def _meaning(df: pd.DataFrame, column: Hashable):
    return infer_column_meaning(str(column), df[column])


def _columns(df: pd.DataFrame, predicate) -> list[Hashable]:
    return [column for column in df.columns if not str(column).startswith("_") and predicate(_meaning(df, column))]


def _first(df: pd.DataFrame, predicate) -> Hashable | None:
    columns = _columns(df, predicate)
    return columns[0] if columns else None


def _requested_columns(df: pd.DataFrame, question: str) -> list[Hashable]:
    """Resolve explicit headers and generic semantic aliases requested by a user."""
    normalized_question = _norm(question)
    requested = [
        column for column in df.columns
        if (
            not str(column).startswith("_")
            # Single-character date components such as "일" occur naturally
            # inside unrelated words (for example 이메일), so they are never a
            # projection unless a future parser resolves them explicitly.
            and len(_norm(str(column))) >= 2
            and _norm(str(column)) in normalized_question
        )
    ]
    for qualifier, terms in _SEMANTIC_REQUEST_TERMS.items():
        if not any(_norm(term) in normalized_question for term in terms):
            continue
        if (
            qualifier == "amount"
            and any(
                _norm(term) in normalized_question
                for term in _SEMANTIC_REQUEST_TERMS["fee_type"]
            )
            and not any(
                _norm(term) in normalized_question
                for term in ("금액", "납부액", "결제액", "후원금", "기부금")
            )
        ):
            # In "회비 구분 미등록", 회비 describes the category header; it
            # is not a request to filter the payment amount as missing.
            continue
        requested.extend(
            _columns(
                df,
                lambda meaning: (
                    (
                        meaning.concept == "category"
                        and meaning.role == "category"
                        and meaning.qualifier == qualifier
                    )
                    or (
                        qualifier == "amount"
                        and (meaning.role == "amount" or meaning.data_type == "money")
                    )
                    or (qualifier == "year" and meaning.role == "year")
                    or (qualifier == "month" and meaning.role == "month")
                    or (
                        qualifier == "registered_date"
                        and meaning.concept == "temporal"
                        and meaning.role in {"date", "registered_date"}
                    )
                ),
            ),
        )
        if qualifier == "fee_type":
            requested.extend(
                column
                for column in df.columns
                if "회비" in _norm(str(column))
                and any(
                    token in _norm(str(column))
                    for token in ("구분", "종류", "유형")
                )
            )
        if (
            qualifier == "payment_time"
            and not any(
                _norm(term) in normalized_question
                for term in _SEMANTIC_REQUEST_TERMS["registered_date"]
            )
        ):
            requested.extend(
                _columns(
                    df,
                    lambda meaning: (
                        meaning.concept == "temporal"
                        and meaning.role in {"year", "month", "date", "year_month"}
                    ),
                )
            )
    return list(dict.fromkeys(requested))


def _value_filters(df: pd.DataFrame, question: str) -> list[FilterCondition]:
    normalized_question = _norm(question)
    filters: list[FilterCondition] = []
    for column in df.columns:
        meaning = _meaning(df, column)
        is_string_identifier = (
            meaning.concept == "identifier" and meaning.data_type == "string"
        )
        if (
            str(column).startswith("_")
            or not is_source_column(df, column)
            or (column_data_type(df, column) != "string" and not is_string_identifier)
        ):
            continue
        values = df[column].dropna().astype(str).str.strip()
        unique = values[values.ne("")].unique().tolist()
        # Exact grounding is bounded to avoid turning arbitrary prose columns into filters.
        if len(unique) > 5000:
            continue
        matched = [value for value in unique if len(_norm(value)) >= 2 and _norm(value) in normalized_question]
        if len(matched) == 1:
            filters.append(FilterCondition(column=str(column), operator="eq", value=matched[0], source_text=matched[0]))
    return filters


def _comparison_filters(
    question: str,
    *,
    money: Hashable | None,
    year: Hashable | None,
    month: Hashable | None,
    cohort: Hashable | None,
) -> list[FilterCondition]:
    """Ground explicit numeric comparisons to only schema-matched columns."""
    filters: list[FilterCondition] = []
    for comparison in parse_grounded_comparisons(question):
        value_text = str(comparison.value_text)
        column: Hashable | None = None
        if comparison.value_kind == "money":
            column = money
        elif "기" in value_text:
            column = cohort
        elif "년" in value_text:
            column = year
        elif "월" in value_text:
            column = month
        if column is None:
            continue
        filters.append(FilterCondition(
            column=str(column),
            operator=comparison.operator,
            value=comparison.value,
            source_text=comparison.source_text,
        ))
    return filters


def _person_filter(
    df: pd.DataFrame,
    person: Hashable,
    question: str,
) -> FilterCondition | None:
    """Ground an exact name or one unambiguous display-name base to a person row."""
    normalized_question = _norm(question)
    values = df[person].dropna().astype(str).str.strip().unique().tolist()
    exact = [value for value in values if len(_norm(value)) >= 2 and _norm(value) in normalized_question]
    if len(exact) == 1:
        return FilterCondition(column=str(person), operator="eq", value=exact[0], source_text=exact[0])

    # A parenthesized/bracketed suffix is supplementary display information,
    # not part of a person's base name.  Only accept it when one stored value
    # has that base; ambiguous bases deliberately fall back to clarification.
    base_matches = []
    for value in values:
        base = _PERSON_DISPLAY_SUFFIX.sub("", value).strip()
        if len(_norm(base)) >= 2 and _norm(base) in normalized_question:
            base_matches.append(value)
    if len(base_matches) == 1:
        return FilterCondition(
            column=str(person),
            operator="eq",
            value=base_matches[0],
            source_text=_PERSON_DISPLAY_SUFFIX.sub("", base_matches[0]).strip(),
        )

    masked_matches: list[tuple[str, str]] = []
    for value in values:
        stored = normalize_person_name(_PERSON_DISPLAY_SUFFIX.sub("", value).strip())
        if not is_masked_name(stored):
            continue
        for start in range(0, max(0, len(normalized_question) - len(stored) + 1)):
            candidate = normalized_question[start:start + len(stored)]
            if not candidate or not all("가" <= char <= "힣" for char in candidate):
                continue
            if all(expected == "*" or expected == actual for expected, actual in zip(stored, candidate)):
                masked_matches.append((value, candidate))
                break
    unique_masked = list(dict.fromkeys(masked_matches))
    if len(unique_masked) == 1:
        value, candidate = unique_masked[0]
        return FilterCondition(column=str(person), operator="eq", value=value, source_text=candidate)
    return None


def has_unmatched_person_amount_reference(
    question: str,
    *,
    dataframes: Mapping[str, pd.DataFrame],
) -> bool:
    """Identify a leading person-like amount lookup absent from the scoped data.

    This is deliberately narrow: it applies only to a Korean name-shaped token
    at the beginning of an amount lookup.  Before reporting no result, the
    candidate is checked against the schema-derived person column and every
    textual cell value so category/value queries are not mistaken for names.
    """
    match = _LEADING_PERSON_AMOUNT_LOOKUP.search(str(question or ""))
    if not match or not dataframes:
        return False
    raw_candidate = match.group(1)
    candidate = _norm(raw_candidate)
    if len(candidate) < 2:
        return False
    if candidate in {_norm(token) for token in _NON_PERSON_LEADING_TOKENS}:
        return False

    for df in dataframes.values():
        person = _first(
            df,
            lambda item: (
                item.concept == "entity"
                and item.role == "entity_name"
                and item.qualifier == "person"
            ),
        )
        if person is not None:
            for value in df[person].dropna().astype(str).str.strip().unique():
                base = _PERSON_DISPLAY_SUFFIX.sub("", value).strip()
                if candidate in {_norm(value), _norm(base)}:
                    return False
                stored = normalize_person_name(base)
                query_name = normalize_person_name(raw_candidate)
                if (
                    is_masked_name(stored)
                    and len(stored) == len(query_name)
                    and all(expected == "*" or expected == actual for expected, actual in zip(stored, query_name))
                ):
                    return False

        for column in df.columns:
            if str(column).startswith("_") or column_data_type(df, column) != "string":
                continue
            values = df[column].dropna().astype(str).str.strip()
            if any(_norm(value) == candidate for value in values[values.ne("")].unique()):
                return False
    return True


def has_unmatched_person_field_reference(
    question: str,
    *,
    dataframes: Mapping[str, pd.DataFrame],
) -> bool:
    """Identify a leading person-like field lookup absent from scoped data.

    This mirrors the amount-lookup fast path, but is deliberately restricted to
    explicit personal-field requests.  It prevents an absent person from being
    sent through repeated LLM planning attempts while preserving reverse field
    lookups and non-person category queries.
    """
    match = _LEADING_PERSON_FIELD_LOOKUP.search(str(question or ""))
    if not match or not dataframes:
        return False
    raw_candidate = match.group(1)
    candidate = _norm(raw_candidate)
    if len(candidate) < 2:
        return False
    if candidate in {_norm(token) for token in _NON_PERSON_LEADING_TOKENS}:
        return False

    for df in dataframes.values():
        person = _first(
            df,
            lambda item: (
                item.concept == "entity"
                and item.role == "entity_name"
                and item.qualifier == "person"
            ),
        )
        if person is not None:
            for value in df[person].dropna().astype(str).str.strip().unique():
                base = _PERSON_DISPLAY_SUFFIX.sub("", value).strip()
                if candidate in {_norm(value), _norm(base)}:
                    return False
                stored = normalize_person_name(base)
                query_name = normalize_person_name(raw_candidate)
                if (
                    is_masked_name(stored)
                    and len(stored) == len(query_name)
                    and all(expected == "*" or expected == actual for expected, actual in zip(stored, query_name))
                ):
                    return False

        for column in df.columns:
            if str(column).startswith("_") or column_data_type(df, column) != "string":
                continue
            values = df[column].dropna().astype(str).str.strip()
            if any(_norm(value) == candidate for value in values[values.ne("")].unique()):
                return False
    return True


def ambiguous_person_lookup_candidates(
    question: str,
    *,
    dataframes: Mapping[str, pd.DataFrame],
) -> tuple[str, ...]:
    """Return person-name candidates only for an ambiguous leading lookup.

    A short partial name must never be treated as absent merely because it is
    not an exact stored value.  Exact and unique display-name-base matches are
    handled by ``_person_filter``; this helper only surfaces multiple possible
    people so the caller can request a full name instead of guessing.
    """
    match = _LEADING_PERSON_AMOUNT_LOOKUP.search(str(question or "")) or _LEADING_PERSON_FIELD_LOOKUP.search(str(question or ""))
    if not match or not dataframes:
        return ()
    raw_candidate = match.group(1)
    candidate = _norm(raw_candidate)
    if len(candidate) < 2:
        return ()

    matches: list[str] = []
    for df in dataframes.values():
        person = _first(
            df,
            lambda item: (
                item.concept == "entity"
                and item.role == "entity_name"
                and item.qualifier == "person"
            ),
        )
        if person is None:
            continue
        for value in df[person].dropna().astype(str).str.strip().unique():
            base = _PERSON_DISPLAY_SUFFIX.sub("", value).strip()
            if candidate in {_norm(value), _norm(base)}:
                return ()
            stored = normalize_person_name(base)
            query_name = normalize_person_name(raw_candidate)
            if (
                is_masked_name(stored)
                and len(stored) == len(query_name)
                and all(expected == "*" or expected == actual for expected, actual in zip(stored, query_name))
            ):
                matches.append(value)
                continue
            if _norm(value).startswith(candidate) or _norm(base).startswith(candidate):
                matches.append(value)
    unique = tuple(dict.fromkeys(matches))
    return unique if len(unique) > 1 else ()


def is_grounded_person_payment_existence_question(
    question: str,
    *,
    dataframes: Mapping[str, pd.DataFrame],
) -> bool:
    """Whether a payment-existence phrase has one grounded person subject."""
    text = str(question or "")
    # A payment-time request can include the same verb (for example "언제
    # 냈어?") but must keep its temporal projection rather than becoming a sum.
    if re.search(r"(?:\uc5b8\uc81c|\ub0a0\uc9dc|\ub4f1\ub85d|\uc2dc\uae30)", text):
        return False
    payment_existence = _PAYMENT_EXISTENCE.search(text) or re.search(
        r"(?:\ub0c8\uc74c|\ub0c8\uc5b4|\ub0c8\ub098|\ub0b8\uac70)\??$",
        text,
    )
    if not payment_existence or len(dataframes) != 1:
        return False
    _, df = next(iter(dataframes.items()))
    person = _first(
        df,
        lambda item: (
            item.concept == "entity"
            and item.role == "entity_name"
            and item.qualifier == "person"
        ),
    )
    return person is not None and _person_filter(df, person, question) is not None


def is_grounded_person_amount_lookup_question(
    question: str,
    *,
    dataframes: Mapping[str, pd.DataFrame],
) -> bool:
    """Whether a leading person subject asks for that person's amount."""

    if not _LEADING_PERSON_AMOUNT_LOOKUP.search(str(question or "")) or len(dataframes) != 1:
        return False
    _, df = next(iter(dataframes.items()))
    person = _first(
        df,
        lambda item: (
            item.concept == "entity"
            and item.role == "entity_name"
            and item.qualifier == "person"
        ),
    )
    return person is not None and _person_filter(df, person, question) is not None


def build_schema_grounded_plan(
    question: str,
    *,
    dataframes: Mapping[str, pd.DataFrame],
    operation_hint: str | None,
) -> QueryPlan | None:
    """Build a conservative plan only from schema roles and literal cell values.

    This is a fallback for a malformed local-model plan, never a replacement for
    validation. It intentionally returns ``None`` when the question cannot be
    grounded without guessing.
    """
    if len(dataframes) != 1:
        return None
    alias, df = next(iter(dataframes.items()))
    person = _first(df, lambda item: item.concept == "entity" and item.role == "entity_name" and item.qualifier == "person")
    money = resolve_amount_column(df, question).selected
    temporal = _first(df, lambda item: item.data_type == "date" or item.role in {"date", "registered_date"})
    year = _first(df, lambda item: item.role == "year")
    month = _first(df, lambda item: item.role == "month")
    cohort = _first(
        df,
        lambda item: item.concept == "category" and item.qualifier == "cohort",
    )
    normalized_question = _norm(question)
    broad_projection = bool(re.search(
        r"(?:전체\s*(?:기록|내역|회원)|모든\s*회원|각\s*회원|회원별)",
        question,
    ))
    complete_grouping = bool(
        _GROUPING_REQUEST.search(question)
        and re.search(r"(?:금액|돈|회비|결제|납부|후원|기부)", question)
    )
    # These shapes explicitly request the full table or every group. Scanning
    # every string cell for an implicit value filter is both unnecessary and
    # expensive on large sheets.
    skip_literal_filter_scan = broad_projection or complete_grouping
    filters = [] if skip_literal_filter_scan else _value_filters(df, question)

    # A range such as "2025년 6월부터 2026년 1월까지" cannot be represented
    # by independent year/month filters: doing so silently means only the
    # first month. The date-filter executor resolves these ranges instead.
    date_spec = parse_date_filter(question)
    if date_spec is not None and date_spec.start_month != date_spec.end_month:
        return None

    requested = _requested_columns(df, question)
    if _MISSING.search(question):
        for column in requested:
            filters = [item for item in filters if item.column != str(column)]
            filters.append(FilterCondition(column=str(column), operator="is_null", source_text=str(column)))

    year_range = _YEAR_RANGE.search(question)
    year_match = _YEAR.search(question)
    if year_range and year is not None:
        filters.append(FilterCondition(
            column=str(year),
            operator="between",
            value=(int(year_range.group(1)), int(year_range.group(2))),
            source_text=year_range.group(0),
        ))
    elif year_match and year is not None:
        filters.append(FilterCondition(column=str(year), operator="eq", value=int(year_match.group(1)), source_text=year_match.group(0)))
    month_match = _MONTH.search(question)
    if month_match and month is not None:
        filters.append(FilterCondition(column=str(month), operator="eq", value=int(month_match.group(1)), source_text=month_match.group(0)))
    cohort_match = _COHORT.search(question)
    if cohort_match and cohort is not None:
        filters = [item for item in filters if item.column != str(cohort)]
        filters.append(FilterCondition(
            column=str(cohort), operator="eq", value=int(cohort_match.group(1)),
            source_text=cohort_match.group(0),
        ))

    # Explicit comparisons must keep their operator (for example, "10만원
    # 이상" is gte, never an exact-money filter).  The helper grounds only
    # money/year/month/cohort comparisons that have a matching schema role.
    comparison_filters = _comparison_filters(
        question, money=money, year=year, month=month, cohort=cohort,
    )
    comparison_columns = {item.column for item in comparison_filters}
    filters = [item for item in filters if item.column not in comparison_columns]
    filters.extend(comparison_filters)

    if money is not None:
        money_literals = [match.group(1) for match in _MONEY.finditer(question)]
        money_literals = [value for value in money_literals if parse_money_value(value) is not None and ("원" in value or parse_money_value(value) >= 10_000)]
        grounded_non_money_values = {
            _norm(item.value)
            for item in filters
            if item.column != str(money) and item.value is not None
        }
        money_literals = [
            value for value in money_literals
            if _norm(value) not in grounded_non_money_values
        ]
        if len(money_literals) == 1 and str(money) not in comparison_columns:
            filters.append(FilterCondition(column=str(money), operator="eq", value=money_literals[0], source_text=money_literals[0]))

    # One grounded person value is a subject filter; projection columns are never filters.
    if person is not None and not skip_literal_filter_scan:
        matched_person = _person_filter(df, person, question)
        if matched_person is not None:
            # Ingested tables can retain several row-scoped representations of
            # one person (original, display, search key, mask pattern). Literal
            # matching those columns independently creates contradictory AND
            # filters, especially when a mask fragment also occurs in a real
            # name. The canonical person column is the sole identity filter.
            person_columns = {
                str(column)
                for column in df.columns
                if (
                    (meaning := _meaning(df, column)).concept == "entity"
                    and meaning.qualifier == "person"
                )
            }
            filters = [item for item in filters if item.column not in person_columns]
            filters.append(matched_person)

    if operation_hint == "lookup_field":
        if not filters:
            return None
        filter_columns = {item.column for item in filters}
        projections = [column for column in requested if str(column) not in filter_columns]
        # A contact-value reverse lookup filters on email/phone and returns the
        # schema-derived person field. No document-specific header is assumed.
        if person is not None and str(person) not in filter_columns and re.search(r"누구|사람|회원", question):
            projections.insert(0, person)
        if not projections:
            return None
        select = tuple(dict.fromkeys((
            *((str(person),) if person is not None and str(person) in filter_columns else ()),
            *(str(column) for column in projections),
        )))
        return QueryPlan(status="ready", dataframe=alias, operation="list", filters=tuple(filters), select=select)

    people_count = bool(_PERSON_COUNT.search(question))
    ordinal_match = _ORDINAL.search(question)
    korean_ordinal_match = _KOREAN_ORDINAL.search(question)
    rank_position = (
        int(ordinal_match.group(1)) if ordinal_match
        else _KOREAN_ORDINAL_VALUES.get(korean_ordinal_match.group(1)) if korean_ordinal_match else None
    )
    limit_match = _LIMIT.search(question)
    explicit_limit = int(limit_match.group(1)) if limit_match else None
    group_order = (
        "asc" if _GROUP_SUM_MINIMUM.search(question)
        else "desc" if _GROUP_SUM_EXTREME.search(question)
        else None
    )
    if rank_position is not None and person is not None and money is not None and re.search(r"(?:사람|회원|인원|누구)", question):
        if _RANK_ASC.search(question) or re.search(r"(?:적게|작게|작은|낮게|낮은|최소)", question):
            group_order = "asc"
        elif _RANK_DESC.search(question) or re.search(r"(?:많이|많은|크게|큰|높게|높은|최대)", question):
            group_order = "desc"
        elif re.search(r"(?:\d+\s*(?:위|등))", question):
            group_order = "desc"
    if group_order is not None and person is not None and money is not None:
        return QueryPlan(
            status="ready",
            dataframe=alias,
            operation="group_sum",
            filters=tuple(filters),
            target=str(money),
            group_by=(str(person),),
            group_order=group_order,
            **({"rank_position": rank_position, "tie_policy": "dense"} if rank_position else {"top_n": explicit_limit or 1}),
        )
    # A plain "전공별 납부액" style request is a complete grouping, not a
    # top-person ranking.  Its grouping column is explicitly named in the
    # question and can be resolved from the schema without an LLM plan.
    group_columns = tuple(
        str(column) for column in requested
        if column not in {person, money}
    )
    if (
        money is not None
        and group_columns
        and _GROUPING_REQUEST.search(question)
        and re.search(r"(?:금액|돈|회비|결제|납부|후원|기부)", question)
    ):
        group_direction = (
            "asc" if _RANK_ASC.search(question)
            else "desc" if _RANK_DESC.search(question)
            else None
        )
        rank_kwargs = (
            {"rank_position": rank_position, "tie_policy": "dense"}
            if rank_position is not None and group_direction is not None
            else {"top_n": explicit_limit}
            if explicit_limit is not None and group_direction is not None
            else {}
        )
        return QueryPlan(
            status="ready", dataframe=alias, operation="group_sum",
            filters=tuple(filters), target=str(money), group_by=group_columns,
            **({"group_order": group_direction} if group_direction is not None else {}),
            **rank_kwargs,
        )
    if operation_hint == "count_records" or people_count or _ROW_COUNT.search(question):
        return QueryPlan(status="ready", dataframe=alias, operation="count", filters=tuple(filters), distinct_by=(str(person),) if people_count and person is not None else ())
    if _MEAN.search(question) and money is not None:
        return QueryPlan(status="ready", dataframe=alias, operation="mean", filters=tuple(filters), target=str(money))
    if _MODE.search(question) and money is not None:
        return QueryPlan(status="ready", dataframe=alias, operation="mode", filters=tuple(filters), target=str(money))
    if (operation_hint == "max_amount" or _MAX_VALUE.search(question)) and money is not None:
        return QueryPlan(status="ready", dataframe=alias, operation="max", filters=tuple(filters), target=str(money))
    if (operation_hint == "min_amount" or _MIN_VALUE.search(question)) and money is not None:
        return QueryPlan(status="ready", dataframe=alias, operation="min", filters=tuple(filters), target=str(money))
    if (
        operation_hint in {"sum_amount", "lookup_amount"}
        or (_SUM.search(question) and money is not None)
        or (_PAYMENT_EXISTENCE.search(question) and money is not None and any(item.column == str(person) for item in filters))
    ):
        if money is None:
            return None
        return QueryPlan(
            status="ready",
            dataframe=alias,
            operation="sum",
            filters=tuple(filters),
            target=str(money),
            result_mode="person_totals" if operation_hint == "lookup_amount" else None,
        )
    if operation_hint in {"list_records", "filter_records", "structured_query"}:
        direction = "desc" if _DESC_ORDER.search(question) else "asc" if _ASC_ORDER.search(question) else None
        if direction is None and rank_position is not None:
            if re.search(r"(?:큰|많은|높은|최대)", question):
                direction = "desc"
            elif re.search(r"(?:작은|적은|낮은|최소)", question):
                direction = "asc"
            elif money is not None and re.search(r"(?:\d+\s*(?:위|등))", question):
                direction = "desc"
        # Fields used only to state a condition (for example, "이메일 없는
        # 사람") are not implicit output columns.  Keep the person label for
        # readable lists and return only explicitly requested projections.
        filter_columns = {item.column for item in filters}
        select = tuple(dict.fromkeys((
            *((str(person),) if person is not None else ()),
            *(str(column) for column in requested if column != person and str(column) not in filter_columns),
        )))
        if (
            operation_hint != "list_records"
            and not filters
            and direction is None
            and not (broad_projection and select)
        ):
            return None
        sort_column = money if money is not None and re.search(r"(?:금액|돈|회비|결제|납부|후원|기부)", question) else temporal
        if direction is not None and sort_column is not None:
            return QueryPlan(
                status="ready", dataframe=alias, operation="list", filters=tuple(filters), select=select,
                sort=({"column": str(sort_column), "direction": direction},),
                **({"rank_position": rank_position, "tie_policy": "dense"} if rank_position else {"limit": explicit_limit} if explicit_limit else {}),
            )
        if rank_position is not None:
            return None
        return QueryPlan(status="ready", dataframe=alias, operation="list", filters=tuple(filters), select=select)
    return None


def build_auto_schema_grounded_plan(
    question: str,
    *,
    dataframes: Mapping[str, pd.DataFrame],
) -> tuple[str | None, QueryPlan | None]:
    """Choose and ground a fast PANDAS plan in one deterministic boundary.

    The operation labels remain the R.JSON contract, while this function owns
    the narrow keyword precedence formerly duplicated in ``main.py``.
    """
    # Cross-month ranges use the dedicated schema-aware date executor.  They
    # must not become independent year/month P.JSON filters because that can
    # silently collapse the range to its first month.
    date_spec = parse_date_filter(question)
    if (
        date_spec is not None
        and not date_spec.error
        and date_spec.start_month != date_spec.end_month
    ):
        return "structured_query", None
    normalized = re.sub(r"\s+", "", question).replace("번쨰", "번째")
    normalized = re.sub(r"\s+", "", question).replace("번쨰", "번째")
    broad_projection = bool(re.search(
        r"(?:전체(?:기록|내역|회원)|모든회원|각회원|회원별)", normalized,
    ))
    missing_request = bool(re.search(
        r"(?:비어있|안적|미입력|미등록|누락|공백|등록되지않|없(?:는|어|어?))",
        normalized,
    ))
    if (
        is_grounded_person_payment_existence_question(
            question, dataframes=dataframes
        )
        or is_grounded_person_amount_lookup_question(
            question, dataframes=dataframes
        )
    ):
        hints = ("lookup_amount", "structured_query")
    elif re.fullmatch(
        r"(?:표의?)?(?:전체|모든|전부)(?:데이터|기록|행|명단|목록|리스트)?"
        r"(?:보여줘|보여|알려줘|조회해줘|확인해줘)?[?!.]*",
        normalized,
    ):
        hints = ("list_records", "structured_query")
    elif broad_projection or missing_request:
        hints = ("structured_query",)
    elif any(token in normalized for token in ("등록날짜", "지급일", "날짜", "언제", "시기")):
        hints = ("lookup_field", "structured_query")
    elif re.search(
        r"(?:돈|금액|회비|결제|납부|후원|기부).{0,8}?"
        r"(?:냈|내었|냈어|냈나요|했어|했나|했나요)",
        normalized,
    ):
        hints = ("lookup_amount", "structured_query")
    elif (
        any(token in normalized for token in ("전화번호", "이메일"))
        and re.search(r"(?:얼마|돈|금액|회비|결제|납부|후원|기부)", normalized)
    ):
        hints = ("lookup_amount", "structured_query")
    elif any(token in normalized for token in ("전화번호", "이메일", "전공", "학과")):
        hints = ("lookup_field", "structured_query")
    elif re.search(r"(?:사람|인원|회원).*?(?:몇명|수)|몇명", normalized):
        hints = ("count_records", "structured_query")
    elif re.search(
        r"(?:최댓값|최대(?:값|액|금액)?|최고(?:값|액|금액)?|"
        r"(?:가장|제일)(?:큰|높은|많은)(?:값|금액|돈|액)|"
        r"(?:값|금액|돈|액).{0,8}?(?:가장|제일)(?:큰|높은|많은))",
        normalized,
    ) and not re.search(r"(?:사람|회원|인원|누구)", normalized):
        hints = ("max_amount", "structured_query")
    elif re.search(
        r"(?:최솟값|최소(?:값|액|금액)?|최저(?:값|액|금액)?|"
        r"(?:가장|제일)(?:작은|낮은)(?:값|금액|돈|액))",
        normalized,
    ):
        hints = ("min_amount", "structured_query")
    elif re.search(
        r"(?:오름차순|내림차순|순서대로|큰순|작은순|많은순|적은순|"
        r"\d+(?:번째|위|등)|첫번째|두번째|세번째|네번째|다섯번째|최신|가장이른)",
        normalized,
    ):
        hints = ("structured_query",)
    elif re.search(r"(?:총합|합계|총액|얼마|금액|돈)", normalized):
        hints = ("sum_amount", "structured_query")
    else:
        hints = ("structured_query",)

    for operation_hint in hints:
        plan = build_schema_grounded_plan(
            question,
            dataframes=dataframes,
            operation_hint=operation_hint,
        )
        if plan is not None:
            return operation_hint, plan
    return None, None
