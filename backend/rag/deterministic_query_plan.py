from __future__ import annotations

import re
import unicodedata
from typing import Hashable, Mapping

import pandas as pd

from pandas_engine.money import parse_money_value
from pandas_engine.plan_validator import column_data_type
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
_MISSING = re.compile(r"(?:비어\s*있|안\s*적|미입력|누락|공백|없(?:는|어|어?))")
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
_ORDINAL = re.compile(r"(?<!\d)(\d+)\s*(?:번째|째)")
_KOREAN_ORDINAL = re.compile(
    r"(첫|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*"
    r"(?:번째|번\s*[째쨰])"
)
_KOREAN_ORDINAL_VALUES = {"첫": 1, "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
_LIMIT = re.compile(r"(?<!\d)(\d+)\s*(?:건|개|명)(?!\d)")
_DESC_ORDER = re.compile(r"(?:내림차순|큰\s*순(?:서(?:대로)?)?|많은\s*순(?:서(?:대로)?)?|높은\s*순(?:서(?:대로)?)?|최신|늦은|가장\s*(?:큰|많은|높은))")
_ASC_ORDER = re.compile(r"(?:오름차순|작은\s*순(?:서(?:대로)?)?|적은\s*순(?:서(?:대로)?)?|낮은\s*순(?:서(?:대로)?)?|빠른|이른|가장\s*(?:작은|적은|낮은))")
_PERSON_DISPLAY_SUFFIX = re.compile(r"\s*[\(\[\{][^\]\)\}]*[\]\)\}]\s*$")
_NON_PERSON_LEADING_TOKENS = {"가장", "제일", "최고", "최저", "사람", "회원", "누구"}

# These are user-facing Korean equivalents for a semantic category, not names
# from any one document. The selected column still has to be inferred from the
# dataframe schema before it can be used in a plan.
_SEMANTIC_REQUEST_TERMS = {
    "department": ("학과", "학부", "전공", "계열", "무슨 과"),
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
        requested.extend(
            _columns(
                df,
                lambda meaning: (
                    meaning.concept == "category"
                    and meaning.role == "category"
                    and meaning.qualifier == qualifier
                ),
            ),
        )
        if qualifier == "payment_time":
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
    money = _first(df, lambda item: item.role == "amount" or item.data_type == "money")
    temporal = _first(df, lambda item: item.data_type == "date" or item.role in {"date", "registered_date"})
    year = _first(df, lambda item: item.role == "year")
    month = _first(df, lambda item: item.role == "month")
    cohort = _first(
        df,
        lambda item: item.concept == "category" and item.qualifier == "cohort",
    )
    normalized_question = _norm(question)
    filters = _value_filters(df, question)

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
        if len(money_literals) == 1:
            filters.append(FilterCondition(column=str(money), operator="eq", value=money_literals[0], source_text=money_literals[0]))

    # One grounded person value is a subject filter; projection columns are never filters.
    if person is not None:
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
        if re.search(r"(?:적게|작게|작은|낮게|낮은|최소)", question):
            group_order = "asc"
        elif re.search(r"(?:많이|많은|크게|큰|높게|높은|최대)", question):
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
        if operation_hint != "list_records" and not filters and direction is None:
            return None
        select = tuple(dict.fromkeys((
            *((str(person),) if person is not None else ()),
            *(str(column) for column in requested if column != person),
        )))
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
