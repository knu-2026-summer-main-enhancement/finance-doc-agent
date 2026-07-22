from __future__ import annotations

import re
import unicodedata
from typing import Hashable, Mapping

import pandas as pd

from pandas_engine.money import parse_money_value
from pandas_engine.plan_validator import column_data_type
from pandas_engine.query_plan import FilterCondition, QueryPlan
from utils.semantic_schema import infer_column_meaning


_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*년")
_YEAR_RANGE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*년?\s*(?:부터|에서|~|〜|-)\s*"
    r"((?:19|20)\d{2})\s*년?\s*(?:까지)?"
)
_MONTH = re.compile(r"(?<!\d)(1[0-2]|[1-9])\s*월")
_MONEY = re.compile(r"(?<!\d)(\d[\d,]*(?:\s*(?:만원|천원|원))?)(?!\d)")
_PERSON_COUNT = re.compile(r"(?:사람|인원|회원).*?(?:몇\s*명|수)|몇\s*명")
_SUM = re.compile(r"(?:총합|합계|총액|얼마(?:야|예|지|냈))")
_MEAN = re.compile(r"(?:평균|평균값|평균액)")
_MODE = re.compile(r"(?:최빈값|최빈액|가장\s*(?:흔한|많이\s*나온)\s*(?:값|금액)?)")
_ROW_COUNT = re.compile(r"(?:몇\s*번|몇\s*회|횟수)")
_MISSING = re.compile(r"(?:비어\s*있|안\s*적|미입력|누락|공백|없(?:는|어|어?))")
_PAYMENT_EXISTENCE = re.compile(
    r"(?:돈|금액|회비|결제|납부|후원|기부).{0,8}?(?:냈|내었|냈어|냈나요|했어|했나|했나요)"
)
_GROUP_SUM_EXTREME = re.compile(
    r"(?:가장|제일|최고).{0,16}?(?:많이|큰|높은).{0,12}?(?:사람|회원|인원)|"
    r"(?:사람|회원|인원).{0,12}?(?:가장|제일|최고).{0,16}?(?:많이|큰|높은)"
)
_GROUP_SUM_MINIMUM = re.compile(
    r"(?:가장|제일)\s*(?:돈|금액|회비|결제)?.{0,8}?(?:적게|작게|낮게|최소로).{0,8}?(?:낸|지급한|결제한)?\s*(?:사람|회원|인원)|"
    r"(?:돈|금액|회비|결제).{0,8}?(?:가장|제일)\s*(?:적게|작게|낮게).{0,8}?(?:낸|지급한|결제한)?\s*(?:사람|회원|인원)"
)
_PERSON_DISPLAY_SUFFIX = re.compile(r"\s*[\(\[\{][^\]\)\}]*[\]\)\}]\s*$")

# These are user-facing Korean equivalents for a semantic category, not names
# from any one document. The selected column still has to be inferred from the
# dataframe schema before it can be used in a plan.
_SEMANTIC_REQUEST_TERMS = {
    "department": ("학과", "학부", "전공", "계열", "무슨 과"),
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
    normalized_question: str,
) -> FilterCondition | None:
    """Ground an exact name or one unambiguous display-name base to a person row."""
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
    return None


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
    year = _first(df, lambda item: item.role == "year")
    month = _first(df, lambda item: item.role == "month")
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
        matched_person = _person_filter(df, person, normalized_question)
        if matched_person is not None and not any(item.column == str(person) for item in filters):
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
    group_order = (
        "asc" if _GROUP_SUM_MINIMUM.search(question)
        else "desc" if _GROUP_SUM_EXTREME.search(question)
        else None
    )
    if group_order is not None and person is not None and money is not None:
        return QueryPlan(
            status="ready",
            dataframe=alias,
            operation="group_sum",
            filters=tuple(filters),
            target=str(money),
            group_by=(str(person),),
            group_order=group_order,
            top_n=1,
        )
    if operation_hint == "count_records" or people_count or _ROW_COUNT.search(question):
        return QueryPlan(status="ready", dataframe=alias, operation="count", filters=tuple(filters), distinct_by=(str(person),) if people_count and person is not None else ())
    if _MEAN.search(question) and money is not None:
        return QueryPlan(status="ready", dataframe=alias, operation="mean", filters=tuple(filters), target=str(money))
    if _MODE.search(question) and money is not None:
        return QueryPlan(status="ready", dataframe=alias, operation="mode", filters=tuple(filters), target=str(money))
    if (
        operation_hint in {"sum_amount", "lookup_amount"}
        or (_SUM.search(question) and money is not None)
        or (_PAYMENT_EXISTENCE.search(question) and money is not None and any(item.column == str(person) for item in filters))
    ):
        if money is None:
            return None
        return QueryPlan(status="ready", dataframe=alias, operation="sum", filters=tuple(filters), target=str(money))
    if operation_hint in {"list_records", "filter_records", "structured_query"}:
        if operation_hint != "list_records" and not filters:
            return None
        select = tuple(dict.fromkeys((
            *((str(person),) if person is not None else ()),
            *(str(column) for column in requested if column != person),
        )))
        return QueryPlan(status="ready", dataframe=alias, operation="list", filters=tuple(filters), select=select)
    return None
