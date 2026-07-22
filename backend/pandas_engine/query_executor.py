from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import pandas as pd

from pandas_engine.date_filter import _to_datetime
from pandas_engine.money import (
    money_series,
    money_unit_for_column,
    parse_money_value,
)
from pandas_engine.plan_validator import (
    PlanValidationResult,
    column_data_type,
)
from pandas_engine.query_plan import (
    FilterCondition,
    QueryPlan,
    ScalarValue,
    SortCondition,
)
from utils.semantic_schema import SYSTEM_COLUMNS, infer_column_meaning
from utils.table_parser import IDENTITY_INTERNAL_COLS, normalize_person_name


_INTERNAL_COLUMNS = set(SYSTEM_COLUMNS) | set(IDENTITY_INTERNAL_COLS)


class QueryPlanExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueryExecutionEvidence:
    dataframe_alias: str
    source_file: str
    filter_logic: str
    filters: tuple[FilterCondition, ...]
    sort: tuple[SortCondition, ...]
    distinct_by: tuple[str, ...]
    group_by: tuple[str, ...]
    source_rows: int
    filtered_rows: int
    unique_people: int | None
    limit: int | None
    top_n: int | None
    rank_position: int | None
    tie_policy: str | None


@dataclass(frozen=True)
class QueryExecutionResult:
    operation: str
    value: object
    source_file: str
    target: str | None
    target_data_type: str | None
    matched_rows: int
    valid_rows: int
    excluded_rows: int
    evidence: QueryExecutionEvidence
    # Exact post-filter/post-distinct rows, retained for structured UI evidence.
    matched_frame: pd.DataFrame
    available_rank_count: int | None = None


def _is_internal_column(column: object) -> bool:
    text = str(column)
    return text in _INTERNAL_COLUMNS or text.startswith("_")


def _actual_column(df: pd.DataFrame, planned_column: str) -> Hashable:
    for column in df.columns:
        if str(column) == planned_column:
            return column
    raise QueryPlanExecutionError(f"검증된 컬럼을 찾을 수 없습니다: {planned_column}")


def _is_person_name_column(df: pd.DataFrame, column: Hashable) -> bool:
    schema = df.attrs.get("semantic_schema")
    if isinstance(schema, dict):
        columns = schema.get("columns")
        mapping = columns.get(str(column)) if isinstance(columns, dict) else None
        if isinstance(mapping, dict):
            if (
                mapping.get("concept") == "entity"
                and mapping.get("role") == "entity_name"
                and mapping.get("qualifier") == "person"
            ):
                return True
    meaning = infer_column_meaning(str(column), df[column])
    return (
        meaning.concept == "entity"
        and meaning.role == "entity_name"
        and meaning.qualifier == "person"
    )


def _typed_series(df: pd.DataFrame, column: Hashable) -> pd.Series:
    data_type = column_data_type(df, column)
    if data_type == "money":
        return money_series(df, str(column))
    if data_type == "number":
        return pd.to_numeric(df[column], errors="coerce")
    if data_type in {"date", "year_month"}:
        return _to_datetime(df[column])
    if data_type == "boolean":
        return df[column].astype("boolean")

    text = df[column].astype("string").str.strip()
    if _is_person_name_column(df, column):
        return text.map(normalize_person_name).astype("string")
    return text


def _typed_value(
    df: pd.DataFrame,
    column: Hashable,
    value: ScalarValue,
) -> object:
    data_type = column_data_type(df, column)
    if data_type == "money":
        return parse_money_value(value, money_unit_for_column(df, str(column)))
    if data_type == "number":
        return float(pd.to_numeric(pd.Series([value]), errors="raise").iloc[0])
    if data_type in {"date", "year_month"}:
        return pd.to_datetime(value, errors="raise")
    if data_type == "boolean":
        return bool(value)
    if _is_person_name_column(df, column):
        return normalize_person_name(value)
    return str(value).strip()


def _filter_mask(df: pd.DataFrame, condition: FilterCondition) -> pd.Series:
    column = _actual_column(df, condition.column)
    series = _typed_series(df, column)
    operator = condition.operator

    if operator in {"is_null", "not_null"}:
        # Spreadsheet exports often represent a missing text cell as an empty
        # string or textual null marker rather than a pandas NaN.
        missing = series.isna()
        if column_data_type(df, column) == "string":
            text = series.astype("string").str.strip().str.casefold()
            missing = missing | text.isin({"", "none", "nan", "nat", "null"}).fillna(False)
        return missing if operator == "is_null" else ~missing

    raw_values = (
        condition.value
        if isinstance(condition.value, tuple)
        else [condition.value]
    )
    values = [
        _typed_value(df, column, value)
        for value in raw_values
        if value is not None
    ]

    if operator == "contains":
        needle = str(values[0])
        return series.astype("string").str.contains(
            needle,
            case=condition.case_sensitive,
            regex=False,
            na=False,
        )
    if operator == "in":
        return series.isin(values).fillna(False)
    if operator == "between":
        return series.between(values[0], values[1], inclusive="both").fillna(False)

    value = values[0]
    if operator == "eq":
        if (
            not condition.case_sensitive
            and column_data_type(df, column) == "string"
            and not _is_person_name_column(df, column)
        ):
            return series.str.casefold().eq(str(value).casefold()).fillna(False)
        return series.eq(value).fillna(False)
    if operator == "ne":
        if (
            not condition.case_sensitive
            and column_data_type(df, column) == "string"
            and not _is_person_name_column(df, column)
        ):
            return series.str.casefold().ne(str(value).casefold()).fillna(False)
        return series.ne(value).fillna(False)
    if operator == "gt":
        return series.gt(value).fillna(False)
    if operator == "gte":
        return series.ge(value).fillna(False)
    if operator == "lt":
        return series.lt(value).fillna(False)
    if operator == "lte":
        return series.le(value).fillna(False)
    raise QueryPlanExecutionError(f"지원하지 않는 필터 연산입니다: {operator}")


def _apply_filters(df: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    if not plan.filters:
        return df.copy()

    initial = plan.filter_logic == "all"
    combined = pd.Series(initial, index=df.index, dtype=bool)
    for condition in plan.filters:
        mask = _filter_mask(df, condition).astype(bool)
        combined = combined & mask if plan.filter_logic == "all" else combined | mask
    return df[combined].copy()


def _sort_rows(df: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    result = df.copy()
    # Stable sorts in reverse priority preserve the order declared in the plan.
    for condition in reversed(plan.sort):
        column = _actual_column(result, condition.column)
        key = _typed_series(result, column)
        ordered_index = key.sort_values(
            ascending=condition.direction == "asc",
            na_position="last",
            kind="stable",
        ).index
        result = result.loc[ordered_index]
    return result


def _visible_columns(df: pd.DataFrame) -> list[Hashable]:
    return [column for column in df.columns if not _is_internal_column(column)]


def _select_records(df: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    if plan.select:
        columns = [_actual_column(df, column) for column in plan.select]
    else:
        columns = _visible_columns(df)
    result = df.loc[:, columns].copy()
    result.attrs.update(df.attrs)
    return result


def _drop_plan_duplicates(df: pd.DataFrame, plan: QueryPlan) -> pd.DataFrame:
    if not plan.distinct_by or df.empty:
        return df
    normalized_keys = pd.DataFrame(
        {
            f"key_{index}": _typed_series(
                df,
                _actual_column(df, column),
            )
            for index, column in enumerate(plan.distinct_by)
        },
        index=df.index,
    )
    keep = ~normalized_keys.duplicated(keep="first")
    return df[keep].copy()


def _target_series(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.Series, Hashable]:
    column = _actual_column(df, target)
    return _typed_series(df, column), column


def execute_query_plan(validation: PlanValidationResult) -> QueryExecutionResult:
    """Execute only a QueryPlan that already passed runtime validation."""

    if not validation.is_executable or validation.dataframe is None:
        details = ", ".join(issue.code for issue in validation.issues) or validation.status
        raise QueryPlanExecutionError(
            f"검증을 통과하지 않은 QueryPlan은 실행할 수 없습니다: {details}"
        )

    plan = validation.plan
    df = validation.dataframe
    source_file = validation.source_file or str(plan.dataframe)
    filtered = _apply_filters(df, plan)
    filtered_rows = int(len(filtered))

    filtered = _drop_plan_duplicates(filtered, plan)

    matched_rows = int(len(filtered))
    person_columns = [
        column for column in filtered.columns if _is_person_name_column(filtered, column)
    ]
    unique_people = (
        max(int(filtered[column].dropna().nunique()) for column in person_columns)
        if person_columns
        else None
    )
    evidence = QueryExecutionEvidence(
        dataframe_alias=str(plan.dataframe),
        source_file=source_file,
        filter_logic=plan.filter_logic,
        filters=plan.filters,
        sort=plan.sort,
        distinct_by=plan.distinct_by,
        group_by=plan.group_by,
        source_rows=int(len(df)),
        filtered_rows=filtered_rows,
        unique_people=unique_people,
        limit=plan.effective_limit,
        top_n=plan.effective_top_n,
        rank_position=plan.rank_position,
        tie_policy=plan.tie_policy,
    )

    if plan.operation == "list":
        rows = _sort_rows(filtered, plan)
        available_rank_count = None
        if plan.rank_position is not None:
            rank_column = _actual_column(rows, plan.sort[0].column)
            ranked_values = _typed_series(rows, rank_column).dropna()
            # Dense ranking: the Nth distinct sorted value returns every tied row.
            distinct_values = ranked_values.drop_duplicates().tolist()
            available_rank_count = len(distinct_values)
            if len(distinct_values) < plan.rank_position:
                rows = rows.iloc[0:0]
            else:
                rows = rows[_typed_series(rows, rank_column).eq(distinct_values[plan.rank_position - 1])]
        elif plan.effective_limit is not None:
            rows = rows.head(plan.effective_limit)
        rows = _select_records(rows, plan)
        return QueryExecutionResult(
            operation="list",
            value=rows,
            source_file=source_file,
            target=None,
            target_data_type=None,
            matched_rows=matched_rows,
            valid_rows=matched_rows,
            excluded_rows=0,
            evidence=evidence,
            matched_frame=filtered,
            available_rank_count=available_rank_count,
        )

    if plan.operation == "count":
        target_data_type = None
        if plan.target:
            target, target_column = _target_series(filtered, plan.target)
            target_data_type = column_data_type(filtered, target_column)
            value = int(target.notna().sum())
            valid_rows = value
        else:
            value = matched_rows
            valid_rows = matched_rows
        return QueryExecutionResult(
            operation="count",
            value=value,
            source_file=source_file,
            target=plan.target,
            target_data_type=target_data_type,
            matched_rows=matched_rows,
            valid_rows=valid_rows,
            excluded_rows=matched_rows - valid_rows,
            evidence=evidence,
            matched_frame=filtered,
        )

    if plan.operation == "group_sum":
        if not plan.target or not plan.group_by:
            raise QueryPlanExecutionError("group_sum에는 대상 컬럼과 그룹 컬럼이 필요합니다.")
        target, target_column = _target_series(filtered, plan.target)
        target_data_type = column_data_type(filtered, target_column)
        group_columns = [_actual_column(filtered, column) for column in plan.group_by]
        grouped_input = pd.DataFrame(index=filtered.index)
        for column in group_columns:
            grouped_input[str(column)] = _typed_series(filtered, column)
        grouped_input[str(plan.target)] = target
        valid_mask = grouped_input[str(plan.target)].notna()
        for column in group_columns:
            valid_mask &= grouped_input[str(column)].notna()
        valid_input = grouped_input[valid_mask]
        valid_rows = int(len(valid_input))
        excluded_rows = matched_rows - valid_rows
        available_rank_count = 0 if plan.rank_position is not None else None
        if valid_input.empty:
            ranked = pd.DataFrame(columns=[*plan.group_by, plan.target])
        else:
            ranked = (
                valid_input.groupby(list(plan.group_by), dropna=False, sort=False)[str(plan.target)]
                .sum()
                .reset_index()
                .sort_values(
                    by=str(plan.target),
                    ascending=(plan.group_order or "desc") == "asc",
                    kind="stable",
                )
            )
            if plan.rank_position is not None:
                distinct_values = ranked[str(plan.target)].drop_duplicates().tolist()
                available_rank_count = len(distinct_values)
                ranked = (
                    ranked[ranked[str(plan.target)].eq(distinct_values[plan.rank_position - 1])]
                    if len(distinct_values) >= plan.rank_position else ranked.iloc[0:0]
                )
            else:
                top_n = plan.effective_top_n or 1
                if top_n == 1 and not ranked.empty:
                    extreme = ranked.iloc[0][str(plan.target)]
                    ranked = ranked[ranked[str(plan.target)].eq(extreme)]
                else:
                    ranked = ranked.head(top_n)
        ranked.attrs.update(filtered.attrs)
        return QueryExecutionResult(
            operation="group_sum",
            value=ranked,
            source_file=source_file,
            target=plan.target,
            target_data_type=target_data_type,
            matched_rows=matched_rows,
            valid_rows=valid_rows,
            excluded_rows=excluded_rows,
            evidence=evidence,
            matched_frame=filtered,
            available_rank_count=available_rank_count,
        )

    if not plan.target:
        raise QueryPlanExecutionError(f"{plan.operation} 연산에 대상 컬럼이 없습니다.")

    target, target_column = _target_series(filtered, plan.target)
    target_data_type = column_data_type(filtered, target_column)
    valid = target.dropna()
    valid_rows = int(valid.size)
    excluded_rows = matched_rows - valid_rows

    if valid.empty:
        value: object = [] if plan.operation == "mode" else None
    elif plan.operation == "sum":
        value = float(valid.sum())
    elif plan.operation == "mean":
        value = float(valid.mean())
    elif plan.operation == "median":
        value = float(valid.median())
    elif plan.operation == "mode":
        value = valid.mode().tolist()
    elif plan.operation in {"min", "max"} and plan.effective_result_mode == "value":
        value = valid.min() if plan.operation == "min" else valid.max()
        if hasattr(value, "item"):
            value = value.item()
    elif plan.operation in {"min", "max"}:
        preordered = _sort_rows(filtered, plan)
        ranking, _ = _target_series(preordered, plan.target)
        ranked_index = ranking.sort_values(
            ascending=plan.operation == "min",
            na_position="last",
            kind="stable",
        ).dropna().index
        ranked_rows = preordered.loc[ranked_index]
        ranked_values, _ = _target_series(ranked_rows, plan.target)
        top_n = plan.effective_top_n or 1
        if top_n == 1:
            # A single extreme keeps every row tied for first place.
            extreme = ranked_values.iloc[0]
            rows = ranked_rows[ranked_values.eq(extreme)]
        else:
            # Top N means N records, not N distinct value bands. The latter can
            # return far more records than the user requested when values tie.
            rows = ranked_rows.head(top_n)
        value = _select_records(rows, plan)
    else:
        raise QueryPlanExecutionError(f"지원하지 않는 연산입니다: {plan.operation}")

    return QueryExecutionResult(
        operation=str(plan.operation),
        value=value,
        source_file=source_file,
        target=plan.target,
        target_data_type=target_data_type,
        matched_rows=matched_rows,
        valid_rows=valid_rows,
        excluded_rows=excluded_rows,
        evidence=evidence,
        matched_frame=filtered,
    )
