from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Mapping

import pandas as pd

from datastore.scope import scoped_mapping
from datastore.state import _df_namespace, _df_sources
from pandas_engine.money import parse_money_value
from pandas_engine.query_plan import FilterCondition, QueryPlan
from utils.semantic_schema import SYSTEM_COLUMNS, infer_data_type
from utils.table_parser import IDENTITY_INTERNAL_COLS


ValidationStatus = Literal[
    "valid",
    "clarification",
    "not_applicable",
    "invalid",
]

_INTERNAL_COLUMNS = set(SYSTEM_COLUMNS) | set(IDENTITY_INTERNAL_COLS)
_NUMERIC_TYPES = {"number", "money"}
_ORDERABLE_TYPES = _NUMERIC_TYPES | {"date", "year_month"}
_STRING_TYPES = {"string"}


@dataclass(frozen=True)
class PlanValidationIssue:
    code: str
    message: str
    field: str = ""
    column: str = ""


@dataclass(frozen=True)
class PlanValidationResult:
    status: ValidationStatus
    plan: QueryPlan
    issues: tuple[PlanValidationIssue, ...] = ()
    dataframe: pd.DataFrame | None = None
    source_file: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"

    @property
    def is_accepted(self) -> bool:
        return self.status in {"valid", "clarification", "not_applicable"}

    @property
    def is_executable(self) -> bool:
        return self.status == "valid" and self.dataframe is not None


def _issue(
    code: str,
    message: str,
    *,
    field: str = "",
    column: str = "",
) -> PlanValidationIssue:
    return PlanValidationIssue(
        code=code,
        message=message,
        field=field,
        column=column,
    )


def column_data_type(df: pd.DataFrame, column: Hashable) -> str:
    schema = df.attrs.get("semantic_schema")
    if isinstance(schema, dict):
        columns = schema.get("columns")
        mapping = columns.get(str(column)) if isinstance(columns, dict) else None
        if isinstance(mapping, dict):
            data_type = str(mapping.get("data_type") or "").strip().casefold()
            if data_type:
                return data_type
    return infer_data_type(df[column])


def _is_internal_column(column: str) -> bool:
    return column in _INTERNAL_COLUMNS or column.startswith("_")


def _referenced_columns(plan: QueryPlan) -> list[tuple[str, str]]:
    references: list[tuple[str, str]] = []
    if plan.target:
        references.append(("target", plan.target))
    references.extend(("select", column) for column in plan.select)
    references.extend(("filters", condition.column) for condition in plan.filters)
    references.extend(("sort", condition.column) for condition in plan.sort)
    references.extend(("distinct_by", column) for column in plan.distinct_by)
    return references


def _condition_values(condition: FilterCondition) -> list[object]:
    if condition.value is None:
        return []
    if isinstance(condition.value, tuple):
        return list(condition.value)
    return [condition.value]


def _values_match_type(condition: FilterCondition, data_type: str) -> bool:
    values = _condition_values(condition)
    if not values:
        return True

    if data_type == "money":
        return all(parse_money_value(value) is not None for value in values)
    if data_type == "number":
        converted = pd.to_numeric(pd.Series(values), errors="coerce")
        return bool(converted.notna().all())
    if data_type in {"date", "year_month"}:
        converted = pd.to_datetime(pd.Series(values), errors="coerce")
        return bool(converted.notna().all())
    if data_type == "boolean":
        return all(isinstance(value, bool) for value in values)
    return True


def _validate_filter_type(
    condition: FilterCondition,
    data_type: str,
) -> list[PlanValidationIssue]:
    issues: list[PlanValidationIssue] = []
    operator = condition.operator

    if operator == "contains" and data_type not in _STRING_TYPES:
        issues.append(
            _issue(
                "incompatible_operator",
                f"contains 연산은 문자열 컬럼에만 사용할 수 있습니다: {condition.column}",
                field="filters",
                column=condition.column,
            )
        )
        return issues

    if operator in {"gt", "gte", "lt", "lte", "between"} and data_type not in _ORDERABLE_TYPES:
        issues.append(
            _issue(
                "incompatible_operator",
                f"{operator} 연산을 적용할 수 없는 컬럼 형식입니다: {condition.column}",
                field="filters",
                column=condition.column,
            )
        )
        return issues

    if not _values_match_type(condition, data_type):
        issues.append(
            _issue(
                "incompatible_value",
                f"필터 값이 컬럼 형식과 맞지 않습니다: {condition.column}",
                field="filters",
                column=condition.column,
            )
        )
    return issues


def validate_query_plan(
    plan: QueryPlan,
    *,
    dataframes: Mapping[str, pd.DataFrame] | None = None,
    source_by_alias: Mapping[str, str] | None = None,
    explicit_dataframe_aliases: set[str] | frozenset[str] | None = None,
) -> PlanValidationResult:
    """Validate a parsed QueryPlan against the current document scope.

    This function never repairs an LLM plan. Any mismatch is returned as an
    explicit issue so an invalid or hallucinated reference cannot be executed.
    """

    if plan.status == "clarification":
        return PlanValidationResult(status="clarification", plan=plan)
    if plan.status == "not_applicable":
        return PlanValidationResult(status="not_applicable", plan=plan)

    all_dataframes = _df_namespace if dataframes is None else dataframes
    all_sources = _df_sources if source_by_alias is None else source_by_alias
    available = scoped_mapping(all_dataframes, all_sources)
    alias = str(plan.dataframe)

    if alias not in available:
        if alias in all_dataframes:
            issue = _issue(
                "dataframe_out_of_scope",
                f"선택한 문서 범위 밖의 데이터프레임입니다: {alias}",
                field="dataframe",
            )
        else:
            issue = _issue(
                "unknown_dataframe",
                f"존재하지 않는 데이터프레임입니다: {alias}",
                field="dataframe",
            )
        return PlanValidationResult(
            status="invalid",
            plan=plan,
            issues=(issue,),
        )

    df = available[alias]
    source_file = all_sources.get(alias, alias)

    available_sources = {
        str(all_sources.get(available_alias, available_alias))
        for available_alias in available
    }
    if len(available_sources) > 1:
        explicit_aliases = set(explicit_dataframe_aliases or ())
        if explicit_aliases != {alias}:
            issue = _issue(
                "ambiguous_document_scope",
                "여러 문서가 조회 대상입니다. 조회할 문서를 하나 선택해 주세요.",
                field="dataframe",
            )
            return PlanValidationResult(
                status="clarification",
                plan=plan,
                issues=(issue,),
            )

    actual_columns = {str(column): column for column in df.columns}
    issues: list[PlanValidationIssue] = []
    valid_references: dict[str, Hashable] = {}

    for field, column in _referenced_columns(plan):
        if _is_internal_column(column):
            issues.append(
                _issue(
                    "internal_column",
                    f"내부 시스템 컬럼은 조회할 수 없습니다: {column}",
                    field=field,
                    column=column,
                )
            )
        elif column not in actual_columns:
            issues.append(
                _issue(
                    "unknown_column",
                    f"실제 표에 존재하지 않는 컬럼입니다: {column}",
                    field=field,
                    column=column,
                )
            )
        else:
            valid_references[column] = actual_columns[column]

    if issues:
        return PlanValidationResult(
            status="invalid",
            plan=plan,
            issues=tuple(issues),
            dataframe=df,
            source_file=source_file,
        )

    if plan.operation in {"sum", "mean", "median"} and plan.target:
        target_type = column_data_type(df, valid_references[plan.target])
        if target_type not in _NUMERIC_TYPES:
            issues.append(
                _issue(
                    "incompatible_target",
                    f"{plan.operation} 연산에는 숫자 또는 금액 컬럼이 필요합니다: {plan.target}",
                    field="target",
                    column=plan.target,
                )
            )

    if plan.operation in {"min", "max"} and plan.target:
        target_type = column_data_type(df, valid_references[plan.target])
        if target_type not in _ORDERABLE_TYPES:
            issues.append(
                _issue(
                    "incompatible_target",
                    f"{plan.operation} 연산에 사용할 수 없는 컬럼 형식입니다: {plan.target}",
                    field="target",
                    column=plan.target,
                )
            )

    for condition in plan.filters:
        data_type = column_data_type(df, valid_references[condition.column])
        issues.extend(_validate_filter_type(condition, data_type))

    if issues:
        return PlanValidationResult(
            status="invalid",
            plan=plan,
            issues=tuple(issues),
            dataframe=df,
            source_file=source_file,
        )

    return PlanValidationResult(
        status="valid",
        plan=plan,
        dataframe=df,
        source_file=source_file,
    )
