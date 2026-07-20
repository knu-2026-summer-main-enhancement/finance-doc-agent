from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


PlanStatus = Literal["ready", "clarification", "not_applicable"]
QueryOperation = Literal[
    "list",
    "count",
    "sum",
    "mean",
    "median",
    "mode",
    "min",
    "max",
]
FilterOperator = Literal[
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "in",
    "between",
    "is_null",
    "not_null",
]
ScalarValue = str | int | float | bool
FilterValue = ScalarValue | tuple[ScalarValue, ...] | None


class _PlanModel(BaseModel):
    """Common strict settings for every object emitted by the query planner."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class FilterCondition(_PlanModel):
    """One filter over an actual DataFrame column.

    Column existence and data-type compatibility are checked later against the
    selected DataFrame. This model only validates the shape of the plan itself.
    """

    column: str = Field(min_length=1)
    operator: FilterOperator
    value: FilterValue = None
    case_sensitive: bool = False
    source_text: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Exact, smallest question span supporting this filter.",
    )

    @model_validator(mode="after")
    def validate_operator_value(self) -> Self:
        if self.operator in {"is_null", "not_null"}:
            if self.value is not None:
                raise ValueError(f"{self.operator} must not include a value")
            return self

        if self.value is None:
            raise ValueError(f"{self.operator} requires a value")

        if self.operator == "between":
            if not isinstance(self.value, tuple) or len(self.value) != 2:
                raise ValueError("between requires exactly two values")
            return self

        if self.operator == "in":
            if not isinstance(self.value, tuple) or not self.value:
                raise ValueError("in requires a non-empty value list")
            return self

        if isinstance(self.value, tuple):
            raise ValueError(f"{self.operator} requires one scalar value")
        if self.operator == "contains" and not isinstance(self.value, str):
            raise ValueError("contains requires a string value")
        return self


class SortCondition(_PlanModel):
    column: str = Field(min_length=1)
    direction: Literal["asc", "desc"] = "asc"


class QueryPlan(_PlanModel):
    """Validated language-independent contract for structured document queries.

    This is intentionally independent of scholarship or donation column names.
    A later runtime validator will compare every column reference with the
    selected DataFrame and its semantic schema before execution.
    """

    status: PlanStatus
    dataframe: str | None = Field(
        default=None,
        description="Alias of a DataFrame inside the currently selected document scope.",
    )
    operation: QueryOperation | None = None
    filters: tuple[FilterCondition, ...] = Field(default_factory=tuple, max_length=20)
    filter_logic: Literal["all", "any"] = "all"
    select: tuple[str, ...] = Field(default_factory=tuple, max_length=30)
    target: str | None = None
    result_mode: Literal["value", "records"] | None = None
    sort: tuple[SortCondition, ...] = Field(default_factory=tuple, max_length=10)
    distinct_by: tuple[str, ...] = Field(default_factory=tuple, max_length=10)
    limit: int | None = Field(default=None, ge=1, le=500)
    top_n: int | None = Field(default=None, ge=1, le=100)
    message: str | None = Field(default=None, max_length=500)
    candidates: tuple[str, ...] = Field(default_factory=tuple, max_length=20)

    @property
    def effective_result_mode(self) -> Literal["value", "records"] | None:
        if self.operation == "list":
            return "records"
        if self.operation in {"count", "sum", "mean", "median", "mode"}:
            return "value"
        if self.operation in {"min", "max"}:
            return self.result_mode or "value"
        return None

    @property
    def effective_limit(self) -> int | None:
        return (self.limit or 100) if self.operation == "list" else None

    @property
    def effective_top_n(self) -> int | None:
        if self.operation in {"min", "max"} and self.effective_result_mode == "records":
            return self.top_n or 1
        return None

    @model_validator(mode="after")
    def validate_status_and_operation(self) -> Self:
        if self.status != "ready":
            if not self.message:
                raise ValueError(f"{self.status} requires a user-facing message")
            if self.operation is not None or self.dataframe is not None:
                raise ValueError(
                    f"{self.status} must not include a dataframe or operation"
                )
            if (
                self.filters
                or self.select
                or self.target is not None
                or self.sort
                or self.distinct_by
                or self.limit is not None
                or self.top_n is not None
                or self.result_mode is not None
            ):
                raise ValueError(f"{self.status} must not include execution fields")
            return self

        if not self.dataframe:
            raise ValueError("ready requires a dataframe")
        if self.operation is None:
            raise ValueError("ready requires an operation")

        scalar_operations = {"count", "sum", "mean", "median", "mode"}
        targeted_operations = {"sum", "mean", "median", "mode", "min", "max"}

        if self.operation == "list":
            if self.target is not None:
                raise ValueError("list must not include a target")
            if self.result_mode not in {None, "records"}:
                raise ValueError("list must return records")
            if self.top_n is not None:
                raise ValueError("list uses limit instead of top_n")
            return self

        if self.operation in targeted_operations and not self.target:
            raise ValueError(f"{self.operation} requires a target column")

        if self.operation in scalar_operations:
            if self.result_mode not in {None, "value"}:
                raise ValueError(f"{self.operation} must return a value")
            if self.select or self.sort or self.limit is not None or self.top_n is not None:
                raise ValueError(
                    f"{self.operation} must not include record-returning fields"
                )
            return self

        # min/max may return only the extreme value or the matching record(s).
        if self.effective_result_mode == "value":
            if self.select or self.sort or self.limit is not None or self.top_n is not None:
                raise ValueError(
                    f"{self.operation} value mode must not include record-returning fields"
                )
        else:
            if self.limit is not None:
                raise ValueError(
                    f"{self.operation} records mode uses top_n instead of limit"
                )
        return self
