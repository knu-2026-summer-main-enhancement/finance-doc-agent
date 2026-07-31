from __future__ import annotations

from typing import Literal


_ENGINE_BY_OPERATION = {
    "list_documents": "DOCUMENTS",
    "filter_records": "PANDAS",
    "compare": "PANDAS",
    "max_person_by_amount": "PANDAS",
    "min_person_by_amount": "PANDAS",
    "list_records": "PANDAS",
    "count_records": "PANDAS",
    "sum_amount": "PANDAS",
    "average_amount": "PANDAS",
    "median_amount": "PANDAS",
    "mode_amount": "PANDAS",
    "max_amount": "PANDAS",
    "min_amount": "PANDAS",
    "lookup_amount": "PANDAS",
    "lookup_field": "PANDAS",
    "structured_query": "PANDAS",
    "document_reason": "VECTOR",
    "document_purpose": "VECTOR",
    "document_criteria": "VECTOR",
    "document_procedure": "VECTOR",
    "document_explain": "VECTOR",
}

PandasStrategy = Literal["DIRECT", "QUERY_PLAN"]


def engines_for_operations(
    operations: list[str] | tuple[str, ...],
) -> list[str]:
    """Map semantic operations to unique execution engines."""

    return list(dict.fromkeys(
        _ENGINE_BY_OPERATION[operation]
        for operation in operations
        if operation in _ENGINE_BY_OPERATION
    ))


def route_operations(
    operations: list[str] | tuple[str, ...],
) -> str:
    """Resolve one operation set; unsupported combinations require GUIDE."""

    engines = engines_for_operations(operations)
    if len(operations) != 1 or len(engines) != 1:
        return "GUIDE"
    return engines[0]


def pandas_strategy_for_operations(
    operations: list[str] | tuple[str, ...],
) -> PandasStrategy | None:
    """Use one validated execution contract for every LLM-routed table query."""

    if route_operations(operations) != "PANDAS":
        return None
    return "QUERY_PLAN"
