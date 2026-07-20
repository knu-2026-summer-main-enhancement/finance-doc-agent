from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


DecisionStatus = Literal["ready", "clarification", "unsupported"]
QuestionOperation = Literal[
    "list_documents",
    "filter_records",
    "compare",
    "max_person_by_amount",
    "min_person_by_amount",
    "list_records",
    "count_records",
    "sum_amount",
    "average_amount",
    "median_amount",
    "mode_amount",
    "max_amount",
    "min_amount",
    "lookup_amount",
    "lookup_field",
    "structured_query",
    "document_reason",
    "document_purpose",
    "document_criteria",
    "document_procedure",
    "document_explain",
]

_VECTOR_OPERATIONS = {
    "document_reason",
    "document_purpose",
    "document_criteria",
    "document_procedure",
    "document_explain",
}


class ClassifiedRequest(BaseModel):
    """One independently answerable request copied from the user question."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    source_text: str = Field(min_length=1, max_length=300)
    operation: QuestionOperation


class QuestionDecision(BaseModel):
    """Strict operation contract emitted by the LLM question engine."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    status: DecisionStatus
    requests: tuple[ClassifiedRequest, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    operations: tuple[QuestionOperation, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    reason: str = Field(min_length=1, max_length=500)
    retrieval_query: str | None = Field(default=None, max_length=1000)
    message: str | None = Field(default=None, max_length=500)
    candidates: tuple[str, ...] = Field(default_factory=tuple, max_length=10)

    @model_validator(mode="before")
    @classmethod
    def derive_operations_from_requests(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        requests = value.get("requests") or []
        if requests and not value.get("operations"):
            derived: list[str] = []
            for request in requests:
                operation = (
                    request.get("operation")
                    if isinstance(request, dict)
                    else getattr(request, "operation", None)
                )
                if operation and operation not in derived:
                    derived.append(operation)
            value = dict(value)
            value["operations"] = derived
        return value

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        if self.status == "ready":
            if not self.operations:
                raise ValueError("ready requires at least one operation")
            if len(set(self.operations)) != len(self.operations):
                raise ValueError("operations must not contain duplicates")
            if self.requests:
                request_operations = tuple(dict.fromkeys(
                    request.operation for request in self.requests
                ))
                if request_operations != self.operations:
                    raise ValueError(
                        "operations must match the operations derived from requests"
                    )
            has_vector_operation = any(
                operation in _VECTOR_OPERATIONS
                for operation in self.operations
            )
            if has_vector_operation and not self.retrieval_query:
                raise ValueError(
                    "document operations require retrieval_query"
                )
            if not has_vector_operation and self.retrieval_query is not None:
                raise ValueError(
                    "retrieval_query is only allowed for document operations"
                )
            if self.message is not None or self.candidates:
                raise ValueError(
                    "ready must not include clarification fields"
                )
            return self

        if self.operations or self.requests:
            raise ValueError(
                f"{self.status} must not include requests or operations"
            )
        if self.retrieval_query is not None:
            raise ValueError(
                f"{self.status} must not include retrieval_query"
            )
        if not self.message:
            raise ValueError(f"{self.status} requires message")
        return self

    @property
    def request_count(self) -> int:
        """Return independent request count while preserving legacy decisions."""
        return len(self.requests) if self.requests else len(self.operations)
