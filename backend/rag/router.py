from __future__ import annotations

from typing import Literal

from rag.question_analyzer import QuestionAnalysis, analyze_question


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
    """Choose direct verified handlers or the generic QueryPlan path."""

    if route_operations(operations) != "PANDAS":
        return None
    return "QUERY_PLAN" if operations[0] == "structured_query" else "DIRECT"


def required_engines(analysis: QuestionAnalysis) -> list[str]:
    """분석된 작업에 필요한 엔진을 작업 순서대로 중복 없이 반환한다."""
    return engines_for_operations(analysis.operations)


def route_analysis(analysis: QuestionAnalysis) -> str:
    """질문을 재분석하지 않고 공통 분석 결과만으로 실행 엔진을 고른다."""
    engines = required_engines(analysis)
    if len(engines) == 1:
        return engines[0]

    # 아래는 명시적 작업을 찾지 못했거나, Guard를 거치지 않은 호출의 폴백 정책이다.
    if analysis.has_vector_override:
        return "VECTOR"
    if analysis.aggregation_intents:
        return "VECTOR" if analysis.has_aggregation_procedure else "PANDAS"
    if analysis.has_pandas_keyword:
        return "VECTOR" if analysis.has_vector_procedure else "PANDAS"
    if analysis.has_scholarship_keyword and not analysis.has_vector_procedure:
        return "PANDAS"
    return "VECTOR"


def _route(question: str, analysis: QuestionAnalysis | None = None) -> str:
    """기존 호출부 호환용 공개 라우팅 함수."""
    return route_analysis(analysis or analyze_question(question))
