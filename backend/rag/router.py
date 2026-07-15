from __future__ import annotations

from rag.question_analyzer import QuestionAnalysis, analyze_question


_ENGINE_BY_OPERATION = {
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
    "document_reason": "VECTOR",
    "document_purpose": "VECTOR",
    "document_criteria": "VECTOR",
    "document_procedure": "VECTOR",
    "document_explain": "VECTOR",
}


def required_engines(analysis: QuestionAnalysis) -> list[str]:
    """분석된 작업에 필요한 엔진을 작업 순서대로 중복 없이 반환한다."""
    return list(dict.fromkeys(
        _ENGINE_BY_OPERATION[operation]
        for operation in analysis.operations
        if operation in _ENGINE_BY_OPERATION
    ))


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
