from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rag.question_analyzer import QuestionAnalysis, analyze_question
from rag.question_decision import QuestionDecision
from rag.router import engines_for_operations


GuardStatus = Literal["PASS", "GUIDE"]


@dataclass
class GuardResult:
    status: GuardStatus
    reason_code: str = ""
    reason: str = ""
    operations: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    analysis: QuestionAnalysis | None = field(default=None, repr=False)


def check_question(question: str) -> GuardResult:
    analysis = analyze_question(question)
    if analysis.is_empty:
        return GuardResult(
            status="GUIDE",
            reason_code="EMPTY_QUESTION",
            reason="질문 내용이 비어 있습니다.",
            suggestions=["조회할 이름, 기수, 발행번호 또는 문서 내용을 입력해 주세요."],
            analysis=analysis,
        )

    if analysis.is_meaningless:
        return GuardResult(
            status="GUIDE",
            reason_code="MEANINGLESS_INPUT",
            reason="문서 조회 의도를 확인하기 어렵습니다.",
            suggestions=["예: 2025-008 출연금액 알려줘", "예: 이 문서의 지급 기준을 설명해줘"],
            analysis=analysis,
        )

    if analysis.is_vague:
        return GuardResult(
            status="GUIDE",
            reason_code="VAGUE_REFERENCE",
            reason="문맥이 필요한 표현이 포함되어 있습니다.",
            suggestions=["이름, 기수, 발행번호, 연도 또는 문서명을 포함해 주세요."],
            analysis=analysis,
        )

    if analysis.is_bare_request:
        return GuardResult(
            status="GUIDE",
            reason_code="AMBIGUOUS_SCOPE",
            reason="조회할 대상이나 범위가 부족합니다.",
            suggestions=["예: 58기 출연자 명단 알려줘", "예: 2025-008 출연금액 알려줘"],
            analysis=analysis,
        )

    operations = analysis.operations
    domains = analysis.domains
    if len(domains) > 1:
        return GuardResult(
            status="GUIDE",
            reason_code="CROSS_ENGINE_QUERY",
            reason="계산·조회와 문서 근거 검색이 한 질문에 함께 포함되어 있습니다.",
            operations=operations,
            domains=domains,
            suggestions=["금액·명단 조회와 이유·기준 검색을 별도 질문으로 나누어 주세요."],
            analysis=analysis,
        )

    supported_extreme_comparison = (
        "compare" in operations
        and {intent.operation for intent in analysis.aggregation_intents} == {"min", "max"}
        and all(intent.target == "value" for intent in analysis.aggregation_intents)
    )
    if "compare" in operations and not supported_extreme_comparison:
        return GuardResult(
            status="GUIDE",
            reason_code="COMPARISON_NOT_SUPPORTED",
            reason="현재 여러 범위의 집계 결과를 직접 비교하는 기능은 지원하지 않습니다.",
            operations=operations,
            domains=domains,
            suggestions=["비교할 각 범위의 집계값을 별도 질문으로 조회해 주세요."],
            analysis=analysis,
        )

    if len(analysis.aggregation_intents) > 1 and not supported_extreme_comparison:
        return GuardResult(
            status="GUIDE",
            reason_code="MULTIPLE_AGGREGATIONS",
            reason="여러 집계 요청이 한 질문에 포함되어 있습니다.",
            operations=operations,
            domains=domains,
            suggestions=["합계, 평균, 최댓값처럼 한 번에 하나의 집계만 요청해 주세요."],
            analysis=analysis,
        )

    if len(operations) > 1 and analysis.has_connector:
        return GuardResult(
            status="GUIDE",
            reason_code="MULTI_OPERATION",
            reason="서로 다른 요청이 한 질문에 함께 포함되어 있습니다.",
            operations=operations,
            domains=domains,
            suggestions=["한 번에 하나의 조회나 계산을 요청해 주세요."],
            analysis=analysis,
        )

    return GuardResult(
        status="PASS",
        reason_code="PASS",
        reason="질문을 처리할 수 있습니다.",
        operations=operations,
        domains=domains,
        analysis=analysis,
    )


def check_question_decision(decision: QuestionDecision) -> GuardResult:
    """Validate LLM operations without re-running regex-based classification."""

    if decision.status != "ready":
        return GuardResult(
            status="GUIDE",
            reason_code=(
                "LLM_CLARIFICATION"
                if decision.status == "clarification"
                else "UNSUPPORTED_QUESTION"
            ),
            reason=decision.message or decision.reason,
            suggestions=list(decision.candidates),
        )

    operations = list(decision.operations)
    engines = engines_for_operations(operations)
    domain_by_engine = {
        "PANDAS": "structured_data",
        "VECTOR": "document_evidence",
        "DOCUMENTS": "document_inventory",
    }
    domains = [
        domain_by_engine[engine]
        for engine in engines
        if engine in domain_by_engine
    ]

    if len(engines) > 1:
        return GuardResult(
            status="GUIDE",
            reason_code="CROSS_ENGINE_QUERY",
            reason="계산·조회와 다른 종류의 문서 요청이 한 질문에 함께 포함되어 있습니다.",
            operations=operations,
            domains=domains,
            suggestions=["각 요청을 별도 질문으로 나누어 주세요."],
        )

    if decision.request_count > 1:
        return GuardResult(
            status="GUIDE",
            reason_code="MULTI_REQUEST",
            reason="서로 다른 답을 요구하는 요청이 한 질문에 함께 포함되어 있습니다.",
            operations=operations,
            domains=domains,
            suggestions=["각 요청을 별도 질문으로 나누어 주세요."],
        )

    aggregation_operations = {
        "count_records",
        "sum_amount",
        "average_amount",
        "median_amount",
        "mode_amount",
        "max_amount",
        "min_amount",
        "max_person_by_amount",
        "min_person_by_amount",
    }
    if len(aggregation_operations.intersection(operations)) > 1:
        return GuardResult(
            status="GUIDE",
            reason_code="MULTIPLE_AGGREGATIONS",
            reason="여러 집계 요청이 한 질문에 포함되어 있습니다.",
            operations=operations,
            domains=domains,
            suggestions=["한 번에 하나의 집계만 요청해 주세요."],
        )

    if len(operations) > 1:
        return GuardResult(
            status="GUIDE",
            reason_code="MULTI_OPERATION",
            reason="서로 다른 요청이 한 질문에 함께 포함되어 있습니다.",
            operations=operations,
            domains=domains,
            suggestions=["한 번에 하나의 조회나 계산을 요청해 주세요."],
        )

    return GuardResult(
        status="PASS",
        reason_code="PASS",
        reason="질문을 처리할 수 있습니다.",
        operations=operations,
        domains=domains,
    )
