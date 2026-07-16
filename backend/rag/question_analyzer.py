from __future__ import annotations

from dataclasses import dataclass, field

from pandas_engine.aggregation import AggregationIntent, detect_aggregation_intents
from pandas_engine.date_filter import DateFilter, parse_date_filter
from rag.question_detectors import (
    QuestionSignals,
    detect_question_signals,
    normalize_question,
    operation_domains,
)


@dataclass
class QuestionAnalysis:
    question: str
    operations: list[str] = field(default_factory=list)
    aggregation_intents: list[AggregationIntent] = field(default_factory=list)
    date_filter: DateFilter | None = None
    domains: list[str] = field(default_factory=list)
    signals: QuestionSignals = field(default_factory=QuestionSignals)
    is_empty: bool = False

    @property
    def is_meaningless(self) -> bool:
        return self.signals.is_meaningless

    @property
    def is_vague(self) -> bool:
        return self.signals.is_vague

    @property
    def is_bare_request(self) -> bool:
        return self.signals.is_bare_request

    @property
    def has_connector(self) -> bool:
        return self.signals.has_connector

    @property
    def has_pandas_keyword(self) -> bool:
        return self.signals.has_pandas_keyword

    @property
    def has_vector_procedure(self) -> bool:
        return self.signals.has_vector_procedure

    @property
    def has_aggregation_procedure(self) -> bool:
        return self.signals.has_aggregation_procedure

    @property
    def has_vector_override(self) -> bool:
        return self.signals.has_vector_override

    @property
    def has_scholarship_keyword(self) -> bool:
        return self.signals.has_scholarship_keyword


def analyze_question(question: str) -> QuestionAnalysis:
    """전문 감지기의 결과를 한 번 수집해 이후 처리 단계에 전달한다."""
    normalized = normalize_question(question)
    if not normalized:
        return QuestionAnalysis(question="", is_empty=True)

    aggregation_intents = detect_aggregation_intents(normalized)
    date_filter = parse_date_filter(normalized)
    signals = detect_question_signals(normalized, aggregation_intents)
    if date_filter and not signals.operations:
        signals.operations.append("filter_records")
    return QuestionAnalysis(
        question=normalized,
        # 분석 결과와 원시 감지 신호가 같은 가변 리스트를 공유하지 않게 한다.
        operations=list(signals.operations),
        aggregation_intents=aggregation_intents,
        date_filter=date_filter,
        domains=operation_domains(signals.operations),
        signals=signals,
    )
