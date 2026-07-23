from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from typing import Literal

import pandas as pd

from datastore.state import _df_namespace, _df_sources
from datastore.scope import scoped_mapping, source_scope_active
from datastore.query import (
    _search_name_pandas,
    _query_pandas_direct,
    _query_all_records,
    _has_explicit_structured_filter,
    has_explicit_masked_name,
)
from pandas_engine.query_executor import (
    QueryPlanExecutionError,
    execute_query_plan,
)
from pandas_engine.formatter import (
    _format_pandas_result,
    _format_scalar_result,
    _format_dataframe_result_for_question,
    _format_query_execution_result,
)
from rag.query_planner import (
    QueryPlannerError,
    generate_validated_query_plan,
)
from rag.question_analyzer import QuestionAnalysis, analyze_question
from rag.deterministic_query_plan import build_schema_grounded_plan
from utils.semantic_schema import infer_column_meaning
from pandas_engine.interactive import build_interactive_result, build_interactive_dataframe

logger = logging.getLogger("uvicorn.error")
_interactive_result: ContextVar[dict | None] = ContextVar("interactive_result", default=None)


def clear_interactive_result() -> None:
    _interactive_result.set(None)


def current_interactive_result() -> dict | None:
    return _interactive_result.get()

_NUMERIC_COMPARISON_FILTER = re.compile(
    r"(?:\d[\d,.]*\s*(?:원|만원|천원|점|명|개)?\s*(?:이상|이하|초과|미만))"
    r"|(?:(?:>=|<=|>|<)\s*\d)",
    re.IGNORECASE,
)


def _format_direct_dataframe_with_evidence(
    df: pd.DataFrame,
    question: str,
    sources: list[str],
) -> str:
    """Expose the same count observability for verified direct-query handlers.

    Direct handlers predate QueryPlan and return only a DataFrame.  Their result
    set is nevertheless deterministic, so report its row count and the
    schema-derived distinct-person count instead of making evaluation fall back
    to unsafe answer-keyword matching.
    """
    person_columns = [
        column
        for column in df.columns
        if (
            (meaning := infer_column_meaning(str(column), df[column])).concept == "entity"
            and meaning.role == "entity_name"
            and meaning.qualifier == "person"
        )
    ]
    lines = [
        _format_dataframe_result_for_question(df, question),
        "",
        "조회 근거:",
        f"- 문서: {', '.join(sources) if sources else '알 수 없음'}",
        "- 실행 방식: 검증된 직접 조회",
        f"- 조건 통과 {len(df):,}건",
    ]
    if person_columns:
        # Some normalized tables retain multiple person-like identity columns
        # (for example a display label plus the actual member name).  The
        # identifier with the broadest non-null population is the least lossy
        # schema-derived representative; this is independent of any document
        # or column name.
        person_column = max(
            person_columns,
            key=lambda column: int(df[column].dropna().nunique()),
        )
        unique_people = int(df[person_column].dropna().nunique())
        lines.append(f"- 조건 충족 고유 인원: {unique_people:,}명")
    answer = "\n".join(lines)
    _interactive_result.set(build_interactive_dataframe(df, answer=answer))
    return answer


async def _answer_query_plan(
    question: str,
    *,
    allow_vector_fallback: bool,
    analysis: QuestionAnalysis | None = None,
    operation_hint: str | None = None,
) -> tuple[str, list[str], str]:
    """Generate, validate, and execute the generic structured-query plan."""

    logger.info(
        "[PANDAS] QueryPlan 생성 중 | hint=%s question=%s",
        operation_hint or "none",
        question[:50],
    )
    early_plan = build_schema_grounded_plan(
        question,
        dataframes=scoped_mapping(_df_namespace, _df_sources),
        operation_hint=operation_hint,
    )
    if early_plan is not None:
        from pandas_engine.plan_validator import validate_query_plan

        early_validation = validate_query_plan(
            early_plan,
            question=question,
            operation_hint=operation_hint,
        )
        if early_validation.is_executable:
            execution = execute_query_plan(early_validation)
            answer = _format_query_execution_result(execution, question)
            _interactive_result.set(build_interactive_result(execution, answer=answer))
            logger.info("[PANDAS] 스키마 기반 선행 계획 실행 | operation=%s", execution.operation)
            return answer, [execution.source_file], "pandas"
    try:
        validation = await generate_validated_query_plan(
            question,
            operation_hint=operation_hint,
        )
    except QueryPlannerError as exc:
        logger.error("[PANDAS] QueryPlan 생성 실패 | err=%s", exc)
        fallback_plan = build_schema_grounded_plan(
            question,
            dataframes=scoped_mapping(_df_namespace, _df_sources),
            operation_hint=operation_hint,
        )
        if fallback_plan is not None:
            from pandas_engine.plan_validator import validate_query_plan

            fallback_validation = validate_query_plan(
                fallback_plan,
                question=question,
                operation_hint=operation_hint,
            )
            if fallback_validation.is_executable:
                execution = execute_query_plan(fallback_validation)
                answer = _format_query_execution_result(execution, question)
                _interactive_result.set(build_interactive_result(execution, answer=answer))
                logger.warning("[PANDAS] 스키마 기반 폴백 계획 실행 | operation=%s", execution.operation)
                return answer, [execution.source_file], "pandas"
        return (
            "질문을 안전한 표 조회 계획으로 변환하지 못했습니다. "
            "조회할 항목과 조건을 조금 더 명확하게 입력해 주세요.",
            [],
            "pandas",
        )

    if validation.status == "clarification":
        message = validation.plan.message or next(
            (issue.message for issue in validation.issues if issue.message),
            "조회할 문서나 항목을 하나로 지정해 주세요.",
        )
        if validation.plan.candidates:
            message += " 후보: " + ", ".join(validation.plan.candidates)
        logger.info("[PANDAS] QueryPlan 추가 확인 필요 | message=%s", message[:200])
        return message, [], "pandas"

    if validation.status == "not_applicable":
        if not allow_vector_fallback:
            return validation.plan.message or "표 조회로 처리할 수 없는 질문입니다.", [], "pandas"
        logger.info("[PANDAS→VECTOR] QueryPlan이 문서 내용 검색으로 판정")
        from rag.vector import _answer_vector

        v_answer, v_sources, _ = await _answer_vector(
            question,
            allow_pandas_fallback=False,
            analysis=analysis,
        )
        return v_answer, v_sources, "vector"

    if not validation.is_executable:
        issue_codes = ", ".join(issue.code for issue in validation.issues)
        logger.warning("[PANDAS] QueryPlan 검증 실패 | issues=%s", issue_codes)
        fallback_plan = build_schema_grounded_plan(
            question,
            dataframes=scoped_mapping(_df_namespace, _df_sources),
            operation_hint=operation_hint,
        )
        if fallback_plan is not None:
            from pandas_engine.plan_validator import validate_query_plan

            fallback_validation = validate_query_plan(
                fallback_plan,
                question=question,
                operation_hint=operation_hint,
            )
            if fallback_validation.is_executable:
                execution = execute_query_plan(fallback_validation)
                answer = _format_query_execution_result(execution, question)
                _interactive_result.set(build_interactive_result(execution, answer=answer))
                logger.warning("[PANDAS] 검증 실패 후 스키마 기반 폴백 실행 | operation=%s", execution.operation)
                return answer, [execution.source_file], "pandas"
        if any(
            issue.code in {"literal_mismatch", "ungrounded_numeric_filter"}
            for issue in validation.issues
        ):
            return (
                "질문의 숫자·단위·비교 조건이 조회 계획에서 달라져 "
                "안전을 위해 실행을 중단했습니다. 같은 질문을 다시 시도해 주세요.",
                [],
                "pandas",
            )
        return (
            "질문을 실제 표의 컬럼과 안전하게 연결하지 못했습니다. "
            "문서에 표시된 항목명과 조회 조건을 확인해 주세요.",
            [],
            "pandas",
        )

    try:
        execution = execute_query_plan(validation)
    except QueryPlanExecutionError as exc:
        logger.error("[PANDAS] QueryPlan 실행 차단 | err=%s", exc)
        return "검증된 표 조회 계획을 실행하지 못했습니다.", [], "pandas"

    logger.info(
        "[PANDAS] QueryPlan 실행 완료 | operation=%s matched=%d source=%s",
        execution.operation,
        execution.matched_rows,
        execution.source_file,
    )
    answer = _format_query_execution_result(execution, question)
    _interactive_result.set(build_interactive_result(execution, answer=answer))
    return answer, [execution.source_file], "pandas"


async def _answer_pandas(
    question: str,
    allow_vector_fallback: bool = True,
    analysis: QuestionAnalysis | None = None,
    strategy: Literal["AUTO", "DIRECT", "QUERY_PLAN"] = "AUTO",
    operation_hint: str | None = None,
) -> tuple[str, list[str], str]:
    clear_interactive_result()
    scoped_dataframes = scoped_mapping(_df_namespace, _df_sources)
    if not scoped_dataframes:
        message = "선택한 문서에서 조회 가능한 표 데이터를 찾을 수 없습니다." if source_scope_active() else "현재 로드된 데이터프레임이 없습니다."
        return message, [], "pandas"

    analysis = analysis or analyze_question(question)

    if strategy == "QUERY_PLAN":
        # R.JSON may deliberately choose QUERY_PLAN for a structured request,
        # but explicit date ranges already have a schema-aware deterministic
        # executor. Run it before P.JSON generation so a cross-year month
        # range cannot be reduced to the first year/month by an LLM plan.
        if analysis.date_filter is not None:
            direct_result, direct_sources = _query_pandas_direct(
                question,
                aggregation_intents=analysis.aggregation_intents,
                date_filter=analysis.date_filter,
            )
            if isinstance(direct_result, pd.DataFrame):
                return (
                    _format_direct_dataframe_with_evidence(
                        direct_result, question, direct_sources
                    ),
                    direct_sources,
                    "pandas",
                )
            return _format_scalar_result(direct_result, question), direct_sources, "pandas"

        # 마스킹 이름은 검증된 전용 검색기가 있다. LLM이 단순 이름 조회를
        # structured_query로 오분류해도, 별도 숫자·범위 조건이 없는 경우에만
        # QueryPlan보다 안전한 직접 검색 결과를 우선한다.
        if (
            operation_hint != "lookup_field"
            and
            has_explicit_masked_name(question)
            and not _has_explicit_structured_filter(question)
            and not _NUMERIC_COMPARISON_FILTER.search(question)
        ):
            name_df, name_sources, name_searched = _search_name_pandas(question)
            if name_df is not None:
                if not source_scope_active() and len(name_sources) > 1:
                    names = ", ".join(name_sources[:5])
                    return (
                        "같은 이름의 기록이 여러 문서에서 발견되었습니다. "
                        f"조회할 문서를 선택해주세요: {names}",
                        name_sources,
                        "pandas",
                    )
                logger.info(
                    "[NAME_SEARCH] QueryPlan 오분류 복구 | rows=%d",
                    len(name_df),
                )
                return (
                    _format_direct_dataframe_with_evidence(name_df, question, name_sources),
                    name_sources,
                    "pandas",
                )
            if name_searched:
                return "조회된 데이터가 없습니다.", [], "pandas"
        return await _answer_query_plan(
            question,
            allow_vector_fallback=allow_vector_fallback,
            analysis=analysis,
            operation_hint=operation_hint,
        )

    # 비교 집계는 그룹 기준을 확정할 전용 실행기가 아직 없으므로 잘못된 단일
    # 집계를 반환하지 않는다. 정상 /chat 경로에서는 Guard가 먼저 안내한다.
    if "compare" in analysis.operations:
        return (
            "현재 여러 범위의 집계 결과를 직접 비교하는 기능은 지원하지 않습니다. "
            "비교할 각 범위의 집계값을 별도로 질문해 주세요.",
            [],
            "pandas",
        )

    # 기본 통계는 LLM 코드 생성이나 VECTOR 검색으로 넘기지 않고 검증된 함수로 계산한다.
    if analysis.aggregation_intents:
        direct_result, direct_sources = _query_pandas_direct(
            question,
            aggregation_intents=analysis.aggregation_intents,
            date_filter=analysis.date_filter,
        )
        if direct_result is None:
            # Person-ranking aggregation may be structurally valid even when
            # the legacy direct aggregator cannot produce its subject payload.
            # Let the schema-grounded QueryPlan handle it instead of turning
            # an unsupported direct shape into a false no-data response.
            person_ranking = any(
                intent.operation in {"min", "max"}
                and intent.target in {"person_total", "row"}
                for intent in analysis.aggregation_intents
            )
            if not person_ranking:
                return "조회된 데이터가 없습니다.", [], "pandas"
        else:
            direct_notice = (
                isinstance(direct_result, dict)
                and direct_result.get("type") == "aggregation_notice"
            )
            person_ranking = any(
                intent.operation in {"min", "max"}
                and intent.target in {"person_total", "row"}
                for intent in analysis.aggregation_intents
            )
            if direct_notice and person_ranking:
                direct_result = None
            else:
                logger.info("[AGGREGATION] 고정 집계 실행 | source=%s", direct_sources)
                return _format_scalar_result(direct_result, question), direct_sources, "pandas"

    # 1단계: 이름 전수 검색 (기존)
    # 숫자 비교가 명시된 복합 필터 질문에서 일반 조건어를 마스킹 이름으로
    # 유사 매칭하지 않는다. 실제 이름 단순 조회는 기존 전수 검색을 유지한다.
    if _NUMERIC_COMPARISON_FILTER.search(question):
        name_df, name_sources, name_searched = None, [], False
    else:
        name_df, name_sources, name_searched = _search_name_pandas(question)
    if name_df is not None:
        if not source_scope_active() and len(name_sources) > 1:
            names = ", ".join(name_sources[:5])
            return (
                f"같은 이름의 기록이 여러 문서에서 발견되었습니다. 조회할 문서를 선택해주세요: {names}",
                name_sources,
                "pandas",
            )
        logger.info("[NAME_SEARCH] %d건 발견, 코드 생성 생략", len(name_df))
        return _format_direct_dataframe_with_evidence(name_df, question, name_sources), name_sources, "pandas"
    if name_searched and re.search(
        r"이라는|라는\s*학생|학생이.{0,20}(?:장학금|받|있)|받았[나어요이]|있[나어]\s*[?？]?$",
        question,
    ):
        # 특정 인물 조회(이라는/학생이...받았어 등) → 이름이 없으면 바로 없음 반환
        logger.info("[NAME_SEARCH] 특정 인물 조회 패턴 — 데이터 없음")
        return "조회된 데이터가 없습니다.", [], "pandas"

    # 2단계: 키워드 직접 조회 (LLM 코드 생성 없음)
    direct_result, direct_sources = _query_pandas_direct(
        question,
        aggregation_intents=analysis.aggregation_intents,
        date_filter=analysis.date_filter,
    )
    if direct_result is not None:
        formatted = _format_pandas_result(direct_result)
        if formatted != "조회된 데이터가 없습니다.":
            logger.info("[DIRECT] 직접 조회 성공 | source=%s", direct_sources)
            if isinstance(direct_result, pd.DataFrame):
                return _format_direct_dataframe_with_evidence(
                    direct_result, question, direct_sources
                ), direct_sources, "pandas"
            # scalar(int/float/str): LLM 우회, 직접 포맷
            return _format_scalar_result(direct_result, question), direct_sources, "pandas"

    # Analyzer가 목록 요청으로 확정한 경우 LLM이 len(df)와 df 반환 사이에서
    # 임의로 선택하지 않도록 선택 문서의 전체 행을 직접 반환한다.
    if "list_records" in analysis.operations:
        list_result, list_sources = _query_all_records()
        if isinstance(list_result, pd.DataFrame):
            logger.info("[LIST_RECORDS] 전체 목록 직접 조회 | source=%s", list_sources)
            return _format_direct_dataframe_with_evidence(
                list_result, question, list_sources
            ), list_sources, "pandas"
        return _format_scalar_result(list_result, question), list_sources, "pandas"

    if _has_explicit_structured_filter(question):
        logger.info("[PANDAS] 명시된 기수/식별번호와 일치하는 데이터 없음")
        return "조회된 데이터가 없습니다.", [], "pandas"

    # 3단계: 검증된 직접 조회로 처리하지 못한 구조화 질문은 Python 코드를
    # 생성하지 않고 제한된 QueryPlan으로 변환한다.
    return await _answer_query_plan(
        question,
        allow_vector_fallback=allow_vector_fallback,
        analysis=analysis,
        operation_hint=operation_hint,
    )
