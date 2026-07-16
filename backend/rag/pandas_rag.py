from __future__ import annotations

import logging
import re

import pandas as pd

from datastore.state import _df_namespace, _df_sources
from datastore.scope import scoped_mapping, source_scope_active, source_is_selected
from datastore.schema import _get_df_schema_filtered
from datastore.query import (
    _search_name_pandas,
    _query_pandas_direct,
    _find_value_locations,
    _has_explicit_structured_filter,
)
from pandas_engine.executor import _exec_pandas_code, _clean_code
from pandas_engine.aggregation import _AGG_COUNT, _AGG_SUM
from pandas_engine.formatter import (
    _format_pandas_result,
    _format_list_result,
    _format_scalar_result,
    _format_dataframe_result_for_question,
)
from rag.prompts import _PANDAS_GEN_TEMPLATE
from rag.question_analyzer import QuestionAnalysis, analyze_question
from core.llm import get_llm_code

logger = logging.getLogger("uvicorn.error")

_NO_VECTOR_FALLBACK = re.compile(r"누가|누구|명단|목록|리스트|몇\s*명|인원")


async def _answer_pandas(
    question: str,
    allow_vector_fallback: bool = True,
    analysis: QuestionAnalysis | None = None,
) -> tuple[str, list[str], str]:
    scoped_dataframes = scoped_mapping(_df_namespace, _df_sources)
    if not scoped_dataframes:
        message = "선택한 문서에서 조회 가능한 표 데이터를 찾을 수 없습니다." if source_scope_active() else "현재 로드된 데이터프레임이 없습니다."
        return message, [], "pandas"

    analysis = analysis or analyze_question(question)

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
        )
        if direct_result is None:
            return "조회된 데이터가 없습니다.", [], "pandas"
        logger.info("[AGGREGATION] 고정 집계 실행 | source=%s", direct_sources)
        return _format_scalar_result(direct_result, question), direct_sources, "pandas"

    # 1단계: 이름 전수 검색 (기존)
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
        return _format_dataframe_result_for_question(name_df, question), name_sources, "pandas"
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
    )
    if direct_result is not None:
        formatted = _format_pandas_result(direct_result)
        if formatted != "조회된 데이터가 없습니다.":
            logger.info("[DIRECT] 직접 조회 성공 | source=%s", direct_sources)
            if isinstance(direct_result, pd.DataFrame):
                return _format_dataframe_result_for_question(direct_result, question), direct_sources, "pandas"
            # scalar(int/float/str): LLM 우회, 직접 포맷
            return _format_scalar_result(direct_result, question), direct_sources, "pandas"

    if _has_explicit_structured_filter(question):
        logger.info("[PANDAS] 명시된 기수/식별번호와 일치하는 데이터 없음")
        return "조회된 데이터가 없습니다.", [], "pandas"

    # 3단계: LLM 코드 생성 (복잡한 질문 폴백) — 관련 DF만 schema에 포함
    schema = _get_df_schema_filtered(question)
    hints = _find_value_locations(question)
    agg_hint = ""
    if _AGG_COUNT.search(question):
        agg_hint = "\n※ 인원수 질문: result = int(len(filtered_df))"
    elif _AGG_SUM.search(question):
        agg_hint = "\n※ 금액 합계 질문: result = float(df['금액컬럼'].sum())"

    prompt_text = _PANDAS_GEN_TEMPLATE.format(schema=schema, hints=hints, question=question) + agg_hint

    logger.info("[PANDAS] 코드 생성 중 | question=%s", question[:50])
    raw_code = await get_llm_code().ainvoke(prompt_text)
    code = _clean_code(raw_code)
    logger.info("[PANDAS] 생성된 코드 | %s", code[:300])

    result = None
    code_err: str | None = None
    try:
        result = _exec_pandas_code(code)
    except Exception as e:
        code_err = str(e)
        logger.error("[PANDAS] 실행 오류 | err=%s | code=%s", e, code[:200])

    # 결과 없거나 오류 → 재시도
    is_empty = result is None or (hasattr(result, "__len__") and len(result) == 0)
    if is_empty or code_err:
        retry_ctx = f"\n이전 코드가 실패했거나 결과가 없었습니다.\n이전 코드:\n{code}"
        if code_err:
            retry_ctx += f"\n오류: {code_err}"
        retry_ctx += "\n조건을 완화(str.contains 사용)하거나 다른 데이터프레임을 사용하세요."

        raw_code2 = await get_llm_code().ainvoke(prompt_text + retry_ctx)
        code2 = _clean_code(raw_code2)
        if code2 and code2 != code:
            logger.info("[PANDAS] 재시도 코드 | %s", code2[:300])
            try:
                result = _exec_pandas_code(code2)
                code = code2
            except Exception as e2:
                logger.error("[PANDAS] 재시도 실패 | err=%s", e2)

    formatted = _format_pandas_result(result)

    if formatted == "조회된 데이터가 없습니다." and allow_vector_fallback:
        if _NO_VECTOR_FALLBACK.search(question):
            logger.info("[PANDAS] 명단형 쿼리 — VECTOR 폴백 건너뜀")
            return formatted, [], "pandas"
        logger.info("[PANDAS→VECTOR] 결과 없음, VECTOR 폴백")
        # 지연 임포트로 순환 참조 방지
        from rag.vector import _answer_vector
        v_answer, v_sources, _ = await _answer_vector(
            question,
            allow_pandas_fallback=False,
            analysis=analysis,
        )
        return v_answer, v_sources, "vector"

    source_files = list({
        _df_sources.get(v, v)
        for v in _df_namespace
        if v in code and source_is_selected(_df_sources.get(v, v))
    })

    if formatted == "조회된 데이터가 없습니다.":
        return formatted, source_files, "pandas"

    if isinstance(result, pd.DataFrame):
        return _format_dataframe_result_for_question(result, question), source_files, "pandas"

    return _format_scalar_result(result, question), source_files, "pandas"
