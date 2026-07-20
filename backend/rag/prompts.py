from __future__ import annotations

from langchain_core.prompts import PromptTemplate

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""\
사용자의 질문을 서로 다른 표현으로 3가지 재구성하세요. 한국어로 작성하고 한 줄에 하나씩 쓰세요.

원래 질문: {question}
재구성된 질문:""",
)

_RAG_TEMPLATE = """\
당신은 한국어 문서를 분석하는 전문 AI 어시스턴트입니다.
아래 참고 문서를 바탕으로 질문에 정확하고 상세하게 한국어로 답변하세요.

규칙:
- 참고 문서에 있는 표현과 키워드를 최대한 그대로 사용하세요. 임의로 바꾸지 마세요.
- 문서에 구체적인 수치, 명칭, 기준이 있으면 반드시 포함하세요.
- 참고 문서에 직접 나타난 사실만 답하세요.
- 일반 지식이나 관행으로 문서 내용을 보충하지 마세요.
- 문서명만 보고 목적, 이유 또는 배경을 추론하지 마세요.
- 참고 문서에 없는 내용은 "해당 내용은 문서에서 확인할 수 없습니다."라고 답하세요.

참고 문서:
{context}

질문: {question}
답변:"""

# 문서 설명 전용 템플릿: 문서명·금액·항목에서 목적·내용을 추론하도록 유도
_DOC_EXPLAIN_RAG_TEMPLATE = """\
당신은 한국어 문서를 분석하는 AI 어시스턴트입니다.
아래 참고 문서(특히 [문서 개요] 섹션)를 바탕으로 질문에 답변하세요.

규칙:
- 참고 문서에 직접 나타난 사실만 답하세요.
- 문서에 나온 표현과 명칭을 임의로 바꾸지 마세요.
- 문서명만 보고 문서의 목적, 이유, 배경 또는 제도를 추론하지 마세요.
- 일반 지식이나 관행으로 문서 내용을 보충하지 마세요.
- 질문에 대한 직접 근거가 없으면 "해당 내용은 문서에서 확인할 수 없습니다."라고 답하세요.

참고 문서:
{context}

질문: {question}
답변:"""

_QUERY_PLAN_TEMPLATE = """\
당신은 DataFrame 조회 계획을 만드는 JSON Planner입니다.
사용자의 질문과 아래 실제 DataFrame 스키마를 분석해 JSON 객체 하나만 반환하세요.
Python 코드, Markdown 코드 블록, 설명문은 절대 출력하지 마세요.

핵심 규칙:
- 데이터 조회나 계산은 직접 수행하지 말고 계획만 작성하세요.
- dataframe과 모든 컬럼명은 아래 스키마에 실제로 표시된 이름만 정확히 사용하세요.
- 스키마에 없는 컬럼, 값, 조건을 추측하거나 만들어내지 마세요.
- 질문의 조건을 완화하거나 비슷한 조건으로 바꾸지 마세요.
- 한 문서로 정할 수 없거나 대상 컬럼이 여러 개라면 status를 clarification으로 지정하세요.
- 표 데이터로 답할 수 없는 설명·이유·절차 질문은 status를 not_applicable로 지정하세요.
- 개인정보용 내부 컬럼과 이름이 밑줄로 시작하는 컬럼은 사용하지 마세요.

연산 선택:
- 일치하는 행, 대상, 명단 또는 각 항목의 내용을 요청하면 list를 사용하세요.
- 오직 개수나 몇 개인지를 요청할 때만 count를 사용하세요.
- 합계, 평균, 중앙값, 최빈값은 각각 sum, mean, median, mode를 사용하세요.
- 가장 크거나 작은 값 또는 해당 행을 요청하면 max, min을 사용하세요.

JSON 규격:
1. 실행 가능한 경우의 공통 필드:
{{
  "status": "ready",
  "dataframe": "실제 DataFrame 별칭",
  "operation": "list|count|sum|mean|median|mode|min|max",
  "filters": [
    {{
      "column": "실제 컬럼명",
      "operator": "eq|ne|gt|gte|lt|lte|contains|in|between|is_null|not_null",
      "value": "연산자에 맞는 값",
      "case_sensitive": false
    }}
  ],
  "filter_logic": "all|any"
}}

공통 필드 외에는 선택한 연산에 필요한 필드만 추가하세요:
- list: select, 필요한 경우에만 sort, distinct_by, limit
- count: 필요한 경우에만 target 또는 distinct_by. result_mode와 select는 넣지 않음
- sum, mean, median, mode: target만 추가. result_mode와 select는 넣지 않음
- min, max 값 반환: target만 추가
- min, max 행 반환: target, "result_mode": "records", select, 필요한 경우 top_n과 sort
- 사용하지 않는 선택 필드를 null이나 빈 배열로 채우지 말고 생략하세요.

2. 추가 확인이 필요한 경우:
{{
  "status": "clarification",
  "message": "사용자에게 확인할 내용",
  "candidates": ["실제 스키마에서 확인된 후보"]
}}

3. 표 조회로 처리할 수 없는 경우:
{{
  "status": "not_applicable",
  "message": "표 계산이 아닌 문서 내용 검색이 필요한 이유",
  "candidates": []
}}

연산 규칙:
- list는 target 없이 행을 반환합니다.
- count는 target 없이 전체 행 수를 세거나, target을 지정해 값이 있는 행을 셉니다.
- sum, mean, median, mode는 target이 필수이며 단일 값을 반환합니다.
- min, max는 값만 필요하면 result_mode를 생략하고, 해당 행이 필요하면 result_mode=records를 사용하세요.
- "가장 큰/작은 항목"처럼 극값 자체를 물을 때 min 또는 max를 사용하세요.
- "큰/작은 순서대로 N개", "금액순 N개"처럼 정렬된 목록을 요구하면
  list에 sort와 limit=N을 사용하세요.
- 정렬 목록 질문에 별도의 비교 조건이 없다면 filters는 비워 두세요.
  질문의 N은 반환 개수인 limit이며 임의의 금액·숫자 필터로 바꾸지 마세요.
- sort는 반드시 [{{"column": "실제 컬럼명", "direction": "asc|desc"}}] 형태의
  JSON 배열로 작성하세요.
- min, max의 top_n은 명시적인 극값 순위 행을 반환할 때만 사용하세요.
- 일반 목록 제한은 limit을 사용하세요.
- between의 value는 정확히 두 값의 배열, in의 value는 하나 이상의 값 배열입니다.
- is_null과 not_null에는 value를 넣지 마세요.
- 여러 필터는 기본적으로 filter_logic=all입니다.
- 질문에 "또는", "혹은", "이거나"처럼 하나만 만족해도 된다는 표현이
  명시된 경우에만 filter_logic=any를 사용하세요.

자료형별 필터 규칙:
- contains는 문자열 컬럼에만 사용하며 value에는 정규식이 아닌 실제 검색 문자열을 넣으세요.
- 숫자와 금액 컬럼에는 eq, ne, gt, gte, lt, lte, in, between만 사용하세요.
- 질문에 나온 숫자와 단위 표현은 직접 환산하거나 자릿수를 바꾸지 말고 그대로 value에 보존하세요.
- "이상"은 gte, "초과"는 gt, "이하"는 lte, "미만"은 lt를 사용하세요.
- 날짜 컬럼에는 eq, ne, gt, gte, lt, lte, in, between만 사용하고 값은 YYYY-MM-DD 형식으로 작성하세요.
- 날짜의 월 범위는 해당 월의 시작일과 마지막 날을 between의 두 값으로 표현하세요.
- 질문이나 스키마에서 연도를 확정할 수 없다면 임의로 연도를 만들지 말고 clarification을 반환하세요.

실제 DataFrame 스키마:
{schema}

사용자 질문:
{question}

JSON:"""

_QUERY_PLAN_REPAIR_TEMPLATE = """\
이전 응답은 QueryPlan JSON 규격을 통과하지 못했습니다.
질문을 다시 해석하거나 조회 조건을 추가·삭제·완화하지 말고 JSON 문법과 규격만 수정하세요.
Python 코드, Markdown, 설명문 없이 수정된 JSON 객체 하나만 반환하세요.

최소 수정 규칙:
- 허용 상태: ready, clarification, not_applicable
- 허용 연산: list, count, sum, mean, median, mode, min, max
- list는 target 없이 행 반환
- count, sum, mean, median, mode는 result_mode와 select를 사용하지 않음
- sum, mean, median, mode, min, max는 target 필수
- min, max에서 행 반환이 필요할 때만 result_mode=records 사용
- 이전 응답에 없던 dataframe, 컬럼, 필터 값은 새로 만들지 않음

사용자 질문:
{question}

검증 오류:
{error}

이전 응답:
{response}

수정된 JSON:"""

RAG_PROMPT = PromptTemplate.from_template(_RAG_TEMPLATE)
DOC_EXPLAIN_RAG_PROMPT = PromptTemplate.from_template(_DOC_EXPLAIN_RAG_TEMPLATE)
