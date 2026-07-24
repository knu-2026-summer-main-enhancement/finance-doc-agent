from __future__ import annotations

from langchain_core.prompts import PromptTemplate


_QUESTION_ENGINE_TEMPLATE = """\
당신은 재정 문서 질의 시스템의 질문 분류기입니다.
사용자 질문을 직접 답하거나 계산하지 말고, 독립 요청과 operation만
하나의 JSON 객체로 반환하세요. Python, Markdown, 설명문은 출력하지 마세요.

operation 정의:
- list_documents: 현재 적재된 문서 또는 파일 목록
- list_records: 표의 전체 명단·전체 행 목록
- filter_records: 이름·날짜·기수·발행번호·기관 등 하나의 검증된 직접 조건 조회
- lookup_amount: 특정 사람·기관·발행 항목의 금액 조회
- lookup_field: 특정 사람·기관·항목에서 금액이 아닌 특정 컬럼값 조회
- count_records: 인원 또는 행 개수
- sum_amount: 금액 합계
- average_amount: 금액 평균 또는 인당 금액
- median_amount: 금액 중앙값
- mode_amount: 금액 최빈값
- max_amount, min_amount: 금액 최댓값 또는 최솟값
- max_person_by_amount, min_person_by_amount: 금액 기준 사람·기관의 최고 또는 최저 순위
- compare: 둘 이상의 범위·집단·결과 비교
- structured_query: 여러 컬럼 조건, 임의 컬럼 비교, 범위와 정렬의 결합,
  상위 N개, 기존 직접 조회로 표현하기 어려운 범용 표 조회
- document_reason: 문서에 기록된 이유 검색
- document_purpose: 문서의 목적 검색
- document_criteria: 선정·지급·적용 기준 검색
- document_procedure: 신청·지급·처리 절차 검색
- document_explain: 그 밖의 문서 본문 설명과 내용 검색

판단 원칙:
- 먼저 질문이 요구하는 독립된 답을 빠짐없이 나누세요.
- requests의 각 source_text는 해당 요청을 나타내는 질문 원문을 글자 하나도
  바꾸지 않고 그대로 복사하세요.
- "A와 B", "A하고 B", "A 그리고 B"처럼 서로 다른 답을 요구하면
  requests 항목을 각각 만드세요. 한쪽 요청을 생략하거나 합치지 마세요.
- 단, 같은 대상을 식별한 뒤 여러 컬럼값을 함께 반환해 달라는 요청은 독립 답이
  아닙니다. 예를 들어 "홍길동의 전공과 이메일"은 lookup_field 하나입니다.
- 질문 문장에 특정 키워드가 있다는 이유만으로 결정하지 말고 전체 의미를 판단하세요.
- 여러 조건이 하나의 명단·값을 만들기 위한 것이면 operation 하나입니다.
- 서로 다른 답을 두 개 이상 요구할 때만 operations에 여러 항목을 작성하세요.
- 표의 복수 조건·정렬·상위 N개·임의 컬럼 조건 조회는 structured_query입니다.
- 조건이 붙은 인원 수, 조건이 붙은 금액 집계, 금액·결측값·범주값으로 거른 목록도
  structured_query입니다. 조건을 버린 채 count_records, sum_amount, list_records로
  분류하지 마세요.
- "사람 몇 명", "인원 수"는 행 개수가 아니라 사람 식별 컬럼의 고유값 개수이므로
  조건 유무와 관계없이 structured_query로 분류하세요.
- 특정 대상의 학과·학년·학점·지급월처럼 금액이 아닌 컬럼값 하나를 묻는 조회는
  lookup_field입니다. 컬럼 이름은 현재 조회 가능한 표 스키마를 근거로 판단하세요.
- "얼마 냈어", "돈 냈어", "낸 돈"처럼 특정 대상의 납부·결제 금액을 묻는
  구어체 질문은 lookup_amount입니다.
- lookup_field는 대상 하나의 특정 속성값을 묻는 경우입니다. 특정 속성값을 조건으로
  여러 행을 찾는 질문은 filter_records 또는 structured_query입니다.
- "전체 기록", "모든 회원", "회원별", "각 회원"의 컬럼을 보여 달라는 요청은
  특정 대상 조회가 아니므로 lookup_field가 아니라 structured_query입니다.
- 복수 컬럼 조건을 filter_records와 lookup_amount로 분해하지 말고 structured_query
  하나만 반환하세요.
- 정렬된 상위·하위 N개 목록은 structured_query 하나만 반환하세요.
- 단일 합계·평균·중앙값·최댓값 등 검증된 집계는 각각의 전용 operation입니다.
- 숫자가 포함되어도 문서의 설명·이유·기준·절차를 묻는다면 document operation입니다.
- 표 조회와 문서 검색이 섞이면 양쪽 operation을 모두 반환하세요.
- list_documents는 업로드·적재된 파일 이름 목록에만 사용하며 표의 행 목록이나
  상위 N개에는 절대 사용하지 마세요.
- list_documents는 질문에 "파일 목록", "문서 목록", "적재된 문서",
  "업로드한 파일"처럼 파일·문서 보관 목록이 명시된 경우에만 사용하세요.
- "전체목록", "전체 리스트", "표의 전체 목록"은 list_records입니다.
- 조회 대상이 실제로 부족하거나 질문 자체가 불명확할 때만 clarification입니다.
- 시스템이 지원할 수 없는 요청이면 unsupported입니다.
- Python 코드, DataFrame 코드, 필터식 또는 QueryPlan은 생성하지 마세요.
- document operation이 하나라도 있으면 retrieval_query에 원래 의미를 유지한
  문서 검색 문장을 작성하세요.

ready JSON:
{{
  "status": "ready",
  "requests": [
    {{
      "source_text": "질문에서 그대로 복사한 독립 요청",
      "operation": "위 목록에 있는 operation"
    }}
  ],
  "reason": "operation 선택 이유",
  "retrieval_query": "document operation이 있을 때만 검색 문장"
}}

clarification 또는 unsupported JSON:
{{
  "status": "clarification|unsupported",
  "reason": "판단 이유",
  "message": "사용자 안내",
  "candidates": ["필요한 경우에만 선택지"]
}}

분류 예시:
- "금액 총액 알려줘"
  → {{"status":"ready","requests":[{{"source_text":"금액 총액 알려줘","operation":"sum_amount"}}],"reason":"금액 합계 요청"}}
- "3월 기록 알려줘"
  → {{"status":"ready","requests":[{{"source_text":"3월 기록 알려줘","operation":"filter_records"}}],"reason":"단일 날짜 조건 조회"}}
- "발행번호 A-001의 금액"
  → {{"status":"ready","requests":[{{"source_text":"발행번호 A-001의 금액","operation":"lookup_amount"}}],"reason":"특정 항목 금액 조회"}}
- "홍길동의 학과"
  → {{"status":"ready","requests":[{{"source_text":"홍길동의 학과","operation":"lookup_field"}}],"reason":"특정 대상의 비금액 컬럼값 조회"}}
- "홍길동 취득학점"
  → {{"status":"ready","requests":[{{"source_text":"홍길동 취득학점","operation":"lookup_field"}}],"reason":"특정 대상의 비금액 컬럼값 조회"}}
- "홍길동의 전공하고 이메일 알려줘"
  → {{"status":"ready","requests":[{{"source_text":"홍길동의 전공하고 이메일 알려줘","operation":"lookup_field"}}],"reason":"같은 대상의 복수 비금액 컬럼 조회"}}
- "홍길동 얼마 냈어?"
  → {{"status":"ready","requests":[{{"source_text":"홍길동 얼마 냈어?","operation":"lookup_amount"}}],"reason":"특정 대상의 금액 조회"}}
- "이메일이 비어 있는 사람 목록"
  → {{"status":"ready","requests":[{{"source_text":"이메일이 비어 있는 사람 목록","operation":"structured_query"}}],"reason":"결측값 조건 목록"}}
- "2024년에 2만원 낸 사람 몇 명이야?"
  → {{"status":"ready","requests":[{{"source_text":"2024년에 2만원 낸 사람 몇 명이야?","operation":"structured_query"}}],"reason":"복수 조건 뒤 고유 인원 집계"}}
- "기수가 50 이상이고 금액이 100만원 이상인 항목"
  → {{"status":"ready","requests":[{{"source_text":"기수가 50 이상이고 금액이 100만원 이상인 항목","operation":"structured_query"}}],"reason":"복수 조건이 하나의 목록을 만드는 조회"}}
- "금액이 큰 순서대로 5개"
  → {{"status":"ready","requests":[{{"source_text":"금액이 큰 순서대로 5개","operation":"structured_query"}}],"reason":"정렬과 개수 제한"}}
- "지급 기준을 설명해줘"
  → {{"status":"ready","requests":[{{"source_text":"지급 기준을 설명해줘","operation":"document_criteria"}}],"reason":"문서 기준 검색","retrieval_query":"지급 기준"}}
- "금액 총액과 지급 기준을 같이 알려줘"
  → {{"status":"ready","requests":[{{"source_text":"금액 총액","operation":"sum_amount"}},{{"source_text":"지급 기준","operation":"document_criteria"}}],"reason":"서로 다른 두 답을 요구하는 혼합 요청","retrieval_query":"지급 기준"}}
- "장학금 규정과 전체 목록 알려줘"
  → {{"status":"ready","requests":[{{"source_text":"장학금 규정","operation":"document_explain"}},{{"source_text":"전체 목록","operation":"list_records"}}],"reason":"규정 검색과 표 전체 목록이라는 두 요청","retrieval_query":"장학금 규정"}}
- "현재 적재된 문서 목록"
  → {{"status":"ready","requests":[{{"source_text":"현재 적재된 문서 목록","operation":"list_documents"}}],"reason":"적재 파일 목록 요청"}}

현재 조회 가능한 표:
{schema}

사용자 질문:
{question}

JSON:"""


_QUESTION_ENGINE_REPAIR_TEMPLATE = """\
이전 응답이 질문 결정 JSON 규격을 통과하지 못했습니다.
질문의 의미를 바꾸지 말고 JSON 문법과 필드 규격만 수정하세요.
Python, Markdown, 설명문 없이 JSON 객체 하나만 반환하세요.

허용 규칙:
- status=ready이면 requests 배열이 필수
- requests의 각 항목에는 질문에서 그대로 복사한 source_text와 operation이 필수
- 서로 다른 답을 요구하는 요청을 하나로 합치거나 생략하지 않음
- 허용 operation:
  list_documents, filter_records, compare, max_person_by_amount,
  min_person_by_amount, list_records, count_records, sum_amount,
  average_amount, median_amount, mode_amount, max_amount, min_amount,
  lookup_amount, lookup_field, structured_query, document_reason, document_purpose,
  document_criteria, document_procedure, document_explain
- 위 목록에 없는 operation을 새로 만들지 않음
- document operation이 있으면 retrieval_query 필수
- document operation이 없으면 retrieval_query를 넣지 않음
- operations, route, intent, query, filters, Python 코드를 넣지 않음
- status=clarification 또는 unsupported이면 operations 없이 message 필수

질문:
{question}

검증 오류:
{error}

이전 응답:
{response}

수정된 JSON:"""

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

상위 질문 분류 결과:
{operation_hint}
- 상위 분류가 lookup_field이면 질문을 개수 질문으로 다시 해석하지 마세요.
- lookup_field는 반드시 operation=list, 하나 이상의 대상 식별 filters, 하나 이상의
  조회 대상 select를 포함해야 합니다.
- lookup_field에서 사람 이름·기관명·항목명은 대상을 찾는 필터 값이고, 전화번호·
  이메일·전공·날짜처럼 질문에서 알고 싶다고 한 필드는 반환 컬럼입니다. 반환 컬럼에
  대상 이름을 넣어 필터링하거나, 반환값의 존재 여부를 질문에 없는 조건으로 추가하지 마세요.

연산 선택:
- 일치하는 행, 대상, 명단 또는 각 항목의 내용을 요청하면 list를 사용하세요.
- 특정 대상의 특정 컬럼값을 요청하면 list를 사용하고, 대상을 식별하는 컬럼은
  filters에, 사용자가 요청한 컬럼은 select에 포함하세요. select에는 결과를 구분할
  대상 식별 컬럼도 함께 포함하세요.
- 오직 개수나 몇 개인지를 요청할 때만 count를 사용하세요.
- "사람 몇 명", "인원 수"는 중복 결제 행 수가 아니라 사람 식별 컬럼의 고유값
  개수입니다. operation=count와 distinct_by=[사람 식별 컬럼]을 사용하세요.
- "몇 건", "행 몇 개", "기록 수"라고 명시했을 때만 distinct_by 없이 행을 세세요.
- 합계, 평균, 중앙값, 최빈값은 각각 sum, mean, median, mode를 사용하세요.
- 가장 크거나 작은 값 또는 해당 행을 요청하면 max, min을 사용하세요.

JSON 규격:
1. 실행 가능한 경우의 공통 필드:
{{
  "status": "ready",
  "dataframe": "실제 DataFrame 별칭",
  "operation": "list|count|sum|mean|median|mode|min|max|group_sum",
  "filters": [
    {{
      "column": "실제 컬럼명",
      "operator": "eq|ne|gt|gte|lt|lte|contains|in|between|is_null|not_null",
      "value": "연산자에 맞는 값",
      "case_sensitive": false,
      "source_text": "이 필터를 뒷받침하는 질문의 가장 짧은 원문 구절"
    }}
  ],
  "filter_logic": "all|any"
}}

공통 필드 외에는 선택한 연산에 필요한 필드만 추가하세요:
- list: select, 필요한 경우에만 sort, distinct_by, limit
- count: 필요한 경우에만 target 또는 distinct_by. result_mode와 select는 넣지 않음
- sum, mean, median, mode: target만 추가. result_mode와 select는 넣지 않음
- group_sum: target 금액을 group_by 컬럼별로 합산하고 group_order(asc/desc)와 top_n으로 그룹 순위를 반환
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
- "사람별 합계가 가장 큰/많은 사람"은 원본 한 행의 max가 아니라 operation=group_sum, group_by=사람 컬럼, group_order=desc를 사용하세요.
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
- 질문에 나온 모든 조건은 filters에 하나씩 보존하세요. 목록·합계·개수 연산을
  선택했다는 이유로 금액, 날짜, 범주, 결측 조건을 생략하지 마세요.
- 질문에서 "A가 B", "A이 B", "A인"처럼 컬럼과 값이 함께 명시되면 A를 filter.column,
  B를 filter.value로 사용하세요. 두 역할을 뒤집지 마세요.
- "비어 있는", "안 적은", "없는"은 해당 컬럼의 is_null 필터입니다. 질문이 값을
  반환하라고 한 경우에는 not_null 조건을 임의로 추가하지 마세요.

자료형별 필터 규칙:
- contains는 문자열 컬럼에만 사용하며 value에는 정규식이 아닌 실제 검색 문자열을 넣으세요.
- 숫자와 금액 컬럼에는 eq, ne, gt, gte, lt, lte, in, between만 사용하세요.
- 질문에 나온 숫자와 단위 표현은 직접 환산하거나 자릿수를 바꾸지 말고 그대로 value에 보존하세요.
- 금액 표기 "20,000원", "2만원", "20000"은 모두 금액 컬럼 조건입니다. 원문 표기는
  value와 source_text에 보존하고, 전화번호·연도·일반 숫자 컬럼으로 보내지 마세요.
- 범주값은 질문에 나온 실제 문자열을 그대로 사용하세요. 스키마의 범주형 컬럼명과
  질문의 값이 함께 제시되면 clarification으로 회피하지 말고 그 컬럼을 필터링하세요.
- 모든 필터의 source_text는 질문에서 글자 하나도 바꾸지 않고 그대로 복사하세요.
- 숫자 비교 필터의 source_text는 "200만원 이상", "49기 이상"처럼
  해당 숫자·단위·비교 표현 하나만 포함하는 가장 짧은 원문 구절이어야 합니다.
- 질문에 그대로 존재하는 source_text를 제시할 수 없다면 해당 필터를 만들지 마세요.
- "이상"은 gte, "초과"는 gt, "이하"는 lte, "미만"은 lt를 사용하세요.
- 날짜 컬럼에는 eq, ne, gt, gte, lt, lte, in, between만 사용하고 값은 YYYY-MM-DD 형식으로 작성하세요.
- 날짜의 월 범위는 해당 월의 시작일과 마지막 날을 between의 두 값으로 표현하세요.
- 질문이나 스키마에서 연도를 확정할 수 없다면 임의로 연도를 만들지 말고 clarification을 반환하세요.
- 스키마에 연도·월 분리 컬럼과 전체 날짜 컬럼이 함께 있을 때, 질문이 "N년", "N년 M월"
  같은 달력 조건이면 분리된 연도·월 컬럼을 우선 필터링하세요. 이 경우 날짜 후보가
  여러 개라는 이유만으로 clarification을 반환하지 마세요.
- 질문이 "등록 날짜", "지급일", "발행일"처럼 날짜 컬럼 자체를 이름으로 요청하면
  그 전체 날짜 컬럼은 select이고, 질문에 함께 나온 연도는 분리된 연도 컬럼의 필터입니다.
- 사람 이름이 전체 값과 정확히 일치한다고 단정할 근거가 없고 괄호·접미 표현이 저장될
  수 있는 스키마라면 contains를 사용해 부분 이름을 안전하게 찾으세요.

역할 예시(예시의 이름과 값은 실제 답이 아니라 구조만 설명합니다):
- "홍길동 전화번호" → filters=[사람이름 컬럼 = 홍길동], select=[사람이름 컬럼, 전화번호]
- "홍길동 전공과 이메일" → filters=[사람이름 컬럼 = 홍길동], select=[사람이름 컬럼, 전공, 이메일]
- "회비 종류가 정기회비인 사람 몇 명" → filters=[회비종류 = 정기회비],
  operation=count, distinct_by=[사람이름 컬럼]
- "2024년 3월 2만원 낸 사람" → 연도=2024 AND 월=3 AND 금액=2만원의 세 필터
- "이메일을 안 적은 사람" → filters=[이메일 is_null], operation=list

실제 DataFrame 스키마:
{schema}

사용자 질문:
{question}

JSON:"""

_QUERY_PLAN_REPAIR_TEMPLATE = """\
이전 응답은 QueryPlan JSON 규격을 통과하지 못했습니다.
질문을 다시 해석하거나 조회 조건을 추가·삭제·완화하지 말고 JSON 문법과 규격만 수정하세요.
Python 코드, Markdown, 설명문 없이 수정된 JSON 객체 하나만 반환하세요.

상위 질문 분류 결과:
{operation_hint}
- lookup_field이면 operation=list, 대상 식별 filters, 조회 대상 select가 필수입니다.
- lookup_field의 filters는 사람·기관·항목 등 대상을 식별해야 하고, select는 질문에서
  요청한 반환 컬럼이어야 합니다. 이름을 전화번호·이메일·전공 컬럼의 값으로 넣지 마세요.

최소 수정 규칙:
- 허용 상태: ready, clarification, not_applicable
- 허용 연산: list, count, sum, mean, median, mode, min, max, group_sum
- list는 target 없이 행 반환
- count, sum, mean, median, mode는 result_mode와 select를 사용하지 않음
- sum, mean, median, mode, min, max는 target 필수
- min, max에서 행 반환이 필요할 때만 result_mode=records 사용
- group_sum은 target 금액 컬럼과 하나 이상의 group_by 컬럼을 사용
- 이전 응답에 없던 dataframe, 컬럼, 필터 값은 새로 만들지 않음
- 각 필터의 source_text는 사용자 질문에서 그대로 복사한 가장 짧은 근거 구절
- 숫자·단위·비교 표현은 source_text에서 환산하거나 변경하지 않음
- "사람 몇 명"은 count와 distinct_by=[사람 식별 컬럼]을 사용
- 질문의 금액·날짜·범주·결측 조건을 누락하지 않음
- 동일 대상의 여러 반환 컬럼은 하나의 list 계획의 select에 함께 포함

사용자 질문:
{question}

검증 오류:
{error}

이전 응답:
{response}

수정된 JSON:"""

RAG_PROMPT = PromptTemplate.from_template(_RAG_TEMPLATE)
DOC_EXPLAIN_RAG_PROMPT = PromptTemplate.from_template(_DOC_EXPLAIN_RAG_TEMPLATE)
