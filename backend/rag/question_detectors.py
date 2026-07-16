from __future__ import annotations

import re
from dataclasses import dataclass, field

from pandas_engine.aggregation import AggregationIntent


_MEANINGFUL_WORDS = re.compile(
    r"장학|지원금|기부|후원|출연|예산|집행|지급|수혜|금액|총액|합계|평균|"
    r"명단|목록|인원|기수|발행번호|기준|조건|목적|이유|절차|규정|내용|"
    r"누구|누가|얼마|중앙값|최빈값|비교|차이|조회|설명",
    re.IGNORECASE,
)
_VAGUE_REFERENCE = re.compile(
    r"이거|그거|저거|이\s*사람|그\s*사람|저\s*사람|아까\s*(?:그거|그\s*사람)|"
    r"저번\s*거|지난\s*거|전에\s*올린\s*거|방금\s*그거",
    re.IGNORECASE,
)
_LOOKUP_WORDS = re.compile(
    r"얼마|금액|총액|합계|명단|목록|인원|기준|조건|목적|내용|절차|방법|"
    r"누구|누가|조회|찾아|보여|알려|설명",
    re.IGNORECASE,
)
_BARE_REQUEST = re.compile(
    r"^(?:얼마|금액|총액|합계|명단|목록|리스트|인원|기준|조건|목적|내용|"
    r"절차|방법|누구|누가|알려줘|보여줘|조회해줘|찾아줘|설명해줘)\s*[?？]?$",
    re.IGNORECASE,
)
_CONNECTOR = re.compile(r"그리고|또한|추가로|동시에|함께|및|하고|랑|이랑|와|과|,|/|&")

_COMPARE = re.compile(r"비교|차이|다른점|같은점", re.IGNORECASE)
_EXPLICIT_LIST = re.compile(r"명단|목록|리스트", re.IGNORECASE)
_DOCUMENT_INVENTORY = re.compile(
    r"(?:전체|모든|전부|적재(?:된)?|등록(?:된)?|업로드(?:된)?|현재)?\s*"
    r"(?:문서|파일)(?:들)?(?:을|를|의|은|는)?\s*"
    r"(?:목록|리스트|보여|알려|조회|확인)",
    re.IGNORECASE,
)
_ENTITY_NOUN = re.compile(r"수혜자|출연자|기부자", re.IGNORECASE)
_AMOUNT = re.compile(
    r"얼마|얼마나|금액|총액|지급액|출연액|기부액|후원액|장학금액|지원금액|예산액|집행액",
    re.IGNORECASE,
)
_REASON = re.compile(r"왜|이유|사유|원인|배경|근거", re.IGNORECASE)
_PURPOSE = re.compile(r"목적|취지|용도", re.IGNORECASE)
_CRITERIA = re.compile(r"기준|조건|자격|요건|규정|선발", re.IGNORECASE)
_PROCEDURE = re.compile(r"절차|방법|서류|신청|문의|제출|접수|심사", re.IGNORECASE)
_HOW_RESULT = re.compile(
    r"어떻게\s*(?:돼|되(?:나요|니|지|는지)?|됩니까|나와|나오(?:나요|니|지|는지)?)",
    re.IGNORECASE,
)
_HOW_PROCEDURE = re.compile(
    r"어떻게\s*(?:신청|제출|접수|선발|지급|처리|계산|산정)",
    re.IGNORECASE,
)
_EXPLAIN = re.compile(r"문서.*(?:내용|설명|요약)|설명해|요약해|어떤\s*문서", re.IGNORECASE)

_PANDAS_KEYWORDS = re.compile(
    r"명단|몇\s*명|\d+\s*명|인원|금액|얼마|통계|집계|합계|총\s*금액|지급액|목록|리스트|누가|누구|현황|조회|어느\s*학과|무슨\s*학과|어느\s*반|종목"
    r"|받았어|받았나|있어\?|있나|있는지|수혜자|받은\s*학생|수혜\s*받",
    re.IGNORECASE,
)
_VECTOR_PROCEDURE = re.compile(
    r"방법|절차|기준|서류|자격|안내|규정|내용|제도|신청|문의|어떻게|왜|이유|달라|같아|차이|비교",
    re.IGNORECASE,
)
_AGG_PROCEDURE = re.compile(r"계산\s*방법|산정\s*방법|방법|절차|기준|규정|공식", re.IGNORECASE)
_VECTOR_OVERRIDE = re.compile(
    r"설명해|설명해줘|목적|문서의\s*내용|내용을\s*설명|어떤\s*내용|몇\s*월|몇\s*년|날짜|작성됐|어느\s*학교"
    r"|어디서|어느\s*기관|기관명|단체명|출처",
    re.IGNORECASE,
)

STRUCTURED_OPERATIONS = frozenset({
    "compare", "max_person_by_amount", "min_person_by_amount", "list_records", "filter_records",
    "count_records", "sum_amount", "average_amount", "median_amount",
    "mode_amount", "max_amount", "min_amount", "lookup_amount",
})
DOCUMENT_INVENTORY_OPERATIONS = frozenset({"list_documents"})
DOCUMENT_OPERATIONS = frozenset({
    "document_reason", "document_purpose", "document_criteria",
    "document_procedure", "document_explain",
})


@dataclass
class QuestionSignals:
    operations: list[str] = field(default_factory=list)
    is_meaningless: bool = False
    is_vague: bool = False
    is_bare_request: bool = False
    has_connector: bool = False
    has_pandas_keyword: bool = False
    has_vector_procedure: bool = False
    has_aggregation_procedure: bool = False
    has_vector_override: bool = False
    has_scholarship_keyword: bool = False


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip())


def _is_meaningless_short_input(question: str) -> bool:
    compact = re.sub(r"[\s!?~.,]+", "", question.lower())
    if len(compact) > 12 or _MEANINGFUL_WORDS.search(question):
        return False
    tokens = re.findall(r"[가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]+", question.lower())
    meaningless = {
        "어이", "어어", "야", "음", "흠", "엥", "뭐", "ㅎㅇ", "하이", "안녕",
        "ㅋㅋ", "ㅎㅎ", "ㅇㅇ", "ㄴㄴ", "ㅇㅋ", "오케이", "test", "asdf",
    }
    return bool(
        compact in meaningless
        or (tokens and all(token in meaningless for token in tokens))
        or re.fullmatch(r"[a-zA-Z]{1,8}", compact)
        or re.fullmatch(r"[ㄱ-ㅎㅏ-ㅣㅋㅎㅠㅜ]+", compact)
    )


def _operation_for_aggregation(intent: AggregationIntent) -> str:
    if intent.operation == "max" and intent.target in {"person_total", "row"}:
        return "max_person_by_amount"
    if intent.operation == "min" and intent.target in {"person_total", "row"}:
        return "min_person_by_amount"
    return {
        "count": "count_records",
        "sum": "sum_amount",
        "mean": "average_amount",
        "per_capita": "average_amount",
        "median": "median_amount",
        "mode": "mode_amount",
        "max": "max_amount",
        "min": "min_amount",
    }[intent.operation]


def _detect_operations(question: str, intents: list[AggregationIntent]) -> list[str]:
    operations: list[str] = []

    def add(operation: str) -> None:
        if operation not in operations:
            operations.append(operation)

    document_inventory = bool(_DOCUMENT_INVENTORY.search(question))
    if document_inventory:
        add("list_documents")

    if _COMPARE.search(question):
        add("compare")
    else:
        # 수혜자·출연자·기부자는 순위 질문의 대상 명사로도 쓰인다. 명시적인
        # 목록 표현이 있거나 집계 의도가 없는 단독 조회일 때만 목록 작업이다.
        if (not document_inventory and _EXPLICIT_LIST.search(question)) or (
            _ENTITY_NOUN.search(question) and not intents
        ):
            add("list_records")
        for intent in intents:
            add(_operation_for_aggregation(intent))
        if _AMOUNT.search(question) and not any(
            operation in operations
            for operation in (
                "sum_amount", "average_amount", "median_amount", "mode_amount",
                "max_amount", "min_amount",
            )
        ):
            add("lookup_amount")

    if _REASON.search(question):
        add("document_reason")
    if _PURPOSE.search(question):
        add("document_purpose")
    if _CRITERIA.search(question):
        add("document_criteria")
    if (
        _PROCEDURE.search(question)
        or _HOW_PROCEDURE.search(question)
        or (
            "어떻게" in question
            and not intents
            and not _HOW_RESULT.search(question)
        )
    ):
        add("document_procedure")
    if _EXPLAIN.search(question):
        add("document_explain")
    return operations


def detect_question_signals(
    question: str,
    aggregation_intents: list[AggregationIntent],
) -> QuestionSignals:
    return QuestionSignals(
        operations=_detect_operations(question, aggregation_intents),
        is_meaningless=_is_meaningless_short_input(question),
        is_vague=bool(_VAGUE_REFERENCE.search(question) and _LOOKUP_WORDS.search(question)),
        is_bare_request=bool(_BARE_REQUEST.fullmatch(question)),
        has_connector=bool(_CONNECTOR.search(question)),
        has_pandas_keyword=bool(_PANDAS_KEYWORDS.search(question)),
        has_vector_procedure=bool(_VECTOR_PROCEDURE.search(question)),
        has_aggregation_procedure=bool(_AGG_PROCEDURE.search(question)),
        has_vector_override=bool(_VECTOR_OVERRIDE.search(question)),
        has_scholarship_keyword="장학" in question,
    )


def operation_domains(operations: list[str]) -> list[str]:
    domains: list[str] = []
    if any(operation in DOCUMENT_INVENTORY_OPERATIONS for operation in operations):
        domains.append("document_inventory")
    if any(operation in STRUCTURED_OPERATIONS for operation in operations):
        domains.append("structured_data")
    if any(operation in DOCUMENT_OPERATIONS for operation in operations):
        domains.append("document_evidence")
    return domains


def is_vector_override_question(question: str) -> bool:
    return bool(_VECTOR_OVERRIDE.search(question or ""))
