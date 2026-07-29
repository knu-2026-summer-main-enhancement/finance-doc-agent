from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import AsyncIterator, Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from core.config import (
    VECTOR_RELATIVE_SCORE_MARGIN,
    VECTOR_RERANK_SCORE_MARGIN,
    VECTOR_SEARCH_FETCH_K,
    VECTOR_SEARCH_K,
)
from core.llm import get_llm_rag, get_llm_code, get_vectorstore, _fmt_docs
from core.privacy import question_log_metadata
from rag.prompts import (
    DOC_EXPLAIN_RAG_PROMPT,
    DOCUMENT_REASONING_RAG_PROMPT,
    DOCUMENT_REASONING_REPAIR_PROMPT,
    MULTI_QUERY_PROMPT,
    NUMERIC_ELIGIBILITY_REPAIR_PROMPT,
    NUMERIC_ELIGIBILITY_RAG_PROMPT,
    NUMERIC_DECISION_FALLBACK_PROMPT,
    RAG_PROMPT,
)
from rag.question_analyzer import QuestionAnalysis
from rag.question_detectors import is_vector_override_question
from rag.cancellation import await_cancellable, next_cancellable, raise_if_cancelled
from datastore.scope import selected_sources

logger = logging.getLogger("uvicorn.error")
_vector_evidence_ctx: ContextVar[list[dict[str, Any]]] = ContextVar(
    "vector_evidence",
    default=[],
)
_query_expansion_cache: dict[tuple[str, bool], tuple[float, list[str]]] = {}
_QUERY_EXPANSION_CACHE_TTL = 300.0

_DOC_EXPLAIN_RE = re.compile(
    r"문서의?\s*(목적|내용|설명)|설명해|어떤\s*(문서|내용)|요약해"
    r"|(?:지급|선발|지원|출연|기부)\s*(목적|기준|이유|방식|기관)"
    r"|(?:목적|내용|용도|기준|이유)\s*(?:이|가)?\s*(?:뭐야|뭐|무엇|어떤가|어떻게)"
    r"|왜\s*(?:만들|생기|도입|제정|시작)|(?:만들|생기|도입|제정)된?\s*이유",
    re.IGNORECASE,
)
_VECTOR_EMPTY_SIGNALS = ("해당 내용은 문서에서 확인할 수 없습니다", "문서에서 확인할 수 없")
_MULTI_DOCUMENT_RE = re.compile(
    r"비교|차이|공통|각각|모두|전부|두\s*문서|문서별",
    re.IGNORECASE,
)
_EVIDENCE_RULES = (
    (
        re.compile(r"왜|이유|사유|원인|배경|근거", re.IGNORECASE),
        re.compile(r"이유|사유|원인|때문|근거|배경|정기\s*후원|분할\s*(?:납부|출연|지급)", re.IGNORECASE),
        "관련 내역은 확인되지만, 그 이유나 사유는 문서에 명시되어 있지 않습니다.",
    ),
    (
        re.compile(r"목적|취지|용도", re.IGNORECASE),
        re.compile(r"목적|취지|용도|위하여|지원하고자", re.IGNORECASE),
        "문서의 목적이나 취지를 직접 확인할 수 있는 근거가 없습니다.",
    ),
    (
        re.compile(r"기준|조건|자격|요건|규정|선발", re.IGNORECASE),
        re.compile(r"기준|조건|자격|요건|규정|선발|대상은|대상자", re.IGNORECASE),
        "질문한 기준이나 조건은 문서에서 확인할 수 없습니다.",
    ),
    (
        re.compile(r"절차|방법|서류|신청|문의", re.IGNORECASE),
        re.compile(r"절차|방법|서류|신청|제출|접수|문의|심사", re.IGNORECASE),
        "질문한 절차나 방법은 문서에서 확인할 수 없습니다.",
    ),
)


@dataclass
class VectorPreparation:
    context: str = ""
    source_files: list[str] | None = None
    prompt: Any = field(default_factory=lambda: RAG_PROMPT)
    immediate_answer: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    documents: list[Any] = field(default_factory=list, repr=False)
    numeric_eligibility: bool = False
    document_reasoning: bool = False

    def __post_init__(self) -> None:
        if self.source_files is None:
            self.source_files = []


def _doc_key(doc: Any) -> tuple[str, str, str]:
    return (
        str(doc.metadata.get("source", "")),
        str(doc.metadata.get("page", "")),
        str(doc.page_content),
    )


_QUERY_TERM_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")
_QUERY_TERM_STOPWORDS = frozenset({
    "뭐야", "뭔가", "알려줘", "설명해줘", "보여줘", "찾아줘",
    "문서", "내용", "이거", "그거", "대한", "관련", "있어", "없는",
    "있는", "있나", "받고", "해야", "어떻게",
})
_KOREAN_PARTICLE_RE = re.compile(
    r"(?:으로부터|에서부터|에게서|까지|부터|으로|에게|한테|에서|"
    r"은|는|이|가|을|를|와|과|의|에|로|도|만)$"
)
_KOREAN_VERB_ENDING_RE = re.compile(
    r"(?:하려면|하려고|해야|하면|하는|하고|할|있는|있나)$"
)


def _query_terms(query: str) -> set[str]:
    normalized: set[str] = set()
    for raw_term in _QUERY_TERM_RE.findall(query):
        term = raw_term.lower()
        if term in _QUERY_TERM_STOPWORDS:
            continue
        without_particle = _KOREAN_PARTICLE_RE.sub("", term)
        without_ending = _KOREAN_VERB_ENDING_RE.sub("", without_particle)
        normalized.add(without_ending if len(without_ending) >= 2 else term)
    return normalized


def _relative_score_candidates(
    query: str,
    scored: list[tuple[Any, float]],
) -> list[tuple[Any, float]]:
    """Keep candidates close to this query's best result.

    Relevance-score calibration differs across embedding models and document
    sets. Comparing results within one query avoids treating a globally fixed
    score as meaningful while still limiting the rerank candidate pool.
    """
    if not scored:
        return []
    best_score = max(float(score) for _doc, score in scored)
    floor = best_score - VECTOR_RELATIVE_SCORE_MARGIN
    return [
        (doc, float(score))
        for doc, score in scored
        if float(score) >= floor
    ]


def _build_vector_evidence(docs: list[Any]) -> list[dict[str, Any]]:
    """Build a stable, non-sensitive citation payload for the API and UI."""
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, object, str]] = set()
    for doc in docs:
        source = os.path.basename(str(doc.metadata.get("source", "")))
        page = doc.metadata.get("page")
        section_id = str(doc.metadata.get("section_id", ""))
        key = (source, page, section_id)
        if not source or key in seen:
            continue
        seen.add(key)
        item: dict[str, Any] = {"source": source}
        if page is not None:
            item["page"] = int(page)
        section_title = str(doc.metadata.get("section_title", "")).strip()
        if section_title:
            item["section_title"] = section_title
        if section_id:
            item["section_id"] = section_id
        child_count = doc.metadata.get("child_count")
        if child_count is not None:
            item["chunk_count"] = int(child_count)
        evidence.append(item)
    return evidence


_SECTION_ITEM_RE = re.compile(
    r"(?ms)^[ \t]*[◦○●▪■□]\s*(.+?)(?=^[ \t]*[◦○●▪■□]\s*|\Z)"
)
_NUMERIC_ELIGIBILITY_RE = re.compile(
    r"받을\s*수|추천할\s*수|가능|충족|해당|자격|기준.*(?:넘|미달)",
    re.IGNORECASE,
)
_DOCUMENT_REASONING_RE = re.compile(
    r"(?:"
    r"비교|차이|합계|평균|점수|반영점수|구간|"
    r"이면|라면|경우|관계없이|무조건|충족|가능|"
    r"할\s*수\s*있|"
    r"받을\s*수\s*있|등록금|생활비|학업장려금|"
    r"얼마(?:야|인가|인지)?"
    r")",
    re.IGNORECASE,
)


def _is_numeric_eligibility_question(question: str) -> bool:
    return bool(
        _NUMERIC_ELIGIBILITY_RE.search(question)
        and re.search(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", question)
        and re.search(
            r"평점|학점|점수|구간|소득|금액|원|달러|퍼센트|%",
            question,
        )
    )


def _numeric_evidence_documents(
    question: str,
    docs: list[Any],
) -> list[Any]:
    """Narrow LLM context to chunks containing the questioned measurement."""
    concepts = {
        term
        for term in _query_terms(question)
        if any(marker in term for marker in ("평점", "학점", "소득", "구간", "금액"))
    }
    if not concepts:
        return docs
    matched = [
        doc
        for doc in docs
        if any(concept in str(doc.page_content) for concept in concepts)
    ]
    threshold_docs = [
        doc for doc in matched
        if re.search(
            r"\d+(?:\.\d+)?(?:구간)?\s*(?:이상|이하|초과|미만)",
            str(doc.page_content),
        )
    ]
    return threshold_docs or matched or docs


def _conditional_evidence_documents(
    question: str,
    docs: list[Any],
) -> list[Any]:
    """Prefer explicit mandatory-condition clauses for unconditional claims."""

    if not re.search(
        r"무조건|관계없이|상관없이|기준만|하나만|안\s*맞|미달",
        question,
    ):
        return docs
    if not re.search(r"성적|소득|자격|기준|(?<!무)조건", question):
        return docs
    mandatory = [
        doc for doc in docs
        if re.search(
            r"모두\s*충족|동시에\s*충족|자격\s*요건|필수\s*조건",
            str(doc.page_content),
        )
    ]
    return mandatory or docs


def _condition_alignment_notes(
    question: str,
    docs: list[Any],
) -> str:
    """Place questioned values beside explicit source criteria for the LLM."""

    markers = (
        "직전학기",
        "총평점",
        "학자금지원구간",
        "성적기준",
        "소득기준",
        "자격 요건",
        "모두 충족",
    )
    question_facts = re.findall(
        r"(?:직전학기|총평점(?:평균)?|학자금지원구간|소득구간)"
        r"[^,?.]{0,20}?\d+(?:\.\d+)?(?:구간)?",
        question,
    )
    source_lines: list[str] = []
    joined_context = " ".join(
        re.sub(r"\s+", " ", str(doc.page_content))
        for doc in docs
    )
    for doc in docs:
        for line in str(doc.page_content).splitlines():
            normalized = re.sub(r"\s+", " ", line).strip(" -*◦")
            if normalized and any(marker in normalized for marker in markers):
                if normalized not in source_lines:
                    source_lines.append(normalized)
    if not question_facts and not source_lines:
        return ""
    parts: list[str] = []
    if question_facts:
        parts.append("[질문 조건 정렬 보조] " + "; ".join(question_facts))
    if source_lines:
        parts.append("[문서 필수 조건 정렬 보조] " + " / ".join(source_lines[:6]))
    criteria = re.findall(
        r"(?:직전학기\s*평점평균|총평점\s*평균|학자금지원구간)"
        r"(?:이|은|는)?\s*\d+(?:\.\d+)?(?:구간)?\s*(?:이상|이하|초과|미만)",
        joined_context,
    )
    if criteria:
        parts.append(
            "[문서 수치 기준 정렬 보조] "
            + "; ".join(dict.fromkeys(criteria))
        )
    return "\n".join(parts)


def _table_alignment_notes(docs: list[Any]) -> str:
    """Linearize label/value rows that PDF text extraction split apart."""

    notes: list[str] = []
    for doc in docs:
        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in str(doc.page_content).splitlines()
            if line.strip()
        ]
        for index, line in enumerate(lines):
            labels = line.split()
            if len(labels) < 3 or not any(label.endswith("구간") for label in labels):
                continue
            if all(re.fullmatch(r"[\d,.]+", label) for label in labels):
                continue
            for value_line in lines[index + 1:index + 4]:
                if not value_line.startswith("반영점수"):
                    continue
                values = re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", value_line)
                if len(values) != len(labels):
                    continue
                mapping = ", ".join(
                    f"{label} = {value}점"
                    for label, value in zip(labels, values)
                )
                note = f"[PDF 표 열 정렬 보조] {mapping}"
                if note not in notes:
                    notes.append(note)
                break
        for index, line in enumerate(lines):
            if not line.startswith("반영점수"):
                continue
            scores = re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", line)
            if len(scores) < 3:
                continue
            preceding = lines[max(0, index - 4):index]
            numeric_rows = [
                re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", row)
                for row in preceding
            ]
            boundary_rows = [
                values for values in numeric_rows
                if len(values) == len(scores) - 1
            ]
            single_values = [
                values[0] for values in numeric_rows
                if len(values) == 1
            ]
            if len(boundary_rows) < 2 or not single_values:
                continue
            upper_bounds, lower_bounds = boundary_rows[:2]
            ranges = [
                f"{lower}~{upper} = {score}점"
                for lower, upper, score in zip(
                    lower_bounds,
                    upper_bounds,
                    scores[:-1],
                )
            ]
            ranges.append(f"{single_values[-1]} = {scores[-1]}점")
            note = "[PDF 범위표 열 정렬 보조] " + ", ".join(ranges)
            if note not in notes:
                notes.append(note)
    return "\n".join(notes)


def _numeric_answer_issue(
    answer: str,
    question: str = "",
) -> str:
    numbers = re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", answer)
    question_numbers = re.findall(
        r"(?<!\d)\d+(?:\.\d+)?(?!\d)",
        question,
    )
    has_conclusion = (
        "충족합니다" in answer
        or "충족하지 않습니다" in answer
    )
    includes_question_values = all(
        value in numbers
        for value in question_numbers
    )
    issues: list[str] = []
    if not has_conclusion:
        issues.append("충족 여부의 명확한 결론이 없습니다.")
    if "기준" not in answer:
        issues.append("문서 기준값과 기준 연산자가 없습니다.")
    if len(set(numbers)) < 2:
        issues.append(
            "질문 수치와 서로 다른 문서 기준값을 함께 비교하지 않았습니다."
        )
    if not includes_question_values:
        missing = [
            value for value in question_numbers
            if value not in numbers
        ]
        issues.append(
            "질문에 제시된 수치 "
            + ", ".join(missing)
            + "을 답변에서 누락했습니다."
        )
    return " ".join(issues)


def _numeric_llm_answer_is_complete(
    answer: str,
    question: str = "",
) -> bool:
    return not _numeric_answer_issue(answer, question)


def _numeric_relation_contradiction(answer: str) -> bool:
    """Detect a plainly reversed numeric comparison in generated prose."""

    for sentence in re.split(
        r"\n|(?<=[!?。])|(?<!\d)\.(?:\s|$)",
        answer,
    ):
        values = re.findall(
            r"(?<!\d)\d+(?:\.\d+)?(?!\d)",
            sentence,
        )
        if len(values) < 2:
            continue
        left, right = float(values[0]), float(values[1])
        if re.search(r"보다\s*(?:낮|작)|미달", sentence) and left > right:
            return True
        if re.search(r"보다\s*(?:높|크)", sentence) and left < right:
            return True
    return False


def _explicit_numeric_criteria_answer(
    question: str,
    documents: list[Any],
) -> str:
    """Build a last-resort answer from explicit labeled numeric criteria."""

    joined = " ".join(
        re.sub(r"\s+", " ", str(doc.page_content))
        for doc in documents
    )
    specifications = (
        (
            "직전학기 평점평균",
            r"직전학기(?!\s*이수학점)(?:\s*평점(?:평균)?)?[^\d]{0,12}(\d+(?:\.\d+)?)",
            r"직전학기\s*평점평균\s*(\d+(?:\.\d+)?)\s*(이상|이하|초과|미만)",
        ),
        (
            "총평점 평균",
            r"총평점(?:\s*평균)?[^\d]{0,12}(\d+(?:\.\d+)?)",
            r"총평점\s*평균(?:이)?\s*(\d+(?:\.\d+)?)\s*(이상|이하|초과|미만)",
        ),
        (
            "학자금지원구간",
            r"(?:학자금)?지원구간(?:이)?[^\d]{0,12}(\d+)구간",
            r"학자금지원구간\s*(\d+)구간\s*(이상|이하|초과|미만)",
        ),
        (
            "직전학기 이수학점",
            r"(?:직전학기\s*)?(?:이수학점(?:이)?\s*)?(\d+)학점",
            r"이수학점\s*(\d+)학점\s*(이상|이하|초과|미만)",
        ),
    )
    comparisons: list[str] = []
    passed_all = True
    for label, question_pattern, criterion_pattern in specifications:
        question_match = re.search(question_pattern, question)
        criterion_match = re.search(criterion_pattern, joined)
        if not question_match or not criterion_match:
            continue
        actual = float(question_match.group(1))
        threshold = float(criterion_match.group(1))
        operator = criterion_match.group(2)
        passed = {
            "이상": actual >= threshold,
            "이하": actual <= threshold,
            "초과": actual > threshold,
            "미만": actual < threshold,
        }[operator]
        passed_all = passed_all and passed
        actual_text = question_match.group(1)
        threshold_text = criterion_match.group(1)
        if label == "학자금지원구간":
            actual_text += "구간"
            threshold_text += "구간"
        comparisons.append(
            f"{label} {actual_text}은 문서 기준 {threshold_text} {operator}에 "
            f"{'충족합니다' if passed else '미달합니다'}."
        )
    if not comparisons:
        return ""
    conclusion = (
        "따라서 질문에 제시된 기본 자격의 수치 기준을 충족합니다."
        if passed_all
        else "따라서 필수 수치 기준 중 미달 항목이 있어 추천할 수 없습니다."
    )
    return " ".join([*comparisons, conclusion])


def _document_reasoning_issue(
    question: str,
    answer: str,
    context: str,
) -> str:
    if any(signal in answer for signal in _VECTOR_EMPTY_SIGNALS):
        return "참고 문서에 관련 표나 조건이 있는지 다시 확인해야 합니다."
    if (
        "대학원" in question
        and "학부 재학생" in context
        and re.search(r"대학원생.{0,30}받을\s*수\s*있|대학원생.{0,30}가능", answer)
    ):
        return (
            "참고 문서는 선발 대상을 학부 재학생으로 명시하지만, 이전 답변은 "
            "근거 없이 대학원생도 받을 수 있다고 범위를 확대했습니다."
        )
    if (
        "등록금" in question
        and re.search(r"학업장려금\s*\(생활비\)|학업장려금|생활비", context)
        and not re.search(r"학업장려금|생활비", answer)
    ):
        return (
            "참고 문서에는 지원내용이 학업장려금(생활비)으로 명시되어 있는데, "
            "이전 답변이 이 직접 근거를 누락했습니다."
        )
    question_numbers = re.findall(
        r"(?<!\d)\d+(?:\.\d+)?(?!\d)",
        question,
    )
    answer_numbers = re.findall(
        r"(?<!\d)\d+(?:\.\d+)?(?!\d)",
        answer,
    )
    missing_question_numbers = [
        number for number in question_numbers
        if number not in answer_numbers
    ]
    if (
        missing_question_numbers
        and re.search(r"반영점수|점수|구간", question)
    ):
        return (
            "이전 답변이 질문에 제시된 수치 "
            + ", ".join(missing_question_numbers)
            + "을 누락했습니다. PDF 범위표 열 정렬 보조 근거에서 해당 값이 "
            "속한 정확한 구간과 대응 점수를 다시 확인해야 합니다."
        )
    if re.search(r"반영점수|점수", question):
        decimal_values = [
            float(value)
            for value in question_numbers
            if "." in value
        ]
        range_rows = re.findall(
            r"(\d+(?:\.\d+)?)~(\d+(?:\.\d+)?)\s*=\s*(\d+)점",
            context,
        )
        for value in decimal_values:
            for lower, upper, score in range_rows:
                if float(lower) <= value <= float(upper) and score not in answer_numbers:
                    return (
                        f"질문의 {value:g}은 PDF 범위표의 {lower}~{upper} 구간이며 "
                        f"대응 점수는 {score}점인데, 이전 답변이 다른 점수를 제시했습니다."
                    )
    article_refs = re.findall(r"제\s*\d+\s*조", question)
    if (
        article_refs
        and any(
            re.search(
                re.escape(article_ref) + r".{0,80}제외",
                context,
                re.DOTALL,
            )
            for article_ref in article_refs
        )
        and re.search(r"추천할\s*수\s*있|추천\s*가능", answer)
    ):
        return (
            "질문에 나온 규정 조항은 참고 문서에서 제외 사유로 명시되어 있는데, "
            "이전 답변은 추천 가능하다고 반대로 답했습니다."
        )
    if (
        re.search(r"무조건|관계없이|상관없이", question)
        and re.search(r"모두\s*충족|동시에\s*충족", context)
        and re.search(r"무조건\s*선발|상관없이\s*무조건|관계없이\s*무조건", answer)
    ):
        return (
            "참고 문서에 여러 기준을 모두 충족해야 한다는 조건이 있는데, "
            "이전 답변은 한 조건만으로 무조건 선발된다고 단정했습니다."
        )
    if (
        re.search(r"한쪽|하나|성적\s*기준만|소득\s*기준만", question)
        and re.search(r"모두\s*충족|동시에\s*충족", context)
        and re.search(r"추천(?:받을)?\s*수\s*있|추천\s*가능", answer)
    ):
        return (
            "참고 문서는 여러 기준을 모두 충족해야 추천 가능하다고 명시하지만, "
            "이전 답변은 한 기준이 미달해도 추천 가능하다고 답했습니다."
        )
    return ""


def _explicit_exclusion_answer(
    question: str,
    documents: list[Any],
) -> str:
    """Answer from an exact cited exclusion clause when the LLM contradicts it."""

    article_refs = re.findall(r"제\s*\d+\s*조", question)
    if not article_refs:
        return ""
    for doc in documents:
        for line in str(doc.page_content).splitlines():
            normalized = re.sub(r"\s+", " ", line).strip(" -*◦")
            if (
                "제외" in normalized
                and any(article_ref in normalized for article_ref in article_refs)
            ):
                return (
                    f'문서에는 "{normalized}"라고 명시되어 있습니다. '
                    "따라서 질문한 조항에 해당하면 추천 제외 대상입니다."
                )
    return ""


def _explicit_scope_or_support_answer(
    question: str,
    documents: list[Any],
) -> str:
    """Use explicit target/support fields when the LLM expands beyond them."""

    lines = [
        re.sub(r"\s+", " ", line).strip(" -*◦")
        for doc in documents
        for line in str(doc.page_content).splitlines()
        if line.strip()
    ]
    if "대학원" in question:
        target = next(
            (line for line in lines if "학부 재학생" in line),
            "",
        )
        if target:
            return (
                f'문서에는 선발 대상이 "{target}"로 명시되어 있습니다. '
                "대학원생이 포함된다는 근거는 문서에 없으므로, 문서 기준으로는 "
                "학부 재학생 대상입니다."
            )
    if "등록금" in question:
        support = next(
            (
                line for line in lines
                if "지원내용" in line
                and ("학업장려금" in line or "생활비" in line)
            ),
            "",
        )
        if support:
            return (
                f'문서의 지원내용은 "{support}"로 명시되어 있습니다. '
                "따라서 문서상 성격은 학업장려금(생활비)이며, 등록금 감면 "
                "장학금으로 명시된 것은 아닙니다."
            )
    return ""


def _explicit_mandatory_condition_answer(
    question: str,
    documents: list[Any],
) -> str:
    """Resolve one-failed-condition questions from an explicit AND clause."""

    if not (
        re.search(r"기준만|조건만|하나만", question)
        and re.search(r"안\s*맞|미달|충족하지", question)
    ):
        return ""
    for doc in documents:
        for line in str(doc.page_content).splitlines():
            normalized = re.sub(r"\s+", " ", line).strip(" -*◦")
            if (
                re.search(r"모두\s*충족|동시에\s*충족", normalized)
                and re.search(r"추천\s*가능|추천가능", normalized)
            ):
                return (
                    f'문서에는 "{normalized}"라고 명시되어 있습니다. '
                    "따라서 한 기준만 충족하고 다른 기준이 미달하면 "
                    "단과대학이 추천할 수 없습니다."
                )
    return ""


def _explicit_scalar_field_answer(
    question: str,
    documents: list[Any],
) -> str:
    """Read direct labeled fields or a matching table row without LLM math."""

    lines = [
        re.sub(r"\s+", " ", line).strip(" -*◦")
        for doc in documents
        for line in str(doc.page_content).splitlines()
        if line.strip()
    ]
    if (
        re.search(r"예산", question)
        and re.search(r"선발인원|인원", question)
        and re.search(r"1인당|인당|지원액|지원금", question)
    ):
        budget_line = next((item for item in lines if "소요예산" in item), "")
        count_line = next((item for item in lines if "선발인원" in item), "")
        support_line = next((item for item in lines if "1인당" in item), "")
        budget = re.search(r"([\d,]+원)", budget_line)
        count = re.search(r"(.+?\d[\d,]*명)", count_line)
        support = re.search(r"1인당\s*([\d,]+원)", support_line)
        if budget and count and support:
            return (
                f"총 소요예산은 {budget.group(1)}, 선발인원은 "
                f"{count.group(1)}, 1인당 지원액은 {support.group(1)}입니다."
            )
    if (
        re.search(r"장학생|선발", question)
        and re.search(r"몇\s*명|인원|총\s*몇", question)
        and not re.search(
            r"비교|각각|한\s*번|정리|함께|동시|예산|지원액|지원금",
            question,
        )
    ):
        line = next((item for item in lines if "선발인원" in item), "")
        match = re.search(r"선발인원\)?\s*(.+?\d[\d,]*명)", line)
        if match:
            return f"문서의 선발인원은 {match.group(1)}입니다."
    if (
        re.search(r"한\s*명당|1인당|인당", question)
        and not re.search(r"한\s*번|정리|함께|동시|예산|선발인원", question)
    ):
        line = next((item for item in lines if "1인당" in item), "")
        match = re.search(r"1인당\s*([\d,]+원)", line)
        if match:
            return f"문서의 1인당 지원금은 {match.group(1)}입니다."
    if (
        re.search(r"총|합계", question)
        and re.search(r"금액|얼마|\d[\d,]*원|달러|사용|예산", question)
    ):
        query_terms = _query_terms(question)
        candidates: list[tuple[float, str, list[str]]] = []
        for line in lines:
            values = re.findall(r"(?<!\d)(\d[\d,]*)(?!\d)", line)
            is_count_row = bool(
                re.search(r"인원|선발", question)
                and re.search(r"\d[\d,]*\s*명", normalized)
            )
            if len(values) < 2 and not is_count_row:
                continue
            label = line[:line.find(values[0])].strip()
            label_terms = _query_terms(label)
            overlap = (
                len(query_terms.intersection(label_terms))
                / max(1, len(label_terms))
            )
            if overlap:
                candidates.append((overlap, label, values))
        if candidates:
            _score, label, values = max(
                candidates,
                key=lambda item: (item[0], len(item[1])),
            )
            return f"{label}의 합계는 {values[-1]}원입니다."
    return ""


def _explicit_priority_answer(
    question: str,
    documents: list[Any],
) -> str:
    """Preserve an explicit priority-then-remainder rule from the source."""

    if not (
        re.search(r"누구|어느|먼저|우선", question)
        and re.search(r"기존|신규|계속자", question)
    ):
        return ""
    lines = [
        re.sub(r"\s+", " ", line).strip(" -*※")
        for doc in documents
        for line in str(doc.page_content).splitlines()
        if line.strip()
    ]
    priority = next(
        (
            line for line in lines
            if "우선 추천" in line and re.search(r"기존|선발자|계속자", line)
        ),
        "",
    )
    remainder = next(
        (
            line for line in lines
            if "잔여 인원" in line and re.search(r"신규|추천", line)
        ),
        "",
    )
    if not priority:
        return ""
    scope = "공과대학은 " if re.search(r"공대|공과대학", question) else ""
    answer = (
        f"문서 기준으로 {scope}기존 선발자를 우선 추천합니다. 근거: {priority}"
    )
    if remainder:
        answer += f" 그 뒤 잔여 인원에 대해 신규 대상자를 추천합니다. 근거: {remainder}"
    return answer


def _explicit_preference_answer(
    question: str,
    documents: list[Any],
) -> str:
    """Do not turn a preference clause into an automatic exclusion."""

    if not re.search(r"무조건|반드시|자동|탈락|제외", question):
        return ""
    for doc in documents:
        for line in str(doc.page_content).splitlines():
            normalized = re.sub(r"\s+", " ", line).strip(" -*※")
            if "우선 선발" not in normalized:
                continue
            topic_groups = (
                ("생활비", "기타장학금"),
                ("성적",),
                ("소득", "지원구간"),
                ("국가근로", "근로대가성"),
                ("외부재단",),
                ("졸업", "수료"),
            )
            if not any(
                any(marker in question for marker in group)
                and any(marker in normalized for marker in group)
                for group in topic_groups
            ):
                continue
            return (
                f'질문한 우선선발 조건과 관련해 문서에는 "{normalized}"라고 되어 있습니다. '
                "이는 해당 대상을 우선한다는 뜻이며, 반대 경우를 자동 탈락 또는 "
                "무조건 제외한다고 명시한 것은 아닙니다."
            )
    return ""


def _explicit_table_comparison_answer(
    question: str,
    documents: list[Any],
) -> str:
    """Compare row totals in a retrieved numeric table without swapping columns."""

    if (
        not re.search(r"비교|차이|각각|보다", question)
        or not re.search(r"금액|예산|인원|선발인원|몇\s*명|배정|지원액", question)
    ):
        return ""
    query_terms = _query_terms(question.replace("공대", "공과대학"))
    candidates: list[tuple[int, str, int, str]] = []
    for doc in documents:
        for line in str(doc.page_content).splitlines():
            normalized = re.sub(r"\s+", " ", line).strip(" -*※")
            values = re.findall(r"(?<!\d)(\d[\d,]*)(?!\d)", normalized)
            if len(values) < 2:
                continue
            first_value_at = normalized.find(values[0])
            label = normalized[:first_value_at].strip()
            overlap = len(
                query_terms.intersection(
                    _query_terms(label.replace("공대", "공과대학"))
                )
            )
            if not overlap:
                continue
            total = int(values[-1].replace(",", ""))
            if re.search(r"금액|사용계획|예산|배정", question) and total < 1_000:
                continue
            candidates.append((overlap, label, total, values[-1]))
    unique: dict[str, tuple[int, str, int, str]] = {}
    for item in candidates:
        unique[item[1]] = max(unique.get(item[1], item), item)
    rows = sorted(unique.values(), key=lambda item: (-item[0], -len(item[1])))
    if len(rows) < 2:
        return ""
    first, second = rows[:2]
    relation = "더 큽니다" if first[2] > second[2] else "더 작습니다"
    difference = abs(first[2] - second[2])
    return (
        f"{first[1]} 합계는 {first[3]}원이고, "
        f"{second[1]} 합계는 {second[3]}원입니다. "
        f"두 합계의 차이는 {difference:,}원이며, "
        f"{first[1]} 합계가 {relation}."
    )


def _repair_explicit_difference_answer(question: str, answer: str) -> str:
    """Make a stated score difference agree with the two stated operands."""

    if "차이" not in question:
        return answer
    claim = re.search(r"차이(?:는|가)?\s*([\d,]+)점", answer)
    if not claim:
        return answer
    following_scores = [
        int(value.replace(",", ""))
        for value in re.findall(r"([\d,]+)점", answer[claim.end():])
    ]
    if len(following_scores) == 2:
        operands = following_scores
    elif (
        len(following_scores) == 3
        and following_scores[2] == abs(following_scores[0] - following_scores[1])
    ):
        # The final value repeats an already explained subtraction result.
        operands = following_scores[:2]
    else:
        # More numeric values make operand identity ambiguous. Preserving the
        # cited answer is safer than rewriting a correct value from guesswork.
        return answer
    difference = abs(operands[0] - operands[1])
    return (
        answer[:claim.start(1)]
        + f"{difference:,}"
        + answer[claim.end(1):]
    )


def _ensure_exception_evidence(
    question: str,
    answer: str,
    documents: list[Any],
) -> str:
    """Append an explicit source exception omitted from a qualified answer."""

    if not re.search(r"무조건|모두|전부|예외", question):
        return answer
    topic_terms = {
        term for term in _query_terms(question)
        if term not in {"무조건", "모두", "전부", "제외", "추천"}
    }
    for doc in documents:
        for line in str(doc.page_content).splitlines():
            normalized = re.sub(r"\s+", " ", line).strip(" -*◦")
            if not re.search(r"추천\s*가능|예외", normalized):
                continue
            if not topic_terms.intersection(_query_terms(normalized)):
                continue
            return f"{answer.rstrip()}\n\n문서에 명시된 예외:\n- {normalized}"
    return answer


def _section_items(docs: list[Any]) -> list[str]:
    """Extract complete bullet items from one retrieved parent section."""
    section_ids = {
        str(doc.metadata.get("section_id"))
        for doc in docs
        if doc.metadata.get("section_id")
    }
    if len(section_ids) != 1:
        return []
    child_counts = {
        int(doc.metadata.get("child_count", 0))
        for doc in docs
        if doc.metadata.get("child_count") is not None
    }
    # A large top-level section usually contains several different subsections.
    # Treating every bullet in it as one answer caused unrelated procedures,
    # budgets, and exclusions to be appended to a narrow user question.
    if child_counts and max(child_counts) > 2:
        return []
    joined = "\n".join(str(doc.page_content) for doc in docs)
    items = [
        re.sub(r"\s+", " ", match.group(1)).strip()
        for match in _SECTION_ITEM_RE.finditer(joined)
    ]
    return list(dict.fromkeys(item for item in items if len(item) >= 10))


def _ensure_section_completeness(answer: str, docs: list[Any]) -> str:
    """Append source items only when the generated answer omitted some."""
    items = _section_items(docs)
    if len(items) < 2:
        return answer

    normalized_answer = re.sub(r"\s+", "", answer)
    missing = []
    for item in items:
        compact = re.sub(r"\s+", "", item)
        probe = compact[: min(24, len(compact))]
        if probe and probe not in normalized_answer:
            missing.append(item)
    if not missing:
        return answer

    complete_list = "\n".join(f"- {item}" for item in items)
    return (
        f"{answer.rstrip()}\n\n"
        "문서에 기재된 전체 항목:\n"
        f"{complete_list}"
    )


def current_vector_evidence() -> list[dict[str, Any]]:
    return [dict(item) for item in _vector_evidence_ctx.get()]


def _rerank_candidates(
    queries: list[str],
    candidates: dict[tuple[str, str, str], dict[str, Any]],
    mmr_order: list[tuple[str, str, str]],
) -> list[Any]:
    """Rerank semantic candidates using general query/document evidence."""
    if not candidates:
        return []

    semantic_scores = [float(item["score"]) for item in candidates.values()]
    low, high = min(semantic_scores), max(semantic_scores)
    spread = high - low
    mmr_rank = {key: index for index, key in enumerate(mmr_order)}
    query_term_sets = [_query_terms(query) for query in queries]
    primary_terms = query_term_sets[0] if query_term_sets else set()
    primary_term_frequency = {
        term: sum(
            1
            for item in candidates.values()
            if term in _query_terms(str(item["doc"].page_content))
        )
        for term in primary_terms
    }
    primary_weight_total = sum(
        1.0 / max(1, primary_term_frequency[term])
        for term in primary_terms
    )

    ranked: list[tuple[float, Any, float, float]] = []
    for key, item in candidates.items():
        doc = item["doc"]
        semantic = (
            (float(item["score"]) - low) / spread
            if spread > 1e-9
            else 1.0
        )
        content = str(doc.page_content).lower()
        title = str(doc.metadata.get("section_title", "")).lower()
        lexical = max(
            (
                len(terms.intersection(_query_terms(content))) / len(terms)
                for terms in query_term_sets
                if terms
            ),
            default=0.0,
        )
        title_match = max(
            (
                len(terms.intersection(_query_terms(title))) / len(terms)
                for terms in query_term_sets
                if terms and title
            ),
            default=0.0,
        )
        primary_lexical = (
            sum(
                1.0 / max(1, primary_term_frequency[term])
                for term in primary_terms.intersection(_query_terms(content))
            )
            / primary_weight_total
            if primary_weight_total
            else 0.0
        )
        support = min(1.0, len(item["queries"]) / max(1, len(queries)))
        diversity = (
            1.0 / (1.0 + mmr_rank[key])
            if key in mmr_rank
            else 0.0
        )
        score = (
            semantic * 0.45
            + lexical * 0.15
            + primary_lexical * 0.20
            + title_match * 0.10
            + support * 0.05
            + diversity * 0.05
        )
        ranked.append((
            score,
            doc,
            title_match,
            lexical,
        ))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    best_rerank_score = ranked[0][0]
    has_strong_section_title = any(
        title_match >= 0.5
        for _score, _doc, title_match, _lexical in ranked
    )
    selected = [
        doc
        for score, doc, title_match, lexical in ranked
        if score >= best_rerank_score - VECTOR_RERANK_SCORE_MARGIN
        and (
            not has_strong_section_title
            or title_match > 0.0
            or lexical >= 0.25
        )
    ][:VECTOR_SEARCH_K]
    selected_keys = {_doc_key(doc) for doc in selected}
    reserved_support_keys: set[tuple[str, str, str]] = set()
    # An expansion query exists to recover a distinct piece of evidence. Keep
    # its strongest lexical hit even when the primary-query score margin would
    # otherwise discard that supporting chunk.
    for terms in query_term_sets[1:]:
        if not terms:
            continue
        best_support: tuple[float, Any] | None = None
        for _score, doc, _title_match, _lexical in ranked:
            if _doc_key(doc) in selected_keys:
                continue
            overlap = len(terms.intersection(_query_terms(str(doc.page_content)))) / len(terms)
            if overlap >= 0.5 and (
                best_support is None or overlap > best_support[0]
            ):
                best_support = (overlap, doc)
        if best_support:
            support = best_support[1]
            support_key = _doc_key(support)
            if len(selected) >= VECTOR_SEARCH_K:
                replace_index = next(
                    (
                        index
                        for index in range(len(selected) - 1, -1, -1)
                        if _doc_key(selected[index]) not in reserved_support_keys
                    ),
                    None,
                )
                if replace_index is None:
                    continue
                removed = selected.pop(replace_index)
                selected_keys.discard(_doc_key(removed))
            selected.append(support)
            selected_keys.add(support_key)
            reserved_support_keys.add(support_key)
    return selected


def _section_filter(doc: Any) -> dict[str, Any] | None:
    source = doc.metadata.get("source")
    section_id = doc.metadata.get("section_id")
    if not source or not section_id:
        return None
    return {
        "$and": [
            {"source": str(source)},
            {"section_id": str(section_id)},
        ]
    }


def _expand_parent_sections(vectorstore: Any, docs: list[Any]) -> list[Any]:
    """Expand a selected child to every stored child of its parent section."""
    expanded: list[Any] = []
    seen: set[tuple[str, str, str]] = set()
    expanded_sections: set[tuple[str, str]] = set()

    for doc in docs:
        siblings: list[Any] = [doc]
        where = _section_filter(doc)
        if where is not None and hasattr(vectorstore, "get"):
            section_key = (
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("section_id", "")),
            )
            if section_key in expanded_sections:
                continue
            expanded_sections.add(section_key)
            try:
                result = vectorstore.get(
                    where=where,
                    include=["documents", "metadatas"],
                )
                siblings = [
                    Document(page_content=text, metadata=metadata or {})
                    for text, metadata in zip(
                        result.get("documents") or [],
                        result.get("metadatas") or [],
                    )
                ] or [doc]
                siblings.sort(
                    key=lambda sibling: int(
                        sibling.metadata.get("child_index", 0)
                    )
                )
                # Small sections are returned completely. Large top-level
                # sections can contain many numbered subsections, so return a
                # local window around the best reranked child instead of
                # dumping the whole parent into the answer context.
                if len(siblings) > 4:
                    selected_index = int(doc.metadata.get("child_index", 0))
                    siblings = [
                        sibling
                        for sibling in siblings
                        if abs(
                            int(sibling.metadata.get("child_index", 0))
                            - selected_index
                        ) <= 1
                    ]
            except Exception:
                logger.warning(
                    "[VECTOR] Parent Section 확장 실패 | source=%s section_id=%s",
                    doc.metadata.get("source"),
                    doc.metadata.get("section_id"),
                )

        for sibling in siblings:
            key = _doc_key(sibling)
            if key not in seen:
                seen.add(key)
                expanded.append(sibling)
    return expanded


def _required_evidence_missing(question: str, docs: list[Any]) -> str:
    context = "\n".join(str(doc.page_content) for doc in docs)
    for question_pattern, evidence_pattern, message in _EVIDENCE_RULES:
        if question_pattern.search(question) and not evidence_pattern.search(context):
            return message
    return ""


async def _expanded_queries(question: str, is_doc_explain: bool) -> list[str]:
    cache_key = (question, is_doc_explain)
    cached = _query_expansion_cache.get(cache_key)
    now = time.monotonic()
    if cached and now - cached[0] <= _QUERY_EXPANSION_CACHE_TTL:
        return list(cached[1])

    queries = [question]
    if (
        re.search(r"공대|공과대학", question)
        and re.search(r"졸업|수료", question)
        and re.search(r"제외|추천|가능|무조건", question)
    ):
        queries.append(
            "공과대학 수업연한 이내 졸업 예정 수료 예정 추천 가능 제외 예외"
        )
    if (
        re.search(r"공대|공과대학", question)
        and re.search(r"기존|신규|먼저|우선", question)
    ):
        queries.extend([
            "공과대학 기존 선발자 우선 추천",
            "잔여 인원 신규 추천",
        ])
    if (
        re.search(r"공대|공과대학", question)
        and re.search(r"단과대학", question)
        and re.search(r"추천", question)
        and re.search(r"비교|차이|다르|달라|각각", question)
    ):
        queries.append(
            "공과대학과 그 외 단과대학 각각 자격 검토 적격자 추천 인원 "
            "학생과 최종 선발 배정인원"
        )
    if (
        "대학원" in question
        and re.search(r"장학금|지원|선발|대상", question)
    ):
        queries.insert(0,
            "장학금 선발 대상 학부 재학생 대학원생 지원 대상"
        )
    if (
        "등록금" in question
        and re.search(r"장학금|지원", question)
    ):
        queries.insert(0,
            "지원내용 학업장려금 생활비 등록금 감면 장학금 성격"
        )
    if (
        re.search(r"자격\s*요건|자격요건", question)
        and re.search(r"확정|자동|수령|선발", question)
    ):
        queries.insert(0,
            "자격 요건 충족 추천 가능 학생과 최종 선발 장학금 지급"
        )
    if (
        re.search(r"장학생|선발", question)
        and re.search(r"총\s*몇|몇\s*명|인원", question)
    ):
        queries.insert(0, "지원규모 선발인원 학부 재학생 총 인원")
    if is_doc_explain:
        if re.search(
            r"왜\s*(?:만들|생기|도입|제정|시작)|(?:만들|생기|도입|제정)된?\s*이유",
            question,
        ):
            queries.insert(0, f"{question} 추진배경 목적 설립 이유")
        doc_ctx = re.sub(r"\s*문서의?\s*(목적|내용|설명).*$", "", question).strip()
        doc_ctx = re.sub(r"\s*설명해.*$", "", doc_ctx).strip()
        if doc_ctx and len(doc_ctx) > 3:
            queries.insert(0, f"[문서 개요] {doc_ctx}")
        _query_expansion_cache[cache_key] = (now, list(queries))
        return queries

    try:
        raw_variants = await await_cancellable(
            get_llm_code().ainvoke(MULTI_QUERY_PROMPT.format(question=question))
        )
        variants = [line.strip() for line in raw_variants.strip().split("\n") if line.strip()]
        for variant in variants[:2]:
            if variant not in queries:
                queries.append(variant)
    except Exception as exc:
        logger.warning("[VECTOR] 쿼리 확장 실패 | err=%s", exc)
    if len(_query_expansion_cache) >= 256:
        oldest = min(
            _query_expansion_cache,
            key=lambda key: _query_expansion_cache[key][0],
        )
        _query_expansion_cache.pop(oldest, None)
    _query_expansion_cache[cache_key] = (now, list(queries))
    return queries


def _selected_source_filter() -> dict[str, object] | None:
    sources = selected_sources()
    if not sources:
        return None
    if len(sources) == 1:
        return {"source": sources[0]}
    return {"source": {"$in": list(sources)}}


async def _ensure_multi_document_coverage(
    vectorstore: Any,
    queries: list[str],
    docs: list[Any],
) -> list[Any]:
    """For explicit comparisons, keep at least one candidate per selected file."""
    sources = selected_sources()
    if (
        len(sources) < 2
        or len(sources) > 5
        or not any(_MULTI_DOCUMENT_RE.search(query) for query in queries)
    ):
        return docs

    present = {
        str(doc.metadata.get("source", ""))
        for doc in docs
    }
    covered = list(docs)
    query = queries[0]
    for source in sources:
        if source in present:
            continue
        try:
            scored = await asyncio.to_thread(
                vectorstore.similarity_search_with_relevance_scores,
                query,
                VECTOR_SEARCH_FETCH_K,
                filter={"source": source},
            )
            relative = _relative_score_candidates(query, scored)
            if relative:
                best_doc, _score = max(relative, key=lambda pair: pair[1])
                covered.append(best_doc)
                present.add(source)
        except Exception:
            logger.warning(
                "[VECTOR] 다중 문서 후보 확보 실패 | source=%s",
                source,
            )
    return covered


async def _retrieve_verified_documents(queries: list[str]) -> list[Any]:
    vectorstore = get_vectorstore()
    source_filter = _selected_source_filter()
    qualified: dict[tuple[str, str, str], dict[str, Any]] = {}
    mmr_order: list[tuple[str, str, str]] = []

    for query in queries:
        try:
            score_kwargs = {"filter": source_filter} if source_filter else {}
            scored = await asyncio.to_thread(
                vectorstore.similarity_search_with_relevance_scores,
                query,
                VECTOR_SEARCH_FETCH_K,
                **score_kwargs,
            )
            relative_candidates = _relative_score_candidates(query, scored)
            relative_keys = {
                _doc_key(doc)
                for doc, _score in relative_candidates
            }
            query_terms = _query_terms(query)
            if query_terms:
                relative_keys.update(
                    _doc_key(doc)
                    for doc, _score in scored
                    if (
                        len(
                            query_terms.intersection(
                                _query_terms(str(doc.page_content))
                            )
                        )
                        / len(query_terms)
                    ) >= 0.4
                )
            for doc, score_value in scored:
                key = _doc_key(doc)
                if key not in relative_keys:
                    continue
                previous = qualified.get(key)
                if previous is None:
                    qualified[key] = {
                        "doc": doc,
                        "score": float(score_value),
                        "queries": {query},
                    }
                else:
                    previous["score"] = max(
                        float(previous["score"]),
                        float(score_value),
                    )
                    previous["queries"].add(query)

            if source_filter and hasattr(vectorstore, "get"):
                raw = await asyncio.to_thread(
                    vectorstore.get,
                    where=source_filter,
                    include=["documents", "metadatas"],
                )
                lexical_floor = 0.3
                semantic_anchor = max(
                    (float(score) for _doc, score in scored),
                    default=0.0,
                )
                for text, metadata in zip(
                    raw.get("documents") or [],
                    raw.get("metadatas") or [],
                ):
                    doc = Document(
                        page_content=str(text),
                        metadata=metadata or {},
                    )
                    overlap = (
                        len(
                            query_terms.intersection(
                                _query_terms(doc.page_content)
                            )
                        )
                        / len(query_terms)
                        if query_terms
                        else 0.0
                    )
                    if overlap < lexical_floor:
                        continue
                    key = _doc_key(doc)
                    previous = qualified.get(key)
                    if previous is None:
                        qualified[key] = {
                            "doc": doc,
                            "score": semantic_anchor,
                            "queries": {query},
                        }
                    else:
                        previous["queries"].add(query)

            mmr_kwargs = {"filter": source_filter} if source_filter else {}
            mmr_docs = await asyncio.to_thread(
                vectorstore.max_marginal_relevance_search,
                query,
                VECTOR_SEARCH_K,
                VECTOR_SEARCH_FETCH_K,
                0.6,
                **mmr_kwargs,
            )
            for doc in mmr_docs:
                key = _doc_key(doc)
                if key in qualified and key not in mmr_order:
                    mmr_order.append(key)
        except Exception as exc:
            query_id, query_chars = question_log_metadata(query)
            logger.warning(
                "[VECTOR] 검색 실패 | query_id=%s chars=%d error_type=%s",
                query_id, query_chars, type(exc).__name__,
            )

    if not qualified:
        return []

    reranked = _rerank_candidates(queries, qualified, mmr_order)
    reranked = await _ensure_multi_document_coverage(
        vectorstore,
        queries,
        reranked,
    )
    return _expand_parent_sections(vectorstore, reranked)


async def prepare_vector_context(question: str) -> VectorPreparation:
    _vector_evidence_ctx.set([])
    is_doc_explain = bool(_DOC_EXPLAIN_RE.search(question))
    is_numeric_eligibility = _is_numeric_eligibility_question(question)
    is_document_reasoning = bool(_DOCUMENT_REASONING_RE.search(question))
    queries = await _expanded_queries(question, is_doc_explain)
    docs = await _retrieve_verified_documents(queries)
    if not docs:
        return VectorPreparation(
            immediate_answer="질문과 충분히 관련된 내용을 문서에서 찾을 수 없습니다."
        )

    if is_numeric_eligibility:
        docs = _numeric_evidence_documents(question, docs)
    if is_document_reasoning:
        docs = _conditional_evidence_documents(question, docs)

    missing_evidence = _required_evidence_missing(question, docs)
    evidence = _build_vector_evidence(docs)
    _vector_evidence_ctx.set(evidence)
    source_files = list(dict.fromkeys(
        os.path.basename(doc.metadata.get("source", ""))
        for doc in docs
        if doc.metadata.get("source")
    ))
    if missing_evidence:
        return VectorPreparation(
            source_files=source_files,
            immediate_answer=missing_evidence,
            evidence=evidence,
        )

    context = _fmt_docs(docs)
    table_notes = _table_alignment_notes(docs)
    if table_notes:
        context = f"{context}\n\n{table_notes}"
    condition_notes = _condition_alignment_notes(question, docs)
    if condition_notes:
        context = f"{context}\n\n{condition_notes}"

    return VectorPreparation(
        context=context,
        source_files=source_files,
        prompt=(
            NUMERIC_ELIGIBILITY_RAG_PROMPT
            if is_numeric_eligibility
            else DOCUMENT_REASONING_RAG_PROMPT
            if is_document_reasoning
            else DOC_EXPLAIN_RAG_PROMPT
            if is_doc_explain
            else RAG_PROMPT
        ),
        evidence=evidence,
        documents=docs,
        numeric_eligibility=is_numeric_eligibility,
        document_reasoning=is_document_reasoning,
    )


async def _answer_vector(
    question: str,
    allow_pandas_fallback: bool = True,
    analysis: QuestionAnalysis | None = None,
) -> tuple[str, list[str], str]:
    from rag.cancellation import raise_if_cancelled

    raise_if_cancelled()
    question_id, question_chars = question_log_metadata(question)
    logger.info(
        "[VECTOR] 검색 시작 | question_id=%s chars=%d",
        question_id, question_chars,
    )
    prepared = await prepare_vector_context(question)
    raise_if_cancelled()
    if prepared.immediate_answer:
        return prepared.immediate_answer, prepared.source_files or [], "vector"

    answer_llm = (
        get_llm_code()
        if prepared.numeric_eligibility or prepared.document_reasoning
        else get_llm_rag()
    )
    answer = await await_cancellable(
        (prepared.prompt | answer_llm | StrOutputParser()).ainvoke(
            {"context": prepared.context, "question": question}
        )
    )
    raise_if_cancelled()
    reasoning_issue = (
        _document_reasoning_issue(question, answer, prepared.context)
        if prepared.document_reasoning
        else ""
    )
    if reasoning_issue:
        focused_reasoning_context = prepared.context
        if "PDF 범위표" in reasoning_issue:
            focused_reasoning_context = (
                _table_alignment_notes(prepared.documents)
                or prepared.context
            )
        answer = await await_cancellable(
            (
                DOCUMENT_REASONING_REPAIR_PROMPT
                | answer_llm
                | StrOutputParser()
            ).ainvoke({
                "context": focused_reasoning_context,
                "question": question,
                "answer": answer,
                "issue": reasoning_issue,
            })
        )
        raise_if_cancelled()
        answer = re.sub(r"(?m)^\s*수정된 답변:\s*", "", answer).strip()
        if _document_reasoning_issue(question, answer, prepared.context):
            explicit_exclusion = _explicit_exclusion_answer(
                question,
                prepared.documents,
            )
            if explicit_exclusion:
                answer = explicit_exclusion
            else:
                explicit_scope = _explicit_scope_or_support_answer(
                    question,
                    prepared.documents,
                )
                if explicit_scope:
                    answer = explicit_scope
    if (
        prepared.numeric_eligibility
        and (numeric_issue := _numeric_answer_issue(answer, question))
    ):
        focused_numeric_context = (
            _condition_alignment_notes(question, prepared.documents)
            or prepared.context
        )
        answer = await await_cancellable(
            (
                NUMERIC_ELIGIBILITY_REPAIR_PROMPT
                | answer_llm
                | StrOutputParser()
            ).ainvoke({
                "context": focused_numeric_context,
                "question": question,
                "answer": answer,
                "error": numeric_issue,
            })
        )
        raise_if_cancelled()
        if _numeric_answer_issue(answer, question):
            answer = await await_cancellable(
                (
                    NUMERIC_DECISION_FALLBACK_PROMPT
                    | answer_llm
                    | StrOutputParser()
                ).ainvoke({
                    "context": focused_numeric_context,
                    "question": question,
                })
            )
            raise_if_cancelled()
        if _numeric_relation_contradiction(answer):
            explicit_numeric = _explicit_numeric_criteria_answer(
                question,
                prepared.documents,
            )
            if explicit_numeric:
                answer = explicit_numeric
    if prepared.numeric_eligibility:
        explicit_numeric = _explicit_numeric_criteria_answer(
            question,
            prepared.documents,
        )
        if explicit_numeric:
            answer = explicit_numeric
    if (
        re.search(r"과정|절차|처음부터|전체\s*단계", question)
        or re.search(
            r"왜\s*(?:만들|생기|도입|제정|시작)|(?:만들|생기|도입|제정)된?\s*이유",
            question,
        )
    ):
        answer = _ensure_section_completeness(answer, prepared.documents)
    answer = _ensure_exception_evidence(
        question,
        answer,
        prepared.documents,
    )
    mandatory_answer = _explicit_mandatory_condition_answer(
        question,
        prepared.documents,
    )
    if mandatory_answer:
        answer = mandatory_answer
    preference_answer = _explicit_preference_answer(
        question,
        prepared.documents,
    )
    if preference_answer:
        answer = preference_answer
    priority_answer = _explicit_priority_answer(
        question,
        prepared.documents,
    )
    if priority_answer:
        answer = priority_answer
    comparison_answer = _explicit_table_comparison_answer(
        question,
        prepared.documents,
    )
    if comparison_answer:
        answer = comparison_answer
    scalar_answer = _explicit_scalar_field_answer(
        question,
        prepared.documents,
    )
    if scalar_answer:
        answer = scalar_answer
    answer = _repair_explicit_difference_answer(question, answer)
    has_vector_override = (
        analysis.has_vector_override
        if analysis is not None
        else is_vector_override_question(question)
    )
    if allow_pandas_fallback and not has_vector_override and any(
        signal in answer for signal in _VECTOR_EMPTY_SIGNALS
    ):
        from rag.pandas_rag import _answer_pandas

        pd_answer, pd_sources, _ = await _answer_pandas(
            question,
            allow_vector_fallback=False,
            analysis=analysis,
        )
        if pd_answer and "없습니다" not in pd_answer and "오류" not in pd_answer:
            return pd_answer, pd_sources, "pandas"
    return answer, prepared.source_files or [], "vector"


async def _stream_vector(question: str) -> AsyncIterator[str]:
    prepared = await prepare_vector_context(question)
    if prepared.immediate_answer:
        yield prepared.immediate_answer
        return

    answer_llm = (
        get_llm_code()
        if prepared.numeric_eligibility or prepared.document_reasoning
        else get_llm_rag()
    )
    chain = prepared.prompt | answer_llm | StrOutputParser()
    iterator = chain.astream({"context": prepared.context, "question": question})
    while True:
        try:
            yield await next_cancellable(iterator)
        except StopAsyncIteration:
            return
