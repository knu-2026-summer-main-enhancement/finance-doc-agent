from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import AsyncIterator, Any

from langchain_core.output_parsers import StrOutputParser

from core.config import VECTOR_MIN_RELEVANCE, VECTOR_SEARCH_FETCH_K, VECTOR_SEARCH_K
from core.llm import get_llm_rag, get_llm_code, get_vectorstore, _fmt_docs
from core.privacy import question_log_metadata
from rag.prompts import RAG_PROMPT, DOC_EXPLAIN_RAG_PROMPT, MULTI_QUERY_PROMPT
from rag.question_analyzer import QuestionAnalysis
from rag.question_detectors import is_vector_override_question
from datastore.scope import selected_sources

logger = logging.getLogger("uvicorn.error")

_DOC_EXPLAIN_RE = re.compile(
    r"문서의?\s*(목적|내용|설명)|설명해|어떤\s*(문서|내용)|요약해"
    r"|(?:지급|선발|지원|출연|기부)\s*(목적|기준|이유|방식|기관)"
    r"|(?:목적|내용|용도|기준|이유)\s*(?:이|가)?\s*(?:뭐야|뭐|무엇|어떤가|어떻게)",
    re.IGNORECASE,
)
_VECTOR_EMPTY_SIGNALS = ("해당 내용은 문서에서 확인할 수 없습니다", "문서에서 확인할 수 없")
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
        re.compile(r"절차|방법|서류|신청|문의|어떻게", re.IGNORECASE),
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

    def __post_init__(self) -> None:
        if self.source_files is None:
            self.source_files = []


def _doc_key(doc: Any) -> tuple[str, str, str]:
    return (
        str(doc.metadata.get("source", "")),
        str(doc.metadata.get("page", "")),
        str(doc.page_content),
    )


def _required_evidence_missing(question: str, docs: list[Any]) -> str:
    context = "\n".join(str(doc.page_content) for doc in docs)
    for question_pattern, evidence_pattern, message in _EVIDENCE_RULES:
        if question_pattern.search(question) and not evidence_pattern.search(context):
            return message
    return ""


async def _expanded_queries(question: str, is_doc_explain: bool) -> list[str]:
    queries = [question]
    if is_doc_explain:
        doc_ctx = re.sub(r"\s*문서의?\s*(목적|내용|설명).*$", "", question).strip()
        doc_ctx = re.sub(r"\s*설명해.*$", "", doc_ctx).strip()
        if doc_ctx and len(doc_ctx) > 3:
            queries.insert(0, f"[문서 개요] {doc_ctx}")
        return queries

    try:
        raw_variants = await get_llm_code().ainvoke(MULTI_QUERY_PROMPT.format(question=question))
        variants = [line.strip() for line in raw_variants.strip().split("\n") if line.strip()]
        for variant in variants[:2]:
            if variant not in queries:
                queries.append(variant)
    except Exception as exc:
        logger.warning("[VECTOR] 쿼리 확장 실패 | err=%s", exc)
    return queries


def _selected_source_filter() -> dict[str, object] | None:
    sources = selected_sources()
    if not sources:
        return None
    if len(sources) == 1:
        return {"source": sources[0]}
    return {"source": {"$in": list(sources)}}


async def _retrieve_verified_documents(queries: list[str]) -> list[Any]:
    vectorstore = get_vectorstore()
    source_filter = _selected_source_filter()
    qualified: dict[tuple[str, str, str], tuple[Any, float]] = {}
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
            for doc, score in scored:
                score_value = float(score)
                if score_value < VECTOR_MIN_RELEVANCE:
                    continue
                key = _doc_key(doc)
                previous = qualified.get(key)
                if previous is None or score_value > previous[1]:
                    qualified[key] = (doc, score_value)

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

    remaining = sorted(
        (item for key, item in qualified.items() if key not in mmr_order),
        key=lambda item: item[1],
        reverse=True,
    )
    docs = [qualified[key][0] for key in mmr_order]
    docs.extend(doc for doc, _ in remaining)
    return docs[:12]


async def prepare_vector_context(question: str) -> VectorPreparation:
    is_doc_explain = bool(_DOC_EXPLAIN_RE.search(question))
    queries = await _expanded_queries(question, is_doc_explain)
    docs = await _retrieve_verified_documents(queries)
    if not docs:
        return VectorPreparation(
            immediate_answer="질문과 충분히 관련된 내용을 문서에서 찾을 수 없습니다."
        )

    missing_evidence = _required_evidence_missing(question, docs)
    source_files = list(dict.fromkeys(
        os.path.basename(doc.metadata.get("source", ""))
        for doc in docs
        if doc.metadata.get("source")
    ))
    if missing_evidence:
        return VectorPreparation(source_files=source_files, immediate_answer=missing_evidence)

    return VectorPreparation(
        context=_fmt_docs(docs),
        source_files=source_files,
        prompt=DOC_EXPLAIN_RAG_PROMPT if is_doc_explain else RAG_PROMPT,
    )


async def _answer_vector(
    question: str,
    allow_pandas_fallback: bool = True,
    analysis: QuestionAnalysis | None = None,
) -> tuple[str, list[str], str]:
    question_id, question_chars = question_log_metadata(question)
    logger.info(
        "[VECTOR] 검색 시작 | question_id=%s chars=%d",
        question_id, question_chars,
    )
    prepared = await prepare_vector_context(question)
    if prepared.immediate_answer:
        return prepared.immediate_answer, prepared.source_files or [], "vector"

    answer = await (prepared.prompt | get_llm_rag() | StrOutputParser()).ainvoke(
        {"context": prepared.context, "question": question}
    )
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

    chain = prepared.prompt | get_llm_rag() | StrOutputParser()
    async for chunk in chain.astream({"context": prepared.context, "question": question}):
        yield chunk
