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
    MULTI_QUERY_PROMPT,
    NUMERIC_ELIGIBILITY_REPAIR_PROMPT,
    NUMERIC_ELIGIBILITY_RAG_PROMPT,
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
    r"|(?:목적|내용|용도|기준|이유)\s*(?:이|가)?\s*(?:뭐야|뭐|무엇|어떤가|어떻게)",
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
})
_KOREAN_PARTICLE_RE = re.compile(
    r"(?:으로부터|에서부터|에게서|까지|부터|으로|에게|한테|에서|"
    r"은|는|이|가|을|를|와|과|의|에|로|도|만)$"
)


def _query_terms(query: str) -> set[str]:
    normalized: set[str] = set()
    for raw_term in _QUERY_TERM_RE.findall(query):
        term = raw_term.lower()
        if term in _QUERY_TERM_STOPWORDS:
            continue
        without_particle = _KOREAN_PARTICLE_RE.sub("", term)
        normalized.add(without_particle if len(without_particle) >= 2 else term)
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
    r"받을\s*수|가능|충족|해당|자격|기준.*(?:넘|미달)",
    re.IGNORECASE,
)


def _is_numeric_eligibility_question(question: str) -> bool:
    return bool(
        _NUMERIC_ELIGIBILITY_RE.search(question)
        and re.search(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", question)
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
    return matched or docs


def _numeric_llm_answer_is_complete(answer: str) -> bool:
    numbers = re.findall(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", answer)
    has_conclusion = (
        "충족합니다" in answer
        or "충족하지 않습니다" in answer
    )
    return has_conclusion and "기준" in answer and len(numbers) >= 2


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

    ranked: list[tuple[float, Any, float]] = []
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
        support = min(1.0, len(item["queries"]) / max(1, len(queries)))
        diversity = (
            1.0 / (1.0 + mmr_rank[key])
            if key in mmr_rank
            else 0.0
        )
        score = (
            semantic * 0.55
            + lexical * 0.20
            + title_match * 0.15
            + support * 0.05
            + diversity * 0.05
        )
        ranked.append((
            score,
            doc,
            title_match,
        ))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    best_rerank_score = ranked[0][0]
    has_strong_section_title = any(
        title_match >= 0.5
        for _score, _doc, title_match in ranked
    )
    return [
        doc
        for score, doc, title_match in ranked
        if score >= best_rerank_score - VECTOR_RERANK_SCORE_MARGIN
        and (
            not has_strong_section_title
            or title_match > 0.0
        )
    ][:VECTOR_SEARCH_K]


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
    if is_doc_explain:
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
    queries = await _expanded_queries(question, is_doc_explain)
    docs = await _retrieve_verified_documents(queries)
    if not docs:
        return VectorPreparation(
            immediate_answer="질문과 충분히 관련된 내용을 문서에서 찾을 수 없습니다."
        )

    if is_numeric_eligibility:
        docs = _numeric_evidence_documents(question, docs)

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

    return VectorPreparation(
        context=_fmt_docs(docs),
        source_files=source_files,
        prompt=(
            NUMERIC_ELIGIBILITY_RAG_PROMPT
            if is_numeric_eligibility
            else DOC_EXPLAIN_RAG_PROMPT
            if is_doc_explain
            else RAG_PROMPT
        ),
        evidence=evidence,
        documents=docs,
        numeric_eligibility=is_numeric_eligibility,
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
        if prepared.numeric_eligibility
        else get_llm_rag()
    )
    answer = await await_cancellable(
        (prepared.prompt | answer_llm | StrOutputParser()).ainvoke(
            {"context": prepared.context, "question": question}
        )
    )
    raise_if_cancelled()
    if (
        prepared.numeric_eligibility
        and not _numeric_llm_answer_is_complete(answer)
    ):
        answer = await await_cancellable(
            (
                NUMERIC_ELIGIBILITY_REPAIR_PROMPT
                | answer_llm
                | StrOutputParser()
            ).ainvoke({
                "context": prepared.context,
                "question": question,
                "answer": answer,
            })
        )
        raise_if_cancelled()
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
    iterator = chain.astream({"context": prepared.context, "question": question})
    while True:
        try:
            yield await next_cancellable(iterator)
        except StopAsyncIteration:
            return
