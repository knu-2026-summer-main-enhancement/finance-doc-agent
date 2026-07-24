from __future__ import annotations

# Ollama LLM, 임베딩, Chroma retriever를 지연 생성해 재사용한다.
# 연결 설정은 core.config에만 두고 호출부는 아래 getter를 통해 접근한다.

from typing import Optional

import chromadb
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma

from core.config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, EMBED_MODEL,
    CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME,
    VECTOR_SEARCH_K, VECTOR_SEARCH_FETCH_K,
)

_llm_rag:  Optional[OllamaLLM] = None
_llm_code: Optional[OllamaLLM] = None
_vectorstore = None
_retriever = None


def get_llm_rag() -> OllamaLLM:
    global _llm_rag
    if _llm_rag is None:
        _llm_rag = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.1,
            num_ctx=4096,
        )
    return _llm_rag


def get_llm_code() -> OllamaLLM:
    global _llm_code
    if _llm_code is None:
        _llm_code = OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.0,
            num_ctx=8192,
        )
    return _llm_code


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        _vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
    return _vectorstore


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_vectorstore().as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": VECTOR_SEARCH_K,
                "fetch_k": VECTOR_SEARCH_FETCH_K,
                "lambda_mult": 0.6,
            },
        )
    return _retriever


def _fmt_docs(docs) -> str:
    parts = []
    for d in docs:
        src  = d.metadata.get("source", "")
        page = d.metadata.get("page", "")
        label = f"[{src} p.{page}]" if page else f"[{src}]"
        parts.append(f"{label}\n{d.page_content}")
    return "\n\n".join(parts)
