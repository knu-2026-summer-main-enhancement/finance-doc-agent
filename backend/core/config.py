from __future__ import annotations

import os

OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
EMBED_MODEL         = os.getenv("EMBED_MODEL", "bge-m3")
CHROMA_HOST         = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT         = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME     = os.getenv("COLLECTION_NAME", "scholarship_rules")
VECTOR_MIN_RELEVANCE = float(os.getenv("VECTOR_MIN_RELEVANCE", "0.35"))
VECTOR_SEARCH_K      = max(1, int(os.getenv("VECTOR_SEARCH_K", "8")))
VECTOR_SEARCH_FETCH_K = max(VECTOR_SEARCH_K, int(os.getenv("VECTOR_SEARCH_FETCH_K", "30")))
DATA_FOLDER         = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
API_KEY             = os.getenv("API_KEY", "")
INGEST_ALLOWED_BASE = os.path.realpath(os.getenv("INGEST_ALLOWED_BASE", DATA_FOLDER))
