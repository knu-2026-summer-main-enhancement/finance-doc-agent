from __future__ import annotations

import json
import logging
import os

import pandas as pd

from utils.parquet_store import DATAFRAME_DIR
from utils.semantic_schema import load_schema_sidecar

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# 공유 mutable state — 다른 파일에서 직접 임포트해 참조 공유
# ---------------------------------------------------------------------------
_df_namespace: dict[str, pd.DataFrame] = {}   # var_name → DataFrame
_df_sources:   dict[str, str]          = {}   # var_name → 원본 파일명
_df_labels:    dict[str, str]          = {}   # var_name → 표시용 레이블
_df_schemas:   dict[str, dict]         = {}   # var_name → 의미 매핑 sidecar
_df_schema_cache: tuple[str, float] | None = None
_SCHEMA_CACHE_TTL = 300


def _load_dataframes():
    """dataframes/ 폴더의 Parquet 파일을 모두 메모리에 로드한다.
    변수명은 df0, df1, df2 ... 형태로 단순화해 LLM이 잘못 잘라 쓰는 것을 방지한다."""
    global _df_namespace, _df_sources, _df_labels, _df_schemas, _df_schema_cache
    if not os.path.exists(DATAFRAME_DIR):
        _df_namespace.clear()
        _df_sources.clear()
        _df_labels.clear()
        _df_schemas.clear()
        _df_schema_cache = None
        return

    entries = []
    for fname in sorted(os.listdir(DATAFRAME_DIR)):
        if not fname.endswith(".parquet"):
            continue
        orig_name = fname[:-len(".parquet")]
        path      = os.path.join(DATAFRAME_DIR, fname)
        meta_path = os.path.join(DATAFRAME_DIR, f"{orig_name}.meta.json")
        try:
            df     = pd.read_parquet(path)
            source = orig_name
            label  = orig_name
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                source = meta.get("source", orig_name)
                label  = meta.get("label", orig_name)
            semantic_schema = load_schema_sidecar(DATAFRAME_DIR, orig_name)
            if semantic_schema:
                df.attrs["semantic_schema"] = semantic_schema
            # 학과 관련 컬럼의 "(N명)" suffix 제거 — LLM 혼란 방지
            for col in df.columns:
                if any(k in col for k in ("학과", "계열", "대상학생")):
                    try:
                        df[col] = df[col].astype(str).str.replace(r"\(\d+명\)", "", regex=True).str.strip()
                    except Exception:
                        pass
            entries.append((df, source, label, semantic_schema))
        except Exception as e:
            logger.warning("DataFrame 로드 실패 | file=%s err=%s", fname, e)

    next_namespace: dict[str, pd.DataFrame] = {}
    next_sources: dict[str, str] = {}
    next_labels: dict[str, str] = {}
    next_schemas: dict[str, dict] = {}
    # df0, df1, df2 ... 로 단순 명명 — LLM이 긴 파일명 기반 변수명을 잘못 잘라 쓰는 문제 방지
    for i, (df, source, label, semantic_schema) in enumerate(entries):
        alias = f"df{i}"
        next_namespace[alias] = df
        next_sources[alias] = source
        next_labels[alias] = label
        if semantic_schema:
            next_schemas[alias] = semantic_schema

    # Keep the previous complete snapshot visible while disk I/O is in
    # progress. Imported modules retain references to these dictionaries, so
    # replace their contents only after the next snapshot is fully prepared.
    _df_namespace.clear()
    _df_namespace.update(next_namespace)
    _df_sources.clear()
    _df_sources.update(next_sources)
    _df_labels.clear()
    _df_labels.update(next_labels)
    _df_schemas.clear()
    _df_schemas.update(next_schemas)
    _df_schema_cache = None

    logger.info("DataFrame %d개 로드 완료", len(_df_namespace))
