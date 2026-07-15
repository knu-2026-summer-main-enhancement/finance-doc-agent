from __future__ import annotations

import json
import logging
import os

import pandas as pd

from utils.semantic_schema import attach_semantic_schema, save_schema_sidecar

# DATAFRAME_DIR: backend/dataframes/
DATAFRAME_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataframes",
)

logger = logging.getLogger("ingest")


def save_dataframe(
    df: pd.DataFrame,
    var_name: str,
    source_file: str,
    label: str = "",
    *,
    file_hash: str = "",
    source_type: str = "",
) -> str:
    """원본 컬럼을 유지한 DataFrame과 의미 매핑 메타데이터를 저장한다."""
    os.makedirs(DATAFRAME_DIR, exist_ok=True)
    schema = attach_semantic_schema(
        df,
        var_name=var_name,
        source_file=source_file,
        file_hash=file_hash,
        source_type=source_type,
        dataframe_dir=DATAFRAME_DIR,
    )
    path = os.path.join(DATAFRAME_DIR, f"{var_name}.parquet")
    df.to_parquet(path, index=False)
    schema_path = save_schema_sidecar(DATAFRAME_DIR, var_name, schema)

    meta_path = os.path.join(DATAFRAME_DIR, f"{var_name}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": source_file,
                "label": label or var_name,
                "rows": len(df),
                "schema_version": schema["schema_version"],
                "document_id": schema["document_id"],
                "table_id": schema["table_id"],
                "mapping_fingerprint": schema["fingerprint"],
                "schema_file": os.path.basename(schema_path),
            },
            f,
            ensure_ascii=False,
        )

    logger.info(
        "DataFrame 저장 | var=%s rows=%d schema=%s unmapped=%d",
        var_name,
        len(df),
        schema["schema_version"],
        len(schema["unmapped_columns"]),
    )
    return path


def drop_dataframe_files(prefix: str):
    """prefix와 정확히 일치하거나 prefix_ 로 시작하는 저장 파일을 삭제한다."""
    if not os.path.exists(DATAFRAME_DIR):
        return
    for fname in os.listdir(DATAFRAME_DIR):
        if not fname.endswith((".parquet", ".meta.json", ".schema.json")):
            continue
        stem = fname
        for ext in (".parquet", ".meta.json", ".schema.json"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        if stem == prefix or stem.startswith(prefix + "_"):
            fpath = os.path.join(DATAFRAME_DIR, fname)
            os.remove(fpath)
            logger.info("DataFrame 파일 삭제: %s", fname)


def drop_dataframe_by_source(source: str) -> int:
    """meta.json의 source 필드를 기준으로 parquet·meta 파일을 삭제한다. 삭제된 쌍 수를 반환."""
    if not os.path.exists(DATAFRAME_DIR):
        return 0
    targets: list[str] = []
    for fname in os.listdir(DATAFRAME_DIR):
        if not fname.endswith(".meta.json"):
            continue
        meta_path = os.path.join(DATAFRAME_DIR, fname)
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if os.path.basename(meta.get("source", "")) == os.path.basename(source):
                stem = fname[: -len(".meta.json")]
                targets.append(stem)
        except Exception:
            continue
    for stem in targets:
        for ext in (".parquet", ".meta.json", ".schema.json"):
            fpath = os.path.join(DATAFRAME_DIR, stem + ext)
            if os.path.exists(fpath):
                os.remove(fpath)
                logger.info("DataFrame 파일 삭제: %s", stem + ext)
    return len(targets)
