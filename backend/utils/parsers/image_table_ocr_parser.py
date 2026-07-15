"""Direct image-table ingestion: image -> records -> DataFrame/ChromaDB."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any

import pandas as pd

from utils.chroma_store import save_to_chroma
from utils.parquet_store import drop_dataframe_files, save_dataframe
from utils.table_parser import _clean_dataframe, classify_name_entity, sanitize_table_name
from utils.text_utils import _make_doc_overview_chunk, _table_to_text_chunks
from utils.parsers.image_table_extractor import (
    EXPECTED_HEADERS,
    ImageTableExtractionError,
    detect_table_grid,
    extract_table_records,
)

logger = logging.getLogger("ingest")

IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}
_ISSUE_PATTERN = re.compile(r"^20\d{2}-\d{3}$")
_COHORT_PATTERN = re.compile(r"^\d{1,3}$")
_AMOUNT_PATTERN = re.compile(r"^\d{1,3}(?:,\d{3})*$")


def detect_table_row_bands(file_path: str) -> list[tuple[int, int]]:
    """Compatibility wrapper used by diagnostics/tests."""
    grid = detect_table_grid(file_path)
    return [(top + 1, bottom) for top, bottom in zip(grid.y_lines, grid.y_lines[1:])]


def detect_table_column_centers(file_path: str) -> list[float]:
    """Compatibility wrapper used by diagnostics/tests."""
    grid = detect_table_grid(file_path)
    return [(left + right) / 2 for left, right in zip(grid.x_lines, grid.x_lines[1:])]


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False


def _validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_cells: list[str] = []
    total = 0
    for index, record in enumerate(records):
        row_errors: list[str] = []
        issue = str(record.get("발행번호", ""))
        date = str(record.get("출연일자", ""))
        cohort = str(record.get("기수", ""))
        name = str(record.get("이름", ""))
        amount = str(record.get("출연금액", ""))
        if not _ISSUE_PATTERN.fullmatch(issue):
            row_errors.append(f"row={index}:발행번호={issue}")
        if not _valid_date(date):
            row_errors.append(f"row={index}:출연일자={date}")
        entity_type = classify_name_entity(name).get("entity_type", "unknown")
        organization_like = entity_type.startswith("organization") or name.startswith(("(주", "㈜", "(유"))
        cohort_is_valid = bool(_COHORT_PATTERN.fullmatch(cohort)) and 1 <= int(cohort) <= 120
        # 단체·법인 행은 원본 표에서도 기수가 비어 있을 수 있다.
        if not cohort_is_valid and not (not cohort and organization_like):
            row_errors.append(f"row={index}:기수={cohort}")
        if not _AMOUNT_PATTERN.fullmatch(amount):
            row_errors.append(f"row={index}:출연금액={amount}")
        else:
            total += int(amount.replace(",", ""))
        record["_ocr_validation_ok"] = not row_errors
        invalid_cells.extend(row_errors)
    return {"invalid_cells": invalid_cells, "calculated_total": total}


def _records_to_dataframe(records: list[dict[str, Any]], source_file: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        raise ImageTableExtractionError("표에서 데이터 행을 추출하지 못했습니다.")
    first = [header for header in EXPECTED_HEADERS if header in df.columns]
    rest = [column for column in df.columns if column not in first]
    df = df[first + rest]
    cleaned = _clean_dataframe(df, source_file=source_file, context_prefix="img_table0")
    if cleaned is None or cleaned.empty:
        raise ImageTableExtractionError("이미지 표 DataFrame 정제 결과가 비어 있습니다.")
    return cleaned


def ingest_image_table(file_path: str, file_hash: str, category: str) -> int:
    """Extract an image table directly and persist Parquet plus Chroma chunks.

    No temporary XLSX is created. The grid is deterministic, merged cells are
    restored from actual border geometry, and PaddleOCR is responsible only for
    recognizing text inside each physical table row.
    """
    source_file = os.path.basename(file_path)
    doc_label = os.path.splitext(source_file)[0]
    safe_name = sanitize_table_name(doc_label)
    output_prefix = f"df_{safe_name}_img"
    logger.info("[IMAGE:grid+paddle] 시작 | file=%s", file_path)

    records, extraction_meta = extract_table_records(file_path)
    quality = _validate_records(records)
    df = _records_to_dataframe(records, source_file)

    drop_dataframe_files(output_prefix)
    var_name = f"{output_prefix}_table0"
    save_dataframe(
        df,
        var_name,
        source_file,
        f"{doc_label} (이미지 표 직접 추출)",
        file_hash=file_hash,
        source_type=os.path.splitext(source_file)[1].lower().lstrip("."),
    )

    chunks = _table_to_text_chunks(df, doc_label)
    overview = _make_doc_overview_chunk(doc_label, source_file, [df])
    if overview:
        chunks.insert(0, overview)
    chroma_count = save_to_chroma(file_path, chunks, file_hash, category) if chunks and file_hash else 0

    logger.info(
        "[IMAGE:grid+paddle] 완료 | file=%s rows=%d invalid_cells=%d total=%d chunks=%d grid=%s",
        file_path,
        len(df),
        len(quality["invalid_cells"]),
        quality["calculated_total"],
        chroma_count,
        extraction_meta,
    )
    if quality["invalid_cells"]:
        logger.warning("[IMAGE] 검증 실패 셀 | %s", quality["invalid_cells"][:30])
    return chroma_count
