"""Direct image-table ingestion: image -> records -> DataFrame/ChromaDB."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from utils.chroma_store import save_to_chroma
from utils.parquet_store import drop_dataframe_files, save_dataframe
from utils.semantic_schema import infer_column_meaning
from utils.table_parser import _clean_dataframe, sanitize_table_name
from utils.text_utils import _make_doc_overview_chunk, _table_to_text_chunks
from utils.parsers.image_table_extractor import (
    ImageTableExtractionError,
    detect_table_grid,
    extract_table_records,
)

logger = logging.getLogger("ingest")

IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}
_OCR_METADATA_COLUMNS = {
    "ocr_row_index",
    "_ocr_confidence_min",
    "_ocr_confidence_avg",
    "_ocr_low_confidence_cells",
    "_ocr_corrections",
    "_ocr_validation_ok",
}


def detect_table_row_bands(file_path: str) -> list[tuple[int, int]]:
    """Compatibility wrapper used by diagnostics/tests."""
    grid = detect_table_grid(file_path)
    return [(top + 1, bottom) for top, bottom in zip(grid.y_lines, grid.y_lines[1:])]


def detect_table_column_centers(file_path: str) -> list[float]:
    """Compatibility wrapper used by diagnostics/tests."""
    grid = detect_table_grid(file_path)
    return [(left + right) / 2 for left, right in zip(grid.x_lines, grid.x_lines[1:])]


def _validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """특정 장부 형식을 강제하지 않고 OCR 결과의 구조적 품질만 확인한다."""
    invalid_cells: list[str] = []
    for index, record in enumerate(records):
        row_errors: list[str] = []
        source_values = [
            str(value or "").strip()
            for key, value in record.items()
            if key not in _OCR_METADATA_COLUMNS
        ]
        if not any(source_values):
            row_errors.append(f"row={index}:빈 행")
        low_confidence = str(record.get("_ocr_low_confidence_cells", "") or "").strip()
        if low_confidence:
            row_errors.append(f"row={index}:낮은 OCR 신뢰도={low_confidence}")
        record["_ocr_validation_ok"] = not row_errors
        invalid_cells.extend(row_errors)

    total = 0
    if records:
        frame = pd.DataFrame(records)
        for column in frame.columns:
            if column in _OCR_METADATA_COLUMNS:
                continue
            meaning = infer_column_meaning(str(column), frame[column])
            if meaning.concept != "measure" or meaning.role != "amount":
                continue
            numeric = pd.to_numeric(
                frame[column]
                .astype(str)
                .str.replace(r"[^0-9.\-]", "", regex=True),
                errors="coerce",
            )
            total += int(numeric.fillna(0).sum())
    return {"invalid_cells": invalid_cells, "calculated_total": total}


def _records_to_dataframe(records: list[dict[str, Any]], source_file: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        raise ImageTableExtractionError("표에서 데이터 행을 추출하지 못했습니다.")
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
