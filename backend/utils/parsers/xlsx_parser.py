from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

import pandas as pd

from utils.table_ingest_pipeline import ingest_dataframe_sheets
from utils.table_parser import sanitize_table_name

logger = logging.getLogger("ingest")


def _confirmed_merged_ranges(
    workbook: Any,
    sheet_name: str,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return physical XLSX merge ranges as zero-based frame coordinates."""

    engine_book = getattr(workbook, "book", None)
    if engine_book is None or sheet_name not in engine_book.sheetnames:
        return ()
    worksheet = engine_book[sheet_name]
    if not hasattr(worksheet, "merged_cells"):
        return ()
    return tuple(
        (
            merged.min_row - 1,
            merged.min_col - 1,
            merged.max_row - 1,
            merged.max_col - 1,
        )
        for merged in worksheet.merged_cells.ranges
    )


def _expand_confirmed_merged_cells(
    frame: pd.DataFrame,
    ranges: Iterable[tuple[int, int, int, int]],
) -> pd.DataFrame:
    """Expand only merge ranges explicitly recorded in the XLSX file."""

    if frame.empty:
        return frame
    result = frame.copy()
    row_count, column_count = result.shape
    for min_row, min_col, max_row, max_col in ranges:
        if min_row >= row_count or min_col >= column_count:
            continue
        value = result.iat[min_row, min_col]
        for row in range(min_row, min(max_row + 1, row_count)):
            for column in range(min_col, min(max_col + 1, column_count)):
                result.iat[row, column] = value
    return result


def _iter_xlsx_sheets(workbook: Any):
    for index, sheet_name in enumerate(workbook.sheet_names):
        raw_frame = workbook.parse(sheet_name, header=None)
        yield (
            index,
            sheet_name,
            _expand_confirmed_merged_cells(
                raw_frame,
                _confirmed_merged_ranges(workbook, sheet_name),
            ),
        )


def ingest_xlsx(
    file_path: str,
    file_hash: str = "",
    category: str = "",
    *,
    source_override: str | None = None,
    label_override: str | None = None,
    var_prefix_override: str | None = None,
    chroma_file_path_override: str | None = None,
    file_type_override: str | None = None,
) -> int:
    logger.info("[XLSX] %s", file_path)
    base_name   = sanitize_table_name(os.path.basename(file_path).rsplit(".", 1)[0])
    source_file = os.path.basename(source_override) if source_override else os.path.basename(file_path)
    doc_label   = label_override or os.path.splitext(source_file)[0]
    df_prefix    = var_prefix_override or f"df_{base_name}"

    # Pandas opens openpyxl workbooks in read-only mode by default, which hides
    # the physical merged-cell ranges required for evidence-based expansion.
    with pd.ExcelFile(
        file_path,
        engine="openpyxl",
        engine_kwargs={"read_only": False},
    ) as workbook:
        sheet_names = workbook.sheet_names
        sheet_frames = _iter_xlsx_sheets(workbook)
        return ingest_dataframe_sheets(
            sheet_frames,
            sheet_count=len(sheet_names),
            source_file=source_file,
            doc_label=doc_label,
            dataframe_prefix=df_prefix,
            source_type=file_type_override or "xlsx",
            chroma_file_path=chroma_file_path_override or file_path,
            file_hash=file_hash,
            category=category,
            chroma_source_override=(
                source_file if source_override else None
            ),
            log_prefix="XLSX",
        )
