from __future__ import annotations

import logging
import math
from collections.abc import Iterable

import pandas as pd

from utils.chroma_store import save_to_chroma
from utils.parquet_store import drop_dataframe_files, save_dataframe
from utils.table_parser import _parse_table
from utils.text_utils import _make_doc_overview_chunk, _table_to_text_chunks


logger = logging.getLogger("ingest")


def _dataframe_to_raw_table(frame: pd.DataFrame) -> list[list[object | None]]:
    """Convert an adapter DataFrame into the neutral table shape parsers use."""

    return [
        [
            None
            if value is None
            or (isinstance(value, float) and math.isnan(value))
            else value
            for value in row
        ]
        for row in frame.values.tolist()
    ]


def ingest_dataframe_sheets(
    sheets: Iterable[tuple[int, str, pd.DataFrame]],
    *,
    sheet_count: int,
    source_file: str,
    doc_label: str,
    dataframe_prefix: str,
    source_type: str,
    chroma_file_path: str,
    file_hash: str = "",
    category: str = "",
    chroma_source_override: str | None = None,
    log_prefix: str = "TABLE",
) -> int:
    """Normalize and persist sheets supplied by any tabular source adapter.

    File-specific adapters only need to return an indexed stream of sheet
    DataFrames. Header detection, table cleanup, semantic schema attachment,
    Parquet persistence, text chunking, and Chroma persistence remain shared.
    """

    drop_dataframe_files(dataframe_prefix)

    chunk_records: list[dict] = []
    parsed_tables: list[pd.DataFrame] = []

    for index, sheet_name, raw_frame in sheets:
        if raw_frame.empty:
            logger.info("빈 시트 건너뜀 | sheet=%s", sheet_name)
            continue

        parsed = _parse_table(
            _dataframe_to_raw_table(raw_frame),
            source_file=source_file,
            context_prefix=f"s{index}",
        )
        if parsed is None:
            logger.warning("%s 파싱 결과 없음 | sheet=%s", log_prefix, sheet_name)
            continue

        parsed_tables.append(parsed)
        variable_name = (
            f"{dataframe_prefix}_s{index}"
            if sheet_count > 1
            else dataframe_prefix
        )
        label = (
            f"{doc_label} - {sheet_name}"
            if sheet_count > 1
            else doc_label
        )
        save_dataframe(
            parsed,
            variable_name,
            source_file,
            label,
            file_hash=file_hash,
            source_type=source_type,
        )
        logger.info(
            "[%s] '%s' 저장 완료 | sheet=%s rows=%d",
            log_prefix,
            variable_name,
            sheet_name,
            len(parsed),
        )
        chunk_records.extend(_table_to_text_chunks(parsed, doc_label))

    if parsed_tables:
        overview = _make_doc_overview_chunk(
            doc_label,
            source_file,
            parsed_tables,
        )
        if overview:
            chunk_records.insert(0, overview)

    if not chunk_records or not file_hash:
        return 0

    count = save_to_chroma(
        chroma_file_path,
        chunk_records,
        file_hash,
        category,
        source_override=chroma_source_override,
        file_type_override=source_type,
    )
    logger.info("[%s] Chroma 저장 완료 | chunks=%d", log_prefix, count)
    return count
