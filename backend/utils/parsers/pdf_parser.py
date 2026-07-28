from __future__ import annotations

import logging
import os
import re
from hashlib import sha1

import pdfplumber

from utils.table_parser import _parse_table, sanitize_table_name
from utils.text_utils import _table_to_text_chunks, _make_doc_overview_chunk, clean_pdf_text, split_into_chunks
from utils.parquet_store import (
    save_dataframe,
    drop_dataframe_by_source,
    drop_dataframe_files,
)
from utils.chroma_store import save_to_chroma

# OCR 선택적 임포트
try:
    from pdf2image import convert_from_path
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

logger = logging.getLogger("ingest")

OCR_DPI  = 300
OCR_LANG = "kor+eng"
PDF_SECTION_SCHEMA_VERSION = "pdf-section-v1"
_SECTION_HEADING_RE = re.compile(
    r"^(?:"
    r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ]"
    r"|(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)[.)]?"
    r"|제\s*\d+\s*장"
    r")\s+\S.{0,78}$",
    re.MULTILINE,
)


def _extract_table_with_confirmed_spans(table) -> list[list[str | None]]:
    """Expand only cells whose PDF geometry physically spans later rows."""

    values = table.extract()
    rows = table.rows
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row.cells):
            if cell is not None or values[row_index][column_index] is not None:
                continue
            current_top = row.bbox[1]
            current_bottom = row.bbox[3]
            for previous_index in range(row_index - 1, -1, -1):
                previous_cell = rows[previous_index].cells[column_index]
                if previous_cell is None:
                    continue
                spans_current_row = (
                    previous_cell[1] < current_top
                    and previous_cell[3] >= current_bottom
                )
                if spans_current_row:
                    values[row_index][column_index] = values[previous_index][column_index]
                break
    return values


def _unparsed_table_to_text(table: list[list[str | None]]) -> str:
    """Preserve heading-like PDF tables that cannot become row data.

    PDF generators often draw section titles as one-row borderless tables.
    Their bounding boxes are removed from page prose to avoid duplicate table
    data, so dropping an unparseable table would also drop a searchable title.
    """
    lines: list[str] = []
    for row in table:
        cells = [re.sub(r"\s+", " ", str(cell)).strip() for cell in row if cell]
        line = " ".join(cells)
        if line and line not in lines:
            lines.append(line)
    return "\n".join(f"PDF 문서의 섹션 제목(검색 키워드): {line}" for line in lines)


def _unparsed_table_lines(table: list[list[str | None]]) -> list[str]:
    """Return the readable labels from a non-tabular PDF table."""
    lines: list[str] = []
    for row in table:
        cells = [re.sub(r"\s+", " ", str(cell)).strip() for cell in row if cell]
        line = " ".join(cells)
        if line and line not in lines:
            lines.append(line)
    return lines


def _detect_section_headings(page_text: str) -> list[str]:
    """Detect top-level numbered headings from the page's readable text.

    Table extraction can return broken glyph mappings even when
    ``page.extract_text()`` is correct. Heading discovery therefore uses the
    page text and a document-agnostic numbering shape, not table cell values or
    scholarship-specific words.
    """
    return list(dict.fromkeys(
        re.sub(r"\s+", " ", match.group(0)).strip()
        for match in _SECTION_HEADING_RE.finditer(page_text)
    ))


def _section_chunks_from_page_text(
    page_text: str,
    headings: list[str],
    *,
    page: int,
) -> list[dict]:
    """Create searchable child chunks with a stable section hierarchy.

    The section itself is the logical parent. Chroma stores its smaller child
    chunks, and the shared ``section_id`` lets retrieval collect siblings
    without relying on a literal heading prefix or a page-wide expansion.
    """
    positions = sorted({
        (page_text.find(heading), heading)
        for heading in headings
        if heading and page_text.find(heading) >= 0
    })
    records: list[dict] = []
    for section_index, (start, heading) in enumerate(positions):
        end = (
            positions[section_index + 1][0]
            if section_index + 1 < len(positions)
            else len(page_text)
        )
        section = page_text[start:end].strip()
        if len(section) < 20:
            continue

        normalized_heading = re.sub(r"\s+", " ", heading).strip()
        section_id = f"p{page}:s{section_index}"
        child_texts = split_into_chunks(section, page=page)
        for child_index, child in enumerate(child_texts):
            text = child["text"]
            if normalized_heading not in text:
                text = f"{normalized_heading}\n{text}"
            records.append({
                "text": f"PDF 문서 섹션:\n{text}",
                "page": page,
                "metadata": {
                    "content_type": "pdf_section_child",
                    "section_schema_version": PDF_SECTION_SCHEMA_VERSION,
                    "section_id": section_id,
                    "parent_id": section_id,
                    "section_title": normalized_heading,
                    "section_index": section_index,
                    "child_index": child_index,
                    "child_count": len(child_texts),
                },
            })
    return records


def _section_continuation_chunks(
    page_text: str,
    *,
    page: int,
    section_id: str,
    section_title: str,
) -> list[dict]:
    """Attach text on a following page to the active parent section."""
    cleaned = page_text.strip()
    if len(cleaned) < 20:
        return []
    children = split_into_chunks(cleaned, page=page)
    return [
        {
            "text": f"PDF 문서 섹션:\n{section_title}\n{child['text']}",
            "page": page,
            "metadata": {
                "content_type": "pdf_section_child",
                "section_schema_version": PDF_SECTION_SCHEMA_VERSION,
                "section_id": section_id,
                "parent_id": section_id,
                "section_title": section_title,
                # Final indices/counts are normalized after every page is read.
                "section_index": int(section_id.rsplit("s", 1)[-1]),
                "child_index": 0,
                "child_count": 0,
            },
        }
        for child in children
    ]


def _normalize_section_child_metadata(chunk_records: list[dict]) -> None:
    """Finalize child ordering after cross-page continuations are attached."""
    grouped: dict[str, list[dict]] = {}
    for record in chunk_records:
        metadata = record.get("metadata") or {}
        section_id = metadata.get("section_id")
        if metadata.get("content_type") == "pdf_section_child" and section_id:
            grouped.setdefault(str(section_id), []).append(record)
    for records in grouped.values():
        for child_index, record in enumerate(records):
            metadata = record["metadata"]
            metadata["child_index"] = child_index
            metadata["child_count"] = len(records)


def _extract_page_texts(file_path: str) -> dict[int, str]:
    page_texts: dict[int, str] = {}
    scanned_pages: list[int] = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                table_bboxes = [tbl.bbox for tbl in page.find_tables()]

                if table_bboxes:
                    def not_in_table(obj):
                        for bbox in table_bboxes:
                            if (obj.get("x0", 0) >= bbox[0] - 1 and
                                    obj.get("x1", 0) <= bbox[2] + 1 and
                                    obj.get("top", 0) >= bbox[1] - 1 and
                                    obj.get("bottom", 0) <= bbox[3] + 1):
                                return False
                        return True

                    raw = page.filter(not_in_table).extract_text() or ""
                else:
                    raw = page.extract_text() or ""

            except Exception:
                logger.exception("pdfplumber 텍스트 추출 실패 | page=%d", page_num)
                raw = ""

            if raw.strip():
                page_texts[page_num] = raw
            else:
                scanned_pages.append(page_num)

    if scanned_pages:
        if not HAS_OCR:
            logger.warning(
                "스캔 페이지 %s 감지됐으나 pytesseract/pdf2image 미설치로 건너뜀 | file=%s",
                scanned_pages, file_path,
            )
        else:
            logger.info("OCR 시작 | file=%s 스캔 페이지=%s", file_path, scanned_pages)
            for page_num in scanned_pages:
                try:
                    images = convert_from_path(
                        file_path, dpi=OCR_DPI,
                        first_page=page_num, last_page=page_num,
                    )
                    ocr_text = pytesseract.image_to_string(images[0], lang=OCR_LANG)
                    if ocr_text.strip():
                        page_texts[page_num] = ocr_text
                        logger.info("OCR 완료 | page=%d chars=%d", page_num, len(ocr_text))
                except Exception:
                    logger.exception("OCR 실패 | page=%d", page_num)

    return page_texts


def ingest_pdf_hybrid(file_path: str, file_hash: str, category: str) -> int:
    logger.info("[PDF] %s", file_path)

    source_file = os.path.basename(file_path)
    safe_stem   = sanitize_table_name(source_file.rsplit(".", 1)[0])
    # Korean-only filenames can collapse to the same ASCII stem (for example,
    # both may become ``tbl_2026``). Keep the source-derived suffix so PDF
    # table storage stays isolated per document.
    safe_name   = f"{safe_stem}_{sha1(source_file.encode('utf-8')).hexdigest()[:8]}"
    doc_label   = os.path.splitext(source_file)[0]

    # Also clean legacy table aliases that may have been created before the
    # source-derived suffix was introduced.
    drop_dataframe_by_source(source_file)
    drop_dataframe_files(f"df_{safe_name}_p")

    page_texts = _extract_page_texts(file_path)

    chunk_records: list[dict] = []
    parsed_tables: list[pd.DataFrame] = []
    for page_num, raw_text in page_texts.items():
        cleaned = clean_pdf_text(raw_text)
        page_chunks = split_into_chunks(cleaned, page=page_num)
        for chunk in page_chunks:
            chunk["metadata"] = {"content_type": "pdf_page_text"}
        chunk_records.extend(page_chunks)

    import pandas as pd

    table_count = 0
    active_section_id: str | None = None
    active_section_title: str | None = None
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.find_tables()
            except Exception:
                logger.exception("PDF 표 추출 실패 | page=%d", page_num)
                continue

            for t_idx, table_object in enumerate(tables):
                try:
                    table = _extract_table_with_confirmed_spans(table_object)
                    df = _parse_table(table, source_file=source_file, context_prefix=f"p{page_num}t{table_count}")
                    if df is None:
                        continue
                    parsed_tables.append(df)
                    var_name = f"df_{safe_name}_p{page_num}t{table_count}"
                    label    = f"{doc_label} (p.{page_num} 표{table_count + 1})"
                    save_dataframe(
                        df,
                        var_name,
                        source_file,
                        label,
                        file_hash=file_hash,
                        source_type="pdf",
                    )
                    logger.info("[PDF] 표 저장 | var=%s rows=%d", var_name, len(df))
                    table_count += 1
                    chunk_records.extend(_table_to_text_chunks(df, doc_label, page_num))
                except Exception:
                    logger.exception("[PDF] 표 저장 실패 | page=%d t=%d", page_num, t_idx)

            full_page_text = page.extract_text() or ""
            section_headings = _detect_section_headings(full_page_text)
            page_section_records: list[dict] = []

            if active_section_id and active_section_title:
                continuation_end = (
                    full_page_text.find(section_headings[0])
                    if section_headings
                    else len(full_page_text)
                )
                page_section_records.extend(_section_continuation_chunks(
                    full_page_text[:continuation_end],
                    page=page_num,
                    section_id=active_section_id,
                    section_title=active_section_title,
                ))

            if section_headings:
                section_records = _section_chunks_from_page_text(
                    full_page_text,
                    section_headings,
                    page=page_num,
                )
                page_section_records.extend(section_records)
                if section_records:
                    last_metadata = section_records[-1]["metadata"]
                    active_section_id = str(last_metadata["section_id"])
                    active_section_title = str(last_metadata["section_title"])

            if page_section_records:
                    # Section children contain the same page prose with a more
                    # useful hierarchy, so remove only the generic page chunks.
                    # Parsed table rows on the page remain untouched.
                    chunk_records = [
                        record
                        for record in chunk_records
                        if not (
                            record.get("page") == page_num
                            and (record.get("metadata") or {}).get("content_type")
                            == "pdf_page_text"
                        )
                    ]
                    chunk_records.extend(page_section_records)
            elif section_headings:
                    fallback_text = _unparsed_table_to_text([
                        [heading] for heading in section_headings
                    ])
                    if fallback_text:
                        chunk_records.append({"text": fallback_text, "page": page_num})

    _normalize_section_child_metadata(chunk_records)

    if parsed_tables:
        overview = _make_doc_overview_chunk(doc_label, source_file, parsed_tables)
        if overview:
            chunk_records.insert(0, overview)

    if not chunk_records and table_count == 0:
        logger.warning("추출 데이터 없음 | file=%s", file_path)

    chroma_count = save_to_chroma(file_path, chunk_records, file_hash, category) if chunk_records else 0
    logger.info("PDF 완료 | file=%s tables=%d chunks=%d", file_path, table_count, chroma_count)
    return chroma_count
