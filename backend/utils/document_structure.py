from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any

from utils.chroma_store import get_chroma_source_records


_TABLE_FILE_TYPES = {"csv", "tsv", "xls", "xlsx"}
_TEXT_FILE_TYPES = {"pdf", "doc", "docx", "hwp", "hwpx", "txt"}
_SECTION_CONTENT_TYPES = {"pdf_section_child", "hwp_section_child"}
_TEXT_CONTENT_TYPES = {
    "pdf_page_text",
    "pdf_section_child",
    "hwp_text_child",
    "hwp_section_child",
}
_SECTION_SUGGESTION_TERMS = (
    "선발 대상", "지원 대상", "대상", "자격", "요건",
    "선발 기준", "지원 기준", "기준", "지원 규모", "일정", "절차",
)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _section_sort_key(section: dict) -> tuple[int, int, str]:
    return (
        _integer(section.get("start_page"), 10**9),
        _integer(section.get("section_index"), 10**9),
        str(section.get("title", "")),
    )


def _clean_section_chunk(text: str, title: str) -> str:
    lines = [line.rstrip() for line in str(text).splitlines()]
    while lines and (
        re.match(r"^\[문서\s*:", lines[0])
        or lines[0].strip() == "PDF 문서 섹션:"
    ):
        lines.pop(0)
    if lines and lines[0].strip() == title.strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def document_structure(
    source: str,
    *,
    file_type: str | None = None,
) -> dict:
    """Summarize capabilities from the chunks actually stored for a document."""
    source = os.path.basename(source)
    records = get_chroma_source_records(source)
    inferred_type = (
        str(file_type or "").lower().lstrip(".")
        or str((records[0].get("metadata") or {}).get("file_type", "")).lower()
        or os.path.splitext(source)[1].lower().lstrip(".")
    )

    section_groups: dict[str, list[dict]] = defaultdict(list)
    pages: set[int] = set()
    table_ids: set[str] = set()
    has_text_content = False

    for record in records:
        metadata = record.get("metadata") or {}
        content_type = str(metadata.get("content_type", ""))
        page = _integer(metadata.get("page"))
        if page > 0:
            pages.add(page)
        section_id = str(metadata.get("section_id", "")).strip()
        section_title = str(metadata.get("section_title", "")).strip()
        if section_id and section_title and content_type in _SECTION_CONTENT_TYPES:
            section_groups[section_id].append(record)
        table_id = str(metadata.get("table_id", "")).strip()
        if table_id:
            table_ids.add(table_id)
        if content_type in _TEXT_CONTENT_TYPES:
            has_text_content = True

    sections: list[dict] = []
    for section_id, children in section_groups.items():
        children.sort(key=lambda item: (
            _integer((item.get("metadata") or {}).get("child_index")),
            _integer((item.get("metadata") or {}).get("page")),
        ))
        first_metadata = children[0].get("metadata") or {}
        section_pages = sorted({
            _integer((child.get("metadata") or {}).get("page"))
            for child in children
            if _integer((child.get("metadata") or {}).get("page")) > 0
        })
        sections.append({
            "section_id": section_id,
            "title": str(first_metadata.get("section_title", "")).strip(),
            "start_page": section_pages[0] if section_pages else None,
            "end_page": section_pages[-1] if section_pages else None,
            "section_index": _integer(first_metadata.get("section_index")),
            "chunk_count": len(children),
        })
    sections.sort(key=_section_sort_key)

    has_sections = bool(sections)
    has_tables = inferred_type in _TABLE_FILE_TYPES or bool(table_ids)
    capabilities = ["semantic_question"] if inferred_type in _TEXT_FILE_TYPES else []
    if has_sections:
        capabilities.extend(["list_sections", "read_section", "summarize_section"])
    if has_tables:
        capabilities.append("list_tables")
    if inferred_type in _TABLE_FILE_TYPES or table_ids:
        capabilities.extend(["list_records", "structured_query"])

    if has_sections and has_tables:
        document_type = "mixed_document"
    elif has_sections:
        document_type = "sectioned_document"
    elif has_tables:
        document_type = "tabular_document"
    else:
        document_type = "unstructured_document"

    return {
        "source": source,
        "file_type": inferred_type,
        "document_type": document_type,
        "features": {
            "has_sections": has_sections,
            "has_tables": has_tables,
            "has_text_content": has_text_content,
        },
        "capabilities": list(dict.fromkeys(capabilities)),
        "statistics": {
            "page_count": max(pages) if pages else 0,
            "section_count": len(sections),
            "table_count": len(table_ids),
            "chunk_count": len(records),
        },
        "sections": sections,
    }


def suggested_section_titles(structure: dict, *, limit: int = 2) -> list[str]:
    """Return representative, stored section titles for document Q&A shortcuts.

    This is intentionally metadata-only: it never infers a title from document
    text or from a document name.  Preference terms only make common document
    questions easier to reach; every returned title is an original section
    title saved during ingestion.
    """
    if limit <= 0 or "list_sections" not in structure.get("capabilities", []):
        return []

    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for index, section in enumerate(structure.get("sections", [])):
        title = str(section.get("title", "")).strip()
        normalized = re.sub(r"\s+", "", title).casefold()
        if not title or normalized in seen:
            continue
        seen.add(normalized)
        preference = next(
            (rank for rank, term in enumerate(_SECTION_SUGGESTION_TERMS) if term in title),
            len(_SECTION_SUGGESTION_TERMS),
        )
        candidates.append((preference, index, title))
    candidates.sort()
    return [title for _, _, title in candidates[:limit]]


def document_section(source: str, section_id: str) -> dict | None:
    """Return one complete section by joining its stored children in order."""
    structure = document_structure(source)
    section = next(
        (
            item
            for item in structure["sections"]
            if item["section_id"] == section_id
        ),
        None,
    )
    if section is None:
        return None

    children = [
        record
        for record in get_chroma_source_records(source)
        if str((record.get("metadata") or {}).get("section_id", "")) == section_id
        and str((record.get("metadata") or {}).get("content_type", ""))
        in _SECTION_CONTENT_TYPES
    ]
    children.sort(key=lambda item: _integer(
        (item.get("metadata") or {}).get("child_index")
    ))
    parts = [
        _clean_section_chunk(record.get("text", ""), section["title"])
        for record in children
    ]
    content = "\n\n".join(part for part in parts if part)
    return {
        **section,
        "source": os.path.basename(source),
        "pages": list(range(
            section["start_page"] or 0,
            (section["end_page"] or section["start_page"] or -1) + 1,
        )),
        "content": content,
    }
