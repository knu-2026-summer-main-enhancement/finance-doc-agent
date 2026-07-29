from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from xml.etree import ElementTree

import pandas as pd
from bs4 import BeautifulSoup

from utils.chroma_store import save_to_chroma
from utils.parquet_store import drop_dataframe_files, save_dataframe
from utils.table_parser import _clean_dataframe, _parse_table, sanitize_table_name
from utils.text_utils import _make_doc_overview_chunk, _table_to_text_chunks, split_into_chunks

logger = logging.getLogger("ingest")

HWP_SECTION_SCHEMA_VERSION = "hwp-section-v5"
_SECTION_HEADING_RE = re.compile(
    r"^\s*(?P<title>(?:[□■◆◇]|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|(?:\d{1,3}\.))\s*\S.{0,100})\s*$"
)
_TOP_LEVEL_TABLE_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d{1,2})\s+(?P<title>[가-힣A-Za-z]\S*(?:\s+\S+){0,8})\s*$"
)
_ATTACHMENT_BOUNDARY_RE = re.compile(
    r"^\s*(?:\[\s*)?붙임\s*\d*(?:\s*\])?.*$"
)


def _hwp5_command(name: str) -> str | None:
    executable = f"{name}.exe" if os.name == "nt" else name
    candidates = (
        shutil.which(executable),
        os.path.join(sys.prefix, "Scripts", executable),
        os.path.join(sys.prefix, "bin", name),
    )
    return next((candidate for candidate in candidates if candidate and os.path.exists(candidate)), None)


def _run_hwp5txt(file_path: str) -> str:
    """Extract readable HWPv5 prose without requiring the Hangul desktop app."""
    command = _hwp5_command("hwp5txt")
    if not command:
        return ""
    try:
        result = subprocess.run(
            [command, file_path],
            capture_output=True,
            timeout=60,
        )
        # pyhwp can emit usable text while returning a non-zero status for a
        # recoverable record. Content is therefore the success criterion.
        text = result.stdout.decode("utf-8", errors="replace")
        text = re.sub(r"^\s*<표>\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if not text and result.returncode != 0:
            logger.warning(
                "[HWP/hwp5txt] 본문 추출 실패 | rc=%d err=%s",
                result.returncode,
                result.stderr.decode("utf-8", errors="replace").strip(),
            )
        return text
    except Exception as exc:
        logger.warning("[HWP/hwp5txt] 본문 추출 실패 | file=%s err=%s", file_path, exc)
        return ""


def _extract_hwpx_text(file_path: str) -> str:
    """Read paragraph text from HWPX's zipped XML package."""
    try:
        with zipfile.ZipFile(file_path) as archive:
            section_names = sorted(
                (
                    name for name in archive.namelist()
                    if re.search(r"(?:^|/)section\d+\.xml$", name, re.IGNORECASE)
                ),
                key=lambda name: [
                    int(part) if part.isdigit() else part.casefold()
                    for part in re.split(r"(\d+)", name)
                ],
            )
            paragraphs: list[str] = []
            for name in section_names:
                root = ElementTree.fromstring(archive.read(name))
                for element in root.iter():
                    if not str(element.tag).endswith("}p"):
                        continue
                    text = "".join(
                        child.text or ""
                        for child in element.iter()
                        if str(child.tag).endswith("}t")
                    ).strip()
                    if text:
                        paragraphs.append(text)
        return "\n".join(paragraphs)
    except Exception as exc:
        logger.warning("[HWPX/xml] 본문 추출 실패 | file=%s err=%s", file_path, exc)
        return ""


def _html_table_grid(table) -> list[list[str | None]]:
    rows = table.find_all("tr")
    if not rows:
        return []
    num_cols = max(
        (sum(int(cell.get("colspan", 1)) for cell in row.find_all(["td", "th"])) for row in rows),
        default=0,
    )
    if num_cols == 0:
        return []
    grid: list[list[str | None]] = [[None] * num_cols for _ in rows]
    for row_index, row in enumerate(rows):
        column_index = 0
        for cell in row.find_all(["td", "th"]):
            while column_index < num_cols and grid[row_index][column_index] is not None:
                column_index += 1
            if column_index >= num_cols:
                break
            rowspan = max(1, int(cell.get("rowspan", 1)))
            colspan = max(1, int(cell.get("colspan", 1)))
            value = re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip() or None
            for target_row in range(row_index, min(row_index + rowspan, len(rows))):
                for target_column in range(column_index, min(column_index + colspan, num_cols)):
                    grid[target_row][target_column] = value
            column_index += colspan
    return grid


def _extract_hwp5_html(
    file_path: str,
    source_file: str,
) -> tuple[list[pd.DataFrame], list[tuple[str, str]]]:
    """Extract independent tables and top-level title-box sections from HWPv5."""
    command = _hwp5_command("hwp5html")
    if not command:
        return [], []
    try:
        with tempfile.TemporaryDirectory(prefix="project1-hwp-") as temp_dir:
            html_path = os.path.join(temp_dir, "document.html")
            result = subprocess.run(
                [command, "--html", "--output", html_path, file_path],
                capture_output=True,
                timeout=90,
            )
            if not os.path.exists(html_path):
                logger.warning(
                    "[HWP/hwp5html] 표 추출 실패 | rc=%d err=%s",
                    result.returncode,
                    result.stderr.decode("utf-8", errors="replace").strip(),
                )
                return [], []
            with open(html_path, "rb") as stream:
                soup = BeautifulSoup(stream.read(), "html.parser")

        frames: list[pd.DataFrame] = []
        current_section_title = ""
        table_index = 0
        for element in soup.find_all(["table", "p"]):
            if element.name == "p" and element.find_parent("table") is not None:
                continue
            value = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
            if not value:
                continue
            if _ATTACHMENT_BOUNDARY_RE.fullmatch(value):
                current_section_title = ""
                continue
            table_heading = (
                _TOP_LEVEL_TABLE_HEADING_RE.fullmatch(value)
                if element.name == "table" and len(value) <= 80
                else None
            )
            paragraph_heading = (
                re.fullmatch(r"(?P<number>\d{1,2})\.\s*(?P<title>\S.{0,100})", value)
                if element.name == "p"
                else None
            )
            heading = table_heading or paragraph_heading
            if heading:
                separator = "." if paragraph_heading else ""
                current_section_title = (
                    f"{heading.group('number')}{separator} "
                    f"{heading.group('title').strip()}"
                )
                continue
            if element.name != "table":
                continue
            grid = _html_table_grid(element)
            if len(grid) < 2 or max((len(row) for row in grid), default=0) < 2:
                continue
            parsed = _parse_table(
                grid,
                source_file=source_file,
                context_prefix=f"t{table_index}",
            )
            table_index += 1
            if parsed is not None and not parsed.empty:
                parsed.attrs["section_title"] = current_section_title
                frames.append(parsed)

        section_parts: list[tuple[str, list[str]]] = []
        current_parts: list[str] | None = None
        for element in soup.find_all(["table", "p"]):
            value = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
            if not value:
                continue
            if _ATTACHMENT_BOUNDARY_RE.fullmatch(value):
                current_parts = None
                continue
            # Prose order in HWP's generated HTML is layout-oriented rather
            # than reading-oriented. hwp5txt supplies prose in the correct
            # order; HTML is used only for title boxes and table-only content.
            if element.name == "p":
                continue
            heading_match = (
                _TOP_LEVEL_TABLE_HEADING_RE.fullmatch(value)
                if element.name == "table" and len(value) <= 80
                else None
            )
            if heading_match:
                title = (
                    f"{heading_match.group('number')} "
                    f"{heading_match.group('title').strip()}"
                )
                current_parts = []
                section_parts.append((title, current_parts))
                continue
            if current_parts is not None:
                current_parts.append(value)

        sections = [
            (title, "\n".join(dict.fromkeys(parts)).strip())
            for title, parts in section_parts
        ]
        return frames, sections
    except Exception as exc:
        logger.warning("[HWP/hwp5html] 표 추출 실패 | file=%s err=%s", file_path, exc)
        return [], []


def _extract_hwp_table_pyhwpx(file_path: str) -> pd.DataFrame | None:
    """Use the existing COM extractor as a fallback for HWPX/unsupported HWP."""
    helper = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hwp_extract.py")
    try:
        result = subprocess.run(
            [sys.executable, helper, file_path],
            capture_output=True,
            timeout=60,
        )
        stdout = result.stdout.decode("utf-8", errors="replace").strip()
        if result.returncode != 0 or not stdout:
            return None
        records = json.loads(stdout)
        if not records:
            return None
        frame = pd.DataFrame(records)
        return frame if not frame.empty else None
    except Exception as exc:
        logger.warning("[HWP/pyhwpx] 표 추출 실패 | file=%s err=%s", file_path, exc)
        return None


def _text_section_chunks(text: str) -> list[dict]:
    if not text:
        return []
    lines = text.splitlines()
    headings = [
        (index, match.group("title").strip())
        for index, line in enumerate(lines)
        if (match := _SECTION_HEADING_RE.match(line))
    ]
    if not headings:
        return [
            {
                **chunk,
                "metadata": {
                    "content_type": "hwp_text_child",
                    "parser_version": HWP_SECTION_SCHEMA_VERSION,
                },
            }
            for chunk in split_into_chunks(text)
        ]

    records: list[dict] = []
    preface_end = headings[0][0]
    if preface_end:
        for child_index, child in enumerate(split_into_chunks("\n".join(lines[:preface_end]))):
            records.append({
                **child,
                "metadata": {
                    "content_type": "hwp_text_child",
                    "parser_version": HWP_SECTION_SCHEMA_VERSION,
                    "child_index": child_index,
                },
            })
    for section_index, (start, title) in enumerate(headings):
        end = headings[section_index + 1][0] if section_index + 1 < len(headings) else len(lines)
        section_text = "\n".join(lines[start:end]).strip()
        section_id = f"s{section_index}"
        children = split_into_chunks(section_text)
        for child_index, child in enumerate(children):
            records.append({
                **child,
                "metadata": {
                    "content_type": "hwp_section_child",
                    "section_schema_version": HWP_SECTION_SCHEMA_VERSION,
                    "parser_version": HWP_SECTION_SCHEMA_VERSION,
                    "section_id": section_id,
                    "parent_id": section_id,
                    "section_title": title,
                    "section_index": section_index,
                    "child_index": child_index,
                    "child_count": len(children),
                },
            })
    return records


def _top_level_section_chunks(sections: list[tuple[str, str]]) -> list[dict]:
    records: list[dict] = []
    for section_index, (title, text) in enumerate(sections):
        section_id = f"s{section_index}"
        section_text = f"{title}\n{text}".strip()
        children = split_into_chunks(section_text)
        if not children:
            children = [{
                "text": (
                    f"한글 문서 섹션 제목: {section_text}. "
                    "관련 상세 내용은 같은 문서의 표 데이터에 저장되어 있습니다."
                ),
                "page": None,
            }]
        for child_index, child in enumerate(children):
            records.append({
                **child,
                "metadata": {
                    "content_type": "hwp_section_child",
                    "section_schema_version": HWP_SECTION_SCHEMA_VERSION,
                    "parser_version": HWP_SECTION_SCHEMA_VERSION,
                    "section_id": section_id,
                    "parent_id": section_id,
                    "section_title": title,
                    "section_index": section_index,
                    "child_index": child_index,
                    "child_count": len(children),
                },
            })
    return records


def _numbered_text_sections(text: str) -> list[tuple[str, str]]:
    """Extract real numbered headings from reading-order HWP text."""
    lines = text.splitlines()
    attachment_index = next(
        (
            index for index, line in enumerate(lines)
            if _ATTACHMENT_BOUNDARY_RE.fullmatch(line)
        ),
        len(lines),
    )
    lines = lines[:attachment_index]
    positions: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        match = re.match(
            r"^\s*(?P<number>\d{1,2})\.\s*(?P<title>\S.{0,100})\s*$",
            line,
        )
        if match:
            positions.append((
                index,
                f"{match.group('number')}. {match.group('title').strip()}",
            ))
    return [
        (
            title,
            "\n".join(
                lines[
                    start + 1:
                    positions[section_index + 1][0]
                    if section_index + 1 < len(positions)
                    else len(lines)
                ]
            ).strip(),
        )
        for section_index, (start, title) in enumerate(positions)
    ]


def _lower_item_blocks(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    attachment_index = next(
        (
            index for index, line in enumerate(lines)
            if _ATTACHMENT_BOUNDARY_RE.fullmatch(line)
        ),
        len(lines),
    )
    lines = lines[:attachment_index]
    positions = [
        (index, match.group("title").strip())
        for index, line in enumerate(lines)
        if (match := _SECTION_HEADING_RE.match(line))
        and match.group("title").lstrip().startswith(("□", "■", "◆", "◇"))
    ]
    return [
        (
            title,
            "\n".join(
                lines[
                    start:
                    positions[item_index + 1][0]
                    if item_index + 1 < len(positions)
                    else len(lines)
                ]
            ).strip(),
        )
        for item_index, (start, title) in enumerate(positions)
    ]


def _heading_terms(title: str) -> set[str]:
    normalized = re.sub(r"^\s*\d{1,2}[.)]?\s*", "", title)
    normalized = normalized.replace("제외", "제한")
    return {
        token
        for token in re.findall(r"[가-힣A-Za-z]{2,}", normalized)
        if token not in {"장학금", "사항"}
    }


def _heading_key(title: str) -> str:
    return re.sub(r"[\W_]+", "", title).casefold()


def _heading_affinity(lower_terms: set[str], top_terms: set[str]) -> int:
    score = 10 * len(lower_terms & top_terms)
    score += sum(
        3
        for lower in lower_terms
        for top in top_terms
        if lower != top and (lower in top or top in lower)
    )
    return score


def _merge_lower_items_into_top_sections(
    sections: list[tuple[str, str]],
    text: str,
) -> list[tuple[str, str]]:
    """Attach box/bullet subheadings to their nearest real numbered section."""
    if not sections:
        return []
    grouped: list[list[str]] = [[] for _ in sections]
    top_terms = [_heading_terms(title) for title, _ in sections]
    current_index = 0
    for lower_title, lower_text in _lower_item_blocks(text):
        lower_terms = _heading_terms(lower_title)
        candidate_scores = {
            current_index: _heading_affinity(
                lower_terms,
                top_terms[current_index],
            )
        }
        for index in range(current_index + 1, len(sections)):
            score = _heading_affinity(lower_terms, top_terms[index])
            candidate_scores[index] = score
            if score > 0 or not sections[index][1].strip():
                break
            # A table-only section can have no matching lower prose. Skip over
            # it so the following textual section can still receive its items.
        best_score = max(candidate_scores.values(), default=0)
        if best_score > 0:
            current_index = max(
                index for index, score in candidate_scores.items()
                if score == best_score
            )
        grouped[current_index].append(lower_text)

    merged: list[tuple[str, str]] = []
    for index, (title, html_content) in enumerate(sections):
        lower_content = "\n\n".join(grouped[index]).strip()
        # Box/bullet prose is cleaner in hwp5txt. Flowchart/schedule sections
        # often exist only as tables, so retain their HTML table text.
        merged.append((title, lower_content or html_content or title))
    return merged


def convert_hwp_to_html_and_ingest(file_path: str, file_hash: str, category: str) -> int:
    """Ingest HWP as a mixed document: structured tables plus sectioned prose."""
    logger.info("[HWP] %s", file_path)
    source_file = os.path.basename(file_path)
    doc_label = os.path.splitext(source_file)[0]
    safe_name = sanitize_table_name(doc_label)
    source_type = os.path.splitext(source_file)[1].lower().lstrip(".")

    drop_dataframe_files(f"df_{safe_name}_t")

    frames, top_level_sections = _extract_hwp5_html(file_path, source_file)
    if not frames:
        fallback = _extract_hwp_table_pyhwpx(file_path)
        cleaned = (
            _clean_dataframe(fallback, source_file=source_file, context_prefix="t0")
            if fallback is not None
            else None
        )
        if cleaned is not None and not cleaned.empty:
            frames = [cleaned]

    text = (
        _extract_hwpx_text(file_path)
        if source_type == "hwpx"
        else _run_hwp5txt(file_path)
    )
    if top_level_sections and text:
        top_level_sections = _merge_lower_items_into_top_sections(
            top_level_sections,
            text,
        )
    elif text:
        top_level_sections = _numbered_text_sections(text)
    chunks = (
        _top_level_section_chunks(top_level_sections)
        if top_level_sections
        else _text_section_chunks(text)
    )
    section_by_title = {
        _heading_key(title): (f"s{index}", index, title)
        for index, (title, _) in enumerate(top_level_sections)
    }
    for table_index, frame in enumerate(frames):
        save_dataframe(
            frame,
            f"df_{safe_name}_t{table_index}",
            source_file,
            f"{doc_label} 표 {table_index + 1}",
            file_hash=file_hash,
            source_type=source_type,
        )
        for chunk in _table_to_text_chunks(frame, doc_label):
            metadata = chunk.setdefault("metadata", {})
            metadata["content_type"] = "hwp_table_row"
            metadata["parser_version"] = HWP_SECTION_SCHEMA_VERSION
            parent = section_by_title.get(
                _heading_key(str(frame.attrs.get("section_title", "")))
            )
            if parent:
                section_id, section_index, section_title = parent
                metadata.update({
                    "section_schema_version": HWP_SECTION_SCHEMA_VERSION,
                    "section_id": section_id,
                    "parent_id": section_id,
                    "section_title": section_title,
                    "section_index": section_index,
                })
            chunks.append(chunk)

    overview = _make_doc_overview_chunk(doc_label, source_file, frames)
    if overview:
        overview.setdefault("metadata", {})["parser_version"] = HWP_SECTION_SCHEMA_VERSION
        chunks.insert(0, overview)

    count = save_to_chroma(file_path, chunks, file_hash, category) if chunks else 0
    if count <= 0:
        raise RuntimeError("HWP에서 검색 가능한 본문이나 표를 추출하지 못했습니다.")
    logger.info(
        "[HWP] 완료 | file=%s tables=%d text_chunks=%d total_chunks=%d",
        file_path,
        len(frames),
        sum(1 for chunk in chunks if chunk.get("metadata", {}).get("content_type", "").startswith("hwp_")),
        count,
    )
    return count
