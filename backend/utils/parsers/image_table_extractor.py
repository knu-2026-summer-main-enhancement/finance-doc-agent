"""OpenCV grid detection and PaddleOCR cell recognition.

The module name is historical; ingestion no longer creates an Excel intermediate.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

os.environ.setdefault("FLAGS_use_onednn", "false")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

logger = logging.getLogger("ingest")
_OCR_LOCK = threading.Lock()

EXPECTED_HEADERS = ("발행번호", "출연일자", "기수", "이름", "출연금액")
_NUMERIC_TRANSLATION = str.maketrans({"O": "0", "o": "0", "Q": "0", "I": "1", "l": "1", "|": "1"})


class ImageTableExtractionError(RuntimeError):
    """Raised when an image cannot be converted into validated table records."""


@dataclass(frozen=True)
class TableGrid:
    image: np.ndarray
    gray: np.ndarray
    x_lines: tuple[int, ...]
    y_lines: tuple[int, ...]

    @property
    def column_count(self) -> int:
        return len(self.x_lines) - 1

    @property
    def row_count(self) -> int:
        return len(self.y_lines) - 1


def _read_image(path: str | Path) -> np.ndarray:
    # cv2.imread can fail on non-ASCII Windows paths; imdecode is path-safe.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageTableExtractionError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def _cluster_positions(indices: np.ndarray, max_gap: int = 2) -> tuple[int, ...]:
    if len(indices) == 0:
        return ()
    groups: list[list[int]] = [[int(indices[0])]]
    for value in indices[1:]:
        value = int(value)
        if value - groups[-1][-1] <= max_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return tuple(round(sum(group) / len(group)) for group in groups)


def detect_table_grid(image_path: str | Path) -> TableGrid:
    """Detect the five-column grid without asking an OCR model to infer layout."""
    image = _read_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    height, width = gray.shape

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 8), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 12)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    y_scores = np.count_nonzero(horizontal, axis=1)
    x_scores = np.count_nonzero(vertical, axis=0)
    y_lines = _cluster_positions(np.where(y_scores >= width * 0.35)[0])
    x_lines = list(_cluster_positions(np.where(x_scores >= height * 0.20)[0]))

    # A border touching the bitmap edge is often lost during thresholding.
    if x_lines and x_lines[0] > width * 0.05:
        x_lines.insert(0, 0)
    if x_lines and x_lines[-1] < width * 0.95:
        x_lines.append(width - 1)

    x_lines = tuple(x_lines)
    if len(x_lines) != 6:
        raise ImageTableExtractionError(
            f"5열 표를 검출하지 못했습니다: column_boundaries={len(x_lines)} ({x_lines})"
        )
    if len(y_lines) < 3:
        raise ImageTableExtractionError(f"표 행을 검출하지 못했습니다: row_boundaries={len(y_lines)}")
    return TableGrid(image=image, gray=gray, x_lines=x_lines, y_lines=y_lines)


@lru_cache(maxsize=1)
def _get_cell_reader() -> Any:
    """Recognition-only model; table geometry already provides detection boxes."""
    # Paddle checks for the optional C++ build cache even though inference never
    # compiles an extension. Hide that misleading warning on CPU-only servers.
    warnings.filterwarnings(
        "ignore",
        message=r"No ccache found\..*",
        module=r"paddle\.utils\.cpp_extension\.extension_utils",
    )
    try:
        from paddleocr import TextRecognition  # type: ignore
    except ImportError as exc:
        raise ImageTableExtractionError("PaddleOCR가 필요합니다.") from exc
    return TextRecognition(
        model_name=os.getenv("PADDLEOCR_REC_MODEL", "korean_PP-OCRv5_mobile_rec"),
        device="cpu",
        enable_mkldnn=False,
    )


def _cell_result(result: Any) -> tuple[str, float]:
    value = getattr(result, "json", result)
    if isinstance(value, dict) and "res" in value:
        value = value["res"]
    if isinstance(value, dict):
        return str(value.get("rec_text", "")).strip(), float(value.get("rec_score", 0.0))
    return "", 0.0


def _clean_numeric(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).translate(_NUMERIC_TRANSLATION)


def normalize_cell(text: str, column: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    if column == 0:
        digits = re.sub(r"[^0-9]", "", _clean_numeric(text))
        return f"{digits[:4]}-{digits[4:7]}" if len(digits) == 7 else ""
    if column == 1:
        digits = re.sub(r"[^0-9]", "", _clean_numeric(text))
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}" if len(digits) == 8 else ""
    if column == 2:
        digits = re.sub(r"[^0-9]", "", _clean_numeric(text))
        return digits if 1 <= len(digits) <= 3 else ""
    if column == 4:
        digits = re.sub(r"[^0-9]", "", _clean_numeric(text))
        return f"{int(digits):,}" if digits else ""
    return text


def _line_crosses_column(grid: TableGrid, y: int, column: int) -> bool:
    x0, x1 = grid.x_lines[column], grid.x_lines[column + 1]
    strip = grid.gray[max(0, y - 1): y + 2, x0 + 2: max(x0 + 3, x1 - 2)]
    if strip.size == 0:
        return False
    # A grid line is normally one pixel high. Averaging the three sampled rows
    # caps a perfect line at about 0.33 and made every boundary look merged.
    # Judge the best individual scanline instead.
    per_scanline_coverage = np.mean(strip < 190, axis=1)
    return float(np.max(per_scanline_coverage)) >= 0.55


def _column_row_groups(grid: TableGrid, row_count: int, column: int) -> list[tuple[int, int]]:
    """Return half-open physical-row ranges for real cells in one column."""
    if column not in {0, 2, 3}:
        return [(index, index + 1) for index in range(row_count)]
    groups: list[tuple[int, int]] = []
    start = 0
    for boundary_index in range(1, row_count):
        if _line_crosses_column(grid, grid.y_lines[boundary_index + 1], column):
            groups.append((start, boundary_index))
            start = boundary_index
    groups.append((start, row_count))
    return groups


def extract_table_records(
    image_path: str | Path,
    *,
    recognizer: Callable[[np.ndarray], tuple[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract a five-column table directly; no XLSX intermediate is created."""
    grid = detect_table_grid(image_path)
    row_bands = list(zip(grid.y_lines[1:-1], grid.y_lines[2:]))
    # Each unit is one real cell. Merged columns use the full vertical span, so
    # text crossing a physical date/amount boundary is never sliced in half.
    cell_units: list[tuple[int, int, int, float]] = []
    cell_images: list[np.ndarray] = []
    for col, (left, right) in enumerate(zip(grid.x_lines, grid.x_lines[1:])):
        for row_start, row_end in _column_row_groups(grid, len(row_bands), col):
            top = row_bands[row_start][0]
            bottom = row_bands[row_end - 1][1]
            cell = grid.image[top + 2:bottom - 1, left + 2:right - 1]
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            dark_y, dark_x = np.where(gray_cell < 180)
            ink = float(len(dark_x))
            if len(dark_x):
                pad = 3
                content_left = max(0, int(dark_x.min()) - pad)
                content_right = min(cell.shape[1], int(dark_x.max()) + pad + 1)
                content_top = max(0, int(dark_y.min()) - pad)
                content_bottom = min(cell.shape[0], int(dark_y.max()) + pad + 1)
                cell = cell[content_top:content_bottom, content_left:content_right]
            target_height = 64
            # Tight content cropping prevents text in a tall merged cell from being
            # reduced to a few pixels by the recognizer's internal resize.
            scale = max(1.0, target_height / max(1, cell.shape[0]))
            image = cv2.resize(cell, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            cell_units.append((row_start, row_end, col, ink))
            cell_images.append(image)

    if recognizer:
        recognized = [recognizer(image) for image in cell_images]
    else:
        reader = _get_cell_reader()
        batch_size = max(1, int(os.getenv("IMAGE_OCR_CELL_BATCH", "64")))
        logger.info(
            "[IMAGE] PaddleOCR 셀 인식 시작 | cells=%d batch_size=%d",
            len(cell_images),
            batch_size,
        )
        recognized = []
        # Paddle inference objects are cached for speed but are not guaranteed to
        # be thread-safe. Other file types may still ingest in parallel.
        with _OCR_LOCK:
            for completed, result in enumerate(
                reader.predict(cell_images, batch_size=batch_size),
                start=1,
            ):
                recognized.append(_cell_result(result))
                if completed % 50 == 0 or completed == len(cell_images):
                    logger.info(
                        "[IMAGE] PaddleOCR 진행 | completed=%d/%d (%.1f%%)",
                        completed,
                        len(cell_images),
                        completed / len(cell_images) * 100,
                    )
        if len(recognized) != len(cell_images):
            raise ImageTableExtractionError(
                f"OCR 결과 수가 셀 수와 다릅니다: cells={len(cell_images)} results={len(recognized)}"
            )

    values: list[list[str]] = [["" for _ in EXPECTED_HEADERS] for _ in row_bands]
    scores: list[list[float]] = [[0.0 for _ in EXPECTED_HEADERS] for _ in row_bands]
    for (row_start, row_end, col, ink), (value, score) in zip(cell_units, recognized):
        if ink < 5:
            value, score = "", 0.0
        normalized = normalize_cell(value, col)
        for row_index in range(row_start, row_end):
            values[row_index][col] = normalized
            scores[row_index][col] = score

    records: list[dict[str, Any]] = []
    for row_index in range(len(row_bands)):
        record: dict[str, Any] = {"ocr_row_index": row_index}
        confidence_values: list[float] = []
        low_cells: list[str] = []
        for col, header in enumerate(EXPECTED_HEADERS):
            value, score = values[row_index][col], scores[row_index][col]
            record[header] = value
            if value:
                confidence_values.append(score)
            if value and score < 0.75:
                low_cells.append(f"{header}:{value}({score:.3f})")
        record["_ocr_confidence_min"] = f"{min(confidence_values):.3f}" if confidence_values else ""
        record["_ocr_confidence_avg"] = (
            f"{sum(confidence_values) / len(confidence_values):.3f}" if confidence_values else ""
        )
        record["_ocr_low_confidence_cells"] = "; ".join(low_cells)
        records.append(record)

    metadata = {
        "physical_rows": len(records),
        "columns": grid.column_count,
        "x_lines": list(grid.x_lines),
        "y_lines": list(grid.y_lines),
    }
    return records, metadata
