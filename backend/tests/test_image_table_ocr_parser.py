from pathlib import Path

from PIL import Image, ImageDraw

from utils.parsers.image_table_ocr_parser import detect_table_column_centers, detect_table_row_bands
from utils.parsers.image_table_extractor import _line_crosses_column, detect_table_grid, normalize_cell
from utils.parsers.image_table_ocr_parser import _validate_records


def test_detect_table_row_bands_with_merged_cells(tmp_path: Path):
    image = Image.new("RGB", (300, 140), "white")
    draw = ImageDraw.Draw(image)
    # 표 외곽선과 열 구분선
    draw.rectangle((0, 10, 299, 130), outline="black", width=2)
    for x in (55, 130, 190, 245):
        draw.line((x, 10, x, 130), fill="black", width=2)
    # 병합된 첫 열에는 닿지 않는 내부 행 구분선
    for y in (30, 50, 70, 90, 110):
        draw.line((55, y, 299, y), fill="black", width=2)

    path = tmp_path / "long_table.png"
    image.save(path)

    bands = detect_table_row_bands(str(path))

    assert len(bands) == 6
    assert all(bottom > top for top, bottom in bands)
    assert len(detect_table_column_centers(str(path))) == 5
    grid = detect_table_grid(str(path))
    assert _line_crosses_column(grid, 30, 3) is True


def test_numeric_normalization_rejects_extra_digits():
    assert normalize_cell("2025-O01", 0) == "2025-001"
    assert normalize_cell("2025-0019", 0) == ""
    assert normalize_cell("2025.01.02", 1) == "2025-01-02"
    assert normalize_cell("1,OOO,OOO", 4) == "1,000,000"


def test_validation_marks_internal_quality_column():
    rows = [{
        "발행번호": "2025-001",
        "출연일자": "2025-01-02",
        "기수": "49",
        "이름": "장*윤",
        "출연금액": "1,000,000",
    }]

    result = _validate_records(rows)

    assert result == {"invalid_cells": [], "calculated_total": 1_000_000}
    assert rows[0]["_ocr_validation_ok"] is True


def test_validation_allows_empty_cohort_for_organization():
    rows = [{
        "발행번호": "2025-009",
        "출연일자": "2025-01-26",
        "기수": "",
        "이름": "현대중공업대공동문회",
        "출연금액": "1,000,000",
    }]

    assert _validate_records(rows)["invalid_cells"] == []
