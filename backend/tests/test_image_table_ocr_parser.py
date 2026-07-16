from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from PIL import Image, ImageDraw

from utils.parsers.image_table_extractor import (
    _line_crosses_column,
    detect_table_grid,
    extract_table_records,
    normalize_cell,
)
from utils.parsers.image_table_ocr_parser import (
    _validate_records,
    detect_table_column_centers,
    detect_table_row_bands,
)


class ImageTableOcrParserTest(unittest.TestCase):
    def test_detect_table_row_bands_with_merged_cells(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "long_table.png"
            image = Image.new("RGB", (300, 140), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 10, 299, 130), outline="black", width=2)
            for x in (55, 130, 190, 245):
                draw.line((x, 10, x, 130), fill="black", width=2)
            # 첫 열만 세로 병합된 표를 만든다.
            for y in (30, 50, 70, 90, 110):
                draw.line((55, y, 299, y), fill="black", width=2)
            image.save(path)

            bands = detect_table_row_bands(str(path))

            self.assertEqual(len(bands), 6)
            self.assertTrue(all(bottom > top for top, bottom in bands))
            self.assertEqual(len(detect_table_column_centers(str(path))), 5)
            grid = detect_table_grid(str(path))
            self.assertTrue(_line_crosses_column(grid, 30, 3))

    def test_numeric_normalization_preserves_unknown_formats(self):
        self.assertEqual(normalize_cell("2025-O01", 0), "2025-001")
        self.assertEqual(normalize_cell("2024-01", 0), "2024-01")
        self.assertEqual(normalize_cell("2025-0019", 0), "2025-0019")
        self.assertEqual(normalize_cell("A-2025-001", 0), "A-2025-001")
        self.assertEqual(normalize_cell("2025.01.02", 1), "2025.01.02")
        self.assertEqual(normalize_cell("1,OOO,OOO", 4), "1,000,000")

    def test_numeric_correction_keeps_raw_ocr_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "corrected_value.png"
            image = Image.new("RGB", (160, 75), "white")
            draw = ImageDraw.Draw(image)
            for x in (0, 80, 159):
                draw.line((x, 5, x, 70), fill="black", width=2)
            for y in (5, 35, 70):
                draw.line((0, y, 159, y), fill="black", width=2)
            for left in (0, 80):
                for top in (5, 35):
                    draw.rectangle((left + 10, top + 8, left + 14, top + 12), fill="black")
            image.save(path)

            recognized = iter([
                ("식별자", 0.99), ("2025-O01", 0.95),
                ("금액", 0.99), ("1,OOO", 0.95),
            ])
            records, _ = extract_table_records(
                path,
                recognizer=lambda _image: next(recognized),
            )

            self.assertEqual(records[0]["식별자"], "2025-001")
            corrections = json.loads(records[0]["_ocr_corrections"])
            self.assertEqual(corrections[0]["raw"], "2025-O01")

    def test_extracts_dynamic_headers_and_column_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "three_columns.png"
            image = Image.new("RGB", (240, 100), "white")
            draw = ImageDraw.Draw(image)
            for x in (0, 80, 160, 239):
                draw.line((x, 10, x, 90), fill="black", width=2)
            for y in (10, 40, 65, 90):
                draw.line((0, y, 239, y), fill="black", width=2)
            # OCR 호출 대상에서 제외되지 않도록 각 셀에 최소한의 잉크를 둔다.
            for left in (0, 80, 160):
                for top in (10, 40, 65):
                    draw.rectangle((left + 10, top + 8, left + 14, top + 12), fill="black")
            image.save(path)

            recognized = iter([
                ("식별자", 0.99), ("A-01", 0.98), ("A-02", 0.97),
                ("설명", 0.99), ("첫 행", 0.98), ("둘째 행", 0.97),
                ("임의 값", 0.99), ("X-10", 0.98), ("Y-20", 0.97),
            ])
            records, metadata = extract_table_records(
                path,
                recognizer=lambda _image: next(recognized),
            )

            self.assertEqual(metadata["columns"], 3)
            self.assertEqual(metadata["headers"], ["식별자", "설명", "임의 값"])
            self.assertEqual(records[0]["식별자"], "A-01")
            self.assertEqual(records[1]["임의 값"], "Y-20")

    def test_validation_marks_internal_quality_column(self):
        rows = [{
            "발행번호": "2025-001",
            "출연일자": "2025-01-02",
            "기수": "49",
            "이름": "장*윤",
            "출연금액": "1,000,000",
        }]

        result = _validate_records(rows)

        self.assertEqual(result, {"invalid_cells": [], "calculated_total": 1_000_000})
        self.assertTrue(rows[0]["_ocr_validation_ok"])

    def test_validation_does_not_require_domain_specific_columns(self):
        rows = [{
            "임의 식별자": "A-15",
            "설명": "범용 표 데이터",
            "점수": "87.5",
        }]

        result = _validate_records(rows)

        self.assertEqual(result["invalid_cells"], [])
        self.assertTrue(rows[0]["_ocr_validation_ok"])


if __name__ == "__main__":
    unittest.main()
