from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from openpyxl import Workbook

from utils.parsers.pdf_parser import (
    _detect_section_headings,
    _extract_table_with_confirmed_spans,
    _normalize_section_child_metadata,
    _section_chunks_from_page_text,
    _section_continuation_chunks,
    _unparsed_table_to_text,
)
from utils.parsers.xlsx_parser import _iter_xlsx_sheets
from utils.table_ingest_pipeline import _dataframe_to_raw_table
from utils.table_parser import _parse_table


class ConfirmedMergedCellTest(unittest.TestCase):
    def test_xlsx_expands_physical_merge_but_preserves_normal_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "merged.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Sheet1"
            sheet.append(["그룹", "금액"])
            sheet.append(["A", 100])
            sheet.append([None, 200])
            sheet.append([None, 300])
            sheet.merge_cells("A2:A3")
            workbook.save(path)

            with pd.ExcelFile(
                path,
                engine="openpyxl",
                engine_kwargs={"read_only": False},
            ) as excel:
                _, _, raw = next(_iter_xlsx_sheets(excel))

            parsed = _parse_table(
                _dataframe_to_raw_table(raw),
                source_file=path.name,
            )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["그룹"].iloc[0], "A")
        self.assertEqual(parsed["그룹"].iloc[1], "A")
        self.assertTrue(pd.isna(parsed["그룹"].iloc[2]))

    def test_pdf_expands_only_geometry_that_spans_the_next_row(self):
        rows = [
            SimpleNamespace(
                bbox=(0, 0, 2, 1),
                cells=[(0, 0, 1, 2), (1, 0, 2, 1)],
            ),
            SimpleNamespace(
                bbox=(0, 1, 2, 2),
                cells=[None, (1, 1, 2, 2)],
            ),
        ]
        table = SimpleNamespace(
            rows=rows,
            extract=lambda: [["A", "100"], [None, "200"]],
        )

        values = _extract_table_with_confirmed_spans(table)

        self.assertEqual(values, [["A", "100"], ["A", "200"]])

    def test_pdf_preserves_blank_without_spanning_geometry(self):
        rows = [
            SimpleNamespace(
                bbox=(0, 0, 2, 1),
                cells=[(0, 0, 1, 1), (1, 0, 2, 1)],
            ),
            SimpleNamespace(
                bbox=(0, 1, 2, 2),
                cells=[None, (1, 1, 2, 2)],
            ),
        ]
        table = SimpleNamespace(
            rows=rows,
            extract=lambda: [["A", "100"], [None, "200"]],
        )

        values = _extract_table_with_confirmed_spans(table)

        self.assertIsNone(values[1][0])

    def test_unparsed_pdf_heading_table_remains_searchable_text(self):
        text = _unparsed_table_to_text([["Ⅰ", None, "추진배경"]])

        self.assertEqual(text, "PDF 문서의 섹션 제목(검색 키워드): Ⅰ 추진배경")
        self.assertGreaterEqual(len(text), 20)

    def test_section_chunk_stops_at_the_next_heading(self):
        page_text = "I Background\nFirst detailed sentence\nSecond detailed sentence\nII Evidence\nRegulation"
        records = _section_chunks_from_page_text(
            page_text,
            ["I Background", "II Evidence"],
            page=1,
        )

        self.assertEqual(len(records), 2)
        self.assertIn("First detailed sentence", records[0]["text"])
        self.assertNotIn("II Evidence", records[0]["text"])
        self.assertEqual(records[0]["metadata"]["content_type"], "pdf_section_child")
        self.assertEqual(records[0]["metadata"]["section_id"], "p1:s0")
        self.assertEqual(records[0]["metadata"]["parent_id"], "p1:s0")
        self.assertEqual(records[0]["metadata"]["section_title"], "I Background")
        self.assertEqual(records[0]["metadata"]["child_index"], 0)

    def test_section_headings_come_from_numbered_page_text(self):
        page_text = (
            "2026 scholarship plan\n"
            "Ⅰ 추진배경\n"
            "first reason\n"
            "Ⅱ 관련근거\n"
            "regulation\n"
            "1. 선발대상\n"
        )

        self.assertEqual(
            _detect_section_headings(page_text),
            ["Ⅰ 추진배경", "Ⅱ 관련근거"],
        )

    def test_long_section_is_split_into_children_with_shared_parent(self):
        page_text = "I Background\n" + "Scholarship support is needed. " * 80
        records = _section_chunks_from_page_text(
            page_text,
            ["I Background"],
            page=2,
        )

        self.assertGreater(len(records), 1)
        self.assertEqual({r["metadata"]["section_id"] for r in records}, {"p2:s0"})
        self.assertEqual(
            [r["metadata"]["child_index"] for r in records],
            list(range(len(records))),
        )
        self.assertTrue(all("I Background" in r["text"] for r in records))
        self.assertTrue(all(r["metadata"]["child_count"] == len(records) for r in records))

    def test_next_page_continuation_keeps_parent_and_final_child_order(self):
        first_page = _section_chunks_from_page_text(
            "Ⅰ Background\n" + "First page sentence. " * 20,
            ["Ⅰ Background"],
            page=1,
        )
        second_page = _section_continuation_chunks(
            "Second page continuation. " * 25,
            page=2,
            section_id="p1:s0",
            section_title="Ⅰ Background",
        )
        records = first_page + second_page

        _normalize_section_child_metadata(records)

        self.assertEqual({r["metadata"]["section_id"] for r in records}, {"p1:s0"})
        self.assertEqual(
            [r["metadata"]["child_index"] for r in records],
            list(range(len(records))),
        )
        self.assertTrue(all(r["metadata"]["child_count"] == len(records) for r in records))


if __name__ == "__main__":
    unittest.main()
