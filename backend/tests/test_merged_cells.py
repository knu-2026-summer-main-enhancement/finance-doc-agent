from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from openpyxl import Workbook

from utils.parsers.pdf_parser import _extract_table_with_confirmed_spans
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


if __name__ == "__main__":
    unittest.main()
