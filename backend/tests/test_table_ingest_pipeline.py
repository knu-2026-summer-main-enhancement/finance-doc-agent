from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import pandas as pd

from utils.table_ingest_pipeline import ingest_dataframe_sheets
from utils.parsers.xlsx_parser import ingest_xlsx


class _FakeWorkbook:
    sheet_names = ["첫 시트", "두 번째 시트"]

    def __init__(self) -> None:
        self.parse = Mock(
            side_effect=[
                pd.DataFrame([["이름"], ["홍길동"]]),
                pd.DataFrame([["이름"], ["김철수"]]),
            ]
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class DataFrameSheetIngestPipelineTest(unittest.TestCase):
    def test_xlsx_adapter_only_reads_workbook_and_delegates_sheets(self):
        workbook = _FakeWorkbook()

        def consume_sheets(sheets, **kwargs):
            supplied = list(sheets)
            self.assertEqual(
                [(index, name) for index, name, _ in supplied],
                [(0, "첫 시트"), (1, "두 번째 시트")],
            )
            self.assertEqual(kwargs["sheet_count"], 2)
            self.assertEqual(kwargs["source_file"], "원본문서.hwpx")
            self.assertEqual(kwargs["doc_label"], "원본 문서")
            self.assertEqual(kwargs["dataframe_prefix"], "df_custom")
            self.assertEqual(kwargs["source_type"], "hwpx")
            self.assertEqual(kwargs["chroma_source_override"], "원본문서.hwpx")
            return 7

        with patch(
            "utils.parsers.xlsx_parser.pd.ExcelFile",
            return_value=workbook,
        ), patch(
            "utils.parsers.xlsx_parser.ingest_dataframe_sheets",
            side_effect=consume_sheets,
        ) as pipeline:
            count = ingest_xlsx(
                "C:/temp/변환결과.xlsx",
                "hash-1",
                "scholarship",
                source_override="원본문서.hwpx",
                label_override="원본 문서",
                var_prefix_override="df_custom",
                chroma_file_path_override="C:/temp/원본문서.hwpx",
                file_type_override="hwpx",
            )

        self.assertEqual(count, 7)
        self.assertEqual(pipeline.call_count, 1)
        self.assertEqual(workbook.parse.call_count, 2)
        self.assertTrue(all(
            call.kwargs == {"header": None}
            for call in workbook.parse.call_args_list
        ))

    def test_multiple_sheets_keep_indices_labels_and_shared_storage_flow(self):
        first_raw = pd.DataFrame([["이름", "금액"], ["홍길동", float("nan")]])
        empty_raw = pd.DataFrame()
        third_raw = pd.DataFrame([["이름", "금액"], ["김철수", 1000]])
        first_parsed = pd.DataFrame({"이름": ["홍길동"], "금액": [None]})
        third_parsed = pd.DataFrame({"이름": ["김철수"], "금액": [1000]})
        parse_table = Mock(side_effect=[first_parsed, third_parsed])
        save_dataframe = Mock()
        chunks = Mock(
            side_effect=[
                [{"text": "첫 번째 행 데이터"}],
                [{"text": "세 번째 행 데이터"}],
            ]
        )
        overview = {"text": "문서 개요 데이터"}

        with patch(
            "utils.table_ingest_pipeline.drop_dataframe_files"
        ) as drop_files, patch(
            "utils.table_ingest_pipeline._parse_table",
            new=parse_table,
        ), patch(
            "utils.table_ingest_pipeline.save_dataframe",
            new=save_dataframe,
        ), patch(
            "utils.table_ingest_pipeline._table_to_text_chunks",
            new=chunks,
        ), patch(
            "utils.table_ingest_pipeline._make_doc_overview_chunk",
            return_value=overview,
        ), patch(
            "utils.table_ingest_pipeline.save_to_chroma",
            return_value=3,
        ) as save_chroma:
            count = ingest_dataframe_sheets(
                [
                    (0, "첫 시트", first_raw),
                    (1, "빈 시트", empty_raw),
                    (2, "세 번째 시트", third_raw),
                ],
                sheet_count=3,
                source_file="장학대장.xlsx",
                doc_label="장학대장",
                dataframe_prefix="df_ledger",
                source_type="xlsx",
                chroma_file_path="C:/data/장학대장.xlsx",
                file_hash="hash-1",
                category="scholarship",
            )

        self.assertEqual(count, 3)
        drop_files.assert_called_once_with("df_ledger")
        self.assertEqual(parse_table.call_count, 2)
        self.assertIsNone(parse_table.call_args_list[0].args[0][1][1])
        self.assertEqual(
            parse_table.call_args_list[0].kwargs["context_prefix"],
            "s0",
        )
        self.assertEqual(
            parse_table.call_args_list[1].kwargs["context_prefix"],
            "s2",
        )
        self.assertEqual(
            [call.args[1] for call in save_dataframe.call_args_list],
            ["df_ledger_s0", "df_ledger_s2"],
        )
        self.assertEqual(
            [call.args[3] for call in save_dataframe.call_args_list],
            ["장학대장 - 첫 시트", "장학대장 - 세 번째 시트"],
        )
        saved_chunks = save_chroma.call_args.args[1]
        self.assertEqual(saved_chunks[0], overview)
        self.assertEqual(len(saved_chunks), 3)

    def test_without_content_hash_skips_chroma_but_keeps_parquet_flow(self):
        raw = pd.DataFrame([["항목"], ["값"]])
        parsed = pd.DataFrame({"항목": ["값"]})

        with patch(
            "utils.table_ingest_pipeline.drop_dataframe_files"
        ), patch(
            "utils.table_ingest_pipeline._parse_table",
            return_value=parsed,
        ), patch(
            "utils.table_ingest_pipeline.save_dataframe"
        ) as save_dataframe, patch(
            "utils.table_ingest_pipeline._table_to_text_chunks",
            return_value=[{"text": "충분히 긴 표 데이터"}],
        ), patch(
            "utils.table_ingest_pipeline._make_doc_overview_chunk",
            return_value=None,
        ), patch(
            "utils.table_ingest_pipeline.save_to_chroma"
        ) as save_chroma:
            count = ingest_dataframe_sheets(
                [(0, "Sheet1", raw)],
                sheet_count=1,
                source_file="목록.xlsx",
                doc_label="목록",
                dataframe_prefix="df_list",
                source_type="xlsx",
                chroma_file_path="C:/data/목록.xlsx",
            )

        self.assertEqual(count, 0)
        save_dataframe.assert_called_once()
        self.assertEqual(save_dataframe.call_args.args[1], "df_list")
        self.assertEqual(save_dataframe.call_args.args[3], "목록")
        save_chroma.assert_not_called()


if __name__ == "__main__":
    unittest.main()
