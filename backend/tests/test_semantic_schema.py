from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from utils.parquet_store import drop_dataframe_files, save_dataframe
from utils.semantic_schema import SCHEMA_VERSION, semantic_columns
from utils.table_parser import _clean_dataframe
from utils.text_utils import _table_to_text_chunks
import datastore.state as dataframe_state


class SemanticSchemaTest(unittest.TestCase):
    def _sample(self) -> pd.DataFrame:
        raw = pd.DataFrame([
            {
                "발행번호": "2025-001",
                "이름": "김철수",
                "출연금액": "1,000,000",
                "출연일자": "2025-01-01",
                "비고": "첫 납부",
            },
            {
                "발행번호": "2025-002",
                "이름": "이영희",
                "출연금액": "500,000",
                "출연일자": "2025-01-02",
                "비고": "",
            },
        ])
        cleaned = _clean_dataframe(
            raw,
            source_file="donations.xlsx",
            context_prefix="s0",
        )
        self.assertIsNotNone(cleaned)
        return cleaned

    def test_save_preserves_columns_and_writes_semantic_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.parquet_store.DATAFRAME_DIR", temp_dir
        ):
            df = self._sample()
            original_columns = ["발행번호", "이름", "출연금액", "출연일자", "비고"]
            path = save_dataframe(
                df,
                "df_donations",
                "donations.xlsx",
                "기부 내역",
                file_hash="abc123",
                source_type="xlsx",
            )

            stored = pd.read_parquet(path)
            for column in original_columns:
                self.assertIn(column, stored.columns)
            for column in (
                "__schema_version", "__document_id", "__table_id", "__row_id",
                "__source_file", "__source_type", "__mapping_fingerprint",
            ):
                self.assertIn(column, stored.columns)
            self.assertEqual(stored["__row_id"].nunique(), len(stored))
            self.assertEqual(set(stored["__schema_version"]), {SCHEMA_VERSION})

            schema_path = os.path.join(temp_dir, "df_donations.schema.json")
            with open(schema_path, encoding="utf-8") as file:
                schema = json.load(file)
            self.assertEqual(schema["columns"]["출연금액"]["concept"], "measure")
            self.assertEqual(schema["columns"]["출연금액"]["role"], "donation_amount")
            self.assertEqual(schema["columns"]["이름"]["role"], "entity_name")
            self.assertIn("비고", schema["unmapped_columns"])

    def test_same_structure_reuses_profile_without_changing_document_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.parquet_store.DATAFRAME_DIR", temp_dir
        ):
            first = self._sample()
            second = self._sample()
            save_dataframe(first, "df_first", "donations.xlsx", file_hash="same-hash")
            save_dataframe(second, "df_second", "donations.xlsx", file_hash="same-hash")

            with open(os.path.join(temp_dir, "df_first.schema.json"), encoding="utf-8") as file:
                first_schema = json.load(file)
            with open(os.path.join(temp_dir, "df_second.schema.json"), encoding="utf-8") as file:
                second_schema = json.load(file)

            self.assertEqual(first_schema["document_id"], second_schema["document_id"])
            self.assertNotEqual(first_schema["table_id"], second_schema["table_id"])
            self.assertEqual(first_schema["fingerprint"], second_schema["fingerprint"])
            self.assertTrue(second_schema["profile_used"])

    def test_loaded_mapping_drives_semantic_column_selection_and_chunk_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.parquet_store.DATAFRAME_DIR", temp_dir
        ):
            df = self._sample()
            save_dataframe(df, "df_donations", "donations.xlsx", file_hash="abc123")
            with open(os.path.join(temp_dir, "df_donations.schema.json"), encoding="utf-8") as file:
                df.attrs["semantic_schema"] = json.load(file)

            self.assertEqual(
                semantic_columns(df, concept="measure", data_type="money"),
                ["출연금액"],
            )
            chunks = _table_to_text_chunks(df, "기부 내역")
            self.assertEqual(len(chunks), len(df))
            self.assertEqual(chunks[0]["metadata"]["row_id"], df.iloc[0]["__row_id"])
            self.assertEqual(chunks[0]["metadata"]["table_id"], df.iloc[0]["__table_id"])
            self.assertNotIn("__row_id", chunks[0]["text"])

    def test_dataframe_loader_attaches_saved_semantic_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.parquet_store.DATAFRAME_DIR", temp_dir
        ), patch("datastore.state.DATAFRAME_DIR", temp_dir):
            df = self._sample()
            save_dataframe(df, "df_donations", "donations.xlsx", file_hash="abc123")
            try:
                dataframe_state._load_dataframes()
                loaded = dataframe_state._df_namespace["df0"]
                self.assertIn("semantic_schema", loaded.attrs)
                self.assertEqual(
                    semantic_columns(loaded, concept="measure", data_type="money"),
                    ["출연금액"],
                )
                self.assertIn("df0", dataframe_state._df_schemas)
            finally:
                dataframe_state._df_namespace.clear()
                dataframe_state._df_sources.clear()
                dataframe_state._df_labels.clear()
                dataframe_state._df_schemas.clear()

    def test_drop_removes_table_sidecars_but_keeps_reusable_profiles(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.parquet_store.DATAFRAME_DIR", temp_dir
        ):
            df = self._sample()
            save_dataframe(df, "df_donations", "donations.xlsx", file_hash="abc123")
            profile_dir = os.path.join(temp_dir, "_schema_profiles")
            self.assertTrue(os.listdir(profile_dir))

            drop_dataframe_files("df_donations")
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "df_donations.parquet")))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "df_donations.meta.json")))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "df_donations.schema.json")))
            self.assertTrue(os.listdir(profile_dir))


if __name__ == "__main__":
    unittest.main()
