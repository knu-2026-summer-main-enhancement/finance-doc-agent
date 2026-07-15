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
from datastore.schema import _build_schema_for_vars


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
            self.assertEqual(schema["columns"]["출연금액"]["role"], "amount")
            self.assertEqual(schema["columns"]["출연금액"]["qualifier"], "donation")
            self.assertEqual(schema["columns"]["이름"]["role"], "entity_name")
            self.assertEqual(schema["columns"]["비고"]["role"], "description")
            self.assertEqual(schema["columns"]["비고"]["qualifier"], "note")
            self.assertNotIn("비고", schema["unmapped_columns"])

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

    def test_compound_headers_do_not_expand_context_words_into_amounts(self):
        raw = pd.DataFrame([
            {
                "후원일": "2026-02-02",
                "후원금": "3,000,000",
                "지급액": "300,000",
                "지급월": "6월",
                "지급기관": "교내 장학위원회",
                "지급목적": "성취 동기 강화",
                "지원분야_비고": "기계설계 / 일시납",
            },
            {
                "후원일": "2026-02-10",
                "후원금": "200,000",
                "지급액": "250,000",
                "지급월": "7월",
                "지급기관": "지역 장학재단",
                "지급목적": "생활비 지원",
                "지원분야_비고": "스마트전기 / 분할납부",
            },
        ])
        cleaned = _clean_dataframe(raw, source_file="ledger.xlsx", context_prefix="s0")
        self.assertIsNotNone(cleaned)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.parquet_store.DATAFRAME_DIR", temp_dir
        ):
            save_dataframe(cleaned, "df_ledger", "ledger.xlsx", file_hash="compound")
            with open(os.path.join(temp_dir, "df_ledger.schema.json"), encoding="utf-8") as file:
                mappings = json.load(file)["columns"]

        self.assertEqual(mappings["후원일"]["role"], "date")
        self.assertEqual(mappings["후원금"]["role"], "amount")
        self.assertEqual(mappings["후원금"]["qualifier"], "sponsorship")
        self.assertEqual(mappings["지급액"]["role"], "amount")
        self.assertEqual(mappings["지급액"]["qualifier"], "payment")
        self.assertEqual(mappings["지급월"]["role"], "period")
        self.assertEqual(mappings["지급기관"]["role"], "entity_name")
        self.assertEqual(mappings["지급기관"]["qualifier"], "organization")
        self.assertEqual(mappings["지급목적"]["role"], "description")
        self.assertEqual(mappings["지급목적"]["qualifier"], "purpose")
        self.assertEqual(mappings["지원분야_비고"]["role"], "category")
        self.assertEqual(mappings["지원분야_비고"]["qualifier"], "field")
        self.assertNotEqual(mappings["지급월"]["concept"], "measure")
        self.assertNotEqual(mappings["지급기관"]["concept"], "measure")
        self.assertNotEqual(mappings["지급목적"]["concept"], "measure")

    def test_compound_identity_and_identifier_headers_are_generalized(self):
        raw = pd.DataFrame([
            {
                "접수코드": "RC26-001",
                "소속회차": "59",
                "기부자명": "안*온",
                "납부방식": "계좌이체",
            },
            {
                "접수코드": "RC26-002",
                "소속회차": "42",
                "기부자명": "노*석",
                "납부방식": "정기이체",
            },
        ])
        cleaned = _clean_dataframe(raw, source_file="sponsors.pdf", context_prefix="p0")
        self.assertIsNotNone(cleaned)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.parquet_store.DATAFRAME_DIR", temp_dir
        ):
            save_dataframe(cleaned, "df_sponsors", "sponsors.pdf", file_hash="identity")
            with open(os.path.join(temp_dir, "df_sponsors.schema.json"), encoding="utf-8") as file:
                mappings = json.load(file)["columns"]

        self.assertEqual(mappings["접수코드"]["role"], "identifier_value")
        self.assertEqual(mappings["소속회차"]["role"], "category")
        self.assertEqual(mappings["소속회차"]["qualifier"], "cohort")
        self.assertEqual(mappings["기부자명"]["role"], "entity_name")
        self.assertEqual(mappings["납부방식"]["role"], "category")
        self.assertEqual(mappings["납부방식"]["qualifier"], "method")

    def test_amount_header_with_text_values_is_not_forced_to_money(self):
        raw = pd.DataFrame({"지원금액": ["미정", "협의 후 결정"]})
        cleaned = _clean_dataframe(raw, source_file="unknown.xlsx", context_prefix="s0")
        self.assertIsNotNone(cleaned)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.parquet_store.DATAFRAME_DIR", temp_dir
        ):
            save_dataframe(cleaned, "df_unknown", "unknown.xlsx", file_hash="text-money")
            with open(os.path.join(temp_dir, "df_unknown.schema.json"), encoding="utf-8") as file:
                mapping = json.load(file)["columns"]["지원금액"]

        self.assertNotEqual(mapping["concept"], "measure")
        self.assertIsNone(mapping["role"])

    def test_unknown_columns_stay_unknown_and_phone_is_tagged_by_value_shape(self):
        raw = pd.DataFrame([
            {
                "학위과정": "석사",
                "취득학점": "42",
                "연락정보": "010-7000-1001",
                "졸업정보": "2026.02",
                "큰금액": "10000000",
            },
            {
                "학위과정": "통합",
                "취득학점": "36",
                "연락정보": "010-7000-1002",
                "졸업정보": "2025.08",
                "큰금액": "20000000",
            },
        ])
        cleaned = _clean_dataframe(raw, source_file="new-layout.xlsx", context_prefix="s0")
        self.assertIsNotNone(cleaned)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.parquet_store.DATAFRAME_DIR", temp_dir
        ):
            save_dataframe(cleaned, "df_new_layout", "new-layout.xlsx", file_hash="unknown")
            with open(os.path.join(temp_dir, "df_new_layout.schema.json"), encoding="utf-8") as file:
                mappings = json.load(file)["columns"]

        self.assertIsNone(mappings["학위과정"]["role"])
        self.assertIsNone(mappings["취득학점"]["role"])
        self.assertEqual(mappings["연락정보"]["concept"], "identifier")
        self.assertEqual(mappings["연락정보"]["sensitivity"], "personal")
        self.assertEqual(mappings["연락정보"]["pii_type"], "phone_number")
        self.assertEqual(mappings["졸업정보"]["concept"], "temporal")
        self.assertEqual(mappings["졸업정보"]["role"], "period")
        self.assertEqual(mappings["졸업정보"]["data_type"], "year_month")
        self.assertEqual(mappings["큰금액"]["concept"], "measure")
        self.assertEqual(mappings["큰금액"]["role"], "amount")
        self.assertEqual(mappings["큰금액"]["sensitivity"], "none")

    def test_sensitive_sample_value_is_hidden_from_llm_schema_description(self):
        raw = pd.DataFrame([
            {"이름": "홍길동", "연락정보": "010-7000-1001", "지급액": "300,000"},
        ])
        cleaned = _clean_dataframe(raw, source_file="contacts.xlsx", context_prefix="s0")
        self.assertIsNotNone(cleaned)

        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "utils.parquet_store.DATAFRAME_DIR", temp_dir
        ), patch("datastore.state.DATAFRAME_DIR", temp_dir):
            save_dataframe(cleaned, "df_contacts", "contacts.xlsx", file_hash="contacts")
            try:
                dataframe_state._load_dataframes()
                schema_text = _build_schema_for_vars({"df0"})
                self.assertNotIn("010-7000-1001", schema_text)
                self.assertIn("[민감정보]", schema_text)
                self.assertIn("sensitivity:personal", schema_text)
            finally:
                dataframe_state._df_namespace.clear()
                dataframe_state._df_sources.clear()
                dataframe_state._df_labels.clear()
                dataframe_state._df_schemas.clear()


if __name__ == "__main__":
    unittest.main()
