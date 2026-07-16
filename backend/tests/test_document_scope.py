from __future__ import annotations

import unittest

import pandas as pd

from datastore.query import _search_name_pandas
from datastore.scope import document_scope, selected_sources, source_is_selected
from datastore.state import _df_labels, _df_namespace, _df_sources
from pandas_engine.executor import _exec_pandas_code
from rag.vector import _selected_source_filter
from utils.table_parser import _clean_dataframe


class DocumentScopeTests(unittest.TestCase):
    def setUp(self):
        _df_namespace.clear()
        _df_sources.clear()
        _df_labels.clear()
        for alias, source, amount in (
            ("df0", "후원대장_이전.xlsx", "100,000"),
            ("df1", "후원대장_현재.xlsx", "200,000"),
        ):
            raw = pd.DataFrame([{"이름": "김철수", "출연금액": amount}])
            df = _clean_dataframe(raw, source_file=source, context_prefix="table0")
            _df_namespace[alias] = df
            _df_sources[alias] = source
            _df_labels[alias] = source

    def tearDown(self):
        _df_namespace.clear()
        _df_sources.clear()
        _df_labels.clear()

    def test_scope_normalizes_paths_and_resets_after_request(self):
        self.assertEqual(selected_sources(), ())
        with document_scope([r"C:\upload\후원대장_현재.xlsx"]):
            self.assertEqual(selected_sources(), ("후원대장_현재.xlsx",))
            self.assertTrue(source_is_selected("후원대장_현재.xlsx"))
            self.assertFalse(source_is_selected("후원대장_이전.xlsx"))
        self.assertEqual(selected_sources(), ())

    def test_name_search_uses_only_selected_document(self):
        with document_scope(["후원대장_현재.xlsx"]):
            rows, sources, searched = _search_name_pandas("김철수 출연금액 알려줘")
        self.assertTrue(searched)
        self.assertIsNotNone(rows)
        self.assertEqual(sources, ["후원대장_현재.xlsx"])
        self.assertEqual(rows["출연금액"].tolist(), ["200,000"])

    def test_unscoped_search_does_not_silently_choose_latest_document(self):
        rows, sources, searched = _search_name_pandas("김철수 출연금액 알려줘")
        self.assertTrue(searched)
        self.assertIsNotNone(rows)
        self.assertEqual(set(sources), {"후원대장_이전.xlsx", "후원대장_현재.xlsx"})
        self.assertEqual(len(rows), 2)

    def test_generated_code_cannot_access_dataframe_outside_scope(self):
        with document_scope(["후원대장_현재.xlsx"]):
            self.assertEqual(_exec_pandas_code("result = len(df1)"), 1)
            with self.assertRaises(NameError):
                _exec_pandas_code("result = len(df0)")

    def test_vector_filter_matches_selected_sources(self):
        with document_scope(["후원대장_이전.xlsx", "후원대장_현재.xlsx"]):
            self.assertEqual(
                _selected_source_filter(),
                {"source": {"$in": ["후원대장_이전.xlsx", "후원대장_현재.xlsx"]}},
            )
        self.assertIsNone(_selected_source_filter())


if __name__ == "__main__":
    unittest.main()
