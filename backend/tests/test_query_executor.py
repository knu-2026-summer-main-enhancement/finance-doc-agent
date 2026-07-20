from __future__ import annotations

import unittest

import pandas as pd

from pandas_engine.plan_validator import validate_query_plan
from pandas_engine.query_executor import (
    QueryPlanExecutionError,
    execute_query_plan,
)
from pandas_engine.query_plan import QueryPlan


class QueryExecutorTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "이름": ["김철수", "이영희", "추＊철", "김철수"],
                "구분": ["개인", "개인", "개인", "단체"],
                "출연금액": ["1,000,000", "500,000", "2,000,000", "잘못된값"],
                "출연일자": [
                    "2025-03-01",
                    "2025-04-15",
                    "2025-03-20",
                    "날짜오류",
                ],
                "기수": [59, 58, 59, 60],
                "비고": [None, "재학생", "졸업생", "법인"],
                "__row_id": ["r1", "r2", "r3", "r4"],
            }
        )
        self.df.attrs["semantic_schema"] = {
            "columns": {
                "이름": {
                    "concept": "entity",
                    "role": "entity_name",
                    "qualifier": "person",
                    "data_type": "string",
                },
                "구분": {"data_type": "string"},
                "출연금액": {"data_type": "money", "unit": "KRW"},
                "출연일자": {"data_type": "date"},
                "기수": {"data_type": "number"},
                "비고": {"data_type": "string"},
            }
        }
        self.dataframes = {"df0": self.df}
        self.sources = {"df0": "후원대장.xlsx"}

    def _execute(self, payload: dict):
        base = {
            "status": "ready",
            "dataframe": "df0",
        }
        base.update(payload)
        plan = QueryPlan.model_validate(base)
        validation = validate_query_plan(
            plan,
            dataframes=self.dataframes,
            source_by_alias=self.sources,
        )
        self.assertTrue(validation.is_executable, validation.issues)
        return execute_query_plan(validation)

    def test_list_applies_filters_sort_selection_and_limit(self):
        result = self._execute(
            {
                "operation": "list",
                "filters": [{"column": "기수", "operator": "gte", "value": 59}],
                "select": ["이름", "기수"],
                "sort": [{"column": "기수", "direction": "desc"}],
                "limit": 2,
            }
        )

        self.assertEqual(result.matched_rows, 3)
        self.assertEqual(result.value.columns.tolist(), ["이름", "기수"])
        self.assertEqual(result.value["기수"].tolist(), [60, 59])
        self.assertEqual(result.evidence.dataframe_alias, "df0")
        self.assertEqual(result.evidence.source_file, "후원대장.xlsx")
        self.assertEqual(result.evidence.source_rows, 4)
        self.assertEqual(result.evidence.filtered_rows, 3)
        self.assertEqual(result.evidence.limit, 2)
        self.assertEqual(result.evidence.filters[0].column, "기수")

    def test_any_filter_logic_uses_or(self):
        result = self._execute(
            {
                "operation": "list",
                "filters": [
                    {"column": "기수", "operator": "eq", "value": 58},
                    {"column": "구분", "operator": "eq", "value": "단체"},
                ],
                "filter_logic": "any",
                "select": ["이름"],
            }
        )

        self.assertEqual(result.matched_rows, 2)
        self.assertEqual(result.value["이름"].tolist(), ["이영희", "김철수"])

    def test_mask_characters_are_normalized_for_exact_person_filter(self):
        result = self._execute(
            {
                "operation": "list",
                "filters": [
                    {"column": "이름", "operator": "eq", "value": "추*철"}
                ],
                "select": ["이름", "출연금액"],
            }
        )

        self.assertEqual(result.matched_rows, 1)
        self.assertEqual(result.value.iloc[0]["이름"], "추＊철")

    def test_distinct_count_is_deterministic(self):
        result = self._execute(
            {
                "operation": "count",
                "distinct_by": ["이름"],
            }
        )

        self.assertEqual(result.value, 3)
        self.assertEqual(result.matched_rows, 3)

    def test_distinct_person_names_use_normalized_mask_characters(self):
        duplicated = pd.concat([self.df, self.df.iloc[[2]]], ignore_index=True)
        duplicated.loc[len(duplicated) - 1, "이름"] = "추*철"
        duplicated.attrs.update(self.df.attrs)
        self.dataframes["df0"] = duplicated

        result = self._execute(
            {
                "operation": "count",
                "distinct_by": ["이름"],
            }
        )

        self.assertEqual(result.value, 3)

    def test_money_statistics_exclude_invalid_values(self):
        expected = {
            "sum": 3_500_000.0,
            "mean": 3_500_000.0 / 3,
            "median": 1_000_000.0,
        }
        for operation, value in expected.items():
            with self.subTest(operation=operation):
                result = self._execute(
                    {
                        "operation": operation,
                        "target": "출연금액",
                    }
                )
                self.assertAlmostEqual(result.value, value)
                self.assertEqual(result.valid_rows, 3)
                self.assertEqual(result.excluded_rows, 1)

    def test_mode_returns_all_tied_values(self):
        result = self._execute(
            {
                "operation": "mode",
                "target": "기수",
            }
        )

        self.assertEqual(result.value, [59])

    def test_min_value_and_top_records(self):
        minimum = self._execute(
            {
                "operation": "min",
                "target": "출연금액",
                "result_mode": "value",
            }
        )
        top_two = self._execute(
            {
                "operation": "max",
                "target": "출연금액",
                "result_mode": "records",
                "select": ["이름", "출연금액"],
                "top_n": 2,
            }
        )

        self.assertEqual(minimum.value, 500_000.0)
        self.assertEqual(top_two.value["이름"].tolist(), ["추＊철", "김철수"])

    def test_top_one_returns_all_rows_tied_at_the_extreme(self):
        tied = pd.concat([self.df, self.df.iloc[[2]]], ignore_index=True)
        tied.loc[len(tied) - 1, "이름"] = "박민수"
        tied.attrs.update(self.df.attrs)
        self.dataframes["df0"] = tied

        result = self._execute(
            {
                "operation": "max",
                "target": "출연금액",
                "result_mode": "records",
                "select": ["이름", "출연금액"],
            }
        )

        self.assertEqual(result.evidence.top_n, 1)
        self.assertEqual(result.value["이름"].tolist(), ["추＊철", "박민수"])

    def test_top_n_over_one_returns_exactly_requested_row_count(self):
        tied = pd.concat(
            [self.df, self.df.iloc[[1]], self.df.iloc[[1]]],
            ignore_index=True,
        )
        tied.loc[3, "이름"] = "동률A"
        tied.loc[4, "이름"] = "동률B"
        tied.attrs.update(self.df.attrs)
        self.dataframes["df0"] = tied

        result = self._execute(
            {
                "operation": "max",
                "target": "출연금액",
                "result_mode": "records",
                "select": ["이름", "출연금액"],
                "top_n": 2,
            }
        )

        self.assertEqual(len(result.value), 2)
        self.assertEqual(result.value["이름"].tolist(), ["추＊철", "김철수"])

    def test_date_between_filter_uses_dates_not_strings(self):
        result = self._execute(
            {
                "operation": "list",
                "filters": [
                    {
                        "column": "출연일자",
                        "operator": "between",
                        "value": ["2025-03-01", "2025-03-31"],
                    }
                ],
                "select": ["이름", "출연일자"],
            }
        )

        self.assertEqual(result.value["이름"].tolist(), ["김철수", "추＊철"])

    def test_null_filter_and_default_selection_hide_internal_columns(self):
        result = self._execute(
            {
                "operation": "list",
                "filters": [{"column": "비고", "operator": "is_null"}],
            }
        )

        self.assertEqual(result.matched_rows, 1)
        self.assertNotIn("__row_id", result.value.columns)

    def test_invalid_plan_cannot_be_executed(self):
        plan = QueryPlan.model_validate(
            {
                "status": "ready",
                "dataframe": "df0",
                "operation": "list",
                "select": ["없는컬럼"],
            }
        )
        validation = validate_query_plan(
            plan,
            dataframes=self.dataframes,
            source_by_alias=self.sources,
        )

        with self.assertRaises(QueryPlanExecutionError):
            execute_query_plan(validation)


if __name__ == "__main__":
    unittest.main()
