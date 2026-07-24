import unittest

import pandas as pd

from rag.deterministic_query_plan import build_schema_grounded_plan
from rag.question_suggestions import (
    build_date_autocomplete_catalog,
    build_person_autocomplete_catalog,
    build_person_prefix_matches,
    build_question_suggestions,
)
from utils.table_parser import _clean_dataframe


def _payment_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "성명": ["김민수", "이서연"],
        "납부금액": [10000, 20000],
        "기수": [49, 50],
        "전화번호": ["010-0000-0000", "010-1111-1111"],
        "이메일": ["minsu@example.com", "seoyeon@example.com"],
        "학과": ["경제학과", "경영학과"],
    })


def _document_dataframe(rows: dict[str, list[object]]) -> pd.DataFrame:
    dataframe = _clean_dataframe(
        pd.DataFrame(rows),
        source_file="test.xlsx",
        context_prefix="sheet0",
    )
    assert dataframe is not None
    return dataframe


class QuestionSuggestionsTest(unittest.TestCase):
    def test_suggestions_only_include_dp_compilable_questions(self):
        dataframes = {"payments": _payment_dataframe()}

        suggestions = build_question_suggestions("가장", dataframes=dataframes)

        self.assertTrue(suggestions)
        self.assertTrue(all(
            build_schema_grounded_plan(
                item["text"],
                dataframes=dataframes,
                operation_hint={
                    "average_amount": "structured_query",
                    "max_person_by_amount": "structured_query",
                    "min_person_by_amount": "structured_query",
                }.get(item["operation"], item["operation"]),
            )
            is not None
            for item in suggestions
            if item["path"] == "fast"
        ))
        self.assertEqual(suggestions[0]["text"], "가장 큰 금액 뭐야?")

    def test_suggestions_do_not_guess_with_multiple_dataframes(self):
        dataframe = _payment_dataframe()

        suggestions = build_question_suggestions(
            "금액",
            dataframes={"first": dataframe, "second": dataframe.copy()},
        )

        self.assertFalse(any(item["path"] == "fast" for item in suggestions))

    def test_person_suggestion_only_echoes_a_grounded_full_input(self):
        dataframes = {"payments": _payment_dataframe()}

        exact = build_question_suggestions("김민수", dataframes=dataframes)
        partial = build_question_suggestions("김", dataframes=dataframes)

        self.assertTrue(any(item["text"] == "김민수 금액 알려줘" for item in exact))
        self.assertFalse(any("김민수" in item["text"] for item in partial))

    def test_person_suggestion_recognizes_particles_and_person_inside_phrase(self):
        dataframes = {"payments": _payment_dataframe()}

        with_particle = build_question_suggestions("김민수는", dataframes=dataframes)
        reversed_order = build_question_suggestions("금액 김민수", dataframes=dataframes)

        self.assertTrue(any(item["text"] == "김민수 금액 알려줘" for item in with_particle))
        self.assertTrue(any(item["text"] == "김민수 금액 알려줘" for item in reversed_order))

    def test_person_autocomplete_catalog_has_only_grounded_names_and_actions(self):
        dataframes = {"payments": _payment_dataframe()}

        catalog = build_person_autocomplete_catalog(dataframes)

        self.assertEqual(catalog["names"], ["김민수", "이서연"])
        self.assertTrue(catalog["actions"])
        self.assertTrue(all(action["suffix"] for action in catalog["actions"]))
        self.assertTrue(all(action["path"] == "fast" for action in catalog["actions"]))

    def test_person_prefix_matches_are_limited_and_mask_aware(self):
        dataframes = {"payments": _payment_dataframe()}

        self.assertEqual(build_person_prefix_matches("김민", dataframes=dataframes), ["김민수"])
        self.assertEqual(build_person_prefix_matches("김", dataframes=dataframes), [])

        masked = {"payments": _payment_dataframe().assign(성명=["추*진", "김민수"])}
        self.assertEqual(build_person_prefix_matches("추교", dataframes=masked), ["추*진"])

    def test_large_person_catalog_switches_to_prefix_mode(self):
        names = [f"김{index:03d}" for index in range(501)]
        dataframes = {"payments": pd.DataFrame({"성명": names, "납부금액": [10000] * len(names)})}

        catalog = build_person_autocomplete_catalog(dataframes)

        self.assertEqual(catalog["mode"], "remote")
        self.assertEqual(catalog["names"], [])
        self.assertEqual(catalog["total"], 501)

    def test_catalog_can_represent_every_eo_operation(self):
        dataframes = {"payments": _payment_dataframe()}
        suggestions = (
            build_question_suggestions("", dataframes=dataframes, limit=100)
            + build_question_suggestions("김민수", dataframes=dataframes, limit=100)
            + build_question_suggestions("49기", dataframes=dataframes, limit=100)
        )
        operations = {item["operation"] for item in suggestions}

        self.assertEqual(operations, {
            "list_documents",
            "filter_records",
            "compare",
            "max_person_by_amount",
            "min_person_by_amount",
            "list_records",
            "count_records",
            "sum_amount",
            "average_amount",
            "median_amount",
            "mode_amount",
            "max_amount",
            "min_amount",
            "lookup_amount",
            "lookup_field",
            "structured_query",
            "document_reason",
            "document_purpose",
            "document_criteria",
            "document_procedure",
            "document_explain",
        })

    def test_date_catalog_supports_complete_date_column(self):
        dataframes = {"payments": _document_dataframe({
            "결제일자": ["2025-01-01", "2026-01-01"],
            "성명": ["김민수", "이서연"],
            "납부금액": [10000, 20000],
        })}

        catalog = build_date_autocomplete_catalog(dataframes)

        self.assertEqual(
            [action["operation"] for action in catalog["actions"]],
            ["filter_records", "sum_amount", "count_records"],
        )

    def test_date_catalog_supports_separate_year_month_columns(self):
        dataframes = {"payments": _document_dataframe({
            "연도": [2025, 2026],
            "지급월": [12, 1],
            "성명": ["김민수", "이서연"],
            "납부금액": [10000, 20000],
        })}

        catalog = build_date_autocomplete_catalog(dataframes)

        self.assertTrue(catalog["actions"])

    def test_date_catalog_omits_amount_action_without_amount_column(self):
        dataframes = {"payments": _document_dataframe({
            "처리날짜": ["2025-12-01", "2026-01-01"],
            "성명": ["김민수", "이서연"],
        })}

        catalog = build_date_autocomplete_catalog(dataframes)
        operations = [action["operation"] for action in catalog["actions"]]

        self.assertEqual(operations, ["filter_records", "count_records"])

    def test_date_catalog_rejects_month_only_and_missing_date_schemas(self):
        month_only = {"payments": _document_dataframe({
            "지급월": [1, 2],
            "성명": ["김민수", "이서연"],
            "납부금액": [10000, 20000],
        })}
        no_date = {"payments": _payment_dataframe()}

        self.assertEqual(build_date_autocomplete_catalog(month_only), {"actions": []})
        self.assertEqual(build_date_autocomplete_catalog(no_date), {"actions": []})

    def test_date_catalog_rejects_ambiguous_complete_date_columns(self):
        dataframes = {"payments": _document_dataframe({
            "신청일자": ["2025-01-01"],
            "지급일자": ["2025-02-01"],
            "성명": ["김민수"],
            "납부금액": [10000],
        })}

        catalog = build_date_autocomplete_catalog(dataframes)
        self.assertTrue(catalog["actions"])
        self.assertEqual(
            {action["lead"] for action in catalog["actions"]},
            {"신청일자 기준", "지급일자 기준"},
        )

    def test_multi_document_suggestions_require_shared_semantic_columns(self):
        first = _payment_dataframe()
        second = _payment_dataframe().rename(columns={"성명": "이름", "납부금액": "결제금액"})

        suggestions = build_question_suggestions("", dataframes={"first": first, "second": second}, limit=100)
        texts = {item["text"] for item in suggestions}

        self.assertIn("선택한 문서 전체 인원 몇 명이야?", texts)
        self.assertIn("선택한 문서 전체 금액 알려줘", texts)


if __name__ == "__main__":
    unittest.main()
