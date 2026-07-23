import unittest

import pandas as pd

from rag.deterministic_query_plan import build_schema_grounded_plan
from rag.question_suggestions import build_person_autocomplete_catalog, build_question_suggestions


def _payment_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "성명": ["김민수", "이서연"],
        "납부금액": [10000, 20000],
        "기수": [49, 50],
        "전화번호": ["010-0000-0000", "010-1111-1111"],
        "이메일": ["minsu@example.com", "seoyeon@example.com"],
        "학과": ["경제학과", "경영학과"],
    })


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

    def test_person_autocomplete_catalog_has_only_grounded_names_and_actions(self):
        dataframes = {"payments": _payment_dataframe()}

        catalog = build_person_autocomplete_catalog(dataframes)

        self.assertEqual(catalog["names"], ["김민수", "이서연"])
        self.assertTrue(catalog["actions"])
        self.assertTrue(all(action["suffix"] for action in catalog["actions"]))
        self.assertTrue(all(action["path"] == "fast" for action in catalog["actions"]))

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


if __name__ == "__main__":
    unittest.main()
