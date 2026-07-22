import unittest

import pandas as pd

from pandas_engine.interactive import build_interactive_result, get_interactive_detail
from pandas_engine.plan_validator import validate_query_plan
from pandas_engine.query_executor import execute_query_plan
from pandas_engine.query_plan import QueryPlan
from utils.semantic_schema import attach_semantic_schema


class InteractiveResultTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "회원명": ["홍길동", "홍길동"],
            "전공": ["컴퓨터공학", "컴퓨터공학"],
            "전화번호": ["010-1111-2222", "010-1111-2222"],
            "결제 금액": [10000, 20000],
            "source": ["test.xlsx", "test.xlsx"],
            "표시명": ["홍길동", "홍길동"],
            "엔티티 타입": ["person", "person"],
            "person_candidate_key": ["internal-1", "internal-2"],
        })
        self.df.attrs["semantic_schema"] = attach_semantic_schema(
            self.df, var_name="df0", source_file="test.xlsx", dataframe_dir=".",
        )
        self.mapping = {"df0": self.df}

    def _execute(self, operation, target=None):
        plan = QueryPlan(status="ready", dataframe="df0", operation=operation, target=target)
        return execute_query_plan(validate_query_plan(plan, dataframes=self.mapping, source_by_alias={"df0": "test.xlsx"}))

    def test_entity_keeps_row_scoped_identity_and_hides_contact_until_detail(self):
        result = build_interactive_result(self._execute("list"))
        self.assertEqual(len(result["entities"]), 2)
        self.assertNotEqual(result["entities"][0]["entity_id"], result["entities"][1]["entity_id"])
        self.assertNotIn("전화번호", [a["column"] for a in result["entities"][0]["attributes"]])
        detail = get_interactive_detail(result["entities"][0]["detail_ref"])
        self.assertIn("전화번호", [a["column"] for a in detail["attributes"]])
        self.assertFalse({"source", "표시명", "엔티티 타입", "person_candidate_key"} & {a["column"] for a in detail["attributes"]})

    def test_mean_has_formula_and_bounded_contributors(self):
        result = build_interactive_result(self._execute("mean", "결제 금액"), page_size=1)
        calculation = result["calculation"]
        self.assertEqual(calculation["formula"]["numerator"], 30000)
        self.assertEqual(calculation["formula"]["denominator"], 2)
        detail = get_interactive_detail(calculation["detail_ref"], limit=1)
        self.assertEqual(detail["page"]["total"], 2)
        self.assertEqual(len(detail["contributors"]), 1)
        self.assertFalse({"source", "표시명", "엔티티 타입", "person_candidate_key", "전화번호"} & set(detail["contributors"][0]))

    def test_inline_segments_use_structured_references_and_hide_text_evidence(self):
        result = build_interactive_result(
            self._execute("mean", "결제 금액"),
            answer="결제 금액 평균은 15,000원입니다.\n\n계산 근거:\n- 내부 근거",
        )
        rendered = "".join(segment["text"] for segment in result["inline_segments"])
        self.assertEqual(rendered, "결제 금액 평균은 15,000원입니다.")
        calculation_segments = [s for s in result["inline_segments"] if s.get("kind") == "calculation"]
        self.assertEqual(len(calculation_segments), 1)
        self.assertEqual(calculation_segments[0]["detail_ref"], result["calculation"]["detail_ref"])

    def test_inline_entity_links_are_not_limited_to_transport_page(self):
        df = pd.DataFrame({"회원명": [f"회원{i:02d}" for i in range(55)], "결제 금액": range(55)})
        df.attrs["semantic_schema"] = attach_semantic_schema(
            df, var_name="df_many", source_file="many.xlsx", dataframe_dir=".",
        )
        plan = QueryPlan(status="ready", dataframe="df_many", operation="list")
        execution = execute_query_plan(validate_query_plan(plan, dataframes={"df_many": df}, source_by_alias={"df_many": "many.xlsx"}))
        answer = "\n".join(df["회원명"].tolist())
        result = build_interactive_result(execution, page_size=50, answer=answer)
        linked_names = [segment["text"] for segment in result["inline_segments"] if segment.get("kind") == "entity"]
        self.assertEqual(len(result["entities"]), 50)
        self.assertEqual(len(linked_names), 55)

    def test_grouped_result_links_name_to_original_person_columns(self):
        df = self.df.copy()
        df["person_candidate_key"] = ["same-person", "same-person"]
        df.attrs["semantic_schema"] = attach_semantic_schema(
            df, var_name="df_group", source_file="group.xlsx", dataframe_dir=".",
        )
        plan = QueryPlan(
            status="ready", dataframe="df_group", operation="group_sum",
            target="결제 금액", group_by=["회원명"], group_order="desc", top_n=1,
        )
        execution = execute_query_plan(validate_query_plan(plan, dataframes={"df_group": df}, source_by_alias={"df_group": "group.xlsx"}))
        result = build_interactive_result(execution, answer="홍길동 30,000")
        name_segments = [segment for segment in result["inline_segments"] if segment.get("kind") == "entity"]
        self.assertEqual(len(name_segments), 1)
        detail = get_interactive_detail(name_segments[0]["detail_ref"])
        self.assertIn("전공", [attribute["column"] for attribute in detail["attributes"]])
        self.assertEqual(len(detail["payment_history"]), 2)
        amounts = [
            field["value"]
            for history in detail["payment_history"]
            for field in history["fields"]
            if field["data_type"] == "money"
        ]
        self.assertEqual(amounts, [10000, 20000])

    def test_cards_exclude_ocr_and_derived_processing_columns(self):
        df = self.df.copy()
        df["_ocr_confidence_min"] = [0.81, 0.92]
        df["_ocr_low_confidence_cells"] = ["회원명", ""]
        df["ocrconfidence"] = [0.81, 0.92]
        df.attrs["semantic_schema"] = attach_semantic_schema(
            df, var_name="df_ocr", source_file="ocr.xlsx", dataframe_dir=".",
        )
        execution = execute_query_plan(
            validate_query_plan(
                QueryPlan(status="ready", dataframe="df_ocr", operation="list"),
                dataframes={"df_ocr": df}, source_by_alias={"df_ocr": "ocr.xlsx"},
            )
        )
        result = build_interactive_result(execution)
        detail = get_interactive_detail(result["entities"][0]["detail_ref"])
        card_columns = {attribute["column"] for attribute in detail["attributes"]}
        self.assertFalse({"_ocr_confidence_min", "_ocr_low_confidence_cells", "ocrconfidence"} & card_columns)
        self.assertFalse({"_ocr_confidence_min", "_ocr_low_confidence_cells", "ocrconfidence"} & set(result["records"][0]))
