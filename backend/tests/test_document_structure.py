from __future__ import annotations

import unittest
from unittest.mock import patch

from utils.document_structure import (
    document_section,
    document_structure,
    suggested_section_titles,
)


def _section_record(
    text: str,
    *,
    section_id: str,
    title: str,
    page: int,
    section_index: int,
    child_index: int,
) -> dict:
    return {
        "text": text,
        "metadata": {
            "file_type": "pdf",
            "content_type": "pdf_section_child",
            "section_id": section_id,
            "section_title": title,
            "section_index": section_index,
            "child_index": child_index,
            "page": page,
        },
    }


class DocumentStructureTests(unittest.TestCase):
    def test_sectioned_pdf_gets_capabilities_from_stored_chunks(self):
        records = [
            _section_record(
                "[문서: plan]\nPDF 문서 섹션:\nⅠ 추진배경\n첫 내용",
                section_id="p1:s0",
                title="Ⅰ 추진배경",
                page=1,
                section_index=0,
                child_index=0,
            ),
            _section_record(
                "[문서: plan]\nPDF 문서 섹션:\nⅣ 향후 추진일정\n일정 내용",
                section_id="p6:s0",
                title="Ⅳ 향후 추진일정",
                page=6,
                section_index=0,
                child_index=0,
            ),
        ]
        with patch(
            "utils.document_structure.get_chroma_source_records",
            return_value=records,
        ):
            metadata = document_structure("plan.pdf")

        self.assertEqual(metadata["document_type"], "sectioned_document")
        self.assertIn("list_sections", metadata["capabilities"])
        self.assertEqual(metadata["statistics"]["section_count"], 2)
        self.assertEqual(metadata["statistics"]["page_count"], 6)
        self.assertEqual(
            [section["title"] for section in metadata["sections"]],
            ["Ⅰ 추진배경", "Ⅳ 향후 추진일정"],
        )

    def test_table_file_does_not_get_section_capability(self):
        records = [{
            "text": "이름: 홍길동 / 금액: 10000",
            "metadata": {"file_type": "xlsx", "table_id": "sheet1"},
        }]
        with patch(
            "utils.document_structure.get_chroma_source_records",
            return_value=records,
        ):
            metadata = document_structure("ledger.xlsx")

        self.assertEqual(metadata["document_type"], "tabular_document")
        self.assertNotIn("list_sections", metadata["capabilities"])
        self.assertIn("structured_query", metadata["capabilities"])

    def test_hwp_sections_and_tables_are_reported_as_mixed_document(self):
        records = [
            {
                "text": "□ 지원대상\n학부 재학생",
                "metadata": {
                    "file_type": "hwp",
                    "content_type": "hwp_section_child",
                    "section_id": "s0",
                    "section_title": "□ 지원대상",
                    "section_index": 0,
                    "child_index": 0,
                },
            },
            {
                "text": "구분: 학부 / 금액: 100만원",
                "metadata": {
                    "file_type": "hwp",
                    "content_type": "hwp_table_row",
                    "table_id": "t0",
                },
            },
        ]
        with patch(
            "utils.document_structure.get_chroma_source_records",
            return_value=records,
        ):
            metadata = document_structure("운영계획.hwp", file_type="hwp")

        self.assertEqual(metadata["document_type"], "mixed_document")
        self.assertTrue(metadata["features"]["has_sections"])
        self.assertTrue(metadata["features"]["has_tables"])
        self.assertTrue(metadata["features"]["has_text_content"])
        self.assertEqual(metadata["statistics"]["section_count"], 1)

    def test_section_detail_joins_children_in_child_order(self):
        records = [
            _section_record(
                "[문서: plan]\nPDF 문서 섹션:\nⅠ 추진배경\n두 번째",
                section_id="p1:s0",
                title="Ⅰ 추진배경",
                page=2,
                section_index=0,
                child_index=1,
            ),
            _section_record(
                "[문서: plan]\nPDF 문서 섹션:\nⅠ 추진배경\n첫 번째",
                section_id="p1:s0",
                title="Ⅰ 추진배경",
                page=1,
                section_index=0,
                child_index=0,
            ),
        ]
        with patch(
            "utils.document_structure.get_chroma_source_records",
            return_value=records,
        ):
            detail = document_section("plan.pdf", "p1:s0")

        self.assertIsNotNone(detail)
        self.assertEqual(detail["pages"], [1, 2])
        self.assertEqual(detail["content"], "첫 번째\n\n두 번째")

    def test_section_shortcuts_use_stored_titles_with_generic_preference(self):
        structure = {
            "capabilities": ["semantic_question", "list_sections"],
            "sections": [
                {"title": "추진 배경"},
                {"title": "향후 추진 일정"},
                {"title": "선발 기준"},
                {"title": "선발 대상"},
            ],
        }

        self.assertEqual(
            suggested_section_titles(structure),
            ["선발 대상", "선발 기준"],
        )


if __name__ == "__main__":
    unittest.main()
