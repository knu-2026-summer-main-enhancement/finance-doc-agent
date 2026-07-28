from __future__ import annotations

import unittest
from unittest.mock import patch

from utils.document_structure import document_section, document_structure


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


if __name__ == "__main__":
    unittest.main()
