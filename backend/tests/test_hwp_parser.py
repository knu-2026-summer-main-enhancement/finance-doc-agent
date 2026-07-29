from __future__ import annotations

import unittest
import tempfile
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from utils.parsers.hwp_parser import (
    _extract_hwpx_text,
    _html_table_grid,
    _merge_lower_items_into_top_sections,
    _numbered_text_sections,
    _text_section_chunks,
    _top_level_section_chunks,
)


class HwpParserTests(unittest.TestCase):
    def test_html_table_grid_expands_rowspan_and_colspan(self):
        soup = BeautifulSoup(
            """
            <table>
              <tr><th rowspan="2">구분</th><th colspan="2">금액</th></tr>
              <tr><th>상한</th><th>하한</th></tr>
              <tr><td>학부</td><td>100</td><td>50</td></tr>
            </table>
            """,
            "html.parser",
        )

        grid = _html_table_grid(soup.table)

        self.assertEqual(grid[0], ["구분", "금액", "금액"])
        self.assertEqual(grid[1], ["구분", "상한", "하한"])
        self.assertEqual(grid[2], ["학부", "100", "50"])

    def test_text_sections_keep_headings_and_ignore_numbered_footnotes(self):
        chunks = _text_section_chunks(
            """
            문서 제목
            □ 지원대상
            학부 재학생
            2) 국가장학금 관련 각주
            □ 필수 서류
            신청서와 추천서
            """
        )
        titles = [
            chunk["metadata"].get("section_title")
            for chunk in chunks
            if chunk["metadata"].get("section_title")
        ]

        self.assertEqual(titles, ["□ 지원대상", "□ 필수 서류"])
        support_chunk = next(
            chunk for chunk in chunks
            if chunk["metadata"].get("section_title") == "□ 지원대상"
        )
        self.assertIn("2) 국가장학금 관련 각주", support_chunk["text"])

    def test_unsectioned_text_still_becomes_searchable_chunks(self):
        chunks = _text_section_chunks("장학금 운영 계획에 대한 일반 설명입니다. " * 8)

        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["metadata"]["content_type"], "hwp_text_child")

    def test_hwpx_text_is_read_from_section_xml(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <hs:sec xmlns:hs="urn:hancom:section" xmlns:hp="urn:hancom:paragraph">
          <hp:p><hp:run><hp:t>□ 지원대상</hp:t></hp:run></hp:p>
          <hp:p><hp:run><hp:t>학부 재학생</hp:t></hp:run></hp:p>
        </hs:sec>
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.hwpx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("Contents/section0.xml", xml)

            text = _extract_hwpx_text(str(path))

        self.assertEqual(text, "□ 지원대상\n학부 재학생")

    def test_top_level_sections_keep_lower_items_inside_parent(self):
        chunks = _top_level_section_chunks([
            (
                "1 장학 개요",
                "□ 목적\n긴급 장학금 지원\n□ 지원대상\n학부 재학생",
            ),
            (
                "2 신청 및 선발 절차",
                "신청\n학생 면담\n장학생 선발",
            ),
        ])

        self.assertEqual(
            {chunk["metadata"]["section_title"] for chunk in chunks},
            {"1 장학 개요", "2 신청 및 선발 절차"},
        )
        overview = next(
            chunk for chunk in chunks
            if chunk["metadata"]["section_title"] == "1 장학 개요"
        )
        self.assertIn("□ 목적", overview["text"])
        self.assertIn("□ 지원대상", overview["text"])

    def test_lower_items_are_grouped_under_numbered_parent_sections(self):
        sections = [
            ("1 장학 개요", "HTML 개요"),
            ("2 장학사정관 및 장학생 선발위원회", ""),
            ("3 장학선발 및 지급 제한", ""),
            ("4 신청 및 선발 절차", "절차 표"),
            ("5 제출 서류", ""),
        ]
        text = """
        □ 목적
        긴급 장학금 지원
        □ 장학사정관
        학생 면담
        □ 장학생 선발 위원회
        위원회 심사
        □ 선발 대상 제외
        서류 미제출자 제외
        □ 장학금 지급 제외 대상자
        휴학자 제외
        □ 필수 서류
        신청서와 추천서
        """

        merged = _merge_lower_items_into_top_sections(sections, text)

        self.assertIn("□ 목적", merged[0][1])
        self.assertIn("□ 장학사정관", merged[1][1])
        self.assertIn("□ 장학생 선발 위원회", merged[1][1])
        self.assertIn("□ 선발 대상 제외", merged[2][1])
        self.assertIn("□ 장학금 지급 제외 대상자", merged[2][1])
        self.assertEqual(merged[3][1], "절차 표")
        self.assertIn("□ 필수 서류", merged[4][1])

    def test_numbered_text_sections_keep_short_table_only_sections(self):
        sections = _numbered_text_sections(
            """
            1. 장학기금 현황
            2. 장학생 배정 원칙
            대학 60%, 경상대 40%
            7. 추진 일정
            8. 지급 방법 : 학생계좌로 입금
            9. 제출서류: 추천서 1부
            붙임 추천서 서식 1부
            □ 추천 사유 또는 의견
            """
        )

        self.assertEqual(
            [title for title, _ in sections],
            [
                "1. 장학기금 현황",
                "2. 장학생 배정 원칙",
                "7. 추진 일정",
                "8. 지급 방법 : 학생계좌로 입금",
                "9. 제출서류: 추천서 1부",
            ],
        )
        self.assertNotIn("추천 사유", sections[-1][1])

    def test_short_top_level_section_still_creates_metadata_chunk(self):
        chunks = _top_level_section_chunks([("1. 장학기금 현황", "")])

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0]["metadata"]["section_title"],
            "1. 장학기금 현황",
        )
        self.assertGreaterEqual(len(chunks[0]["text"]), 20)


if __name__ == "__main__":
    unittest.main()
