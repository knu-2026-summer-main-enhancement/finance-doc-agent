from __future__ import annotations

import unittest

import pandas as pd

from utils.table_parser import _parse_table


class TableCleanupTest(unittest.TestCase):
    def test_blank_row_and_trailing_annotations_are_not_promoted_to_records(self):
        raw = [
            ["연번", "이름", "전화번호", "장학금액"],
            [1, "김하늘", "010-7000-1001", 1_000_000],
            [2, "이바다", "010-7000-1002", 2_000_000],
            [None, None, None, None],
            ["이 문장은 표 아래에 있는 충분히 긴 안내 문구입니다.", None, None, None],
            ["두 번째로 이어지는 충분히 긴 안내 문구입니다.", None, None, None],
        ]

        result = _parse_table(raw, source_file="ledger.xlsx", context_prefix="s0")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["이름"].tolist(), ["김하늘", "이바다"])

    def test_repeated_long_annotation_is_removed_without_phrase_keywords(self):
        note = "모든 열에 복제되어 들어온 충분히 긴 문장 형태의 각주입니다."
        raw = [
            ["번호", "이름", "금액"],
            [1, "김하늘", 1_000_000],
            [note, note, note],
        ]

        result = _parse_table(raw, source_file="ledger.xlsx", context_prefix="s0")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["이름"], "김하늘")

    def test_unconfirmed_vertical_blanks_are_not_inherited(self):
        raw = [
            ["접수코드", "후원일", "기부자명", "후원금"],
            ["RC-001", "2026-03-01", "김*수", "500,000"],
            [None, "2026-03-03", None, "1,000,000"],
        ]

        result = _parse_table(raw, source_file="sponsors.pdf", context_prefix="p0")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["접수코드"], "RC-001")
        self.assertTrue(pd.isna(result.iloc[1]["접수코드"]))
        self.assertEqual(result.iloc[0]["기부자명"], "김*수")
        self.assertTrue(pd.isna(result.iloc[1]["기부자명"]))
        self.assertEqual(result["후원금"].tolist(), ["500,000", "1,000,000"])
        self.assertEqual(result["entity_type"].tolist(), ["person_masked", "unknown"])

    def test_descriptive_total_row_is_removed_from_amount_records(self):
        raw = [
            ["접수코드", "후원일", "기부자명", "후원금"],
            ["RC-001", "2026-03-01", "김*수", "500,000"],
            ["RC-002", "2026-03-02", "이*희", "1,000,000"],
            [None, None, "총 후원금", "1,500,000"],
        ]

        result = _parse_table(raw, source_file="sponsors.pdf", context_prefix="p0")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        amounts = pd.to_numeric(result["후원금"].str.replace(",", "", regex=False))
        self.assertEqual(int(amounts.sum()), 1_500_000)

    def test_horizontal_gaps_are_not_guessed_as_merged_cells(self):
        raw = [
            ["구분1", "구분2", "값"],
            ["A", None, "B"],
            ["한 셀에만 존재하는 충분히 긴 안내 문장입니다.", None, None],
        ]

        result = _parse_table(raw, source_file="generic.xlsx", context_prefix="s0")

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result.iloc[0]["구분2"]))


if __name__ == "__main__":
    unittest.main()
