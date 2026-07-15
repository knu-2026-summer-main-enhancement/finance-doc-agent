from __future__ import annotations

import unittest

import pandas as pd

from pandas_engine.money import money_series, parse_money_value


class MoneyParserTests(unittest.TestCase):
    def test_parses_supported_money_formats(self):
        cases = {
            "1000000": 1_000_000.0,
            "1,000,000": 1_000_000.0,
            "1,000,000원": 1_000_000.0,
            "₩1,000,000": 1_000_000.0,
            "KRW 1,000,000": 1_000_000.0,
            "100만원": 1_000_000.0,
            "100천원": 100_000.0,
            "0": 0.0,
            "000": 0.0,
            "-100,000": -100_000.0,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(parse_money_value(raw), expected)

    def test_rejects_ambiguous_or_damaged_values(self):
        for raw in ("500,", "1,OOO,OOO", "미정", "100,000 / 200,000", "-"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_money_value(raw))

    def test_schema_unit_is_applied_without_column_name_hardcoding(self):
        df = pd.DataFrame({"임의의 측정값": [100, "25", "1만원", "미정"]})
        df.attrs["semantic_schema"] = {
            "columns": {"임의의 측정값": {"unit": "KRW_10000"}}
        }
        values = money_series(df, "임의의 측정값")
        self.assertEqual(values.iloc[0], 1_000_000.0)
        self.assertEqual(values.iloc[1], 250_000.0)
        self.assertEqual(values.iloc[2], 10_000.0)
        self.assertTrue(pd.isna(values.iloc[3]))

    def test_zero_negative_and_small_values_are_not_discarded(self):
        df = pd.DataFrame({"금액": ["0", "500", "-200", "100만원", "미정"]})
        values = money_series(df, "금액")
        self.assertEqual(values.dropna().tolist(), [0.0, 500.0, -200.0, 1_000_000.0])
        self.assertEqual(float(values.sum()), 1_000_300.0)


if __name__ == "__main__":
    unittest.main()
