from __future__ import annotations

import math
import numbers
import re
from typing import Any

import pandas as pd


_MONEY_RE = re.compile(
    r"^\s*(?P<sign>[+-]?)\s*(?P<prefix>₩|KRW)?\s*"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"(?P<unit>원|천원|만원)?\s*$",
    re.IGNORECASE,
)
_UNIT_MULTIPLIERS = {
    "KRW": 1.0,
    "KRW_1000": 1_000.0,
    "KRW_10000": 10_000.0,
    "원": 1.0,
    "천원": 1_000.0,
    "만원": 10_000.0,
}


def parse_money_value(value: Any, unit_hint: str | None = None) -> float | None:
    """Parse one complete money value without guessing or repairing OCR text."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return numeric * _UNIT_MULTIPLIERS.get(str(unit_hint or "").upper(), 1.0)

    text = str(value).strip()
    if not text or text.casefold() in {"none", "nan", "null", "<na>", "-"}:
        return None
    match = _MONEY_RE.fullmatch(text)
    if match is None:
        return None

    numeric = float(match.group("number").replace(",", ""))
    if match.group("sign") == "-":
        numeric = -numeric
    explicit_unit = match.group("unit")
    multiplier = (
        _UNIT_MULTIPLIERS[explicit_unit]
        if explicit_unit
        else _UNIT_MULTIPLIERS.get(str(unit_hint or "").upper(), 1.0)
    )
    return numeric * multiplier


def money_unit_for_column(df: pd.DataFrame, column: str) -> str | None:
    """Return a schema/header unit; never infer units from domain-specific aliases."""
    schema = df.attrs.get("semantic_schema")
    if isinstance(schema, dict):
        columns = schema.get("columns", {})
        mapping = columns.get(column) if isinstance(columns, dict) else None
        if isinstance(mapping, dict):
            unit = str(mapping.get("unit") or "").upper()
            if unit in {"KRW", "KRW_1000", "KRW_10000"}:
                return unit

    header = str(column)
    if "만원" in header:
        return "KRW_10000"
    if "천원" in header:
        return "KRW_1000"
    return None


def money_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric Series using the same parser throughout the service."""
    if df is None or column not in df.columns:
        return pd.Series(dtype="float64")
    unit_hint = money_unit_for_column(df, column)
    parsed = df[column].map(lambda value: parse_money_value(value, unit_hint))
    return pd.to_numeric(parsed, errors="coerce")


def money_values(df: pd.DataFrame, column: str) -> list[float]:
    if df is None or df.empty or column not in df.columns:
        return []
    return [float(value) for value in money_series(df, column).dropna().tolist()]
