from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from utils.semantic_schema import infer_column_meaning, infer_data_type, semantic_columns


_YEAR = r"(?:19|20)\d{2}"
_MONTH = r"(?:1[0-2]|0?[1-9])"
_ALL_YEARS_RE = re.compile(r"모든\s*연도|전체\s*연도|연도\s*상관없이", re.IGNORECASE)
_MONTH_RANGE_RE = re.compile(
    rf"(?:(?P<year>{_YEAR})\s*년?\s*)?"
    rf"(?P<start>{_MONTH})\s*월?\s*"
    rf"(?:~|～|−|–|—|-|부터|에서)\s*"
    rf"(?:(?P<end_year>{_YEAR})\s*년?\s*)?"
    rf"(?P<end>{_MONTH})\s*월(?:\s*까지|\s*사이)?",
    re.IGNORECASE,
)
_SINGLE_MONTH_RE = re.compile(
    rf"(?:(?P<year>{_YEAR})\s*년\s*)?(?P<month>{_MONTH})\s*월",
    re.IGNORECASE,
)
_DATE_HEADER_HINTS = (
    "일자", "날짜", "지급일", "출연일", "후원일", "기부일", "입금일", "납입일", "등록일",
)
_QUESTION_ROLE_HINTS = (
    (("지급", "받은", "수령"), ("지급", "수령")),
    (("출연",), ("출연",)),
    (("후원",), ("후원",)),
    (("기부",), ("기부",)),
    (("납부", "입금", "낸", "냈"), ("납부", "입금", "출연", "기부", "후원")),
)


@dataclass(frozen=True)
class DateFilter:
    start_month: int
    end_month: int
    year: int | None = None
    end_year: int | None = None
    all_years: bool = False
    expression: str = ""
    error: str = ""


@dataclass
class DateFilterResult:
    rows: pd.DataFrame | None
    column: str | None = None
    matched_rows: int = 0
    invalid_rows: int = 0
    evidence: dict[str, object] | None = None
    message: str = ""


def _matched_int(matched: re.Match[str], group: str) -> int | None:
    value = matched.group(group)
    return int(value) if value else None


def parse_date_filter(question: str) -> DateFilter | None:
    """질문에서 월 또는 월 범위를 구조화한다. 특정 문서 값은 사용하지 않는다."""
    text = str(question or "").strip()
    matched = _MONTH_RANGE_RE.search(text)
    if matched:
        year = _matched_int(matched, "year")
        end_year = _matched_int(matched, "end_year") or year
        start_month = int(matched.group("start"))
        end_month = int(matched.group("end"))
        error = ""
        if year is None and start_month > end_month:
            error = "연도를 생략한 상태에서는 연도를 넘는 월 범위를 판단할 수 없습니다. 시작 연도와 종료 연도를 지정해 주세요."
        elif year is not None and end_year is not None and (end_year, end_month) < (year, start_month):
            error = "종료 시점이 시작 시점보다 빠릅니다. 날짜 범위를 다시 확인해 주세요."
        return DateFilter(
            start_month=start_month,
            end_month=end_month,
            year=year,
            end_year=end_year,
            all_years=bool(_ALL_YEARS_RE.search(text)),
            expression=matched.group(0),
            error=error,
        )

    matched = _SINGLE_MONTH_RE.search(text)
    if not matched:
        return None
    month = int(matched.group("month"))
    year = _matched_int(matched, "year")
    return DateFilter(
        start_month=month,
        end_month=month,
        year=year,
        end_year=year,
        all_years=bool(_ALL_YEARS_RE.search(text)),
        expression=matched.group(0),
    )


def date_column_candidates(df: pd.DataFrame) -> list[str]:
    candidates = semantic_columns(
        df,
        concept="temporal",
        roles={"date", "year_month", "month"},
    )
    for column in df.columns:
        if str(column).startswith("_") or column in candidates:
            continue
        header = re.sub(r"\s+", "", str(column))
        meaning = infer_column_meaning(str(column), df[column])
        if (
            (meaning.concept == "temporal" and meaning.role in {"date", "year_month", "month"})
            or any(hint in header for hint in _DATE_HEADER_HINTS)
            or infer_data_type(df[column]) == "date"
        ):
            candidates.append(str(column))
    return candidates


def _temporal_role(df: pd.DataFrame, column: str) -> str:
    schema = df.attrs.get("semantic_schema")
    if isinstance(schema, dict):
        columns = schema.get("columns")
        mapping = columns.get(column) if isinstance(columns, dict) else None
        if isinstance(mapping, dict) and mapping.get("concept") == "temporal":
            return str(mapping.get("role") or "")
    return str(infer_column_meaning(column, df[column]).role or "")


def _component_column(df: pd.DataFrame, role: str) -> str | None:
    matches: list[str] = []
    for column in df.columns:
        if str(column).startswith("_"):
            continue
        if _temporal_role(df, str(column)) == role:
            matches.append(str(column))
    return matches[0] if len(matches) == 1 else None


def resolve_date_column(df: pd.DataFrame, question: str) -> tuple[str | None, list[str]]:
    candidates = date_column_candidates(df)
    if len(candidates) <= 1:
        return (candidates[0] if candidates else None), candidates

    question_text = str(question or "")
    spec = parse_date_filter(question_text)
    # A fully specified month range needs one chronological key.  Without an
    # explicitly named business date header, one complete date/year-month
    # column is the only candidate that can represent it without reducing the
    # range to its first year/month component.
    complete_candidates = [
        column for column in candidates
        if _temporal_role(df, column) in {"date", "year_month"}
    ]
    normalized_question = re.sub(r"\s+", "", question_text)
    explicitly_named = [
        column for column in candidates
        if len(re.sub(r"\s+", "", column)) >= 2
        and re.sub(r"\s+", "", column) in normalized_question
    ]
    range_ready_complete = []
    if spec is not None and spec.year is not None:
        start_key = spec.year * 100 + spec.start_month
        end_key = (spec.end_year or spec.year) * 100 + spec.end_month
        for column in complete_candidates:
            parsed = _to_datetime(df[column])
            valid = parsed.dropna()
            if valid.empty:
                continue
            keys = valid.dt.year * 100 + valid.dt.month
            if int(keys.min()) <= start_key and int(keys.max()) >= end_key:
                range_ready_complete.append(column)
    component_candidates = [
        column for column in candidates
        if _temporal_role(df, column) == "month" and _component_column(df, "year") is not None
    ]
    if (
        spec is not None
        and (spec.start_month != spec.end_month or spec.year != spec.end_year)
        and not explicitly_named
        and len(range_ready_complete) == 1
    ):
        return range_ready_complete[0], candidates
    if (
        spec is not None
        and (spec.start_month != spec.end_month or spec.year != spec.end_year)
        and not explicitly_named
        and not range_ready_complete
        and len(component_candidates) == 1
    ):
        return component_candidates[0], candidates
    # More detailed data is not automatically more relevant. When a document
    # has both 신청일자 and 지급월, silently preferring the full date would
    # change the user's intended business meaning. Only question evidence may
    # break a tie between multiple temporal columns.
    scores = {column: 0 for column in candidates}
    for column in candidates:
        normalized_column = re.sub(r"\s+", "", column)
        if len(normalized_column) >= 2 and normalized_column in normalized_question:
            scores[column] += 10
    for question_words, column_words in _QUESTION_ROLE_HINTS:
        if not any(word in question_text for word in question_words):
            continue
        for column in candidates:
            normalized = re.sub(r"\s+", "", str(column))
            scores[column] += sum(1 for word in column_words if word in normalized)

    best_score = max(scores.values(), default=0)
    selected = [
        column for column, score in scores.items()
        if score == best_score and score > 0
    ]
    return (selected[0] if len(selected) == 1 else None), candidates


def _to_integer_component(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    missing = numeric.isna()
    if missing.any():
        extracted = text.loc[missing].str.extract(
            r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:년|월|일)?\s*$",
            expand=False,
        )
        numeric.loc[missing] = pd.to_numeric(extracted, errors="coerce")
    return numeric


def _to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    text = series.astype("string").str.strip()
    normalized = (
        text.str.replace(r"\s*년\s*", "-", regex=True)
        .str.replace(r"\s*월\s*", "-", regex=True)
        .str.replace(r"\s*일\s*$", "", regex=True)
        .str.replace(".", "-", regex=False)
        .str.replace("/", "-", regex=False)
    )
    try:
        parsed = pd.to_datetime(normalized, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(normalized, errors="coerce")

    numeric = pd.to_numeric(text, errors="coerce")
    compact_date = text.str.fullmatch(r"\d{8}", na=False)
    if compact_date.any():
        parsed.loc[compact_date] = pd.to_datetime(text.loc[compact_date], format="%Y%m%d", errors="coerce")
    excel_serial = numeric.between(20_000, 80_000, inclusive="both") & ~compact_date
    if excel_serial.any():
        parsed.loc[excel_serial] = pd.to_datetime(
            numeric.loc[excel_serial], unit="D", origin="1899-12-30", errors="coerce"
        )
    return parsed


def _period_label(spec: DateFilter, years: list[int]) -> str:
    if spec.year is not None:
        end_year = spec.end_year or spec.year
        if spec.year == end_year and spec.start_month == spec.end_month:
            return f"{spec.year}년 {spec.start_month}월"
        return f"{spec.year}년 {spec.start_month}월~{end_year}년 {spec.end_month}월"
    month_label = f"{spec.start_month}월" if spec.start_month == spec.end_month else f"{spec.start_month}~{spec.end_month}월"
    if spec.all_years:
        return f"모든 연도의 {month_label}"
    return f"{years[0]}년 {month_label}" if years else month_label


def _date_range_mask(parsed: pd.Series, spec: DateFilter) -> pd.Series:
    if spec.year is None:
        return parsed.dt.month.between(
            spec.start_month,
            spec.end_month,
            inclusive="both",
        )

    start_key = spec.year * 100 + spec.start_month
    end_key = (spec.end_year or spec.year) * 100 + spec.end_month
    date_key = parsed.dt.year * 100 + parsed.dt.month
    return date_key.between(start_key, end_key, inclusive="both")


def _date_filter_evidence(
    column: str,
    period: str,
    matched_rows: int,
    invalid_rows: int,
    year_column: str | None = None,
) -> dict[str, object]:
    evidence = {
        "column": column,
        "period": period,
        "matched_rows": matched_rows,
        "invalid_date_rows": invalid_rows,
    }
    if year_column:
        evidence["year_column"] = year_column
    return evidence


def _apply_month_component_filter(
    df: pd.DataFrame,
    column: str,
    spec: DateFilter,
) -> DateFilterResult:
    months = _to_integer_component(df[column])
    valid_months = months.between(1, 12, inclusive="both")
    invalid_rows = int((~valid_months).sum())
    year_column = _component_column(df, "year")
    years = _to_integer_component(df[year_column]) if year_column else None
    available_years = (
        sorted({int(value) for value in years.dropna().unique()})
        if years is not None
        else []
    )

    if spec.year is not None and years is None:
        return DateFilterResult(
            None,
            column=column,
            invalid_rows=invalid_rows,
            message=(
                f"{column} 컬럼에는 월 정보만 있어 {spec.year}년을 구분할 수 없습니다. "
                "연도 컬럼이 포함된 문서를 사용하거나 연도를 제외해 주세요."
            ),
        )
    if spec.year is None and years is not None and not spec.all_years and len(available_years) > 1:
        year_text = ", ".join(str(year) for year in available_years)
        return DateFilterResult(
            None,
            column=column,
            invalid_rows=invalid_rows,
            message=f"이 문서에는 여러 연도가 있습니다({year_text}). 연도를 지정하거나 '모든 연도'라고 질문해 주세요.",
        )

    if spec.year is not None and years is not None:
        start_key = spec.year * 100 + spec.start_month
        end_key = (spec.end_year or spec.year) * 100 + spec.end_month
        valid_years = years.between(1900, 2100, inclusive="both")
        invalid_rows = int((~(valid_months & valid_years)).sum())
        mask = valid_months & valid_years & (years * 100 + months).between(
            start_key,
            end_key,
            inclusive="both",
        )
    else:
        mask = valid_months & months.between(
            spec.start_month,
            spec.end_month,
            inclusive="both",
        )

    rows = df[mask.fillna(False)].copy()
    period = _period_label(spec, available_years)
    evidence = _date_filter_evidence(
        column,
        period,
        len(rows),
        invalid_rows,
        year_column,
    )
    rows.attrs.update(df.attrs)
    rows.attrs["date_filter_evidence"] = evidence
    return DateFilterResult(
        rows,
        column=column,
        matched_rows=len(rows),
        invalid_rows=invalid_rows,
        evidence=evidence,
    )


def apply_date_filter(df: pd.DataFrame, spec: DateFilter, question: str) -> DateFilterResult:
    if spec.error:
        return DateFilterResult(None, message=spec.error)

    column, candidates = resolve_date_column(df, question)
    if not candidates:
        return DateFilterResult(None, message="이 문서에는 날짜로 판단할 수 있는 컬럼이 없습니다.")
    if column is None:
        return DateFilterResult(
            None,
            message=f"날짜 기준을 하나로 결정할 수 없습니다. 사용할 날짜 컬럼을 지정해 주세요: {', '.join(candidates)}",
        )

    if _temporal_role(df, column) == "month":
        return _apply_month_component_filter(df, column, spec)

    parsed = _to_datetime(df[column])
    invalid_rows = int(parsed.isna().sum())
    available_years = sorted({int(value) for value in parsed.dropna().dt.year.unique()})
    if not available_years:
        return DateFilterResult(None, column=column, invalid_rows=invalid_rows, message="변환 가능한 날짜 데이터가 없습니다.")
    if spec.year is None and not spec.all_years and len(available_years) > 1:
        years = ", ".join(str(year) for year in available_years)
        return DateFilterResult(
            None,
            column=column,
            invalid_rows=invalid_rows,
            message=f"이 문서에는 여러 연도의 날짜가 있습니다({years}). 연도를 지정하거나 '모든 연도'라고 질문해 주세요.",
        )

    mask = _date_range_mask(parsed, spec)
    rows = df[mask.fillna(False)].copy()
    period = _period_label(spec, available_years)
    evidence = _date_filter_evidence(column, period, len(rows), invalid_rows)
    rows.attrs.update(df.attrs)
    rows.attrs["date_filter_evidence"] = evidence
    return DateFilterResult(
        rows,
        column=column,
        matched_rows=len(rows),
        invalid_rows=invalid_rows,
        evidence=evidence,
    )
