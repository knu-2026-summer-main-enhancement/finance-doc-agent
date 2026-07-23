from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


SCHEMA_VERSION = "2.3"
SYSTEM_COLUMNS = (
    "__schema_version",
    "__document_id",
    "__table_id",
    "__row_id",
    "__source_file",
    "__source_type",
    "__mapping_fingerprint",
)

_PROFILE_DIR_NAME = "_schema_profiles"
_EMPTY_VALUES = {"", "none", "nan", "nat", "null"}
_MONEY_VALUE_RE = re.compile(r"^[+-]?\s*(?:₩|krw)?\s*\d[\d,]*(?:\.\d+)?\s*(?:원|만원|천원)?$", re.IGNORECASE)
_DATE_VALUE_RE = re.compile(
    r"^(?:19|20)\d{2}[-./년]\s*\d{1,2}[-./월]\s*\d{1,2}(?:일)?$"
)
_YEAR_MONTH_VALUE_RE = re.compile(r"^(?:19|20)\d{2}(?:[-./년]\s*(?:0?[1-9]|1[0-2])(?:월)?)$")
_PHONE_VALUE_RE = re.compile(r"^(?=.*[- ])(?:\+?82[- ]?)?0?\d{1,2}[- ]?\d{3,4}[- ]?\d{4}$")


@dataclass(frozen=True)
class ColumnMeaning:
    concept: str
    role: str | None
    data_type: str
    unit: str | None
    confidence: float
    mapping_source: str
    qualifier: str | None = None
    sensitivity: str = "none"
    pii_type: str | None = None
    is_derived: bool = False


def make_document_id(source_file: str, file_hash: str = "") -> str:
    key = f"{os.path.basename(source_file).casefold()}::{file_hash or 'no-hash'}"
    return "doc_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def make_table_id(document_id: str, table_name: str) -> str:
    key = f"{document_id}::{table_name}"
    return "tbl_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _normal_header(column: Any) -> str:
    return re.sub(r"[\s_()\[\]{}]+", "", str(column or "")).casefold()


def _nonempty_values(series: pd.Series, limit: int = 100) -> list[str]:
    values: list[str] = []
    for value in series.dropna().head(limit):
        text = str(value).strip()
        if text.casefold() not in _EMPTY_VALUES:
            values.append(text)
    return values


def infer_data_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "number"

    values = _nonempty_values(series)
    if not values:
        return "empty"

    money_ratio = sum(bool(_MONEY_VALUE_RE.fullmatch(value)) for value in values) / len(values)
    if money_ratio >= 0.8:
        return "number"

    date_ratio = sum(bool(_DATE_VALUE_RE.fullmatch(value)) for value in values) / len(values)
    if date_ratio >= 0.8:
        return "date"
    return "string"


def _money_unit(header: str, values: list[str]) -> str | None:
    joined = " ".join([header, *values[:10]])
    if "만원" in joined:
        return "KRW_10000"
    if "천원" in joined:
        return "KRW_1000"
    if "원" in joined or "₩" in joined or "krw" in joined.casefold():
        return "KRW"
    return None


def _value_match_ratio(values: list[str], pattern: re.Pattern[str]) -> float:
    if not values:
        return 0.0
    return sum(bool(pattern.fullmatch(value)) for value in values) / len(values)


def _integer_range_ratio(values: list[str], minimum: int, maximum: int) -> float:
    """Return the ratio of values representing integers in the given range."""

    if not values:
        return 0.0
    matched = 0
    for value in values:
        normalized = re.sub(r"\s*(?:년|월|일)\s*$", "", value).strip()
        try:
            number = float(normalized)
        except (TypeError, ValueError):
            continue
        if number.is_integer() and minimum <= int(number) <= maximum:
            matched += 1
    return matched / len(values)


def _is_row_sequence(values: list[str]) -> bool:
    if len(values) < 2:
        return False
    if not all(re.fullmatch(r"\d+", value) for value in values):
        return False
    numbers = [int(value) for value in values]
    return numbers == list(range(numbers[0], numbers[0] + len(numbers)))


def _amount_qualifier(header: str, values: list[str]) -> tuple[str | None, float] | None:
    """금액형 헤더와 실제 값 형식이 모두 일치할 때만 세부 힌트를 반환한다."""
    amount_suffixes = ("금액", "금", "액", "비용", "사업비")
    if not header.endswith(amount_suffixes):
        return None

    # 값이 존재하는데 통화/숫자 형태가 아니면 '지급기관', '지원분야' 같은
    # 문맥 단어를 금액으로 확대 해석하지 않는다.
    money_ratio = _value_match_ratio(values, _MONEY_VALUE_RE)
    if values and money_ratio < 0.8:
        return None

    if any(token in header for token in ("출연금", "기부금")):
        qualifier = "donation"
    elif any(token in header for token in ("후원금", "후원액")):
        qualifier = "sponsorship"
    elif any(token in header for token in ("집행액", "집행금")):
        qualifier = "executed"
    elif any(token in header for token in ("잔액", "잔여액", "불용액")):
        qualifier = "remaining"
    elif any(token in header for token in ("본예산", "추경예산", "예산액", "예산금")):
        qualifier = "budget"
    elif any(token in header for token in ("장학금", "장학액")):
        qualifier = "scholarship"
    elif any(token in header for token in ("지원금", "지원액")):
        qualifier = "support"
    elif any(token in header for token in ("지급금", "지급액")):
        qualifier = "payment"
    elif any(token in header for token in ("교부액", "배정액")):
        qualifier = "allocated"
    else:
        qualifier = None
    return qualifier, 0.95 if qualifier else 0.8


def _deterministic_meaning(column: str, series: pd.Series, is_derived: bool) -> ColumnMeaning:
    header = _normal_header(column)
    inferred_type = infer_data_type(series)
    values = _nonempty_values(series)

    def meaning(
        concept: str,
        role: str | None,
        data_type: str = inferred_type,
        unit: str | None = None,
        confidence: float = 0.95,
        *,
        qualifier: str | None = None,
        sensitivity: str = "none",
        pii_type: str | None = None,
    ) -> ColumnMeaning:
        return ColumnMeaning(
            concept=concept,
            role=role,
            data_type=data_type,
            unit=unit,
            confidence=confidence,
            mapping_source="deterministic",
            qualifier=qualifier,
            sensitivity=sensitivity,
            pii_type=pii_type,
            is_derived=is_derived,
        )

    if header == "ocrrowindex":
        return meaning("quality", "extraction_row_index", "number")
    if header == "ocrconfidencemin":
        return meaning("quality", "ocr_confidence_min", "number")
    if header == "ocrconfidenceavg":
        return meaning("quality", "ocr_confidence_avg", "number")
    if header == "ocrlowconfidencecells":
        return meaning("quality", "ocr_low_confidence_cells", "string")
    if header == "ocrvalidationok":
        return meaning("quality", "ocr_validation_ok", "boolean")
    if header == "source":
        return meaning("metadata", "source_file", "string")
    if header == "rowindex":
        return meaning("metadata", "source_row_index", "number")
    if header == "rowcontext":
        return meaning("metadata", "row_context", "string")
    if header == "rowuid":
        # 현재 row_uid에는 원본 행의 이름·금액 등이 포함될 수 있다.
        return meaning(
            "identifier", "legacy_row_id", "string",
            sensitivity="personal", pii_type="derived_record_identifier",
        )
    if header == "personcandidatekey":
        return meaning(
            "identifier", "person_candidate_key", "string",
            sensitivity="personal", pii_type="person_candidate_key",
        )
    if header == "성명마스킹패턴":
        return meaning(
            "entity", "name_mask_pattern", "string",
            qualifier="person", sensitivity="personal", pii_type="person_name",
        )
    if header == "성명마스킹여부":
        return meaning("entity", "is_name_masked", "boolean")

    phone_ratio = _value_match_ratio(values, _PHONE_VALUE_RE)
    if values and phone_ratio >= 0.8:
        return meaning(
            "identifier", "identifier_value", "string",
            qualifier="contact", sensitivity="personal", pii_type="phone_number",
        )

    if header in {"발행번호", "발급번호", "접수번호", "문서번호", "관리번호"}:
        return meaning("identifier", "identifier_value", "string", qualifier="document")
    if any(token in header for token in ("학번", "수험번호")):
        return meaning(
            "identifier", "identifier_value", "string",
            qualifier="education", sensitivity="personal", pii_type="education_identifier",
        )
    if header.endswith("코드"):
        return meaning("identifier", "identifier_value", "string")
    if header in {"연번", "순번"} or (header == "번호" and _is_row_sequence(values)):
        return meaning("identifier", "row_number")
    if header.endswith("번호"):
        return meaning("identifier", "identifier_value", "string")

    if values and _value_match_ratio(values, _YEAR_MONTH_VALUE_RE) >= 0.8:
        return meaning("temporal", "year_month", "year_month", qualifier="year_month")

    # A component is classified only when both its header and actual values
    # agree. This prevents columns such as 학년 or 영업일 from becoming dates.
    if (
        (header == "년" or header.endswith(("연도", "년도")))
        and _integer_range_ratio(values, 1900, 2100) >= 0.8
    ):
        return meaning("temporal", "year", "number", qualifier="year")
    if (
        (header == "월" or (header.endswith("월") and not header.endswith("연월")))
        and _integer_range_ratio(values, 1, 12) >= 0.8
    ):
        return meaning("temporal", "month", "number", qualifier="month")
    if header == "일" and _integer_range_ratio(values, 1, 31) >= 0.8:
        return meaning("temporal", "day", "number", qualifier="day")

    if (
        inferred_type == "date"
        or any(token in header for token in ("일자", "날짜", "지급일", "출연일", "후원일", "입금일", "납입일", "등록일"))
    ):
        return meaning("temporal", "date", "date", qualifier="date")
    if header.endswith(("기간", "시기", "연월")):
        return meaning("temporal", "period", inferred_type, qualifier="period")

    if header == "기수" or header.endswith("회차"):
        return meaning("category", "category", qualifier="cohort")
    if any(token in header for token in ("학과", "학부", "전공", "계열")):
        return meaning("category", "category", qualifier="department")
    if (
        header in {"성명", "이름", "회원명", "학생명", "수혜자명", "기부자", "후원자", "출연자", "표시명", "성명원문", "성명검색키"}
        or header.endswith("자명")
    ):
        return meaning(
            "entity", "entity_name", "string",
            qualifier="person", sensitivity="personal", pii_type="person_name",
        )
    if header in {"기관명", "단체명", "회사명", "법인명"} or header.endswith("기관"):
        return meaning("entity", "entity_name", "string", qualifier="organization")
    if header == "entitytype":
        return meaning("entity", "entity_type", "string")

    if "분야" in header:
        return meaning("category", "category", "string", qualifier="field")
    if header.endswith(("방식", "방법")):
        return meaning("category", "category", "string", qualifier="method")
    if header.endswith("목적"):
        return meaning("description", "description", "string", qualifier="purpose")
    if header.endswith("기준"):
        return meaning("description", "description", "string", qualifier="criteria")
    if header.endswith("비고"):
        return meaning("description", "description", "string", qualifier="note")

    amount = _amount_qualifier(header, values)
    if amount:
        qualifier, confidence = amount
        return meaning(
            "measure",
            "amount",
            "money",
            _money_unit(str(column), values),
            confidence,
            qualifier=qualifier,
        )

    return meaning(
        "unknown",
        None,
        inferred_type,
        confidence=0.4 if inferred_type != "empty" else 0.0,
    )


def infer_column_meaning(
    column: str,
    series: pd.Series,
    *,
    is_derived: bool = False,
) -> ColumnMeaning:
    """저장 전 정제 단계에서도 같은 의미 추론 규칙을 재사용한다."""
    return _deterministic_meaning(column, series, is_derived)


def schema_fingerprint(df: pd.DataFrame, source_columns: list[str]) -> str:
    descriptors = [
        f"{_normal_header(column)}:{infer_data_type(df[column])}"
        for column in source_columns
        if column in df.columns
    ]
    payload = f"{SCHEMA_VERSION}|" + "|".join(descriptors)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _source_columns_for(df: pd.DataFrame | pd.Series) -> list[str]:
    explicit = df.attrs.get("source_columns")
    if explicit:
        return [str(column) for column in explicit if not str(column).startswith("__")]

    # 구버전 Parquet은 attrs가 저장되지 않는다. 공통 정제가 뒤에 붙인 첫 파생
    # 컬럼을 기준으로 앞부분을 원본 추출 컬럼으로 복원한다.
    axis_columns = df.columns if isinstance(df, pd.DataFrame) else df.index
    columns = [str(column) for column in axis_columns if not str(column).startswith("__")]
    derived_anchors = {
        "성명_원문", "표시명", "entity_type", "_row_index", "row_uid",
        "person_candidate_key",
    }
    positions = [index for index, column in enumerate(columns) if column in derived_anchors]
    if positions:
        first_derived = min(positions)
        if "source" in columns[:first_derived]:
            first_derived = columns.index("source")
        return columns[:first_derived]
    return columns


def is_source_column(df: pd.DataFrame | pd.Series, column: object) -> bool:
    """Whether a column came from the source table rather than enrichment."""
    key = str(column)
    if key.startswith("_") or key in SYSTEM_COLUMNS:
        return False
    explicit = df.attrs.get("source_columns")
    if explicit:
        return key in {str(item) for item in explicit}
    schema = df.attrs.get("semantic_schema")
    if isinstance(schema, dict):
        columns = schema.get("columns")
        mapping = columns.get(key) if isinstance(columns, dict) else None
        if isinstance(mapping, dict) and "is_derived" in mapping:
            return not bool(mapping.get("is_derived"))
    return key in set(_source_columns_for(df))


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def build_column_mappings(
    df: pd.DataFrame,
    source_columns: list[str],
    profile_path: str,
) -> tuple[dict[str, dict[str, Any]], bool]:
    profile = _read_json(profile_path)
    profile_mappings = profile.get("columns", {}) if profile else {}
    mappings: dict[str, dict[str, Any]] = {}
    profile_used = False

    for column in df.columns:
        if str(column).startswith("__"):
            continue
        is_derived = str(column) not in source_columns
        cached = profile_mappings.get(str(column))
        if isinstance(cached, dict):
            value = dict(cached)
            if value.get("mapping_source") != "human":
                value["mapping_source"] = "profile"
            value["is_derived"] = is_derived
            mappings[str(column)] = value
            profile_used = True
            continue
        mappings[str(column)] = asdict(
            _deterministic_meaning(str(column), df[column], is_derived)
        )
    return mappings, profile_used


def attach_semantic_schema(
    df: pd.DataFrame,
    *,
    var_name: str,
    source_file: str,
    file_hash: str = "",
    source_type: str = "",
    dataframe_dir: str,
) -> dict[str, Any]:
    """원본 컬럼을 유지하면서 시스템 ID와 컬럼 의미 매핑을 추가한다."""
    source_columns = _source_columns_for(df)
    document_id = make_document_id(source_file, file_hash)
    table_id = make_table_id(document_id, var_name)
    fingerprint = schema_fingerprint(df, [c for c in source_columns if c in df.columns])
    resolved_type = source_type or os.path.splitext(source_file)[1].lower().lstrip(".")

    df["__schema_version"] = SCHEMA_VERSION
    df["__document_id"] = document_id
    df["__table_id"] = table_id
    df["__row_id"] = [
        "row_" + hashlib.sha256(
            f"{table_id}::{row.get('row_uid', position)}".encode("utf-8")
        ).hexdigest()[:20]
        for position, (_, row) in enumerate(df.iterrows())
    ]
    df["__source_file"] = source_file
    df["__source_type"] = resolved_type
    df["__mapping_fingerprint"] = fingerprint

    profile_dir = os.path.join(dataframe_dir, _PROFILE_DIR_NAME)
    profile_path = os.path.join(profile_dir, f"{fingerprint}.json")
    mappings, profile_used = build_column_mappings(df, source_columns, profile_path)

    if not os.path.exists(profile_path):
        _atomic_write_json(profile_path, {
            "schema_version": SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "verification_status": "auto_generated",
            "source_columns": source_columns,
            "columns": mappings,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "table_id": table_id,
        "fingerprint": fingerprint,
        "source_file": source_file,
        "source_type": resolved_type,
        "source_columns": source_columns,
        "profile_used": profile_used,
        "columns": mappings,
        "unmapped_columns": [
            column
            for column, mapping in mappings.items()
            if mapping.get("role") is None and not mapping.get("is_derived")
        ],
    }


def save_schema_sidecar(dataframe_dir: str, var_name: str, schema: dict[str, Any]) -> str:
    path = os.path.join(dataframe_dir, f"{var_name}.schema.json")
    _atomic_write_json(path, schema)
    return path


def load_schema_sidecar(dataframe_dir: str, var_name: str) -> dict[str, Any] | None:
    return _read_json(os.path.join(dataframe_dir, f"{var_name}.schema.json"))


def semantic_columns(
    df: pd.DataFrame,
    *,
    concept: str | None = None,
    roles: set[str] | None = None,
    data_type: str | None = None,
    min_confidence: float = 0.75,
) -> list[str]:
    """로드된 sidecar 의미 정보에서 안전하게 사용할 원본 컬럼을 찾는다."""
    schema = df.attrs.get("semantic_schema")
    if not isinstance(schema, dict):
        return []
    mappings = schema.get("columns", {})
    result: list[str] = []
    for column, mapping in mappings.items():
        if column not in df.columns or not isinstance(mapping, dict):
            continue
        if float(mapping.get("confidence", 0.0) or 0.0) < min_confidence:
            continue
        if concept is not None and mapping.get("concept") != concept:
            continue
        if roles is not None and mapping.get("role") not in roles:
            continue
        if data_type is not None and mapping.get("data_type") != data_type:
            continue
        result.append(column)
    return result
