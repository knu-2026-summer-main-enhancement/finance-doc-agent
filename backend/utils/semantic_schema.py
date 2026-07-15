from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd


SCHEMA_VERSION = "1.1"
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


@dataclass(frozen=True)
class ColumnMeaning:
    concept: str
    role: str | None
    data_type: str
    unit: str | None
    confidence: float
    mapping_source: str
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
    ) -> ColumnMeaning:
        return ColumnMeaning(
            concept=concept,
            role=role,
            data_type=data_type,
            unit=unit,
            confidence=confidence,
            mapping_source="deterministic",
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
        return meaning("identifier", "legacy_row_id", "string")
    if header == "personcandidatekey":
        return meaning("identifier", "person_candidate_key", "string")
    if header == "성명마스킹패턴":
        return meaning("entity", "name_mask_pattern", "string")
    if header == "성명마스킹여부":
        return meaning("entity", "is_name_masked", "boolean")
    if header in {"발행번호", "발급번호", "접수번호", "문서번호", "관리번호"}:
        return meaning("identifier", "issue_number", "string")
    if any(token in header for token in ("일자", "날짜", "지급일", "출연일", "입금일", "납입일", "등록일")):
        return meaning("temporal", "transaction_date", "date")
    if header in {"기수", "회차"}:
        return meaning("category", "cohort")
    if any(token in header for token in ("학과", "학부", "전공", "계열")):
        return meaning("category", "department")
    if header in {"성명", "이름", "학생명", "수혜자명", "기부자", "후원자", "출연자", "표시명", "성명원문", "성명검색키"}:
        return meaning("entity", "entity_name", "string")
    if header in {"기관명", "단체명", "회사명", "법인명"}:
        return meaning("entity", "organization_name", "string")
    if header == "entitytype":
        return meaning("entity", "entity_type", "string")

    amount_role: str | None = None
    if any(token in header for token in ("출연금", "기부금")):
        amount_role = "donation_amount"
    elif "후원" in header:
        amount_role = "sponsorship_amount"
    elif any(token in header for token in ("집행액", "집행금")):
        amount_role = "executed_amount"
    elif any(token in header for token in ("잔액", "잔여액", "불용액")):
        amount_role = "remaining_amount"
    elif any(token in header for token in ("본예산", "추경예산", "예산액", "예산")):
        amount_role = "budget_amount"
    elif "장학" in header:
        amount_role = "scholarship_amount"
    elif "지원" in header:
        amount_role = "support_amount"
    elif "지급" in header:
        amount_role = "payment_amount"
    elif any(token in header for token in ("교부액", "배정액")):
        amount_role = "allocated_amount"
    elif any(token in header for token in ("금액", "액", "비용", "사업비")):
        amount_role = "amount"

    if amount_role:
        return meaning(
            "measure",
            amount_role,
            "money",
            _money_unit(str(column), values),
            0.95 if amount_role != "amount" else 0.8,
        )

    return meaning(
        "unknown",
        None,
        inferred_type,
        confidence=0.4 if inferred_type != "empty" else 0.0,
    )


def schema_fingerprint(df: pd.DataFrame, source_columns: list[str]) -> str:
    descriptors = [
        f"{_normal_header(column)}:{infer_data_type(df[column])}"
        for column in source_columns
        if column in df.columns
    ]
    payload = f"{SCHEMA_VERSION}|" + "|".join(descriptors)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _source_columns_for(df: pd.DataFrame) -> list[str]:
    explicit = df.attrs.get("source_columns")
    if explicit:
        return [str(column) for column in explicit if not str(column).startswith("__")]

    # 구버전 Parquet은 attrs가 저장되지 않는다. 공통 정제가 뒤에 붙인 첫 파생
    # 컬럼을 기준으로 앞부분을 원본 추출 컬럼으로 복원한다.
    columns = [str(column) for column in df.columns if not str(column).startswith("__")]
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
