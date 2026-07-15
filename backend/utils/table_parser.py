from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# 공통 패턴/상수
# ---------------------------------------------------------------------------
_AGGREGATE_ROW_RE = re.compile(
    r"^(합\s*계|소\s*계|총\s*계|합\s*산|계|장학금\s*계|total|subtotal)$",
    re.IGNORECASE,
)
_AMOUNT_FORMULA_RE = re.compile(r"\d+명\s*[*×x]\s*\d+", re.IGNORECASE)
_DIGIT_AMOUNT_RE = re.compile(r"^\d[\d,]*$|^\d[\d,]*만원$|^\d[\d,]*원$")

NAME_COLS = (
    "성명", "이름", "학생명", "수혜자명", "학생이름", "수혜자", "명단", "학생",
    "이_름", "성_명", "기부자", "후원자", "출연자",
)
AMOUNT_COL_KEYWORDS = (
    "출연금액", "출연금", "지급액", "지급금액", "장학금액", "수혜금액", "지원금액",
    "후원금액", "기부금액", "납입금액", "입금액", "금액", "장학금", "지원액", "수혜액",
)
DATE_COL_KEYWORDS = ("일자", "날짜", "지급일", "출연일", "입금일", "납입일", "등록일")

IDENTITY_INTERNAL_COLS = {
    "성명_원문", "성명_검색키", "성명_마스킹패턴", "성명_마스킹여부",
    "row_uid", "person_candidate_key", "_row_index", "_row_context", "_질문이름", "_매칭유형",
}

_MASK_CHARS_RE = re.compile(r"[＊○●Oo0xX×]")
_PERSON_REAL_RE = re.compile(r"^[가-힣]{2,5}$")
_PERSON_MASK_RE = re.compile(r"^[가-힣*]{2,6}$")
_LEADING_COHORT_RE = re.compile(r"^\s*(\d{1,3})\s*(?:회|기)\s+(.+)$")

_ORG_KEYWORDS = (
    "(주)", "㈜", "주식회사", "유한회사", "재단", "재단법인", "사단법인", "협회", "장학회",
    "동문회", "총동문회", "대공동문회", "동기회", "회장단", "위원회", "조합", "법인",
    "회사", "기업", "은행", "중공업", "공업", "산업", "건설", "전기", "전자", "기계",
    "학교", "대학교", "대학", "고등학교", "학회", "공단", "재단", "센터", "연구소",
)
_DEPARTMENT_WORDS = (
    "기계과", "전기과", "전자과", "건축과", "화학과", "컴퓨터과", "정보통신과", "자동차과",
    "디자인과", "토목과", "산업설비과", "로봇과", "메카트로닉스과", "인공지능전공",
)
_DEPARTMENT_SUFFIX_RE = re.compile(r"[가-힣A-Za-z0-9]+(?:학과|학부|전공|계열|공학과)$")


def _as_text(value) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)

def _cell_val(cell: Any) -> str:
    return str(cell).strip() if cell is not None else ""


def sanitize_table_name(name: str) -> str:
    original = name
    name = re.sub(r"[^\x00-\x7F]", "", name)
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    name = name.lower()[:32].rstrip("_")
    if not name:
        name = "tbl_" + hashlib.md5(original.encode("utf-8")).hexdigest()[:8]
    elif name[0].isdigit():
        name = "tbl_" + name
    return name


def sanitize_column_name(col: str) -> str | None:
    col = str(col).strip()
    if not col or col in ("None", "nan"):
        return None
    col = re.sub(r"[^\w가-힣]", "_", col, flags=re.UNICODE)
    col = re.sub(r"_+", "_", col).strip("_")
    col = col[:40]
    if not col:
        return None
    if col[0].isdigit():
        col = "col_" + col
    return col


def normalize_mask_chars(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if s.lower() in ("none", "nan"):
        return ""
    s = re.sub(r"\s+", "", s)
    s = _MASK_CHARS_RE.sub("*", s)
    # 연속 마스킹 문자(**)는 원본 정보이므로 줄이지 않는다. 예: 송**랑
    return s


def normalize_person_name(value: Any) -> str:
    """사람 이름 검색용 정규화. 기관명 판별 뒤 사람 값에만 적용한다."""
    return normalize_mask_chars(value)


def _contains_hangul(value: str) -> bool:
    return any("가" <= ch <= "힣" for ch in value)


def is_masked_name(value: Any) -> bool:
    key = normalize_person_name(value)
    return "*" in key and bool(_PERSON_MASK_RE.fullmatch(key)) and _contains_hangul(key)


def looks_like_person_name(value: Any) -> bool:
    key = normalize_person_name(value)
    if not key or not _contains_hangul(key):
        return False
    if is_masked_name(key):
        return True
    return bool(_PERSON_REAL_RE.fullmatch(key))


def make_mask_pattern(value: Any) -> str:
    key = normalize_person_name(value)
    if not key:
        return ""
    if "*" in key:
        return key
    if re.fullmatch(r"[가-힣]{3,5}", key):
        return key[0] + "*" * max(1, len(key) - 2) + key[-1]
    return key


def _has_org_indicator(value: str) -> bool:
    compact = re.sub(r"\s+", "", str(value or ""))
    if not compact:
        return False
    if any(k in compact for k in _ORG_KEYWORDS):
        return True
    # "49회 동기회 기계과"처럼 회차 + 단체/학과 단서가 있으면 기관/단체로 본다.
    if re.search(r"\d{1,3}회", compact) and any(k in compact for k in ("동기", "동문", "학과", "전공", "계열")):
        return True
    return False


def _is_department_like(value: str) -> bool:
    compact = re.sub(r"\s+", "", str(value or ""))
    if compact in _DEPARTMENT_WORDS:
        return True
    return bool(_DEPARTMENT_SUFFIX_RE.fullmatch(compact))


def _split_leading_cohort(value: str) -> tuple[str, str]:
    m = _LEADING_COHORT_RE.match(str(value or ""))
    if not m:
        return "", str(value or "").strip()
    return f"{m.group(1)}회", m.group(2).strip()


def classify_name_entity(value: Any, record: dict[str, Any] | pd.Series | None = None) -> dict[str, Any]:
    """이름/성명 계열 셀 값을 사람/기관/unknown으로 분류한다.

    원본값은 절대 삭제하지 않는다. 파생 컬럼만 생성한다.
    """
    original = "" if value is None else str(value).strip()
    if not original or original.lower() in ("none", "nan"):
        return {
            "display_name": "", "person_name": "", "organization_name": "",
            "entity_type": "unknown", "cohort_from_name": "",
        }

    cohort, rest = _split_leading_cohort(original)
    target = rest if cohort else original
    target = target.strip()

    # 기관/단체는 사람 이름보다 먼저 판별한다. 예: (주)금**, 49회 동기회 기계과
    if _has_org_indicator(original):
        return {
            "display_name": original,
            "person_name": "",
            "organization_name": original,
            "entity_type": "organization_masked" if "*" in normalize_mask_chars(original) else "organization",
            "cohort_from_name": cohort,
        }

    # 값 자체가 학과/전공만 나타내면 사람 이름이 아니다.
    if _is_department_like(original):
        return {
            "display_name": original,
            "person_name": "",
            "organization_name": original,
            "entity_type": "department",
            "cohort_from_name": cohort,
        }

    # "56회 금하영"처럼 앞 회차 + 뒤 사람명은 사람으로 분리한다.
    person_candidate = target
    if looks_like_person_name(person_candidate):
        person_key = normalize_person_name(person_candidate)
        return {
            "display_name": person_key,
            "person_name": person_key,
            "organization_name": "",
            "entity_type": "person_masked" if is_masked_name(person_key) else "person_real",
            "cohort_from_name": cohort,
        }

    # 길거나 복합적인 한글 문자열은 억지로 성명으로 확정하지 않는다.
    return {
        "display_name": original,
        "person_name": "",
        "organization_name": "",
        "entity_type": "unknown",
        "cohort_from_name": cohort,
    }


def _find_name_col(df: pd.DataFrame) -> str | None:
    return next((c for c in df.columns if c in NAME_COLS), None)


def _find_first_col(df: pd.DataFrame, keywords: tuple[str, ...]) -> str | None:
    return next((c for c in df.columns if any(k in str(c) for k in keywords)), None)


def find_amount_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if any(k in str(c) for k in AMOUNT_COL_KEYWORDS)]


def find_first_amount_column(df: pd.DataFrame) -> str | None:
    cols = find_amount_columns(df)
    return cols[0] if cols else None


def _safe_part(value: Any) -> str:
    s = str(value).strip() if value is not None else ""
    if s.lower() in ("", "none", "nan"):
        return ""
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[|:/\\]", "_", s)
    return s[:80]


def _first_value_by_keywords(row: pd.Series, keywords: tuple[str, ...]) -> str:
    for col in row.index:
        if any(k in str(col) for k in keywords):
            v = _safe_part(row.get(col, ""))
            if v:
                return v
    return ""


def _make_row_uid_for_row(row: pd.Series, source_file: str, context_prefix: str, row_pos: int) -> str:
    amount = ""
    for col in row.index:
        if any(k in str(col) for k in AMOUNT_COL_KEYWORDS):
            amount = _safe_part(row.get(col, ""))
            if amount:
                break
    date = _first_value_by_keywords(row, DATE_COL_KEYWORDS)
    parts = [
        _safe_part(source_file),
        _safe_part(context_prefix),
        f"row_{row_pos:04d}",
        _safe_part(row.get("발행번호", row.get("번호", ""))),
        date,
        _safe_part(row.get("기수", "")),
        _safe_part(row.get("학과", row.get("전공", ""))),
        _safe_part(row.get("표시명", row.get("성명_검색키", row.get("성명", row.get("이름", ""))))),
        amount,
    ]
    compact = "::".join(p for p in parts if p)
    return compact or f"{_safe_part(source_file) or 'unknown'}::{context_prefix or 'ctx'}::row_{row_pos:04d}"


def _make_person_candidate_key_for_row(row: pd.Series, df: pd.DataFrame) -> str:
    if not str(row.get("성명_검색키", "")).strip():
        return ""
    parts = [_safe_part(row.get("성명_마스킹패턴", row.get("성명_검색키", "")))]
    for keywords in (("학과", "계열", "학부", "전공"), ("학년",), ("생년", "생일", "월일"), ("기수",)):
        col = _find_first_col(df, keywords)
        if col:
            parts.append(_safe_part(row.get(col, "")))
    return "::".join(p for p in parts if p)


def _fill_empty_series(existing: pd.Series, fill_values: list[Any]) -> pd.Series:
    result = existing.copy()
    fill = pd.Series(fill_values, index=result.index)
    empty = result.astype(str).str.strip().isin(["", "None", "nan", "NaN"])
    result.loc[empty] = fill.loc[empty]
    return result


def add_identity_columns(
    df: pd.DataFrame,
    source_file: str = "",
    context_prefix: str = "",
    row_offset: int = 0,
) -> pd.DataFrame:
    """원본 컬럼은 보존하고 검색/식별용 파생 컬럼만 추가한다."""
    if df is None or df.empty:
        return df

    df = df.copy()
    name_col = _find_name_col(df)

    if source_file and "source" not in df.columns:
        df["source"] = source_file

    if name_col:
        originals = df[name_col].astype(str).where(df[name_col].notna(), "")
        classified = [classify_name_entity(v) for v in originals]

        display_names = [c["display_name"] for c in classified]
        person_names = [c["person_name"] for c in classified]
        org_names = [c["organization_name"] for c in classified]
        entity_types = [c["entity_type"] for c in classified]
        cohorts = [c["cohort_from_name"] for c in classified]

        df["성명_원문"] = originals
        df["표시명"] = display_names
        df["entity_type"] = entity_types

        if "기관명" in df.columns:
            df["기관명"] = _fill_empty_series(df["기관명"].astype(str), org_names)
        else:
            df["기관명"] = org_names

        # 원본 이름 컬럼이 '성명'인 경우 원본 보존을 위해 덮어쓰지 않는다.
        # 원본 이름 컬럼이 '이름' 등인 경우 사람 행에 한해 별도 성명 컬럼을 만든다.
        if name_col != "성명":
            if "성명" in df.columns:
                df["성명"] = _fill_empty_series(df["성명"].astype(str), person_names)
            else:
                df["성명"] = person_names

        if any(cohorts):
            if "기수" in df.columns:
                df["기수"] = _fill_empty_series(df["기수"].astype(str), cohorts)
            else:
                df["기수"] = cohorts

        search_keys = [normalize_person_name(v) if v else "" for v in person_names]
        df["성명_검색키"] = search_keys
        df["성명_마스킹패턴"] = [make_mask_pattern(v) if v else "" for v in search_keys]
        df["성명_마스킹여부"] = [is_masked_name(v) if v else False for v in search_keys]
    else:
        if "표시명" not in df.columns:
            df["표시명"] = ""
        if "entity_type" not in df.columns:
            df["entity_type"] = "unknown"

    df["_row_index"] = [row_offset + i for i in range(len(df))]
    df["_row_context"] = context_prefix or ""

    df["row_uid"] = [
        _make_row_uid_for_row(row, source_file or str(row.get("source", "")), context_prefix, row_offset + i)
        for i, (_, row) in enumerate(df.iterrows())
    ]
    df["person_candidate_key"] = [_make_person_candidate_key_for_row(row, df) for _, row in df.iterrows()]
    return df


def _is_footer_row(row: pd.Series) -> bool:
    vals = [str(v).strip() for v in row if str(v).strip() not in ("", "None", "nan")]
    if len(vals) < 2:
        return False
    non_amount = [v for v in vals if not _DIGIT_AMOUNT_RE.match(v)]
    if any(re.search(r"[가-힣]", v) for v in non_amount):
        return False
    return all(_DIGIT_AMOUNT_RE.match(v) for v in vals) and len(set(vals)) <= 3


def _ffill_merged_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """세로 병합셀 보정. 금액 컬럼은 누락 OCR 오인 방지를 위해 세로 채움에서 제외한다."""
    if df.empty:
        return df
    fill_cols = [c for c in df.columns if not any(k in str(c) for k in AMOUNT_COL_KEYWORDS)]
    if fill_cols:
        df[fill_cols] = df[fill_cols].ffill(axis=0)
    return df


def _clean_dataframe(
    df: pd.DataFrame,
    source_file: str = "",
    context_prefix: str = "",
    row_offset: int = 0,
) -> pd.DataFrame:
    """인제스트 시 공통 정제: 집계 행 제거 + 원본 보존 + 파생 검색/식별 컬럼 보강."""
    if df is None or df.empty:
        return df

    df = df.copy()
    source_columns = [str(column) for column in df.columns]

    agg_mask = pd.Series(False, index=df.index)
    for col in df.columns:
        vals = df[col].astype(str).str.strip()
        agg_mask |= vals.apply(lambda v: bool(_AGGREGATE_ROW_RE.match(_as_text(v))) or bool(_AMOUNT_FORMULA_RE.search(_as_text(v))))

    if agg_mask.any():
        logger.info("집계 행 제거: %d행", int(agg_mask.sum()))
        df = df[~agg_mask].reset_index(drop=True)

    footer_mask = df.apply(_is_footer_row, axis=1)
    if footer_mask.any():
        logger.info("숫자전용 footer 행 제거: %d행", int(footer_mask.sum()))
        df = df[~footer_mask].reset_index(drop=True)

    seq_col = next((c for c in df.columns if any(k in str(c) for k in ("연번", "순번", "번호", "순"))), None)
    if seq_col:
        try:
            while len(df) > 1:
                seq_vals = pd.to_numeric(df[seq_col], errors="coerce")
                last, prev = seq_vals.iloc[-1], seq_vals.iloc[-2]
                if pd.notna(last) and pd.notna(prev) and last == prev:
                    logger.info("말미 중복 순번 행 제거: %s=%s", seq_col, df.iloc[-1][seq_col])
                    df = df.iloc[:-1].reset_index(drop=True)
                else:
                    break
        except Exception:
            pass

    for col in df.columns:
        if any(k in str(col) for k in ("학과", "계열", "학부", "전공", "대상학생")):
            try:
                cleaned = df[col].astype(str).str.replace(r"\(\d+명\)", "", regex=True).str.strip()
                df[col] = cleaned.where(~cleaned.isin({"None", "nan", ""}), None)
            except Exception:
                pass

    if df.empty:
        return df

    result = add_identity_columns(
        df,
        source_file=source_file,
        context_prefix=context_prefix,
        row_offset=row_offset,
    )
    result.attrs["source_columns"] = source_columns
    return result


def _parse_table(
    raw_table: list[list],
    source_file: str = "",
    context_prefix: str = "",
    row_offset: int = 0,
    horizontal_ffill_data: bool = True,
) -> "pd.DataFrame | None":
    """병합 셀(None) 처리 + 2행 헤더 자동 탐지 후 DataFrame 반환.

    image OCR에서 만든 raw_table은 빈 칸이 실제 공백일 수 있으므로
    horizontal_ffill_data=False로 호출해 좌우 채움 오염을 막는다.
    """
    if not raw_table or len(raw_table) < 2:
        return None

    ncols = max(len(r) for r in raw_table)
    table = [list(r) + [None] * (ncols - len(r)) for r in raw_table]

    header_idx = 0
    for i, row in enumerate(table):
        if sum(1 for c in row if _cell_val(c)) >= max(1, ncols * 0.4):
            header_idx = i
            break

    h1 = table[header_idx]
    data_start = header_idx + 1

    if data_start < len(table):
        h2 = table[data_start]
        empty_pos = [j for j in range(ncols) if not _cell_val(h1[j])]
        fills = sum(1 for j in empty_pos if _cell_val(h2[j]))
        if empty_pos and fills >= len(empty_pos) * 0.5:
            merged = [_cell_val(h2[j]) if not _cell_val(h1[j]) else _cell_val(h1[j]) for j in range(ncols)]
            data_start += 1
        else:
            merged = [_cell_val(c) for c in h1]
    else:
        merged = [_cell_val(c) for c in h1]

    filled_headers: list[str] = []
    last = ""
    for v in merged:
        last = v if v else last
        filled_headers.append(last)

    seen: dict[str, int] = {}
    headers: list[str] = []
    for j, h in enumerate(filled_headers):
        name = sanitize_column_name(h) or f"col_{j}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        headers.append(name)

    def ffill_row(row: list[Any]) -> list[Any]:
        result, last = [], None
        for cell in row:
            v = _cell_val(cell)
            if v:
                last = v
            result.append(last)
        return result

    if horizontal_ffill_data:
        data_rows = [ffill_row(r) for r in table[data_start:]]
    else:
        data_rows = [[_cell_val(c) or None for c in r] for r in table[data_start:]]

    df = pd.DataFrame(data_rows, columns=headers)
    df = df.replace("", None)
    df = _ffill_merged_like_columns(df)
    df = df.dropna(how="all").replace("\n", " ", regex=True)
    if df.empty:
        return None
    df = _clean_dataframe(df, source_file=source_file, context_prefix=context_prefix, row_offset=row_offset)
    return df if df is not None and not df.empty else None
