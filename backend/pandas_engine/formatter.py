from __future__ import annotations

import re
from typing import Any

import pandas as pd

from pandas_engine.aggregation import (
    _AGG_COUNT,
    amount_column_candidates,
    amount_column_clarification,
    resolve_amount_column,
)
from pandas_engine.money import money_values
from utils.table_parser import IDENTITY_INTERNAL_COLS

_INTERNAL_COLS = set(IDENTITY_INTERNAL_COLS)
_DISPLAY_ORDER = (
    "source", "발행번호", "출연일자", "지급일", "날짜", "기수", "학과", "전공", "학년",
    "이름", "성명", "표시명", "기관명", "entity_type",
    "출연금액", "지급액", "금액", "장학금액", "지원금액", "수혜금액", "지급처", "생년월일", "원문행",
)
_AMOUNT_QUESTION_RE = re.compile(r"얼마|금액|총액|합계|출연금|지급액|장학금|지원금|수혜금|후원금|기부금")
_SUM_WORD_RE = re.compile(r"총|합계|전체|누적|합산|모두|다")


def _is_internal_col(col: str) -> bool:
    return col in _INTERNAL_COLS or str(col).startswith("_")


def _find_amount_cols(df: pd.DataFrame) -> list[str]:
    return amount_column_candidates(df)


def _format_number(value: Any) -> str:
    try:
        f = float(value)
        if pd.isna(f):
            return str(value)
        if f == int(f):
            return f"{int(f):,}"
        return f"{f:,.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def _display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = [c for c in _DISPLAY_ORDER if c in df.columns and not _is_internal_col(c)]
    cols += [c for c in df.columns if c not in cols and not _is_internal_col(c)]
    return df.loc[:, cols]


def _format_amount_payload(payload: dict[str, Any]) -> str:
    label = str(payload.get("label") or "금액")
    value = payload.get("value", "")
    agg = str(payload.get("agg") or "")
    number = _format_number(value)
    if agg == "sum":
        return f"{label} 합계는 {number}입니다."
    if agg == "max":
        return f"최대 {label}은 {number}입니다."
    if agg == "min":
        return f"최소 {label}은 {number}입니다."
    if agg == "per":
        return f"1인당 {label}은 {number}입니다."
    return f"{label}은 {number}입니다."


def _format_aggregation_payload(payload: dict[str, Any]) -> str:
    operation = str(payload.get("operation") or "")
    label = str(payload.get("label") or "금액")
    invalid_rows = int(payload.get("invalid_rows") or 0)
    warning = f" 숫자로 변환하지 못한 {invalid_rows}개 행은 계산에서 제외했습니다." if invalid_rows else ""

    if operation == "count":
        value = int(payload.get("value") or 0)
        unit = str(payload.get("unit") or "건")
        return f"총 {value:,}{unit}입니다."

    if operation == "mode":
        values = payload.get("values") or []
        numbers = ", ".join(f"{_format_number(value)}원" for value in values)
        return f"{label} 최빈값은 {numbers}입니다.{warning}"

    subjects = payload.get("subjects") or []
    if subjects:
        scope = str(payload.get("scope") or "")
        if scope == "person_total":
            order_word = "가장 적은" if operation == "min" else "가장 많은"
            if len(subjects) == 1:
                subject = subjects[0]
                return (
                    f"{subject.get('name', '이름 정보 없음')}의 누적 {label}이 "
                    f"{_format_number(subject.get('value'))}원으로 {order_word} 금액입니다.{warning}"
                )
            lines = [
                f"- {subject.get('name', '이름 정보 없음')}: {_format_number(subject.get('value'))}원"
                for subject in subjects
            ]
            heading = "누적 금액 하위" if operation == "min" else "누적 금액 상위"
            return f"{heading} 결과입니다.\n" + "\n".join(lines) + warning

        order_word = "가장 작은" if operation == "min" else "가장 큰"
        lines: list[str] = []
        for subject in subjects:
            details = [
                str(subject.get(col))
                for col in ("발행번호", "출연일자", "지급일", "날짜")
                if subject.get(col)
            ]
            suffix = f" ({', '.join(details)})" if details else ""
            lines.append(
                f"- {subject.get('name', '이름 정보 없음')}: "
                f"{_format_number(subject.get('value'))}원{suffix}"
            )
        return f"한 번에 {order_word} {label}을 기록한 항목입니다.\n" + "\n".join(lines) + warning

    value = _format_number(payload.get("value"))
    if operation == "sum":
        answer = f"{label} 합계는 {value}원입니다."
    elif operation == "mean":
        answer = f"{label} 평균은 {value}원입니다."
    elif operation == "median":
        answer = f"{label} 중앙값은 {value}원입니다."
    elif operation == "per_capita":
        count = int(payload.get("people_count") or 0)
        count_text = f" {count:,}명을 기준으로 계산했습니다." if count else ""
        answer = f"1인당 평균 {label}은 {value}원입니다.{count_text}"
    elif operation == "max":
        answer = f"최대 {label}은 {value}원입니다."
    elif operation == "min":
        answer = f"최소 {label}은 {value}원입니다."
    else:
        answer = f"{label}은 {value}원입니다."
    return answer + warning


def _format_pandas_result(result: object) -> str:
    if result is None:
        return "조회된 데이터가 없습니다."
    if isinstance(result, dict) and result.get("type") == "aggregation_notice":
        return str(result.get("message") or "집계 결과를 확인할 수 없습니다.")
    if isinstance(result, dict) and result.get("type") == "aggregation":
        return _format_aggregation_payload(result)
    if isinstance(result, dict) and result.get("type") == "amount":
        return _format_amount_payload(result)
    if hasattr(result, "item"):
        result = result.item()
    if isinstance(result, (int, float)):
        return str(result)
    if isinstance(result, pd.Series):
        result = result.reset_index().to_dict("records")
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return "조회된 데이터가 없습니다."
        return _mask_warning(result) + _display_df(result).to_string(index=False)
    if isinstance(result, list):
        if not result:
            return "조회된 데이터가 없습니다."
        if isinstance(result[0], dict):
            df = pd.DataFrame(result)
            return _mask_warning(df) + _display_df(df).to_string(index=False)
        return "\n".join(str(r) for r in result)
    return str(result)


def _format_list_result(df: pd.DataFrame) -> str:
    """DataFrame 명단 결과를 LLM 우회로 직접 포맷."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return "조회된 데이터가 없습니다."
    warning = _mask_warning(df)
    display = _display_df(df)
    header = f"총 {len(display)}건\n"
    if "source" in display.columns:
        try:
            sort_cols = ["source"]
            if "성명" in display.columns:
                sort_cols.append("성명")
            elif "이름" in display.columns:
                sort_cols.append("이름")
            display = display.sort_values(by=sort_cols)
        except Exception:
            pass
    return warning + header + display.to_string(index=False)


def _format_scalar_result(result: object, question: str) -> str:
    """int/float/str/dict scalar를 LLM 없이 자연스러운 문장으로 포맷."""
    if isinstance(result, dict) and result.get("type") == "aggregation_notice":
        return str(result.get("message") or "집계 결과를 확인할 수 없습니다.")
    if isinstance(result, dict) and result.get("type") == "aggregation":
        return _format_aggregation_payload(result)
    if isinstance(result, dict) and result.get("type") == "amount":
        return _format_amount_payload(result)
    if hasattr(result, "item"):
        result = result.item()
    if isinstance(result, int):
        if _AGG_COUNT.search(question):
            return f"총 {result}명입니다."
        return str(result)
    if isinstance(result, float):
        if result == int(result):
            return _format_scalar_result(int(result), question)
        return _format_number(result)
    if isinstance(result, str):
        if re.search(r"\d+만원", result):
            return f"금액은 {result}입니다."
        return result
    return str(result)

# ---------------------------------------------------------------------------
# 구조적 보강: 금액/기관/마스킹 답변 템플릿 일반화
# ---------------------------------------------------------------------------
def _amount_values_from_df(df: pd.DataFrame, col: str) -> list[float]:
    return money_values(df, col)


def _format_payment_breakdown(values: list[float]) -> str:
    """Format installment amounts without hard-coding any person or amount.

    Small result sets keep the original payment order.  Larger result sets are
    compressed by identical amount so the answer does not become excessively long.
    """
    if not values:
        return ""

    if len(values) <= 5:
        joined = ", ".join(f"{_format_number(v)}원" for v in values)
        return f"총 {len(values)}회에 걸쳐 {joined}으로 납부했습니다."

    counts: dict[float, int] = {}
    order: list[float] = []
    for value in values:
        key = float(value)
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1

    parts: list[str] = []
    for value in order:
        count = counts[value]
        if count == 1:
            parts.append(f"{_format_number(value)}원 1회")
        else:
            parts.append(f"{_format_number(value)}원씩 {count}회")
    return f"총 {len(values)}회에 걸쳐 " + ", ".join(parts) + " 납부했습니다."


def _clean_identity_value(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    return re.sub(r"\s+", "", text)


def _representative_name(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    for col in ("기관명", "표시명", "성명", "이름", "성명_검색키"):
        if col in df.columns:
            vals = [str(v).strip() for v in df[col].dropna().tolist() if _clean_identity_value(v)]
            if vals:
                return vals[0]
    return ""


def _mask_warning(df: pd.DataFrame) -> str:
    if df is None or df.empty or "_매칭유형" not in df.columns:
        return ""
    kinds = set(df["_매칭유형"].dropna().astype(str))
    # 사용자가 마스킹 이름으로 직접 물은 경우에는 경고하지 않는다.
    if "masked_candidate_match" in kinds:
        qnames = [str(v).strip() for v in df.get("_질문이름", pd.Series(dtype=str)).dropna().astype(str).tolist() if str(v).strip()]
        qname = qnames[0] if qnames else "입력한 이름"
        pattern_vals = [str(v).strip() for v in df.get("_질문마스킹패턴", pd.Series(dtype=str)).dropna().astype(str).tolist() if str(v).strip()]
        if pattern_vals:
            masked = pattern_vals[0]
        else:
            masked = _representative_name(df)
        return f"문서에는 해당 항목이 마스킹된 이름으로만 확인됩니다. 동일 인물로 확정할 수는 없지만, {masked} 항목의 "
    return ""


# ---------------------------------------------------------------------------
# 발행번호/기수/이름을 함께 사용한 엔터티 분리 포맷
# - 동일 이름이라도 발행번호 또는 기수가 다르면 별도 항목으로 출력한다.
# - 같은 발행번호 + 같은 이름 + 같은 기수의 여러 행만 분할 납부로 합산한다.
# ---------------------------------------------------------------------------
_FORMAT_IDENTIFIER_HINTS = (
    "id", "번호", "식별", "관리번호", "발행번호", "발급번호", "접수번호", "등록번호", "문서번호",
)


def _fmt_identifier_norm(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"[^0-9A-Z가-힣]", "", text)


def _find_identifier_col(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    preferred = ("발행번호", "발급번호", "접수번호", "관리번호", "식별번호", "등록번호", "문서번호", "ID", "id")
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        name = re.sub(r"\s+", "", str(col)).lower()
        if str(col).startswith("_") or name in {"rowuid", "personcandidatekey"}:
            continue
        if name == "id" or any(h in name for h in _FORMAT_IDENTIFIER_HINTS if h != "id"):
            return str(col)
    return ""


def _cohort_value(row: pd.Series) -> str:
    for col in ("기수", "회차", "기수_원문", "cohort_from_name"):
        if col not in row.index:
            continue
        text = _clean_identity_value(row.get(col))
        match = re.search(r"\d{1,3}", text)
        if match:
            return str(int(match.group()))
    return ""


def _row_identity_name(row: pd.Series) -> str:
    for col in ("성명_검색키", "기관명", "표시명", "성명", "이름", "성명_원문"):
        if col in row.index:
            value = str(row.get(col) or "").strip()
            if _clean_identity_value(value):
                return value
    return ""


def _entity_groups(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    id_col = _find_identifier_col(df)
    order: list[tuple[str, str, str]] = []
    buckets: dict[tuple[str, str, str], list[Any]] = {}
    names: dict[tuple[str, str, str], str] = {}
    for idx, row in df.iterrows():
        identifier = str(row.get(id_col) or "").strip() if id_col else ""
        if identifier.lower() in {"nan", "none", "null"}:
            identifier = ""
        name = _row_identity_name(row)
        cohort = _cohort_value(row)
        key = (_fmt_identifier_norm(identifier), _clean_identity_value(name), cohort)
        if key not in buckets:
            buckets[key] = []
            names[key] = name
            order.append(key)
        buckets[key].append(idx)
    groups: list[dict[str, Any]] = []
    for key in order:
        rows = df.loc[buckets[key]].copy()
        groups.append({
            "identifier": str(rows[id_col].iloc[0]).strip() if id_col and id_col in rows.columns else "",
            "name": names[key],
            "cohort": key[2],
            "rows": rows,
        })
    return groups


def _group_label(group: dict[str, Any]) -> str:
    name = str(group.get("name") or "").strip()
    cohort = str(group.get("cohort") or "").strip()
    identifier = str(group.get("identifier") or "").strip()
    base = name
    if cohort and not re.search(rf"(?<!\d){re.escape(cohort)}\s*회", base):
        base = f"{cohort}회 {base}".strip()
    if identifier:
        return f"{base} ({identifier})" if base else identifier
    return base or "식별 정보 없음"


def _question_mentions_group_identifier(question: str, group: dict[str, Any]) -> bool:
    identifier = _fmt_identifier_norm(group.get("identifier"))
    return bool(identifier and identifier in _fmt_identifier_norm(question))


def _group_is_organization(group: dict[str, Any]) -> bool:
    rows = group.get("rows")
    if not isinstance(rows, pd.DataFrame) or rows.empty:
        return False
    if "entity_type" in rows.columns and rows["entity_type"].astype(str).str.contains(
        "organization|department", case=False, regex=True, na=False
    ).any():
        return True
    if "성명_검색키" in rows.columns:
        return rows["성명_검색키"].astype(str).map(_clean_identity_value).eq("").all()
    return False


def _format_multi_entity_amount(groups: list[dict[str, Any]], col: str) -> str:
    ids = {str(g.get("identifier") or "").strip() for g in groups if str(g.get("identifier") or "").strip()}
    if len(ids) == 1:
        heading = f"{next(iter(ids))}에서 {len(groups)}개 항목이 조회되었습니다."
    else:
        heading = f"동일한 이름으로 {len(groups)}개 발행번호 항목이 조회되었습니다."
    lines: list[str] = []
    for group in groups:
        rows = group["rows"]
        values = _amount_values_from_df(rows, col)
        label = _group_label(group)
        if len(values) > 1:
            lines.append(
                f"- {label}: 합계 {_format_number(sum(values))}원; {_format_payment_breakdown(values)}"
            )
        elif values:
            lines.append(f"- {label}: {_format_number(values[0])}원")
        else:
            lines.append(f"- {label}: 금액 정보 없음")
    return heading + "\n" + "\n".join(lines)


def _format_dataframe_for_amount_question(df: pd.DataFrame, question: str) -> str | None:
    amount_selection = resolve_amount_column(df, question)
    if not amount_selection.candidates or not _AMOUNT_QUESTION_RE.search(question):
        return None
    if amount_selection.selected is None:
        return amount_column_clarification(amount_selection.candidates)
    col = amount_selection.selected
    groups = _entity_groups(df)
    if not groups:
        display = _display_df(df)
        return f"총 {len(display)}건\n" + display.to_string(index=False)
    if len(groups) > 1:
        return _mask_warning(df) + _format_multi_entity_amount(groups, col)

    group = groups[0]
    rows = group["rows"]
    values = _amount_values_from_df(rows, col)
    if not values:
        display = _display_df(df)
        return f"총 {len(display)}건\n" + display.to_string(index=False)

    name = str(group.get("name") or "").strip()
    label = _group_label(group)
    use_identifier_label = _question_mentions_group_identifier(question, group)
    if use_identifier_label:
        prefix = f"{label}의 "
    elif _group_is_organization(group) and name:
        prefix = f"{name}의 "
    else:
        prefix = ""

    explicit_sum = _SUM_WORD_RE.search(question) is not None
    if len(values) > 1 or explicit_sum:
        answer = (
            f"{prefix}{col} 합계는 {_format_number(sum(values))}원입니다. "
            f"{_format_payment_breakdown(values)}"
        )
    else:
        answer = f"{prefix}{col}은 {_format_number(values[0])}입니다."
    return _mask_warning(df) + answer


def _format_dataframe_result_for_question(df: pd.DataFrame, question: str) -> str:
    if df is None or df.empty:
        return "조회된 데이터가 없습니다."
    amount_answer = _format_dataframe_for_amount_question(df, question)
    if amount_answer:
        return amount_answer
    return _format_list_result(df)
