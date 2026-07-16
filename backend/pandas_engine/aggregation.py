from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from pandas_engine.money import money_series
from utils.table_parser import AMOUNT_COL_KEYWORDS, normalize_person_name
from utils.semantic_schema import semantic_columns


_AGG_COUNT = re.compile(
    r"몇\s*명|몇명|총\s*인원|전체\s*인원|인원수|인원은|명이야|명인가|"
    r"사람.{0,8}(?:수|몇\s*명)|몇\s*건|건수|총\s*몇\s*건|"
    r"(?:기록|내역|지급|출연).{0,8}(?:몇\s*건|건수|횟수)",
    re.IGNORECASE,
)
_AGG_SUM_EXPLICIT = re.compile(
    r"총\s*(?:액|금액|합계)|전체\s*(?:금액|합계)|합계\s*(?:금액|액)?|합산|"
    r"(?:모두|모든|전부|다).{0,15}?(?:합하면|합한|더하면|더한|합산|합친)|"
    r"(?:합하면|합한|더하면|더한|합산하면|합친).{0,12}?(?:얼마|금액|값)|"
    r"총\s*(?:얼마(?:야|인지|니|인가)?|얼만지)|모두\s*얼마(?:야|인지|니|인가)?",
    re.IGNORECASE,
)
_AGG_CUMULATIVE = re.compile(
    r"누적.{0,10}?(?:액|합계|금액|출연금|기부금|후원금|장학금|지원금|수혜금)",
    re.IGNORECASE,
)
_AGG_SUM = re.compile(
    rf"(?:{_AGG_SUM_EXPLICIT.pattern})|(?:{_AGG_CUMULATIVE.pattern})",
    re.IGNORECASE,
)
_AGG_AVG = re.compile(
    r"평균(?:값|\s*금액)?|평균적으로|산술\s*평균|"
    r"(?:금액|돈|출연금|기부금|지급액).{0,10}?평균",
    re.IGNORECASE,
)
_AGG_MEDIAN = re.compile(
    r"중앙값|중간값|중앙\s*금액|정중앙|"
    r"(?:금액|돈|출연금|기부금|지급액).{0,10}?(?:중앙|중간)\s*값",
    re.IGNORECASE,
)
_AGG_MODE = re.compile(
    r"최빈값|(?:가장|제일).{0,12}?흔한|"
    r"(?:가장|제일).{0,15}?(?:자주|많이).{0,10}?(?:나온|등장한)|"
    r"빈도.{0,12}?(?:가장|제일)?.{0,8}?높은",
    re.IGNORECASE,
)
_RANK_MAX = re.compile(
    r"(?:가장|제일).{0,15}?(?:많이|많은|높은|큰)",
    re.IGNORECASE,
)
_RANK_MIN = re.compile(
    r"(?:가장|제일).{0,15}?(?:적게|적은|낮은|작은|덜)",
    re.IGNORECASE,
)
_AGG_MAX = re.compile(
    r"최댓값|최대(?:값|액|\s*금액)?|최고(?:액|\s*금액)?|"
    rf"(?:{_RANK_MAX.pattern})|상위\s*\d+|1\s*위",
    re.IGNORECASE,
)
_AGG_MIN = re.compile(
    r"최솟값|최소(?:값|액|\s*금액)?|최저(?:액|\s*금액)?|"
    rf"(?:{_RANK_MIN.pattern})|하위\s*\d+",
    re.IGNORECASE,
)
_AGG_PER = re.compile(
    r"1\s*인당|인당|한\s*(?:사람|명|학생)당|"
    r"(?:학생|수혜자|기부자|후원자|출연자).{0,10}?한\s*(?:사람|명).{0,10}?(?:평균|얼마)",
    re.IGNORECASE,
)

_ROW_COUNT_RE = re.compile(r"몇\s*건|건수|기록|횟수|회수", re.IGNORECASE)
_PERSON_RE = re.compile(
    r"누가|누구|사람|개인|학생|수혜자|기부자|후원자|출연자|납부자|\d+\s*명",
    re.IGNORECASE,
)
_SINGLE_EVENT_RE = re.compile(
    r"한\s*번에|한\s*건|1\s*회에|개별\s*(?:건|납부|출연)|건당|회당",
    re.IGNORECASE,
)
_TOP_N_RE = re.compile(r"(?:상위|최고)\s*(\d+)\s*(?:명|개|건)?", re.IGNORECASE)
_BOTTOM_N_RE = re.compile(r"(?:하위|최저)\s*(\d+)\s*(?:명|개|건)?", re.IGNORECASE)


@dataclass(frozen=True)
class AggregationIntent:
    operation: str
    target: str = "value"
    top_n: int = 1
    count_unit: str = ""


@dataclass(frozen=True)
class AmountColumnSelection:
    candidates: tuple[str, ...]
    selected: str | None


def amount_column_candidates(df: pd.DataFrame) -> list[str]:
    """공통 스키마와 실제 헤더에서 금액 컬럼 후보를 찾는다."""
    mapped = semantic_columns(df, concept="measure", data_type="money")
    if mapped:
        return mapped
    return [
        str(column)
        for column in df.columns
        if any(keyword in str(column) for keyword in AMOUNT_COL_KEYWORDS)
    ]


def _comparison_tokens(value: object) -> list[str]:
    """문서별 단어 목록 없이 질문·헤더의 실제 구성 토큰을 만든다."""
    text = str(value or "").casefold().replace("_", " ")
    return [
        token
        for token in re.findall(r"[0-9]+[a-z가-힣]+|[a-z]+[0-9]+|[a-z가-힣]+|[0-9]+", text)
        if len(token) >= 2
    ]


def _amount_column_score(question: str, column: str) -> int:
    question_tokens = _comparison_tokens(question)
    column_tokens = _comparison_tokens(column)
    score = 0
    for question_token in question_tokens:
        matches = [
            min(len(question_token), len(column_token))
            for column_token in column_tokens
            if question_token in column_token or column_token in question_token
        ]
        score += max(matches, default=0)
    return score


def resolve_amount_column(df: pd.DataFrame, question: str) -> AmountColumnSelection:
    """실제 후보 헤더와 질문이 명확히 연결될 때만 금액 컬럼을 선택한다."""
    candidates = tuple(amount_column_candidates(df))
    if len(candidates) == 1:
        return AmountColumnSelection(candidates, candidates[0])
    if not candidates:
        return AmountColumnSelection(candidates, None)

    ranked = sorted(
        ((_amount_column_score(question, column), index, column)
         for index, column in enumerate(candidates)),
        key=lambda item: (-item[0], item[1]),
    )
    best_score, _, best_column = ranked[0]
    second_score = ranked[1][0]
    # 일반적인 공통어 하나가 우연히 더 맞는 정도로는 자동 선택하지 않는다.
    selected = best_column if best_score >= 2 and best_score - second_score >= 2 else None
    return AmountColumnSelection(candidates, selected)


def amount_column_clarification(candidates: tuple[str, ...] | list[str]) -> str:
    labels = ", ".join(str(column) for column in candidates)
    return f"금액 항목이 여러 개입니다. 다음 중 계산할 항목을 질문에 포함해 주세요: {labels}"


def detect_aggregation_intents(question: str) -> list[AggregationIntent]:
    """질문에 포함된 모든 기본 통계 연산을 의미 충돌 없이 반환한다."""
    text = str(question or "")
    if not text.strip():
        return []

    intents: list[AggregationIntent] = []

    def add(intent: AggregationIntent) -> None:
        signature = (intent.operation, intent.target, intent.top_n, intent.count_unit)
        if all(
            (item.operation, item.target, item.top_n, item.count_unit) != signature
            for item in intents
        ):
            intents.append(intent)

    if _AGG_COUNT.search(text):
        unit = "rows" if _ROW_COUNT_RE.search(text) else "people"
        add(AggregationIntent("count", count_unit=unit))
    if _AGG_MEDIAN.search(text):
        add(AggregationIntent("median"))

    mode_matched = bool(_AGG_MODE.search(text))
    if mode_matched:
        add(AggregationIntent("mode"))

    per_capita_matched = bool(_AGG_PER.search(text))
    if per_capita_matched:
        add(AggregationIntent("per_capita", target="person_total"))
    elif _AGG_AVG.search(text):
        add(AggregationIntent("mean"))

    top_match = _TOP_N_RE.search(text)
    bottom_match = _BOTTOM_N_RE.search(text)
    max_matched = bool(_AGG_MAX.search(text) or top_match)
    min_matched = bool(_AGG_MIN.search(text) or bottom_match)
    # "가장 많이 등장한 값"은 최댓값이 아니라 최빈값이다.
    if max_matched and not mode_matched:
        top_n = max(1, int(top_match.group(1))) if top_match else 1
        if _PERSON_RE.search(text):
            target = "row" if _SINGLE_EVENT_RE.search(text) else "person_total"
        else:
            target = "value"
        add(AggregationIntent("max", target=target, top_n=top_n))
    if min_matched:
        top_n = max(1, int(bottom_match.group(1))) if bottom_match else 1
        if _PERSON_RE.search(text):
            target = "row" if _SINGLE_EVENT_RE.search(text) else "person_total"
        else:
            target = "value"
        add(AggregationIntent("min", target=target, top_n=top_n))

    # "누적 금액 상위"의 누적은 그룹 기준이며 별도의 합계 요청이 아니다.
    explicit_sum = bool(_AGG_SUM_EXPLICIT.search(text))
    cumulative_sum = bool(_AGG_CUMULATIVE.search(text))
    if explicit_sum or (cumulative_sum and not (max_matched or min_matched)):
        add(AggregationIntent("sum"))
    return intents


def detect_aggregation_intent(question: str) -> AggregationIntent | None:
    """단일 실행 경로용 대표 집계 의도를 반환한다.

    복수 연산 질문은 Guard가 먼저 안내 대상으로 처리한다.
    """
    intents = detect_aggregation_intents(question)
    return intents[0] if intents else None


def is_aggregation_question(question: str) -> bool:
    return detect_aggregation_intent(question) is not None


def aggregation_notice(message: str, *, kind: str = "error") -> dict[str, object]:
    return {"type": "aggregation_notice", "kind": kind, "message": message}


def _amount_series(
    df: pd.DataFrame,
    amount_column: str,
) -> tuple[pd.Series, str, int]:
    numeric = money_series(df, amount_column)
    label = str(amount_column)
    return numeric, label, int(numeric.isna().sum())


def _identity_series(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    display = pd.Series("", index=df.index, dtype=object)
    for col in ("표시명", "이름", "성명", "기관명", "성명_원문"):
        if col not in df.columns:
            continue
        values = df[col].fillna("").astype(str).str.strip()
        empty = display.astype(str).str.strip().isin(["", "None", "nan", "NaN"])
        display.loc[empty & values.ne("")] = values.loc[empty & values.ne("")]

    key = pd.Series("", index=df.index, dtype=object)
    # 같은 마스킹 이름이라도 기수/문맥이 다른 사람은 기존 후보 키로 분리한다.
    for col in ("person_candidate_key", "성명_검색키", "기관명", "표시명", "이름", "성명"):
        if col not in df.columns:
            continue
        values = df[col].fillna("").astype(str).map(normalize_person_name)
        empty = key.astype(str).str.strip().eq("")
        key.loc[empty & values.ne("")] = values.loc[empty & values.ne("")]
    return key, display


def _person_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "entity_type" not in df.columns:
        return df
    entity_types = df["entity_type"].astype(str)
    person_mask = entity_types.str.contains(
        r"^person(?:_|$)", case=False, regex=True, na=False
    )
    known_mask = entity_types.str.contains(
        r"^(?:person|organization|department)(?:_|$)", case=False, regex=True, na=False
    )
    # 구형 데이터처럼 전부 unknown이면 이름 컬럼을 fallback으로 사용한다. 반대로
    # 단체로 분류된 행만 있다면 이를 사람으로 되돌리지 않는다.
    return df[person_mask].copy() if known_mask.any() else df


def aggregate_rows(
    rows: pd.DataFrame,
    sources: list[str],
    intent: AggregationIntent,
    *,
    question: str = "",
    count_valid_name_rows: Callable[[pd.DataFrame], int],
) -> tuple[object, list[str]]:
    """이미 선택·필터링된 행에 기본 통계를 적용한다.

    데이터 저장소와 결합되지 않도록 금액 정규화와 레거시 이름 카운트 함수는
    호출자가 주입한다. 이 모듈은 집계 계산과 결과 payload만 책임진다.
    """
    if rows is None or rows.empty:
        return aggregation_notice("조건에 맞는 데이터가 없습니다."), []

    working = rows.copy()
    if intent.count_unit == "people" or intent.target == "person_total":
        working = _person_rows(working)

    key, display = _identity_series(working)

    if intent.operation == "count":
        if intent.count_unit == "rows":
            value, unit = int(len(working)), "건"
        else:
            valid = key.astype(str).str.strip().ne("")
            value = int(key[valid].nunique()) if valid.any() else count_valid_name_rows(working)
            unit = "명"
        return {
            "type": "aggregation",
            "operation": "count",
            "value": value,
            "unit": unit,
            "matched_rows": int(len(working)),
            "sources": list(dict.fromkeys(sources)),
        }, sources

    amount_selection = resolve_amount_column(working, question)
    if not amount_selection.candidates:
        return aggregation_notice("집계할 수 있는 금액 컬럼이 없습니다."), sources
    if amount_selection.selected is None:
        return aggregation_notice(
            amount_column_clarification(amount_selection.candidates),
            kind="clarification",
        ), sources

    numeric, label, invalid_rows = _amount_series(
        working,
        amount_selection.selected,
    )
    valid = numeric.dropna()
    if valid.empty:
        return aggregation_notice("집계할 수 있는 금액 데이터가 없습니다."), sources

    common = {
        "type": "aggregation",
        "operation": intent.operation,
        "label": label,
        "matched_rows": int(len(working)),
        "valid_rows": int(valid.size),
        "invalid_rows": invalid_rows,
        "sources": list(dict.fromkeys(sources)),
    }
    scalar_operations = {
        "sum": valid.sum,
        "mean": valid.mean,
        "median": valid.median,
    }
    if intent.operation in scalar_operations:
        return {**common, "value": float(scalar_operations[intent.operation]())}, sources
    if intent.operation == "mode":
        return {**common, "values": [float(value) for value in valid.mode().tolist()]}, sources

    valid_identity = key.astype(str).str.strip().ne("")
    if intent.operation == "per_capita" or intent.target == "person_total":
        grouped_input = pd.DataFrame({
            "key": key[valid_identity],
            "name": display[valid_identity],
            "amount": numeric[valid_identity],
        }).dropna(subset=["amount"])
        if grouped_input.empty:
            return aggregation_notice("사람별로 집계할 이름 정보가 없습니다."), sources
        grouped = grouped_input.groupby("key", as_index=False).agg(
            name=("name", "first"), value=("amount", "sum")
        )
        duplicate_names = grouped["name"].astype(str).duplicated(keep=False)
        for idx in grouped[duplicate_names].index:
            raw_key = str(grouped.at[idx, "key"])
            suffix = raw_key.rsplit("::", 1)[1] if "::" in raw_key else ""
            if suffix:
                grouped.at[idx, "name"] = f"{grouped.at[idx, 'name']} ({suffix}기)"
        if intent.operation == "per_capita":
            return {
                **common,
                "value": float(grouped["value"].mean()),
                "people_count": int(len(grouped)),
            }, sources

        ascending = intent.operation == "min"
        grouped = grouped.sort_values("value", ascending=ascending, kind="stable")
        if intent.top_n == 1:
            extreme = grouped["value"].min() if ascending else grouped["value"].max()
            selected = grouped[grouped["value"] == extreme]
        else:
            selected = grouped.head(intent.top_n)
        subjects = [
            {"name": str(row["name"] or row["key"]), "value": float(row["value"])}
            for _, row in selected.iterrows()
        ]
        return {**common, "scope": "person_total", "subjects": subjects, "top_n": intent.top_n}, sources

    extreme = valid.min() if intent.operation == "min" else valid.max()
    if intent.target == "row":
        subjects: list[dict[str, object]] = []
        for idx in numeric[numeric == extreme].index:
            name = str(display.get(idx, "") or "").strip()
            item: dict[str, object] = {"name": name or "이름 정보 없음", "value": float(extreme)}
            for col in ("발행번호", "출연일자", "지급일", "날짜"):
                if col not in working.columns:
                    continue
                value = str(working.at[idx, col] or "").strip()
                if value and value.lower() not in {"nan", "none"}:
                    item[col] = value
            subjects.append(item)
        return {**common, "scope": "row", "value": float(extreme), "subjects": subjects}, sources
    return {**common, "value": float(extreme), "scope": "value"}, sources
