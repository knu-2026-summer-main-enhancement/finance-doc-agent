from __future__ import annotations

import logging
import re

import pandas as pd

from datastore.state import _df_namespace, _df_sources, _df_labels
from datastore.scope import scoped_mapping, source_scope_active
from pandas_engine.aggregation import (
    AggregationIntent,
    aggregate_rows,
    aggregation_notice as _aggregation_notice,
    amount_column_clarification,
    detect_aggregation_intents,
    resolve_amount_column,
)
from pandas_engine.money import money_series
from pandas_engine.date_filter import DateFilter, apply_date_filter
from utils.table_parser import normalize_person_name, make_mask_pattern, is_masked_name, AMOUNT_COL_KEYWORDS
from utils.semantic_schema import semantic_columns

logger = logging.getLogger("uvicorn.error")


def _scoped_dataframes() -> dict[str, pd.DataFrame]:
    return scoped_mapping(_df_namespace, _df_sources)

# ---------------------------------------------------------------------------
# 이름 검색용 상수
# ---------------------------------------------------------------------------
_NAME_COLS     = ("성명", "이름", "학생명", "수혜자명", "학생이름", "수혜자", "명단", "학생", "이_름", "성_명")
_NAME_COLS_SET = frozenset(_NAME_COLS)
_AMOUNT_COLS = AMOUNT_COL_KEYWORDS

_NON_NAME_WORDS = frozenset([
    "장학금", "장학", "전기과", "건축과", "기계과", "화학과", "컴퓨터",
    "학과", "학년", "학생", "신입생", "재학생", "대상자", "수혜자",
    "성적", "우수자", "금액", "명단", "목록", "정보", "대학교",
    "이상", "이하", "미만", "해당", "지급", "기준", "선발",
    "알려줘", "알려주", "주세요", "해줘", "계열", "바이오", "화학",
    "동문장학", "동문회", "실습품", "확인서", "기능대회", "지원금",
    "공무원", "검도부", "관악부", "운동부", "축구부",
    "동기회", "총동문회", "대공동문회", "재단", "협회", "장학회", "주식회사", "중공업", "회사",
    "학년말", "성적우수", "총인원", "총금액", "얼마야", "얼마",
    "수령자", "수령확인", "출전선수", "학교운동", "스마트공간",
    "출연자", "기부자", "후원자", "발행번호", "발급번호", "접수번호",
])

_KR_PARTICLES = frozenset("의이가을를은는에도로과와며서")

# source-label 검색용 제외 단어
_SOURCE_STOP_WORDS = frozenset([
    "학생", "이름", "알려줘", "알려주", "주세요", "해줘", "누구", "누구야",
    "몇명", "인원", "총인원", "총금액", "얼마야", "얼마",
])

_AMOUNT_IN_FILENAME_RE = re.compile(r"(\d[\d,]*)만원")
_MONTH_IN_FILENAME_RE  = re.compile(r"(\d{1,2})월")

_MASKED_NAME_TOKEN_RE = re.compile(r"[가-힣]\s*[*＊○●Oo0xX×]\s*[가-힣*＊○●Oo0xX×]{0,3}")
_PERSON_LIST_RE = re.compile(r"사람|개인|학생|수혜자|기부자|후원자|출연자|납부자")
_INTERNAL_COLS = {
    "성명_원문", "성명_검색키", "성명_마스킹패턴", "성명_마스킹여부",
    "row_uid", "person_candidate_key", "_row_index", "_row_context",
    "_매칭유형", "_질문이름",
}
_PERSON_IDENTITY_COLUMNS = (
    "person_candidate_key", "성명_검색키", "표시명", "이름", "성명", "성명_원문",
)
_PERSON_LIST_COLUMNS = (
    "source", "기수", "학과", "전공", "학년", "표시명", "이름", "성명",
)


def has_explicit_masked_name(question: str) -> bool:
    """Return whether the user explicitly supplied a supported masked name."""
    return bool(_MASKED_NAME_TOKEN_RE.search(str(question or "")))


def _is_internal_col(col: str) -> bool:
    return col in _INTERNAL_COLS or str(col).startswith("_")


def _extract_name_candidates(question: str) -> list[str]:
    """질문에서 실명/마스킹 이름 후보를 순서 보존으로 추출."""
    seen: set[str] = set()
    candidates: list[str] = []

    for w in _MASKED_NAME_TOKEN_RE.findall(question):
        clean = normalize_person_name(_strip_kr_particle(w))
        if clean and clean not in _NON_NAME_WORDS and clean not in seen:
            candidates.append(clean)
            seen.add(clean)

    for w in re.findall(r"[가-힣]{2,5}", question):
        clean = _strip_kr_particle(w)
        if clean not in _NON_NAME_WORDS and clean not in seen:
            candidates.append(clean)
            seen.add(clean)
    return candidates


def _series_norm(series: pd.Series) -> pd.Series:
    return series.astype(str).map(normalize_person_name)


def _has_valid_amount(rows: pd.DataFrame, amount_cols: list[str]) -> pd.DataFrame:
    if not amount_cols or rows.empty:
        return rows
    masks = [money_series(rows, column).notna() for column in amount_cols if column in rows]
    if not masks:
        return rows.iloc[0:0]
    valid = pd.concat(masks, axis=1).any(axis=1)
    return rows[valid]


_AMOUNT_QUESTION_RE = re.compile(r"얼마|금액|총액|합계|출연금|지급액|장학금|지원금|수혜금|후원금|기부금")
def _is_amount_question(question: str) -> bool:
    return bool(_AMOUNT_QUESTION_RE.search(question))


def _find_amount_cols(df: pd.DataFrame) -> list[str]:
    mapped = semantic_columns(df, concept="measure", data_type="money")
    if mapped:
        return mapped
    return [c for c in df.columns if any(k in str(c) for k in _AMOUNT_COLS)]


def _amount_payload(label: str, value: object, agg: str | None = None) -> dict[str, object]:
    return {"type": "amount", "label": label, "value": value, "agg": agg or ""}


def _source_relevance(src: str, context_words: set[str]) -> int:
    return sum(1 for w in context_words if w and w in src)


def _strip_kr_particle(word: str) -> str:
    if len(word) >= 3 and word[-1] in _KR_PARTICLES:
        return word[:-1]
    return word


def _count_valid_name_rows(df: pd.DataFrame) -> int:
    """이름 컬럼이 있으면 비어있지 않은 행만 카운트, 없으면 전체 행 수."""
    name_col = next((c for c in df.columns if c in _NAME_COLS_SET), None)
    if name_col:
        valid = df[name_col].astype(str).str.strip()
        cnt = int((~valid.isin(["", "None", "nan", "NaN"])).sum())
        return cnt if cnt > 0 else len(df)
    return len(df)


def _expand_명단_column(df: pd.DataFrame) -> pd.DataFrame:
    """df2/df3처럼 '명단' 컬럼에 이름이 뭉쳐 있는 경우 행을 개별 이름으로 분리한다.

    원본 형식: "1반 22번 최성욱 2반 22번 추승민 ..."
    결과: 학과·성명·생년월일 컬럼으로 펼쳐진 DataFrame
    """
    if '명단' not in df.columns:
        return df
    rows: list[dict] = []
    for _, row in df.iterrows():
        명단_text  = str(row.get('명단', ''))
        생년월일_text = str(row.get('생년월일', ''))
        names     = re.findall(r'\d+반\s*\d+번\s*([가-힣]{2,4})', 명단_text)
        birthdates = re.findall(r'\d{6}', 생년월일_text)
        if names:
            for i, name in enumerate(names):
                rows.append({
                    '학과':   str(row.get('학과', '')),
                    '성명':   name,
                    '생년월일': birthdates[i] if i < len(birthdates) else '',
                })
        else:
            rows.append(row.to_dict())
    return pd.DataFrame(rows) if rows else df


def _find_filter_conditions(question: str) -> dict[str, list[tuple[str, str]]]:
    """질문 키워드를 실제 DataFrame 셀 값과 대조해 {alias: [(col, value), ...]} 반환."""
    dataframes = _scoped_dataframes()
    if not dataframes:
        return {}

    candidates: list[str] = []
    seen: set[str] = set()
    for w in re.findall(r"[가-힣]{2,10}", question):
        stripped = _strip_kr_particle(w)
        for cand in dict.fromkeys([w, stripped]):
            if cand in seen or len(cand) < 2:
                continue
            # ~과 학과명은 NON_NAME_WORDS 제외 대상 (전기과, 친환경자동차과 등)
            if cand not in _NON_NAME_WORDS or (cand.endswith("과") and len(cand) >= 3):
                candidates.append(cand)
                seen.add(cand)

    for m in re.findall(r"20\d{2}|[1-4]학년", question):
        if m not in seen:
            candidates.append(m)
            seen.add(m)

    if not candidates:
        return {}

    result: dict[str, list[tuple[str, str]]] = {}
    visited: set[tuple[str, str]] = set()

    for cand in candidates[:10]:
        for alias, df in dataframes.items():
            for col in df.columns:
                if _is_internal_col(col):
                    continue
                if (alias, col) in visited:
                    continue
                try:
                    if df[col].astype(str).str.contains(re.escape(cand), na=False).any():
                        result.setdefault(alias, []).append((col, cand))
                        visited.add((alias, col))
                        break
                except Exception:
                    continue

    return result


def _find_dfs_by_source_label(question: str) -> list[str]:
    """데이터 셀 매칭이 없을 때 소스명·레이블을 키워드로 검색해 관련 alias 목록 반환."""
    words: set[str] = set()
    for w in re.findall(r"[가-힣]{2,}|20\d{2}|\d+월|\d+분기", question):
        stripped = _strip_kr_particle(w)
        # 조사를 모두 제거한 어근까지 추가 ("상반기에서" → "상반기")
        fully = w
        while len(fully) >= 3 and fully[-1] in _KR_PARTICLES:
            fully = fully[:-1]
        for cand in dict.fromkeys([w, stripped, fully]):
            if cand not in _SOURCE_STOP_WORDS and len(cand) >= 2:
                words.add(cand)

    scored: list[tuple[str, int]] = []
    for alias in _scoped_dataframes():
        text = (_df_sources.get(alias, "") + " " + _df_labels.get(alias, ""))
        score = sum(1 for w in words if w in text)
        if score > 0:
            scored.append((alias, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [a for a, _ in scored]


def _find_value_locations(question: str) -> str:
    """_find_filter_conditions 결과를 LLM 프롬프트용 힌트 문자열로 변환."""
    conditions = _find_filter_conditions(question)
    if not conditions:
        return ""
    hints = [
        f"'{val}' → {alias}['{col}'] (파일: {_df_sources.get(alias, alias)})"
        for alias, cond_list in conditions.items()
        for col, val in cond_list
    ]
    return "데이터 위치 힌트 (질문 맥락에 맞는 DataFrame을 선택하세요):\n" + "\n".join(
        f"  {h}" for h in hints
    )


def _extract_total_from_source(alias: str) -> str | None:
    """소스 파일명에서 총액 정보 추출 (예: '-760만원.pdf' → '760만원')."""
    src = _df_sources.get(alias, "")
    m = _AMOUNT_IN_FILENAME_RE.search(src)
    return f"{m.group(1)}만원" if m else None


def _extract_month_from_source(source: str) -> str:
    """파일명에서 지출월 추출 (예: '3월' → '3월')."""
    m = _MONTH_IN_FILENAME_RE.search(source)
    return f"{m.group(1)}월" if m else ""


def _extract_recipient_from_dfs(aliases: list[str]) -> str:
    """DataFrame의 '지급처' 컬럼에서 대표 지급처명 추출."""
    for alias in aliases:
        df = _df_namespace.get(alias)
        if df is None:
            continue
        col = next((c for c in df.columns if "지급처" in c), None)
        if col:
            vals = df[col].dropna()
            vals = vals[vals.astype(str).str.strip().ne("")]
            if not vals.empty:
                return str(vals.iloc[0]).strip()
    return ""


def _query_pandas_direct_base(question: str) -> tuple[object, list[str]]:
    """LLM 코드 생성 없이 키워드 매핑으로 직접 pandas 조회."""
    conditions = _find_filter_conditions(question)
    year_in_q = re.search(r"20\d{2}", question)
    year_str   = year_in_q.group() if year_in_q else None

    def _extract_year_from_alias(alias: str) -> int:
        src = _df_sources.get(alias, "") + _df_labels.get(alias, "")
        years = re.findall(r"20(\d{2})", src)
        return max(int(y) for y in years) if years else 0

    _src_keywords = set(re.findall(r"[가-힣]{2,}", question))

    def _src_relevance(alias: str) -> int:
        src = _df_sources.get(alias, "") + " " + _df_labels.get(alias, "")
        return sum(1 for w in _src_keywords if w in src)

    def _pick_best_alias(aliases: list[str]) -> str:
        """1순위: 소스명 키워드 유사도, 2순위: 연도(질문 연도 → 최신), 3순위: 조건 수."""
        if year_str:
            year_matched = [
                a for a in aliases
                if year_str in (_df_sources.get(a, "") + _df_labels.get(a, ""))
            ]
            if year_matched:
                return max(year_matched, key=lambda a: (_src_relevance(a), len(conditions.get(a, []))))

        def _score(a: str) -> tuple[int, int, int]:
            return (_src_relevance(a), _extract_year_from_alias(a), len(conditions.get(a, [])))

        return max(aliases, key=_score)

    grade_m = re.search(r"([1-4])학년", question)

    def _apply_grade_filter(df: pd.DataFrame) -> pd.DataFrame:
        if not grade_m:
            return df
        grade_col = next((c for c in df.columns if "학년" in c), None)
        if grade_col:
            try:
                return df[df[grade_col].astype(str).str.contains(grade_m.group(1), na=False)]
            except Exception:
                pass
        return df

    if conditions:
        condition_sources = list(dict.fromkeys(
            _df_sources.get(alias, alias) for alias in conditions
        ))
        if len(condition_sources) > 1:
            return _aggregation_notice(
                "조건과 일치하는 기록이 여러 문서에 있습니다. 조회할 문서를 선택해주세요: "
                + ", ".join(condition_sources[:5]),
                kind="clarification",
            ), condition_sources
        best_alias = _pick_best_alias(list(conditions.keys()))
        df = _df_namespace[best_alias]

        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in conditions[best_alias]:
            mask &= df[col].astype(str).str.contains(re.escape(val), na=False)
        filtered = _apply_grade_filter(df[mask])

    else:
        # 소스명 기반 fallback
        src_aliases = _find_dfs_by_source_label(question)
        if not src_aliases:
            return None, []

        best_alias = _pick_best_alias(src_aliases) if year_str else src_aliases[0]
        df = _df_namespace[best_alias]
        filtered = _apply_grade_filter(df)

    if filtered.empty:
        return None, []

    source = _df_sources.get(best_alias, best_alias)

    # ── 단순 금액 질문: 실제 금액 컬럼명을 유지해 반환 ─────────────────────────
    if _is_amount_question(question):
        amount_selection = resolve_amount_column(filtered, question)
        if len(amount_selection.candidates) > 1 and amount_selection.selected is None:
            return _aggregation_notice(
                amount_column_clarification(amount_selection.candidates),
                kind="clarification",
            ), [source]
        if amount_selection.selected:
            amount_col = amount_selection.selected
            nums = money_series(filtered, amount_col).dropna()
            if not nums.empty:
                if len(nums) == 1 or len(filtered) == 1:
                    return _amount_payload(amount_col, float(nums.iloc[0]), None), [source]
        # 여러 행/여러 금액 컬럼이면 표로 반환해 formatter가 실제 컬럼명을 보여주게 한다.
        return filtered, [source]

    # 명단 컬럼이 있으면 개별 이름 행으로 변환
    return _expand_명단_column(filtered), [source]

# ---------------------------------------------------------------------------
# 구조화된 표 조회용 공통 규칙
# - 특정 이름/기관/금액을 코드에 박지 않는다.
# ---------------------------------------------------------------------------
def _lookup_norm(value: object) -> str:
    text = normalize_person_name(value)
    text = re.sub(r"\s+", "", str(text or ""))
    text = re.sub(r"[?？!！.,，。]", "", text)
    return text


def _source_year(source: str) -> int:
    years = re.findall(r"20\d{2}", str(source or ""))
    return max((int(y) for y in years), default=0)


def _select_best_source_rows(result: pd.DataFrame, question: str) -> pd.DataFrame:
    if result is None or result.empty or "source" not in result.columns:
        return result
    q_years = re.findall(r"20\d{2}", question)
    if q_years:
        year = q_years[-1]
        mask = result["source"].astype(str).str.contains(year, regex=False, na=False)
        if mask.any():
            return result[mask].copy()
    sources = list(result["source"].dropna().astype(str).unique())
    if len(sources) <= 1:
        return result
    # 문서 범위가 없는 질문에서 임의로 최신 문서를 선택하지 않는다. 여러 문서의
    # 동명이인은 호출부가 출처를 보여주거나 문서 선택을 요청할 수 있게 그대로 둔다.
    return result


def _masked_shape(value: object) -> str:
    key = normalize_person_name(value)
    return "".join("*" if ch == "*" else ("가" if "가" <= ch <= "힣" else ch) for ch in key)


# fuzzy를 좁게 유지해 최소 OCR 거리 후보만 남긴다.
def _masked_name_distance(query_mask: object, stored_mask: object) -> int | None:
    q = normalize_person_name(query_mask)
    s = normalize_person_name(stored_mask)
    if not q or not s or "*" not in q or "*" not in s:
        return None
    if len(q) != len(s) or _masked_shape(q) != _masked_shape(s) or q[0] != s[0]:
        return None
    total = 0
    diff_count = 0
    for qc, sc in zip(q, s):
        if qc == "*" or sc == "*":
            continue
        if qc != sc:
            diff_count += 1
            total += abs(ord(qc) - ord(sc))
    if diff_count == 0:
        return 0
    if diff_count > 1:
        return None
    return total


def _nearest_masked_rows(df: pd.DataFrame, series: pd.Series, key: str, max_distance: int = 180) -> pd.DataFrame:
    distances = series.map(lambda v: _masked_name_distance(key, v))
    valid = distances.dropna()
    if valid.empty:
        return df.iloc[0:0]
    min_d = int(valid.min())
    if min_d > max_distance:
        return df.iloc[0:0]
    return df[distances == min_d]


def _search_name_pandas_core(question: str) -> tuple[pd.DataFrame | None, list[str], bool]:
    candidates = _extract_name_candidates(question)
    if not candidates:
        return None, [], False

    context_words = {w for w in re.findall(r"[가-힣]{2,}|20\d{2}|\d+월", question)} - _NON_NAME_WORDS
    exact_matches: list[tuple[pd.DataFrame, str, int]] = []
    masked_direct_matches: list[tuple[pd.DataFrame, str, int]] = []
    masked_candidate_matches: list[tuple[pd.DataFrame, str, int]] = []
    fallback_matches: list[tuple[pd.DataFrame, str, int]] = []

    for var_name, df in _scoped_dataframes().items():
        name_col = next((c for c in df.columns if c in _NAME_COLS_SET), None)
        if name_col is None and "성명_검색키" not in df.columns:
            continue
        src = _df_sources.get(var_name, var_name)
        score = _source_relevance(src + " " + _df_labels.get(var_name, ""), context_words)
        amount_cols = _find_amount_cols(df)
        search_series = _series_norm(df["성명_검색키"]) if "성명_검색키" in df.columns else None
        pattern_series = _series_norm(df["성명_마스킹패턴"]) if "성명_마스킹패턴" in df.columns else None
        mask_flag = df["성명_마스킹여부"].astype(bool) if "성명_마스킹여부" in df.columns else pd.Series(False, index=df.index)

        for cand in candidates:
            search_key = normalize_person_name(cand)
            pattern_key = make_mask_pattern(cand)
            if not search_key:
                continue

            if search_series is not None:
                rows = df[search_series == search_key]
                rows = _has_valid_amount(rows, amount_cols)
                if not rows.empty:
                    rows = rows.copy()
                    rows["_매칭유형"] = "masked_direct_match" if is_masked_name(search_key) else "exact_match"
                    rows["_질문이름"] = cand
                    target = masked_direct_matches if is_masked_name(search_key) else exact_matches
                    target.append((rows, src, score))
                    continue

                if is_masked_name(search_key):
                    rows = _nearest_masked_rows(df, search_series, search_key)
                    rows = _has_valid_amount(rows, amount_cols)
                    if not rows.empty:
                        rows = rows.copy()
                        rows["_매칭유형"] = "masked_direct_match"
                        rows["_질문이름"] = cand
                        rows["_질문마스킹패턴"] = search_key
                        masked_direct_matches.append((rows, src, score))
                        continue

            if pattern_series is not None and pattern_key and pattern_key != search_key and not is_masked_name(search_key):
                rows = df[(pattern_series == pattern_key) & (mask_flag == True)]  # noqa: E712
                rows = _has_valid_amount(rows, amount_cols)
                if rows.empty:
                    rows = _nearest_masked_rows(df, pattern_series, pattern_key)
                    if not rows.empty and "성명_마스킹여부" in rows.columns:
                        rows = rows[rows["성명_마스킹여부"].astype(bool) == True]  # noqa: E712
                    rows = _has_valid_amount(rows, amount_cols)
                if not rows.empty:
                    rows = rows.copy()
                    rows["_매칭유형"] = "masked_candidate_match"
                    rows["_질문이름"] = cand
                    rows["_질문마스킹패턴"] = pattern_key
                    masked_candidate_matches.append((rows, src, score))
                    continue

            if search_series is None and name_col:
                try:
                    name_series = _series_norm(df[name_col])
                    rows = df[name_series.str.contains(search_key, na=False, regex=False)]
                except Exception:
                    continue
                rows = _has_valid_amount(rows, amount_cols)
                if not rows.empty:
                    rows = rows.copy()
                    rows["_매칭유형"] = "fallback_name_match"
                    rows["_질문이름"] = cand
                    fallback_matches.append((rows, src, score))
                    continue

    selected = exact_matches or masked_direct_matches or masked_candidate_matches or fallback_matches
    if not selected:
        return None, [], True
    selected.sort(key=lambda x: x[2], reverse=True)
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    for rows, src, _ in selected:
        rows = rows.copy()
        if "source" not in rows.columns:
            rows.insert(0, "source", src)
        frames.append(rows)
        if src not in sources:
            sources.append(src)
    result = pd.concat(frames, ignore_index=True)
    if "row_uid" in result.columns:
        result = result.drop_duplicates(subset=["row_uid"])
    else:
        result = result.drop_duplicates()
    result = _select_best_source_rows(result, question)
    sources = list(result.get("source", pd.Series(dtype=str)).dropna().astype(str).unique())
    logger.info("[NAME_SEARCH_STRUCT_NARROW] %d건 매칭 | sources=%s", len(result), sources[:5])
    return result, sources, True


# ---------------------------------------------------------------------------
# 검색 조건 보강: 질문에 포함된 기수와 기관/단체 부분명을 구조적으로 반영한다.
# 특정 사람명·단체명·금액은 사용하지 않는다.
# ---------------------------------------------------------------------------
_COHORT_QUERY_RE = re.compile(r"(?<!\d)(\d{1,3})\s*(?:회|기)(?!\d)")


def _question_cohort(question: str) -> str:
    match = _COHORT_QUERY_RE.search(str(question or ""))
    return match.group(1) if match else ""


def _cohort_key(value: object) -> str:
    text = str(value or "").strip()
    match = re.search(r"\d{1,3}", text)
    return str(int(match.group(0))) if match else ""


def _filter_rows_by_question_cohort(rows: pd.DataFrame, question: str) -> pd.DataFrame:
    """질문에 `N회` 또는 `N기`가 있을 때 결과를 같은 기수로 제한한다.

    사용자가 기수를 명시했다면 일치하지 않는 다른 기수 행으로 대체하지 않는다.
    """
    if rows is None or rows.empty:
        return rows
    cohort = _question_cohort(question)
    if not cohort:
        return rows

    masks: list[pd.Series] = []
    for col in ("기수", "회차", "기수_원문", "cohort_from_name"):
        if col in rows.columns:
            masks.append(rows[col].map(_cohort_key) == cohort)

    # 기관명/표시명 자체에 회차나 기수가 포함된 문서도 지원한다.
    for col in ("기관명", "표시명", "이름"):
        if col in rows.columns:
            masks.append(rows[col].astype(str).str.replace(r"\s+", "", regex=True).str.contains(
                rf"(?<!\d){re.escape(cohort)}(?:회|기)", regex=True, na=False
            ))

    if not masks:
        return rows.iloc[0:0].copy()
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined = combined | mask
    return rows[combined].copy()


def _find_cohort_exact(question: str) -> tuple[pd.DataFrame | None, list[str]]:
    cohort = _question_cohort(question)
    if not cohort:
        return None, []

    frames: list[pd.DataFrame] = []
    for alias, df in _scoped_dataframes().items():
        rows = _filter_rows_by_question_cohort(df, question)
        if rows.empty:
            continue
        src = _df_sources.get(alias, alias)
        rows = rows.copy()
        if "source" not in rows.columns:
            rows.insert(0, "source", src)
        rows["_매칭유형"] = "cohort_exact_match"
        frames.append(rows)

    if not frames:
        return None, []
    result = pd.concat(frames, ignore_index=True)
    if "row_uid" in result.columns:
        result = result.drop_duplicates(subset=["row_uid"])
    result = _select_best_source_rows(result, question)
    sources = list(result.get("source", pd.Series(dtype=str)).dropna().astype(str).unique())
    return result, sources


def _organization_display_series(df: pd.DataFrame) -> pd.Series:
    """Return one canonical display value for each organization-like row.

    기관명 파생 컬럼이 비어 있는 오래된/unknown 행도 원본 표시명·이름을
    보존해 검색할 수 있도록 한다. 사람으로 확정된 행은 아래 후보 마스크에서
    제외되므로 이 함수는 값 선택만 담당한다.
    """
    result = pd.Series("", index=df.index, dtype=object)
    for col in ("기관명", "표시명", "이름", "성명_원문"):
        if col not in df.columns:
            continue
        values = df[col].astype(str)
        valid = values.map(_lookup_norm).ne("")
        empty = result.astype(str).map(_lookup_norm).eq("")
        result.loc[empty & valid] = values.loc[empty & valid]
    return result


def _organization_candidate_mask(df: pd.DataFrame) -> pd.Series:
    """Select rows that can safely participate in organization/compound-name lookup.

    특정 단체명을 하드코딩하지 않는다. 대신 공통 파생 컬럼을 사용해
    `사람으로 확정된 행`을 제외하고, 표시 가능한 원본 이름이 있는 행을 후보로
    삼는다. 이 규칙은 기존 분류기가 unknown으로 남긴 복합 단체명도
    지원하면서, 마스킹 사람 이름의 부분검색 오탐을 막는다.
    """
    display = _organization_display_series(df)
    has_display = display.map(_lookup_norm).ne("")

    person_mask = pd.Series(False, index=df.index)
    if "성명_검색키" in df.columns:
        person_mask = person_mask | df["성명_검색키"].astype(str).map(_lookup_norm).ne("")
    if "entity_type" in df.columns:
        person_mask = person_mask | df["entity_type"].astype(str).str.contains(
            r"^person(?:_|$)", case=False, regex=True, na=False
        )

    explicit_org = pd.Series(False, index=df.index)
    if "기관명" in df.columns:
        explicit_org = explicit_org | df["기관명"].astype(str).map(_lookup_norm).ne("")
    if "entity_type" in df.columns:
        explicit_org = explicit_org | df["entity_type"].astype(str).str.contains(
            "organization|department", case=False, regex=True, na=False
        )

    # unknown 복합명은 사람으로 확정되지 않았고 표시값이 있을 때만 허용한다.
    # 실제 매칭은 아래 _find_org_exact의 exact/suffix/contains 순위와 유일성 검사로
    # 한 번 더 제한된다.
    unknown_compound = has_display & ~person_mask
    return has_display & ~person_mask & (explicit_org | unknown_compound)


def _find_org_exact(question: str) -> tuple[pd.DataFrame | None, list[str]]:
    """Search organization/compound labels before broad keyword aggregation.

    우선순위는 정규화 전체일치 > 접미 구문일치 > 포함일치다. 최상위 순위에
    서로 다른 대표명이 둘 이상 남으면 모호한 질문으로 간주해 결과를 확정하지
    않는다. 특정 회차·학과·단체명 목록은 사용하지 않는다.
    """
    phrase = _question_lookup_phrase(question)
    target = _lookup_norm(phrase)
    if len(target) < 2:
        return None, []

    ranked_frames: list[tuple[int, pd.DataFrame, str]] = []
    for alias, df in _scoped_dataframes().items():
        candidate_mask = _organization_candidate_mask(df)
        if not candidate_mask.any():
            continue

        candidates = df[candidate_mask].copy()
        display = _organization_display_series(candidates)
        normalized = display.map(_lookup_norm)

        rank = pd.Series(99, index=candidates.index, dtype=int)
        rank[normalized == target] = 0
        rank[(rank == 99) & normalized.str.endswith(target, na=False)] = 1
        if len(target) >= 3:
            rank[(rank == 99) & normalized.str.contains(re.escape(target), regex=True, na=False)] = 2

        valid = rank < 99
        if not valid.any():
            continue
        best_rank = int(rank[valid].min())
        rows = candidates[rank == best_rank].copy()
        rows["_기관대표명"] = display.loc[rows.index].values
        src = _df_sources.get(alias, alias)
        if "source" not in rows.columns:
            rows.insert(0, "source", src)
        ranked_frames.append((best_rank, rows, src))

    if not ranked_frames:
        return None, []

    global_rank = min(rank for rank, _, _ in ranked_frames)
    selected = [(rows, src) for rank, rows, src in ranked_frames if rank == global_rank]
    combined = pd.concat([rows for rows, _ in selected], ignore_index=True)

    # 같은 단체의 다중 납부 행은 유지하되, 서로 다른 단체가 함께 잡히면
    # broad direct search에 넘겨 임의 합계를 방지한다.
    entity_names = combined["_기관대표명"].astype(str).map(_lookup_norm)
    distinct_names = [name for name in entity_names.unique().tolist() if name]
    if len(distinct_names) != 1:
        return None, []

    representative = combined["_기관대표명"].dropna().astype(str)
    full_name = representative.iloc[0] if not representative.empty else phrase
    if "기관명" not in combined.columns:
        combined["기관명"] = full_name
    else:
        # 부분 검색으로 잡힌 unknown 행도 formatter가 완전한 대표명을 쓰도록 한다.
        combined["기관명"] = full_name

    combined["_매칭유형"] = (
        "organization_exact_match" if global_rank == 0 else "organization_partial_match"
    )
    combined["_질문이름"] = phrase
    combined = _filter_rows_by_question_cohort(combined, question)
    combined = _select_best_source_rows(combined, question)
    sources = list(combined.get("source", pd.Series(dtype=str)).dropna().astype(str).unique())
    if not sources:
        sources = list(dict.fromkeys(src for _, src in selected))
    return combined.drop(columns=["_기관대표명"], errors="ignore"), sources


# ---------------------------------------------------------------------------
# 식별번호 기반 조회 보강
# - 특정 ID 형식/값을 하드코딩하지 않고 ID/번호 계열 컬럼의 실제 값으로 매칭한다.
# - 이름 질문에 식별번호가 함께 있으면 이름 결과도 해당 번호로 제한한다.
# ---------------------------------------------------------------------------
_IDENTIFIER_COL_HINTS = (
    "id", "번호", "식별", "관리번호", "발행번호", "발급번호", "접수번호", "등록번호", "문서번호",
)
_QUESTION_IDENTIFIER_RE = re.compile(
    r"(?<!\d)(?:20\d{2}[-\s]?\d{3,}|[A-Za-z]{1,10}[-\s]?\d{2,})(?!\d)",
    re.IGNORECASE,
)


def _identifier_norm(value: object) -> str:
    text = str(value or "").strip().upper()
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"[^0-9A-Z가-힣]", "", text)


def _identifier_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        name = re.sub(r"\s+", "", str(col)).lower()
        if str(col).startswith("_") or name in {"rowuid", "personcandidatekey"}:
            continue
        if name == "id" or any(hint in name for hint in _IDENTIFIER_COL_HINTS if hint != "id"):
            cols.append(col)
    return cols


def _question_identifier_targets(question: str) -> list[str]:
    qkey = _identifier_norm(question)
    if not qkey:
        return []
    matches = [
        key
        for key in (_identifier_norm(match.group(0)) for match in _QUESTION_IDENTIFIER_RE.finditer(question))
        if key
    ]
    for df in _scoped_dataframes().values():
        for col in _identifier_columns(df):
            for raw in df[col].dropna().astype(str).unique().tolist():
                key = _identifier_norm(raw)
                if len(key) >= 4 and key in qkey and key not in matches:
                    matches.append(key)
    return list(dict.fromkeys(matches))


def _has_explicit_structured_filter(question: str) -> bool:
    return bool(_question_cohort(question) or _QUESTION_IDENTIFIER_RE.search(str(question or "")))


def _filter_rows_by_question_identifier(rows: pd.DataFrame, question: str) -> pd.DataFrame:
    if rows is None or rows.empty:
        return rows
    targets = _question_identifier_targets(question)
    if not targets:
        return rows
    masks: list[pd.Series] = []
    for col in _identifier_columns(rows):
        series = rows[col].map(_identifier_norm)
        masks.append(series.isin(targets))
    if not masks:
        return rows.iloc[0:0]
    combined = masks[0].copy()
    for mask in masks[1:]:
        combined = combined | mask
    return rows[combined].copy()


def _find_identifier_exact(question: str) -> tuple[pd.DataFrame | None, list[str]]:
    targets = _question_identifier_targets(question)
    if not targets:
        return None, []
    target = targets[0]
    frames: list[pd.DataFrame] = []
    for alias, df in _scoped_dataframes().items():
        masks: list[pd.Series] = []
        matched_col = ""
        for col in _identifier_columns(df):
            mask = df[col].map(_identifier_norm) == target
            if mask.any():
                masks.append(mask)
                if not matched_col:
                    matched_col = str(col)
        if not masks:
            continue
        combined = masks[0].copy()
        for mask in masks[1:]:
            combined = combined | mask
        rows = df[combined].copy()
        if rows.empty:
            continue
        src = _df_sources.get(alias, alias)
        if "source" not in rows.columns:
            rows.insert(0, "source", src)
        rows["_매칭유형"] = "identifier_exact_match"
        rows["_질문ID"] = target
        rows["_식별컬럼"] = matched_col
        frames.append(rows)
    if not frames:
        return None, []
    result = pd.concat(frames, ignore_index=True)
    if "row_uid" in result.columns:
        result = result.drop_duplicates(subset=["row_uid"])
    else:
        result = result.drop_duplicates()
    result = _select_best_source_rows(result, question)
    sources = list(result.get("source", pd.Series(dtype=str)).dropna().astype(str).unique())
    return result, sources


# 기관명 질문 끝의 조사/금액 표현을 일반적으로 제거한다.
_LOOKUP_ENDING_RE = re.compile(
    r"(?:"
    r"알려\s*줘|알려\s*주세요|보여\s*줘|확인해\s*줘|해\s*줘|주세요|"
    r"얼마(?:야|인가요|입니까)?|"
    r"출연금액|출연금|지급액|장학금액|장학금|지원금액|지원금|수혜금액|수혜금|"
    r"후원금액|후원금|기부금액|기부금|총금액|총액|합계|금액"
    r")(?:은|는|이|가|을|를|도|만)?\s*$"
)


def _question_lookup_phrase(question: str) -> str:
    text = str(question or "").strip()
    text = re.sub(r"[?？!！。.,，]+$", "", text).strip()
    previous = None
    while text and text != previous:
        previous = text
        text = _LOOKUP_ENDING_RE.sub("", text).strip()
    return re.sub(r"\s+", " ", text)


def _search_name_pandas(question: str) -> tuple[pd.DataFrame | None, list[str], bool]:
    """이름 검색 후 질문의 기수와 식별번호 조건을 순서대로 적용한다."""
    rows, sources, searched = _search_name_pandas_core(question)
    if rows is None or rows.empty:
        return rows, sources, searched

    if _question_cohort(question):
        rows = _filter_rows_by_question_cohort(rows, question)
        if rows is None or rows.empty:
            return rows, [], True

    targets = _question_identifier_targets(question)
    if targets:
        rows = _filter_rows_by_question_identifier(rows, question)
        if rows is None or rows.empty:
            return rows, [], True
    sources = list(rows.get("source", pd.Series(dtype=str)).dropna().astype(str).unique()) or sources
    return rows, sources, searched


def _concat_aggregation_aliases(aliases: list[str]) -> tuple[pd.DataFrame | None, list[str]]:
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    for alias in aliases:
        df = _df_namespace.get(alias)
        if df is None or df.empty:
            continue
        src = _df_sources.get(alias, alias)
        rows = df.copy()
        if "source" not in rows.columns:
            rows.insert(0, "source", src)
        frames.append(rows)
        if src not in sources:
            sources.append(src)
    if not frames:
        return None, []
    result = pd.concat(frames, ignore_index=True, sort=False)
    if "row_uid" in result.columns:
        result = result.drop_duplicates(subset=["row_uid"])
    return result, sources


def _query_all_records() -> tuple[object, list[str]]:
    """선택 문서가 하나일 때 해당 원본의 전체 표 행을 직접 반환한다.

    목록 요청을 LLM Pandas 코드에 맡기지 않기 위한 결정론적 실행 경로다.
    한 원본에 시트·표가 여러 개면 같은 source의 DataFrame을 합치고, 서로
    다른 원본이 함께 범위에 들어오면 임의로 섞지 않고 문서 선택을 요청한다.
    """
    sources_to_aliases: dict[str, list[str]] = {}
    for alias in _scoped_dataframes():
        source = _df_sources.get(alias, alias)
        sources_to_aliases.setdefault(source, []).append(alias)

    if not sources_to_aliases:
        return _aggregation_notice("조회 가능한 표 데이터가 없습니다."), []

    if len(sources_to_aliases) > 1:
        names = ", ".join(list(sources_to_aliases)[:5])
        return _aggregation_notice(
            f"전체 목록을 조회할 문서를 하나 선택해주세요: {names}",
            kind="clarification",
        ), list(sources_to_aliases)

    aliases = next(iter(sources_to_aliases.values()))
    rows, sources = _concat_aggregation_aliases(aliases)
    if rows is None or rows.empty:
        return _aggregation_notice("선택한 문서에 조회할 표 데이터가 없습니다."), sources
    return rows, sources


def _source_aliases_from_question(question: str) -> list[str]:
    aliases = _find_dfs_by_source_label(question)
    if aliases:
        return aliases

    # 영문·숫자가 포함된 파일명(test2025 등)도 직접 지정할 수 있게 한다.
    qkey = re.sub(r"[^0-9A-Za-z가-힣]", "", str(question or "")).lower()
    matched: list[str] = []
    for alias in _scoped_dataframes():
        src = _df_sources.get(alias, "")
        label = _df_labels.get(alias, "")
        stem = re.sub(r"\.[^.]+$", "", src)
        keys = {
            re.sub(r"[^0-9A-Za-z가-힣]", "", value).lower()
            for value in (src, stem, label)
            if value
        }
        if any(len(key) >= 3 and key in qkey for key in keys):
            matched.append(alias)
    return matched


def _resolve_aggregation_scope(question: str) -> tuple[pd.DataFrame | None, list[str], dict[str, object] | None]:
    """집계할 행을 결정한다. 한 원본만 로드된 경우에는 문서명을 생략해도 된다."""
    conditions = _find_filter_conditions(question)
    if conditions:
        condition_sources = list(dict.fromkeys(
            _df_sources.get(alias, alias) for alias in conditions
        ))
        if len(condition_sources) > 1:
            names = ", ".join(condition_sources[:5])
            return None, [], _aggregation_notice(
                f"조건과 일치하는 기록이 여러 문서에 있습니다. 조회할 문서를 선택해주세요: {names}",
                kind="clarification",
            )
        context_words = set(re.findall(r"[0-9A-Za-z가-힣]{2,}", question))

        def score(alias: str) -> tuple[int, int, int]:
            src = _df_sources.get(alias, "") + " " + _df_labels.get(alias, "")
            year = max([int(y) for y in re.findall(r"20\d{2}", src)] or [0])
            return (len(conditions.get(alias, [])), _source_relevance(src, context_words), year)

        best_alias = max(conditions, key=score)
        df = _df_namespace[best_alias]
        mask = pd.Series(True, index=df.index)
        for col, value in conditions[best_alias]:
            mask &= df[col].astype(str).str.contains(re.escape(value), na=False)
        rows = df[mask].copy()
        src = _df_sources.get(best_alias, best_alias)
        if "source" not in rows.columns:
            rows.insert(0, "source", src)
        return rows, [src], None

    source_aliases = _source_aliases_from_question(question)
    if source_aliases:
        distinct = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in source_aliases))
        if len(distinct) == 1:
            rows, sources = _concat_aggregation_aliases(source_aliases)
            return rows, sources, None
        names = ", ".join(distinct[:5])
        return None, [], _aggregation_notice(
            f"조회할 문서를 하나로 특정해주세요. 일치하는 문서: {names}", kind="clarification"
        )

    sources_to_aliases: dict[str, list[str]] = {}
    for alias in _scoped_dataframes():
        sources_to_aliases.setdefault(_df_sources.get(alias, alias), []).append(alias)
    if len(sources_to_aliases) == 1:
        aliases = next(iter(sources_to_aliases.values()))
        rows, sources = _concat_aggregation_aliases(aliases)
        return rows, sources, None

    names = ", ".join(list(sources_to_aliases)[:5])
    return None, [], _aggregation_notice(
        f"조회할 문서를 지정해주세요. 현재 선택 가능한 문서: {names}", kind="clarification"
    )


def _aggregate_rows(
    rows: pd.DataFrame,
    sources: list[str],
    intent: AggregationIntent,
    question: str,
) -> tuple[object, list[str]]:
    return aggregate_rows(
        rows,
        sources,
        intent,
        question=question,
        count_valid_name_rows=_count_valid_name_rows,
    )


def _query_aggregation(question: str, intent: AggregationIntent) -> tuple[object, list[str]]:
    # 마스킹 이름이 명시된 경우에만 이름 전용 검색기를 사용한다. 일반 집계 문장의
    # 서술어를 이름 후보로 오해해 개인 행만 선택하는 것을 막는다.
    if _MASKED_NAME_TOKEN_RE.search(question):
        name_rows, name_sources, _ = _search_name_pandas(question)
        if name_rows is not None and not name_rows.empty:
            return _aggregate_rows(name_rows, name_sources, intent, question)
        return _aggregation_notice("조건에 맞는 이름을 찾지 못했습니다."), []

    rows, sources, notice = _resolve_aggregation_scope(question)
    if notice is not None:
        return notice, []
    if rows is None or rows.empty:
        return _aggregation_notice("조건에 맞는 데이터가 없습니다."), []
    return _aggregate_rows(rows, sources, intent, question)


def _filter_known_people(rows: pd.DataFrame) -> pd.DataFrame:
    """엔터티 분류 결과가 있는 경우에만 개인 행으로 제한한다."""
    if "entity_type" not in rows.columns:
        return rows
    entity_types = rows["entity_type"].astype(str)
    known = entity_types.str.contains(
        r"^(?:person|organization|department)(?:_|$)",
        case=False,
        regex=True,
        na=False,
    )
    if not known.any():
        return rows
    is_person = entity_types.str.contains(
        r"^person(?:_|$)", case=False, regex=True, na=False
    )
    return rows[is_person].copy()


def _apply_date_query_conditions(
    rows: pd.DataFrame,
    *,
    alias: str,
    question: str,
    conditions: dict[str, list[tuple[str, str]]],
    identifier_requested: bool,
    cohort_requested: bool,
    person_list_requested: bool,
) -> pd.DataFrame | None:
    """날짜로 거른 행에 기존 값·식별번호·기수 조건을 순서대로 적용한다."""
    if conditions:
        if alias not in conditions:
            return None
        for column, value in conditions[alias]:
            rows = rows[rows[column].astype(str).str.contains(re.escape(value), na=False)]
    if identifier_requested:
        rows = _filter_rows_by_question_identifier(rows, question)
    if cohort_requested:
        rows = _filter_rows_by_question_cohort(rows, question)
    if person_list_requested:
        rows = _filter_known_people(rows)
    return rows


def _unique_person_list(rows: pd.DataFrame) -> pd.DataFrame:
    """기간 명단에서 동일 인물의 반복 납부 행을 한 번만 표시한다."""
    key_column = next(
        (column for column in _PERSON_IDENTITY_COLUMNS if column in rows.columns),
        None,
    )
    if not key_column:
        return rows

    keys = rows[key_column].astype(str).map(normalize_person_name)
    keep = keys.str.strip().ne("") & ~keys.duplicated(keep="first")
    result = rows[keep].copy()
    display_columns = [column for column in _PERSON_LIST_COLUMNS if column in result.columns]
    return result.loc[:, display_columns] if display_columns else result


def _date_evidence(evidences: list[dict[str, object]]) -> dict[str, object]:
    if len(evidences) == 1:
        return evidences[0]
    return {"items": evidences}


def _query_date_filtered(
    question: str,
    date_filter: DateFilter,
    intent: AggregationIntent | None,
) -> tuple[object, list[str]]:
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    evidences: list[dict[str, object]] = []
    messages: list[str] = []
    conditions = _find_filter_conditions(question)
    identifier_requested = bool(_question_identifier_targets(question))
    cohort_requested = bool(_question_cohort(question))
    person_list_requested = intent is None and bool(_PERSON_LIST_RE.search(question))

    for alias, df in _scoped_dataframes().items():
        result = apply_date_filter(df, date_filter, question)
        if result.rows is None:
            if result.message:
                messages.append(result.message)
            continue
        source = _df_sources.get(alias, alias)
        rows = _apply_date_query_conditions(
            result.rows.copy(),
            alias=alias,
            question=question,
            conditions=conditions,
            identifier_requested=identifier_requested,
            cohort_requested=cohort_requested,
            person_list_requested=person_list_requested,
        )
        if rows is None or rows.empty:
            continue
        if "source" not in rows.columns:
            rows.insert(0, "source", source)
        frames.append(rows)
        if source not in sources:
            sources.append(source)
        if result.evidence:
            evidences.append({**result.evidence, "source": source})

    if not frames:
        message = next(iter(dict.fromkeys(messages)), "조건에 맞는 날짜 데이터가 없습니다.")
        return _aggregation_notice(message, kind="clarification"), []
    if len(sources) > 1 and not source_scope_active():
        return _aggregation_notice(
            "날짜 조건을 적용할 문서를 선택해주세요: " + ", ".join(sources[:5]),
            kind="clarification",
        ), sources

    rows = pd.concat(frames, ignore_index=True, sort=False)
    if "row_uid" in rows.columns:
        rows = rows.drop_duplicates(subset=["row_uid"])
    rows.attrs["date_filter_evidence"] = _date_evidence(evidences)

    if rows.empty:
        return _aggregation_notice("해당 날짜 조건과 일치하는 데이터가 없습니다."), sources
    if intent:
        payload, payload_sources = _aggregate_rows(rows, sources, intent, question)
        if isinstance(payload, dict) and payload.get("type") == "aggregation":
            payload["date_filter"] = rows.attrs["date_filter_evidence"]
        return payload, payload_sources
    if person_list_requested:
        rows = _unique_person_list(rows)
        rows.attrs["date_filter_evidence"] = _date_evidence(evidences)
    return rows, sources


def _query_pandas_direct(
    question: str,
    aggregation_intents: list[AggregationIntent] | None = None,
    date_filter: DateFilter | None = None,
) -> tuple[object, list[str]]:
    """식별번호, 기관명, 일반 조건 순으로 직접 조회한다."""
    # None은 독립 호출의 하위 호환 경로이고, 빈 목록은 이미 분석했지만 집계가
    # 감지되지 않았다는 뜻이다. 정상 /chat 흐름에서는 Analyzer 결과가 전달된다.
    if aggregation_intents is None:
        aggregation_intents = detect_aggregation_intents(question)
    if len(aggregation_intents) > 1:
        return _aggregation_notice(
            "한 번에 하나의 집계 기준을 요청해 주세요.",
            kind="clarification",
        ), []
    intent = aggregation_intents[0] if aggregation_intents else None
    if date_filter is None:
        from pandas_engine.date_filter import parse_date_filter
        date_filter = parse_date_filter(question)
    if date_filter is not None:
        return _query_date_filtered(question, date_filter, intent)
    identifier_targets = _question_identifier_targets(question)
    if identifier_targets:
        id_df, id_sources = _find_identifier_exact(question)
        if id_df is not None and not id_df.empty:
            if intent:
                return _aggregate_rows(id_df, id_sources, intent, question)
            return id_df, id_sources
        return None, []

    if _question_cohort(question):
        cohort_df, cohort_sources = _find_cohort_exact(question)
        if cohort_df is not None and not cohort_df.empty:
            if intent:
                return _aggregate_rows(cohort_df, cohort_sources, intent, question)
            return cohort_df, cohort_sources
        return None, []

    org_df, org_sources = _find_org_exact(question)
    if org_df is not None and not org_df.empty:
        if intent:
            return _aggregate_rows(org_df, org_sources, intent, question)
        return org_df, org_sources

    if intent:
        return _query_aggregation(question, intent)

    return _query_pandas_direct_base(question)
