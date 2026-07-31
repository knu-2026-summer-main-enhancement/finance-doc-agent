from __future__ import annotations

import asyncio
import logging
import re
from contextvars import ContextVar
from typing import Literal

import pandas as pd

from core.privacy import question_log_metadata
from datastore.state import _df_namespace, _df_sources
from datastore.scope import scoped_mapping, source_scope_active
from datastore.query import (
    _search_name_pandas,
    _query_pandas_direct,
    _query_all_records,
    _has_explicit_structured_filter,
    has_explicit_masked_name,
)
from pandas_engine.query_executor import (
    QueryPlanExecutionError,
    execute_query_plan,
)
from pandas_engine.aggregation import amount_column_clarification, resolve_amount_column
from pandas_engine.money import money_series
from pandas_engine.formatter import (
    _format_pandas_result,
    _format_scalar_result,
    _format_dataframe_result_for_question,
    _format_query_execution_result,
)
from rag.query_planner import (
    QueryPlannerError,
    generate_validated_query_plan,
)
from rag.question_analyzer import QuestionAnalysis, analyze_question
from rag.deterministic_query_plan import build_schema_grounded_plan
from rag.cancellation import await_cancellable, raise_if_cancelled
from utils.semantic_schema import infer_column_meaning
from pandas_engine.interactive import (
    build_interactive_result,
    build_interactive_dataframe,
    build_interactive_people_list,
)
from pandas_engine.query_plan import QueryPlan
from utils.table_parser import normalize_person_name

logger = logging.getLogger("uvicorn.error")
_interactive_result: ContextVar[dict | None] = ContextVar("interactive_result", default=None)


def clear_interactive_result() -> None:
    _interactive_result.set(None)


def current_interactive_result() -> dict | None:
    return _interactive_result.get()

_NUMERIC_COMPARISON_FILTER = re.compile(
    r"(?:\d[\d,.]*\s*(?:원|만원|천원|점|명|개)?\s*(?:이상|이하|초과|미만))"
    r"|(?:(?:>=|<=|>|<)\s*\d)",
    re.IGNORECASE,
)
_AMBIGUOUS_SUMMARY_QUESTION = re.compile(
    r"^\s*(?:총|전체|합계|총합|얼마)\s*[?!。.]*\s*$"
)
_YEAR_COMPARISON = re.compile(
    r"(?<!\d)((?:19|20)\d{2})\s*년.*?(?<!\d)((?:19|20)\d{2})\s*년"
)
_COMPARISON_WORD = re.compile(r"비교|차이|증감|증가|감소")
_YEAR_NON_PAYER = re.compile(
    r"(?<!\d)(?P<paid_year>(?:19|20)\d{2})\s*년(?:에|도)?"
    r".{0,30}?(?:낸|납부(?:한)?|결제(?:한)?)"
    r".{0,40}?(?:중(?:에)?|가운데(?:에서)?|에서)\s*"
    r"(?<!\d)(?P<missing_year>(?:19|20)\d{2})\s*년(?:에|도)?"
    r".{0,30}?(?:안\s*낸|안낸|미납|납부\s*안|결제\s*안|내지\s*않)",
)
_LOWEST_YEAR_AMOUNT = re.compile(
    r"(?:가장|제일|최저).{0,12}(?:적|낮).{0,12}(?:년도|연도|년)|"
    r"(?:년도|연도|년).{0,12}(?:가장|제일|최저).{0,12}(?:적|낮)"
)
_HIGHEST_YEAR_AMOUNT = re.compile(
    r"(?:가장|제일|최고).{0,12}(?:많|크|높).{0,12}(?:년도|연도|년)|"
    r"(?:년도|연도|년).{0,12}(?:가장|제일|최고).{0,12}(?:많|크|높)"
)
_LOWEST_MONTH_AMOUNT = re.compile(
    r"(?:가장|제일|최저).{0,12}(?:적|낮).{0,12}(?:월|달)|(?:월|달).{0,12}(?:가장|제일|최저).{0,12}(?:적|낮)"
)
_HIGHEST_MONTH_AMOUNT = re.compile(
    r"(?:가장|제일|최고).{0,12}(?:많|크|높).{0,12}(?:월|달)|(?:월|달).{0,12}(?:가장|제일|최고).{0,12}(?:많|크|높)"
)
_EXTREME_PERIOD_COUNT = re.compile(
    r"(?:횟수|기록\s*수|건수).{0,18}(?:가장|제일|최고|최저).{0,18}(?:년도|연도|년|월|달)|"
    r"(?:년도|연도|년|월|달).{0,18}(?:횟수|기록\s*수|건수).{0,18}(?:가장|제일|최고|최저)"
)
_MOST_FREQUENT_PERSON = re.compile(
    r"(?:횟수|몇\s*번|가장\s*자주|제일\s*자주).{0,20}(?:많은|많이|높은)?.{0,20}(?:사람|회원|인원)|"
    r"(?:사람|회원|인원).{0,20}(?:횟수|몇\s*번|가장\s*자주|제일\s*자주)"
)


def _year_series(df: pd.DataFrame) -> pd.Series | None:
    """Return a schema-grounded year series without relying on header names."""

    for column in df.columns:
        meaning = infer_column_meaning(str(column), df[column])
        if meaning.concept == "temporal" and meaning.role == "year":
            return pd.to_numeric(df[column], errors="coerce").astype("Int64")
    for column in df.columns:
        meaning = infer_column_meaning(str(column), df[column])
        if meaning.concept == "temporal" and meaning.role in {"date", "year_month"}:
            dates = pd.to_datetime(df[column], errors="coerce")
            if dates.notna().any():
                return dates.dt.year.astype("Int64")
    return None


def _month_series(df: pd.DataFrame) -> pd.Series | None:
    for column in df.columns:
        meaning = infer_column_meaning(str(column), df[column])
        if meaning.concept == "temporal" and meaning.role == "month":
            return pd.to_numeric(df[column], errors="coerce").astype("Int64")
    for column in df.columns:
        meaning = infer_column_meaning(str(column), df[column])
        if meaning.concept == "temporal" and meaning.role in {"date", "year_month"}:
            dates = pd.to_datetime(df[column], errors="coerce")
            if dates.notna().any():
                return dates.dt.month.astype("Int64")
    return None


def _person_name_column(df: pd.DataFrame) -> object | None:
    for column in df.columns:
        meaning = infer_column_meaning(str(column), df[column])
        if (
            meaning.concept == "entity"
            and meaning.role == "entity_name"
            and meaning.qualifier == "person"
        ):
            return column
    return None


def _source_row_count(dataframes: dict[str, pd.DataFrame]) -> int:
    return sum(len(df) for df in dataframes.values())


def _compact_breakdown(values: dict[object, float | int], suffix: str) -> str:
    return ", ".join(
        f"{key}{suffix} {value:,.0f}" for key, value in sorted(values.items())
    )


def _explicit_year_pair(question: str) -> tuple[int, int] | None:
    years = [int(value) for value in re.findall(r"(?<!\d)((?:19|20)\d{2})\s*년", question)]
    unique_years = list(dict.fromkeys(years))
    if len(unique_years) != 2:
        return None
    return unique_years[0], unique_years[1]


def _collect_year_person_totals(
    question: str,
    dataframes: dict[str, pd.DataFrame],
    first_year: int,
    second_year: int,
) -> tuple[dict[int, dict[str, tuple[str, float]]], list[str], str | None]:
    """Collect unique people and their paid totals for two years in one source."""

    sources = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in dataframes))
    if len(sources) != 1:
        return {}, sources, "문서 내 비교는 한 번에 한 문서를 선택했을 때만 지원합니다. 비교할 문서를 하나 선택해 주세요."

    people = {first_year: {}, second_year: {}}
    usable_frames = 0
    for df in dataframes.values():
        years = _year_series(df)
        person_column = _person_name_column(df)
        if years is None or person_column is None:
            continue
        selection = resolve_amount_column(df, question)
        if len(selection.candidates) > 1 and selection.selected is None:
            return {}, sources, amount_column_clarification(selection.candidates)
        amounts = money_series(df, selection.selected) if selection.selected else None
        usable_frames += 1
        for year in people:
            mask = years.eq(year).fillna(False)
            if amounts is not None:
                mask = mask & amounts.gt(0).fillna(False)
            for index, raw_name in df.loc[mask, person_column].dropna().items():
                display_name = str(raw_name).strip()
                normalized_name = normalize_person_name(display_name)
                if not normalized_name:
                    continue
                amount = float(amounts.loc[index]) if amounts is not None and pd.notna(amounts.loc[index]) else 0.0
                existing = people[year].get(normalized_name)
                people[year][normalized_name] = (
                    existing[0] if existing else display_name,
                    (existing[1] if existing else 0.0) + amount,
                )
    if not usable_frames:
        return {}, sources, "선택한 문서에서 연도와 사람 이름 컬럼을 함께 찾지 못했습니다."
    return people, sources, None


def _format_person_list(title: str, people: list[tuple[str, float]], sources: list[str], evidence: str) -> tuple[str, list[str], str]:
    lines = [title, ""]
    if people:
        lines.extend(f"- {name}" for name, _ in people[:200])
        if len(people) > 200:
            lines.append(f"- 외 {len(people) - 200:,}명")
    else:
        lines.append("- 해당 없음")
    lines.extend(("", "조회 근거:", f"- 문서: {sources[0]}", f"- 비교 조건: {evidence}"))
    return "\n".join(lines), sources, "pandas"


def _answer_year_person_comparison(
    question: str,
    dataframes: dict[str, pd.DataFrame],
) -> tuple[str, list[str], str] | None:
    """Answer yearly people-set and person-total comparison questions."""

    pair = _explicit_year_pair(question)
    if pair is None:
        return None
    first_year, second_year = pair
    compact = re.sub(r"\s+", "", question)
    is_people_comparison = any(token in compact for token in ("인원비교", "비교해줘", "비교해주세요"))
    kind: str | None = None
    if re.search(r"둘다.*(?:낸|납부)|모두.*(?:낸|납부)|양쪽.*(?:낸|납부)", compact):
        kind = "both"
    elif re.search(r"안냈다가.*(?:낸|납부)", compact):
        kind = "later_only"
    elif re.search(r"더많이낸|금액이늘|증가.*사람", compact):
        kind = "increased"
    elif re.search(r"금액이줄|덜낸|감소.*사람", compact):
        kind = "decreased"
    elif "인원" in compact and is_people_comparison:
        kind = "count"
    if kind is None:
        return None

    people, sources, error = _collect_year_person_totals(question, dataframes, first_year, second_year)
    if error:
        return error, sources, "pandas"
    first, second = people[first_year], people[second_year]
    first_names, second_names = set(first), set(second)

    if kind == "count":
        difference = len(second_names) - len(first_names)
        answer = "\n".join((
            f"{first_year}년과 {second_year}년 납부 인원을 비교했습니다.",
            "",
            f"- {first_year}년: {len(first_names):,}명",
            f"- {second_year}년: {len(second_names):,}명",
            f"- 변화: {difference:+,}명",
            "",
            "조회 근거:",
            f"- 문서: {sources[0]}",
            f"- 비교 조건: {first_year}년 납부 기록, {second_year}년 납부 기록",
        ))
        return answer, sources, "pandas"
    if kind == "both":
        names = sorted(first_names & second_names)
        response = _format_person_list(
            f"{first_year}년과 {second_year}년에 모두 납부한 사람은 {len(names):,}명입니다.",
            [(first[name][0], first[name][1]) for name in names],
            sources,
            f"{first_year}년과 {second_year}년 납부 기록 모두 있음",
        )
        answer, _, _ = response
        source_frame = pd.concat(list(dataframes.values()), ignore_index=True)
        interactive = build_interactive_people_list(
            [first[name][0] for name in names], source_frame, answer,
        )
        if interactive is not None:
            _interactive_result.set(interactive)
        return response
    if kind == "later_only":
        names = sorted(second_names - first_names)
        return _format_person_list(
            f"{first_year}년에는 납부 기록이 없고 {second_year}년에 납부한 사람은 {len(names):,}명입니다.",
            [(second[name][0], second[name][1]) for name in names],
            sources,
            f"{first_year}년 납부 기록에는 없고 {second_year}년 납부 기록에는 있음",
        )

    shared = first_names & second_names
    if kind == "increased":
        names = sorted(name for name in shared if second[name][1] > first[name][1])
        title = f"{first_year}년보다 {second_year}년에 더 많이 납부한 사람은 {len(names):,}명입니다."
        evidence = f"같은 사람의 {first_year}년·{second_year}년 납부 금액 합계 비교, 증가한 경우만 표시"
    else:
        names = sorted(name for name in shared if second[name][1] < first[name][1])
        title = f"{first_year}년보다 {second_year}년에 납부 금액이 줄어든 사람은 {len(names):,}명입니다."
        evidence = f"같은 사람의 {first_year}년·{second_year}년 납부 금액 합계 비교, 감소한 경우만 표시"
    return _format_person_list(title, [(first[name][0], first[name][1]) for name in names], sources, evidence)


def _answer_period_payment_comparison(
    question: str,
    dataframes: dict[str, pd.DataFrame],
) -> tuple[str, list[str], str] | None:
    """Compare two months or the first/second half of one document."""

    month_values = [int(value) for value in re.findall(r"(?<!\d)(1[0-2]|[1-9])\s*월", question)]
    unique_months = list(dict.fromkeys(month_values))
    compact = re.sub(r"\s+", "", question)
    half_requested = "상반기" in compact and "하반기" in compact
    if not (len(unique_months) == 2 or half_requested) or not _COMPARISON_WORD.search(question):
        return None
    labels = (f"{unique_months[0]}월", f"{unique_months[1]}월") if unique_months else ("상반기", "하반기")
    sources = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in dataframes))
    if len(sources) != 1:
        return "문서 내 기간 비교는 한 번에 한 문서를 선택했을 때만 지원합니다. 비교할 문서를 하나 선택해 주세요.", sources, "pandas"

    totals = [0.0, 0.0]
    people = [set(), set()]
    usable_frames = 0
    for df in dataframes.values():
        months = _month_series(df)
        selection = resolve_amount_column(df, question)
        if months is None or selection.selected is None:
            if months is not None and len(selection.candidates) > 1:
                return amount_column_clarification(selection.candidates), sources, "pandas"
            continue
        usable_frames += 1
        amounts = money_series(df, selection.selected)
        person_column = _person_name_column(df)
        masks = (
            (months.le(6).fillna(False), months.ge(7).fillna(False))
            if half_requested else
            (months.eq(unique_months[0]).fillna(False), months.eq(unique_months[1]).fillna(False))
        )
        for index, mask in enumerate(masks):
            paid_mask = mask & amounts.gt(0).fillna(False)
            totals[index] += float(amounts[paid_mask].sum())
            if person_column is not None:
                people[index].update(
                    normalize_person_name(name)
                    for name in df.loc[paid_mask, person_column].dropna().astype(str)
                    if normalize_person_name(name)
                )
    if not usable_frames:
        return "선택한 문서에서 월 또는 날짜와 금액 컬럼을 함께 찾지 못했습니다.", sources, "pandas"
    difference = totals[1] - totals[0]
    rate = None if totals[0] == 0 else difference / totals[0] * 100
    change = f"{difference:+,.0f}원" + (f" ({rate:+.1f}%)" if rate is not None else "")
    return "\n".join((
        f"{labels[0]}과 {labels[1]} 납부 현황을 비교했습니다.",
        "",
        f"- {labels[0]}: {totals[0]:,.0f}원, {len(people[0]):,}명",
        f"- {labels[1]}: {totals[1]:,.0f}원, {len(people[1]):,}명",
        f"- 금액 변화: {change}",
        f"- 인원 변화: {len(people[1]) - len(people[0]):+,}명",
        "",
        "조회 근거:",
        f"- 문서: {sources[0]}",
        f"- 원본 행 수: {_source_row_count(dataframes):,}개",
        f"- 비교 조건: {labels[0]}, {labels[1]}",
        "- 적용 조건: 결제 금액이 0원 초과",
    )), sources, "pandas"


def _answer_group_payment_comparison(
    question: str,
    dataframes: dict[str, pd.DataFrame],
) -> tuple[str, list[str], str] | None:
    """Summarize payment people, total, and available rate by department/cohort."""

    compact = re.sub(r"\s+", "", question)
    group_qualifier = "department" if "학과별" in compact else "cohort" if "기수별" in compact else None
    if group_qualifier is None or not any(token in compact for token in ("비교", "납부율", "납부인원", "금액")):
        return None
    sources = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in dataframes))
    if len(sources) != 1:
        return "학과·기수별 비교는 한 번에 한 문서를 선택했을 때만 지원합니다. 비교할 문서를 하나 선택해 주세요.", sources, "pandas"
    groups: dict[str, dict[str, object]] = {}
    amount_labels: set[str] = set()
    usable_frames = 0
    for df in dataframes.values():
        group_column = next((column for column in df.columns if infer_column_meaning(str(column), df[column]).qualifier == group_qualifier), None)
        person_column = _person_name_column(df)
        selection = resolve_amount_column(df, question)
        if group_column is None or person_column is None or selection.selected is None:
            if group_column is not None and person_column is not None and len(selection.candidates) > 1:
                return amount_column_clarification(selection.candidates), sources, "pandas"
            continue
        usable_frames += 1
        amount_labels.add(selection.selected)
        amounts = money_series(df, selection.selected)
        for index, raw_group in df[group_column].dropna().items():
            group = str(raw_group).strip()
            raw_name = df.at[index, person_column]
            name = normalize_person_name(str(raw_name).strip()) if pd.notna(raw_name) else ""
            if not group or not name:
                continue
            entry = groups.setdefault(group, {"all": set(), "paid": set(), "amount": 0.0})
            entry["all"].add(name)
            amount = float(amounts.loc[index]) if pd.notna(amounts.loc[index]) else 0.0
            if amount > 0:
                entry["paid"].add(name)
                entry["amount"] += amount
    if not usable_frames:
        label = "학과" if group_qualifier == "department" else "기수"
        return f"선택한 문서에서 {label}, 사람 이름, 금액 컬럼을 함께 찾지 못했습니다.", sources, "pandas"
    label = "학과" if group_qualifier == "department" else "기수"
    lines = [f"{label}별 납부 현황입니다.", ""]
    for group, entry in sorted(groups.items(), key=lambda item: (-float(item[1]["amount"]), item[0])):
        total_people = len(entry["all"])
        paid_people = len(entry["paid"])
        rate = paid_people / total_people * 100 if total_people else 0.0
        lines.append(f"- {group}: 납부 {paid_people:,}명 / 전체 {total_people:,}명 ({rate:.1f}%), {float(entry['amount']):,.0f}원")
    lines.extend((
        "", "조회 근거:", f"- 문서: {sources[0]}",
        f"- 원본 행 수: {_source_row_count(dataframes):,}개",
        "- 적용 조건: 결제 금액이 0원 초과인 행을 납부 인원·금액에 반영",
        f"- 대상 컬럼: {', '.join(sorted(amount_labels))}",
        f"- 비교 기준: {label}별 고유 인원과 납부 금액",
    ))
    return "\n".join(lines), sources, "pandas"


def _answer_year_nonpayer_comparison(
    question: str,
    dataframes: dict[str, pd.DataFrame],
) -> tuple[str, list[str], str] | None:
    """List people recorded in the first year's payments but not the second."""

    match = _YEAR_NON_PAYER.search(question)
    if match is None:
        return None
    paid_year = int(match.group("paid_year"))
    missing_year = int(match.group("missing_year"))
    if paid_year == missing_year:
        return "서로 다른 두 연도를 입력해 주세요.", [], "pandas"

    sources = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in dataframes))
    if len(sources) != 1:
        return (
            "연도별 미납자 비교는 한 번에 한 문서를 선택했을 때만 지원합니다. 비교할 문서를 하나 선택해 주세요.",
            sources,
            "pandas",
        )

    paid_people: dict[str, str] = {}
    later_people: set[str] = set()
    usable_frames = 0
    for df in dataframes.values():
        years = _year_series(df)
        person_column = _person_name_column(df)
        if years is None or person_column is None:
            continue
        usable_frames += 1
        amounts = None
        selection = resolve_amount_column(df, question)
        if selection.selected is not None:
            amounts = money_series(df, selection.selected)
        for year, collection in ((paid_year, paid_people), (missing_year, later_people)):
            mask = years.eq(year).fillna(False)
            if amounts is not None:
                mask = mask & amounts.gt(0).fillna(False)
            for raw_name in df.loc[mask, person_column].dropna().astype(str):
                display_name = raw_name.strip()
                normalized_name = normalize_person_name(display_name)
                if not normalized_name:
                    continue
                if isinstance(collection, dict):
                    collection.setdefault(normalized_name, display_name)
                else:
                    collection.add(normalized_name)

    if not usable_frames:
        return (
            "선택한 문서에서 연도와 사람 이름 컬럼을 함께 찾지 못했습니다.",
            sources,
            "pandas",
        )

    missing_people = [
        display_name for normalized_name, display_name in paid_people.items()
        if normalized_name not in later_people
    ]
    missing_people.sort(key=lambda value: value.casefold())
    lines = [
        f"{paid_year}년에 납부했지만 {missing_year}년에는 납부 기록이 없는 사람은 {len(missing_people):,}명입니다.",
        "",
    ]
    if missing_people:
        lines.extend(f"- {name}" for name in missing_people[:200])
        if len(missing_people) > 200:
            lines.append(f"- 외 {len(missing_people) - 200:,}명")
    else:
        lines.append("- 해당 없음")
    lines.extend((
        "",
        "조회 근거:",
        f"- 문서: {sources[0]}",
        f"- 비교 조건: {paid_year}년 납부 기록에는 있고 {missing_year}년 납부 기록에는 없음",
        f"- 인원 수: {paid_year}년 {len(paid_people):,}명, {missing_year}년 {len(later_people):,}명",
    ))
    return "\n".join(lines), sources, "pandas"


def _answer_year_amount_comparison(
    question: str,
    dataframes: dict[str, pd.DataFrame],
) -> tuple[str, list[str], str] | None:
    """Compare the same amount field for two explicit years in one document."""

    match = _YEAR_COMPARISON.search(question)
    if match is None or not _COMPARISON_WORD.search(question):
        return None
    first_year, second_year = (int(value) for value in match.groups())
    if first_year == second_year:
        return "서로 다른 두 연도를 입력해 주세요.", [], "pandas"

    sources = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in dataframes))
    if len(sources) != 1:
        return (
            "문서 내 비교는 한 번에 한 문서를 선택했을 때만 지원합니다. 비교할 문서를 하나 선택해 주세요.",
            sources,
            "pandas",
        )

    totals = {first_year: 0.0, second_year: 0.0}
    rows = {first_year: 0, second_year: 0}
    amount_labels: set[str] = set()
    usable_frames = 0
    for df in dataframes.values():
        years = _year_series(df)
        selection = resolve_amount_column(df, question)
        if years is None or selection.selected is None:
            if years is not None and len(selection.candidates) > 1:
                return amount_column_clarification(selection.candidates), sources, "pandas"
            continue
        usable_frames += 1
        amount_labels.add(selection.selected)
        amounts = money_series(df, selection.selected)
        for year in totals:
            mask = years.eq(year).fillna(False)
            totals[year] += float(amounts[mask].sum())
            rows[year] += int(mask.sum())

    if not usable_frames:
        return (
            "선택한 문서에서 연도와 금액 컬럼을 함께 찾지 못했습니다.",
            sources,
            "pandas",
        )

    difference = totals[second_year] - totals[first_year]
    rate = None if totals[first_year] == 0 else difference / totals[first_year] * 100
    label = next(iter(amount_labels), "금액")
    change = f"{difference:+,.0f}원"
    if rate is not None:
        change += f" ({rate:+.1f}%)"
    answer = "\n".join((
        f"{first_year}년과 {second_year}년 {label}을 비교했습니다.",
        "",
        f"- {first_year}년: {totals[first_year]:,.0f}원 ({rows[first_year]:,}건)",
        f"- {second_year}년: {totals[second_year]:,.0f}원 ({rows[second_year]:,}건)",
        f"- 변화: {change}",
        "",
        "조회 근거:",
        f"- 문서: {sources[0]}",
        f"- 비교 조건: {first_year}년, {second_year}년",
        f"- 대상 컬럼: {label}",
    ))
    return answer, sources, "pandas"


def _answer_lowest_year_amount(
    question: str,
    dataframes: dict[str, pd.DataFrame],
) -> tuple[str, list[str], str] | None:
    """Find the year with the smallest summed payment amount, not the smallest row."""

    is_highest = bool(_HIGHEST_YEAR_AMOUNT.search(question))
    if not is_highest and not _LOWEST_YEAR_AMOUNT.search(question):
        return None
    sources = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in dataframes))
    if len(sources) != 1:
        return "연도별 금액 비교는 한 번에 한 문서를 선택했을 때만 지원합니다. 비교할 문서를 하나 선택해 주세요.", sources, "pandas"
    totals: dict[int, float] = {}
    amount_labels: set[str] = set()
    usable_frames = 0
    for df in dataframes.values():
        years = _year_series(df)
        selection = resolve_amount_column(df, question)
        if years is None or selection.selected is None:
            if years is not None and len(selection.candidates) > 1:
                return amount_column_clarification(selection.candidates), sources, "pandas"
            continue
        usable_frames += 1
        amount_labels.add(selection.selected)
        amounts = money_series(df, selection.selected)
        for year in sorted(years.dropna().unique()):
            year_number = int(year)
            mask = years.eq(year_number).fillna(False)
            totals[year_number] = totals.get(year_number, 0.0) + float(amounts[mask].sum())
    if not usable_frames or not totals:
        return "선택한 문서에서 연도와 금액 컬럼을 함께 찾지 못했습니다.", sources, "pandas"
    extreme = max(totals.values()) if is_highest else min(totals.values())
    years = sorted(year for year, total in totals.items() if total == extreme)
    label = next(iter(amount_labels), "금액")
    year_text = ", ".join(f"{year}년" for year in years)
    comparison_label = "가장 많은" if is_highest else "가장 적은"
    extreme_label = "최댓값" if is_highest else "최솟값"
    return "\n".join((
        f"연도별 {label} 합계가 {comparison_label} 때는 {year_text}입니다.",
        "",
        f"- {year_text}: {extreme:,.0f}원",
        f"- 비교한 연도: {', '.join(f'{year}년' for year in sorted(totals))}",
        "",
        "조회 근거:",
        f"- 문서: {sources[0]}",
        f"- 원본 행 수: {_source_row_count(dataframes):,}개",
        f"- 계산 방식: 연도별 {label} 합계를 계산한 뒤 {extreme_label} 선택",
        f"- 대상 컬럼: {label}",
        f"- 연도별 합계: {_compact_breakdown(totals, '년')}",
    )), sources, "pandas"


def _answer_extreme_month_amount(
    question: str,
    dataframes: dict[str, pd.DataFrame],
) -> tuple[str, list[str], str] | None:
    """Find the month with the largest/smallest summed payment amount."""

    is_highest = bool(_HIGHEST_MONTH_AMOUNT.search(question))
    if not is_highest and not _LOWEST_MONTH_AMOUNT.search(question):
        return None
    sources = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in dataframes))
    if len(sources) != 1:
        return "월별 금액 비교는 한 번에 한 문서를 선택했을 때만 지원합니다. 비교할 문서를 하나 선택해 주세요.", sources, "pandas"
    totals: dict[int, float] = {}
    amount_labels: set[str] = set()
    usable_frames = 0
    for df in dataframes.values():
        months = _month_series(df)
        selection = resolve_amount_column(df, question)
        if months is None or selection.selected is None:
            if months is not None and len(selection.candidates) > 1:
                return amount_column_clarification(selection.candidates), sources, "pandas"
            continue
        usable_frames += 1
        amount_labels.add(selection.selected)
        amounts = money_series(df, selection.selected)
        for month in sorted(months.dropna().unique()):
            month_number = int(month)
            mask = months.eq(month_number).fillna(False)
            totals[month_number] = totals.get(month_number, 0.0) + float(amounts[mask].sum())
    if not usable_frames or not totals:
        return "선택한 문서에서 월 또는 날짜와 금액 컬럼을 함께 찾지 못했습니다.", sources, "pandas"
    extreme = max(totals.values()) if is_highest else min(totals.values())
    months = sorted(month for month, total in totals.items() if total == extreme)
    label = next(iter(amount_labels), "금액")
    month_text = ", ".join(f"{month}월" for month in months)
    comparison_label = "가장 많은" if is_highest else "가장 적은"
    extreme_label = "최댓값" if is_highest else "최솟값"
    return "\n".join((
        f"월별 {label} 합계가 {comparison_label} 때는 {month_text}입니다.",
        "",
        f"- {month_text}: {extreme:,.0f}원",
        f"- 비교한 월: {', '.join(f'{month}월' for month in sorted(totals))}",
        "",
        "조회 근거:",
        f"- 문서: {sources[0]}",
        f"- 원본 행 수: {_source_row_count(dataframes):,}개",
        f"- 계산 방식: 월별 {label} 합계를 계산한 뒤 {extreme_label} 선택",
        f"- 대상 컬럼: {label}",
        f"- 월별 합계: {_compact_breakdown(totals, '월')}",
    )), sources, "pandas"


def _answer_extreme_period_count(
    question: str,
    dataframes: dict[str, pd.DataFrame],
) -> tuple[str, list[str], str] | None:
    """Find the period with the greatest/smallest number of records."""

    if not _EXTREME_PERIOD_COUNT.search(question):
        return None
    is_highest = bool(re.search(r"가장\s*(?:많|높)|제일\s*(?:많|높)|최고", question))
    is_month = bool(re.search(r"월|달", question)) and not bool(re.search(r"(?:년도|연도|년)", question))
    sources = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in dataframes))
    if len(sources) != 1:
        return "기간별 횟수 비교는 한 번에 한 문서를 선택했을 때만 지원합니다. 비교할 문서를 하나 선택해 주세요.", sources, "pandas"
    counts: dict[int, int] = {}
    usable_frames = 0
    payment_only = bool(re.search(r"납부|결제|낸", question))
    for df in dataframes.values():
        periods = _month_series(df) if is_month else _year_series(df)
        if periods is None:
            continue
        mask = periods.notna()
        if payment_only:
            selection = resolve_amount_column(df, question)
            if selection.selected is None:
                if len(selection.candidates) > 1:
                    return amount_column_clarification(selection.candidates), sources, "pandas"
                continue
            mask = mask & money_series(df, selection.selected).gt(0).fillna(False)
        usable_frames += 1
        for value, count in periods[mask].value_counts().items():
            counts[int(value)] = counts.get(int(value), 0) + int(count)
    if not usable_frames or not counts:
        label = "월" if is_month else "연도"
        return f"선택한 문서에서 {label} 또는 기록 컬럼을 찾지 못했습니다.", sources, "pandas"
    extreme = max(counts.values()) if is_highest else min(counts.values())
    values = sorted(value for value, count in counts.items() if count == extreme)
    suffix = "월" if is_month else "년"
    period_text = ", ".join(f"{value}{suffix}" for value in values)
    comparison_label = "가장 많은" if is_highest else "가장 적은"
    return "\n".join((
        f"{comparison_label} 기록 횟수의 기간은 {period_text}입니다.",
        "",
        f"- {period_text}: {extreme:,}건",
        f"- 비교한 기간: {', '.join(f'{value}{suffix}' for value in sorted(counts))}",
        "",
        "조회 근거:",
        f"- 문서: {sources[0]}",
        f"- 원본 행 수: {_source_row_count(dataframes):,}개",
        f"- 계산 방식: {'월' if is_month else '연도'}별 기록 행 수를 집계한 뒤 {'최댓값' if is_highest else '최솟값'} 선택",
        f"- 적용 조건: {'결제 금액이 0원 초과' if payment_only else '없음'}",
        f"- 기간별 기록 수: {_compact_breakdown(counts, suffix)}",
    )), sources, "pandas"


def _answer_extreme_group_amount(
    question: str,
    dataframes: dict[str, pd.DataFrame],
) -> tuple[str, list[str], str] | None:
    """Rank a semantic group by summed amount without requiring the word '별'."""

    group_terms = (("학과", "department"), ("기수", "cohort"), ("회비 구분", "fee_type"))
    match = next(((label, qualifier) for label, qualifier in group_terms if label in question), None)
    if match is None or not re.search(r"가장|제일|최고|최저", question):
        return None
    label, qualifier = match
    is_highest = bool(re.search(r"가장\s*(?:많|크|높)|제일\s*(?:많|크|높)|최고", question))
    sources = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in dataframes))
    if len(sources) != 1:
        return f"{label}별 금액 비교는 한 번에 한 문서를 선택했을 때만 지원합니다. 비교할 문서를 하나 선택해 주세요.", sources, "pandas"
    totals: dict[str, float] = {}
    amount_labels: set[str] = set()
    usable_frames = 0
    for df in dataframes.values():
        group_column = next((column for column in df.columns if infer_column_meaning(str(column), df[column]).qualifier == qualifier), None)
        selection = resolve_amount_column(df, question)
        if group_column is None or selection.selected is None:
            if group_column is not None and len(selection.candidates) > 1:
                return amount_column_clarification(selection.candidates), sources, "pandas"
            continue
        usable_frames += 1
        amount_labels.add(selection.selected)
        amounts = money_series(df, selection.selected)
        for group, amount in amounts.groupby(df[group_column].astype("string").str.strip()).sum().items():
            if pd.isna(group) or not str(group):
                continue
            totals[str(group)] = totals.get(str(group), 0.0) + float(amount)
    if not usable_frames or not totals:
        return f"선택한 문서에서 {label}와 금액 컬럼을 함께 찾지 못했습니다.", sources, "pandas"
    extreme = max(totals.values()) if is_highest else min(totals.values())
    groups = sorted(group for group, total in totals.items() if total == extreme)
    group_text = ", ".join(groups)
    comparison_label = "가장 많은" if is_highest else "가장 적은"
    return "\n".join((
        f"금액 합계가 {comparison_label} {label}는 {group_text}입니다.",
        "",
        f"- {group_text}: {extreme:,.0f}원",
        "",
        "조회 근거:",
        f"- 문서: {sources[0]}",
        f"- 원본 행 수: {_source_row_count(dataframes):,}개",
        f"- 계산 방식: {label}별 금액 합계를 계산한 뒤 {'최댓값' if is_highest else '최솟값'} 선택",
        f"- 대상 컬럼: {', '.join(sorted(amount_labels))}",
        f"- 비교한 {label} 수: {len(totals):,}개",
        f"- {label}별 합계: {', '.join(f'{group} {amount:,.0f}원' for group, amount in sorted(totals.items()))}",
    )), sources, "pandas"


def _answer_most_frequent_person(
    question: str,
    dataframes: dict[str, pd.DataFrame],
) -> tuple[str, list[str], str] | None:
    """Rank people by record count when the question explicitly asks frequency."""

    if not _MOST_FREQUENT_PERSON.search(question):
        return None
    counts: dict[str, tuple[str, int]] = {}
    sources = list(dict.fromkeys(_df_sources.get(alias, alias) for alias in dataframes))
    usable_frames = 0
    payment_only = bool(re.search(r"납부|결제|낸", question))
    for df in dataframes.values():
        person_column = _person_name_column(df)
        if person_column is None:
            continue
        mask = df[person_column].notna()
        if payment_only:
            selection = resolve_amount_column(df, question)
            if selection.selected is None:
                if len(selection.candidates) > 1:
                    return amount_column_clarification(selection.candidates), sources, "pandas"
                continue
            mask = mask & money_series(df, selection.selected).gt(0).fillna(False)
        usable_frames += 1
        for raw_name in df.loc[mask, person_column].astype(str):
            display_name = raw_name.strip()
            normalized_name = normalize_person_name(display_name)
            if not normalized_name:
                continue
            previous = counts.get(normalized_name)
            counts[normalized_name] = (
                previous[0] if previous else display_name,
                (previous[1] if previous else 0) + 1,
            )
    if not usable_frames or not counts:
        return "선택한 문서에서 사람 이름 컬럼을 찾지 못했습니다.", sources, "pandas"
    is_least = bool(re.search(r"가장\s*(?:적|작|낮)|제일\s*(?:적|작|낮)|최소", question))
    frequency = (
        min(count for _, count in counts.values())
        if is_least else max(count for _, count in counts.values())
    )
    people = sorted(name for name, count in counts.values() if count == frequency)
    label = "가장 적은" if is_least else "가장 많은"
    lines = [f"기록 횟수가 {label} 사람은 {len(people):,}명입니다.", ""]
    lines.extend(f"- {name}: {frequency:,}건" for name in people)
    lines.extend((
        "",
        "조회 근거:",
        f"- 문서: {', '.join(sources)}",
        f"- 원본 행 수: {_source_row_count(dataframes):,}개",
        f"- 계산 방식: 회원명별 기록 행 수를 집계한 뒤 {'최솟값' if is_least else '최댓값'} 선택",
        f"- 적용 조건: {'결제 금액이 0원 초과' if payment_only else '없음'}",
        f"- 비교한 고유 인원: {len(counts):,}명",
    ))
    return "\n".join(lines), sources, "pandas"


def _answer_extreme_value_comparison(
    question: str,
    analysis: QuestionAnalysis,
) -> tuple[str, list[str], str] | None:
    """Compare one minimum and one maximum value without an LLM-generated plan."""

    intents = analysis.aggregation_intents
    if (
        {intent.operation for intent in intents} != {"min", "max"}
        or any(intent.target != "value" for intent in intents)
    ):
        return None
    payloads: dict[str, dict] = {}
    sources: list[str] = []
    for intent in intents:
        result, result_sources = _query_pandas_direct(
            question,
            aggregation_intents=[intent],
            date_filter=analysis.date_filter,
        )
        if not isinstance(result, dict) or result.get("type") != "aggregation":
            return None
        payloads[intent.operation] = result
        sources.extend(result_sources)
    minimum = float(payloads["min"]["value"])
    maximum = float(payloads["max"]["value"])
    label = str(payloads["max"].get("label") or payloads["min"].get("label") or "금액")
    answer = (
        f"{label} 최댓값은 {maximum:,.0f}원이고 최솟값은 {minimum:,.0f}원입니다. "
        f"두 값의 차이는 {maximum - minimum:,.0f}원입니다."
    )
    return answer, list(dict.fromkeys(sources)), "pandas"


def _format_direct_dataframe_with_evidence(
    df: pd.DataFrame,
    question: str,
    sources: list[str],
) -> str:
    """Expose the same count observability for verified direct-query handlers.

    Direct handlers predate QueryPlan and return only a DataFrame.  Their result
    set is nevertheless deterministic, so report its row count and the
    schema-derived distinct-person count instead of making evaluation fall back
    to unsafe answer-keyword matching.
    """
    person_columns = [
        column
        for column in df.columns
        if (
            (meaning := infer_column_meaning(str(column), df[column])).concept == "entity"
            and meaning.role == "entity_name"
            and meaning.qualifier == "person"
        )
    ]
    if re.search(r"(?:전체|모든)\s*(?:회원|사람|인원).*(?:보여|목록|명단|리스트|조회)", question):
        if person_columns:
            person_column = max(
                person_columns,
                key=lambda column: int(df[column].dropna().nunique()),
            )
            df = df.drop_duplicates(subset=[person_column], keep="first").copy()
            person_columns = [person_column]
    lines = [
        _format_dataframe_result_for_question(df, question),
        "",
        "조회 근거:",
        f"- 문서: {', '.join(sources) if sources else '알 수 없음'}",
        "- 실행 방식: 검증된 직접 조회",
        f"- 조건 통과 {len(df):,}건",
    ]
    if person_columns:
        # Some normalized tables retain multiple person-like identity columns
        # (for example a display label plus the actual member name).  The
        # identifier with the broadest non-null population is the least lossy
        # schema-derived representative; this is independent of any document
        # or column name.
        person_column = max(
            person_columns,
            key=lambda column: int(df[column].dropna().nunique()),
        )
        unique_people = int(df[person_column].dropna().nunique())
        lines.append(f"- 조건 충족 고유 인원: {unique_people:,}명")
    answer = "\n".join(lines)
    _interactive_result.set(
        build_interactive_dataframe(df, answer=answer, question=question)
    )
    return answer


async def _answer_query_plan(
    question: str,
    *,
    allow_vector_fallback: bool,
    analysis: QuestionAnalysis | None = None,
    operation_hint: str | None = None,
    prepared_plan: QueryPlan | None = None,
) -> tuple[str, list[str], str]:
    """Generate, validate, and execute the generic structured-query plan."""

    question_id, question_chars = question_log_metadata(question)
    raise_if_cancelled()
    logger.info(
        "[PANDAS] QueryPlan 생성 중 | hint=%s question_id=%s chars=%d",
        operation_hint or "none", question_id, question_chars,
    )
    early_plan = prepared_plan or build_schema_grounded_plan(
        question,
        dataframes=scoped_mapping(_df_namespace, _df_sources),
        operation_hint=operation_hint,
    )
    if early_plan is not None:
        from pandas_engine.plan_validator import validate_query_plan

        early_validation = validate_query_plan(
            early_plan,
            question=question,
            operation_hint=operation_hint,
        )
        if early_validation.is_executable:
            execution = await await_cancellable(
                asyncio.to_thread(execute_query_plan, early_validation)
            )
            raise_if_cancelled()
            answer = _format_query_execution_result(execution, question)
            _interactive_result.set(
                build_interactive_result(execution, answer=answer, question=question)
            )
            logger.info("[PANDAS] 스키마 기반 선행 계획 실행 | operation=%s", execution.operation)
            return answer, [execution.source_file], "pandas"
    try:
        validation = await generate_validated_query_plan(
            question,
            operation_hint=operation_hint,
        )
        raise_if_cancelled()
    except QueryPlannerError as exc:
        logger.error("[PANDAS] QueryPlan 생성 실패 | err=%s", exc)
        fallback_plan = build_schema_grounded_plan(
            question,
            dataframes=scoped_mapping(_df_namespace, _df_sources),
            operation_hint=operation_hint,
        )
        if fallback_plan is not None:
            from pandas_engine.plan_validator import validate_query_plan

            fallback_validation = validate_query_plan(
                fallback_plan,
                question=question,
                operation_hint=operation_hint,
            )
            if fallback_validation.is_executable:
                execution = await await_cancellable(
                    asyncio.to_thread(execute_query_plan, fallback_validation)
                )
                raise_if_cancelled()
                answer = _format_query_execution_result(execution, question)
                _interactive_result.set(
                    build_interactive_result(execution, answer=answer, question=question)
                )
                logger.warning("[PANDAS] 스키마 기반 폴백 계획 실행 | operation=%s", execution.operation)
                return answer, [execution.source_file], "pandas"
        return (
            "질문을 안전한 표 조회 계획으로 변환하지 못했습니다. "
            "조회할 항목과 조건을 조금 더 명확하게 입력해 주세요.",
            [],
            "pandas",
        )

    if validation.status == "clarification":
        message = validation.plan.message or next(
            (issue.message for issue in validation.issues if issue.message),
            "조회할 문서나 항목을 하나로 지정해 주세요.",
        )
        if validation.plan.candidates:
            message += " 후보: " + ", ".join(validation.plan.candidates)
        logger.info(
            "[PANDAS] QueryPlan 추가 확인 필요 | message_chars=%d",
            len(message),
        )
        return message, [], "pandas"

    if validation.status == "not_applicable":
        if not allow_vector_fallback:
            return validation.plan.message or "표 조회로 처리할 수 없는 질문입니다.", [], "pandas"
        logger.info("[PANDAS→VECTOR] QueryPlan이 문서 내용 검색으로 판정")
        from rag.vector import _answer_vector

        v_answer, v_sources, _ = await _answer_vector(
            question,
            allow_pandas_fallback=False,
            analysis=analysis,
        )
        raise_if_cancelled()
        return v_answer, v_sources, "vector"

    if not validation.is_executable:
        issue_codes = ", ".join(issue.code for issue in validation.issues)
        logger.warning("[PANDAS] QueryPlan 검증 실패 | issues=%s", issue_codes)
        fallback_plan = build_schema_grounded_plan(
            question,
            dataframes=scoped_mapping(_df_namespace, _df_sources),
            operation_hint=operation_hint,
        )
        if fallback_plan is not None:
            from pandas_engine.plan_validator import validate_query_plan

            fallback_validation = validate_query_plan(
                fallback_plan,
                question=question,
                operation_hint=operation_hint,
            )
            if fallback_validation.is_executable:
                execution = await await_cancellable(
                    asyncio.to_thread(execute_query_plan, fallback_validation)
                )
                raise_if_cancelled()
                answer = _format_query_execution_result(execution, question)
                _interactive_result.set(
                    build_interactive_result(execution, answer=answer, question=question)
                )
                logger.warning("[PANDAS] 검증 실패 후 스키마 기반 폴백 실행 | operation=%s", execution.operation)
                return answer, [execution.source_file], "pandas"
        if any(
            issue.code in {"literal_mismatch", "ungrounded_numeric_filter"}
            for issue in validation.issues
        ):
            return (
                "질문의 숫자·단위·비교 조건이 조회 계획에서 달라져 "
                "안전을 위해 실행을 중단했습니다. 같은 질문을 다시 시도해 주세요.",
                [],
                "pandas",
            )
        return (
            "질문을 실제 표의 컬럼과 안전하게 연결하지 못했습니다. "
            "문서에 표시된 항목명과 조회 조건을 확인해 주세요.",
            [],
            "pandas",
        )

    try:
        execution = await await_cancellable(
            asyncio.to_thread(execute_query_plan, validation)
        )
        raise_if_cancelled()
    except QueryPlanExecutionError as exc:
        logger.error("[PANDAS] QueryPlan 실행 차단 | err=%s", exc)
        return "검증된 표 조회 계획을 실행하지 못했습니다.", [], "pandas"

    logger.info(
        "[PANDAS] QueryPlan 실행 완료 | operation=%s matched=%d source=%s",
        execution.operation,
        execution.matched_rows,
        execution.source_file,
    )
    answer = _format_query_execution_result(execution, question)
    _interactive_result.set(
        build_interactive_result(execution, answer=answer, question=question)
    )
    return answer, [execution.source_file], "pandas"


async def _answer_pandas(
    question: str,
    allow_vector_fallback: bool = True,
    analysis: QuestionAnalysis | None = None,
    strategy: Literal["AUTO", "DIRECT", "QUERY_PLAN"] = "AUTO",
    operation_hint: str | None = None,
    prepared_plan: QueryPlan | None = None,
) -> tuple[str, list[str], str]:
    clear_interactive_result()
    raise_if_cancelled()
    if _AMBIGUOUS_SUMMARY_QUESTION.fullmatch(question):
        return (
            "무엇을 확인할지 알려주세요. 예: 총 인원, 총 기록 수, 총 금액",
            [],
            "pandas",
        )
    scoped_dataframes = scoped_mapping(_df_namespace, _df_sources)
    if not scoped_dataframes:
        message = "선택한 문서에서 조회 가능한 표 데이터를 찾을 수 없습니다." if source_scope_active() else "현재 로드된 데이터프레임이 없습니다."
        return message, [], "pandas"

    most_frequent_person = _answer_most_frequent_person(question, scoped_dataframes)
    if most_frequent_person is not None:
        return most_frequent_person

    extreme_period_count = _answer_extreme_period_count(question, scoped_dataframes)
    if extreme_period_count is not None:
        return extreme_period_count

    extreme_group_amount = _answer_extreme_group_amount(question, scoped_dataframes)
    if extreme_group_amount is not None:
        return extreme_group_amount

    year_nonpayer_comparison = _answer_year_nonpayer_comparison(question, scoped_dataframes)
    if year_nonpayer_comparison is not None:
        return year_nonpayer_comparison

    year_person_comparison = _answer_year_person_comparison(question, scoped_dataframes)
    if year_person_comparison is not None:
        return year_person_comparison

    period_comparison = _answer_period_payment_comparison(question, scoped_dataframes)
    if period_comparison is not None:
        return period_comparison

    group_comparison = _answer_group_payment_comparison(question, scoped_dataframes)
    if group_comparison is not None:
        return group_comparison

    lowest_year_amount = _answer_lowest_year_amount(question, scoped_dataframes)
    if lowest_year_amount is not None:
        return lowest_year_amount

    extreme_month_amount = _answer_extreme_month_amount(question, scoped_dataframes)
    if extreme_month_amount is not None:
        return extreme_month_amount

    year_comparison = _answer_year_amount_comparison(question, scoped_dataframes)
    if year_comparison is not None:
        return year_comparison

    analysis = analysis or analyze_question(question)

    if "compare" in analysis.operations:
        comparison = _answer_extreme_value_comparison(question, analysis)
        if comparison is not None:
            return comparison
        return (
            "현재 이 비교 조건은 한 번에 안전하게 계산할 수 없습니다. "
            "비교할 대상과 범위를 조금 더 명확하게 입력해 주세요.",
            [],
            "pandas",
        )

    if strategy == "QUERY_PLAN":
        # R.JSON may deliberately choose QUERY_PLAN for a structured request,
        # but explicit date ranges already have a schema-aware deterministic
        # executor. Run it before P.JSON generation so a cross-year month
        # range cannot be reduced to the first year/month by an LLM plan.
        if analysis.date_filter is not None:
            direct_result, direct_sources = _query_pandas_direct(
                question,
                aggregation_intents=analysis.aggregation_intents,
                date_filter=analysis.date_filter,
            )
            if isinstance(direct_result, pd.DataFrame):
                return (
                    _format_direct_dataframe_with_evidence(
                        direct_result, question, direct_sources
                    ),
                    direct_sources,
                    "pandas",
                )
            return _format_scalar_result(direct_result, question), direct_sources, "pandas"

        # 마스킹 이름은 검증된 전용 검색기가 있다. LLM이 단순 이름 조회를
        # structured_query로 오분류해도, 별도 숫자·범위 조건이 없는 경우에만
        # QueryPlan보다 안전한 직접 검색 결과를 우선한다.
        if (
            operation_hint != "lookup_field"
            and
            has_explicit_masked_name(question)
            and not _has_explicit_structured_filter(question)
            and not _NUMERIC_COMPARISON_FILTER.search(question)
        ):
            name_df, name_sources, name_searched = _search_name_pandas(question)
            if name_df is not None:
                if not source_scope_active() and len(name_sources) > 1:
                    names = ", ".join(name_sources[:5])
                    return (
                        "같은 이름의 기록이 여러 문서에서 발견되었습니다. "
                        f"조회할 문서를 선택해주세요: {names}",
                        name_sources,
                        "pandas",
                    )
                logger.info(
                    "[NAME_SEARCH] QueryPlan 오분류 복구 | rows=%d",
                    len(name_df),
                )
                return (
                    _format_direct_dataframe_with_evidence(name_df, question, name_sources),
                    name_sources,
                    "pandas",
                )
            if name_searched:
                return "조회된 데이터가 없습니다.", [], "pandas"
        return await _answer_query_plan(
            question,
            allow_vector_fallback=allow_vector_fallback,
            analysis=analysis,
            operation_hint=operation_hint,
            prepared_plan=prepared_plan,
        )

    # 기본 통계는 LLM 코드 생성이나 VECTOR 검색으로 넘기지 않고 검증된 함수로 계산한다.
    if analysis.aggregation_intents:
        direct_result, direct_sources = await await_cancellable(
            asyncio.to_thread(
                _query_pandas_direct,
                question,
                aggregation_intents=analysis.aggregation_intents,
                date_filter=analysis.date_filter,
            )
        )
        raise_if_cancelled()
        if direct_result is None:
            # Person-ranking aggregation may be structurally valid even when
            # the legacy direct aggregator cannot produce its subject payload.
            # Let the schema-grounded QueryPlan handle it instead of turning
            # an unsupported direct shape into a false no-data response.
            person_ranking = any(
                intent.operation in {"min", "max"}
                and intent.target in {"person_total", "row"}
                for intent in analysis.aggregation_intents
            )
            if not person_ranking:
                return "조회된 데이터가 없습니다.", [], "pandas"
        else:
            direct_notice = (
                isinstance(direct_result, dict)
                and direct_result.get("type") == "aggregation_notice"
            )
            person_ranking = any(
                intent.operation in {"min", "max"}
                and intent.target in {"person_total", "row"}
                for intent in analysis.aggregation_intents
            )
            if direct_notice and person_ranking:
                direct_result = None
            else:
                logger.info("[AGGREGATION] 고정 집계 실행 | source=%s", direct_sources)
                return _format_scalar_result(direct_result, question), direct_sources, "pandas"

    # 1단계: 이름 전수 검색 (기존)
    # 숫자 비교가 명시된 복합 필터 질문에서 일반 조건어를 마스킹 이름으로
    # 유사 매칭하지 않는다. 실제 이름 단순 조회는 기존 전수 검색을 유지한다.
    if _NUMERIC_COMPARISON_FILTER.search(question):
        name_df, name_sources, name_searched = None, [], False
    else:
        name_df, name_sources, name_searched = _search_name_pandas(question)
    if name_df is not None:
        if not source_scope_active() and len(name_sources) > 1:
            names = ", ".join(name_sources[:5])
            return (
                f"같은 이름의 기록이 여러 문서에서 발견되었습니다. 조회할 문서를 선택해주세요: {names}",
                name_sources,
                "pandas",
            )
        logger.info("[NAME_SEARCH] %d건 발견, 코드 생성 생략", len(name_df))
        return _format_direct_dataframe_with_evidence(name_df, question, name_sources), name_sources, "pandas"
    if name_searched and re.search(
        r"이라는|라는\s*학생|학생이.{0,20}(?:장학금|받|있)|받았[나어요이]|있[나어]\s*[?？]?$",
        question,
    ):
        # 특정 인물 조회(이라는/학생이...받았어 등) → 이름이 없으면 바로 없음 반환
        logger.info("[NAME_SEARCH] 특정 인물 조회 패턴 — 데이터 없음")
        return "조회된 데이터가 없습니다.", [], "pandas"

    # 2단계: 키워드 직접 조회 (LLM 코드 생성 없음)
    direct_result, direct_sources = await await_cancellable(
        asyncio.to_thread(
            _query_pandas_direct,
            question,
            aggregation_intents=analysis.aggregation_intents,
            date_filter=analysis.date_filter,
        )
    )
    raise_if_cancelled()
    if direct_result is not None:
        formatted = _format_pandas_result(direct_result)
        if formatted != "조회된 데이터가 없습니다.":
            logger.info("[DIRECT] 직접 조회 성공 | source=%s", direct_sources)
            if isinstance(direct_result, pd.DataFrame):
                return _format_direct_dataframe_with_evidence(
                    direct_result, question, direct_sources
                ), direct_sources, "pandas"
            # scalar(int/float/str): LLM 우회, 직접 포맷
            return _format_scalar_result(direct_result, question), direct_sources, "pandas"

    # Analyzer가 목록 요청으로 확정한 경우 LLM이 len(df)와 df 반환 사이에서
    # 임의로 선택하지 않도록 선택 문서의 전체 행을 직접 반환한다.
    if "list_records" in analysis.operations:
        list_result, list_sources = _query_all_records()
        if isinstance(list_result, pd.DataFrame):
            logger.info("[LIST_RECORDS] 전체 목록 직접 조회 | source=%s", list_sources)
            return _format_direct_dataframe_with_evidence(
                list_result, question, list_sources
            ), list_sources, "pandas"
        return _format_scalar_result(list_result, question), list_sources, "pandas"

    if _has_explicit_structured_filter(question):
        logger.info("[PANDAS] 명시된 기수/식별번호와 일치하는 데이터 없음")
        return "조회된 데이터가 없습니다.", [], "pandas"

    # 3단계: 검증된 직접 조회로 처리하지 못한 구조화 질문은 Python 코드를
    # 생성하지 않고 제한된 QueryPlan으로 변환한다.
    return await _answer_query_plan(
        question,
        allow_vector_fallback=allow_vector_fallback,
        analysis=analysis,
        operation_hint=operation_hint,
    )
