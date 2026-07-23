"""
골드셋 기반 RAG 시스템 평가 스크립트

사용법:
    # feature/update-models 브랜치 평가 (기본)
    python eval.py --tag update-models

    # feature/experiment 브랜치 평가 (sql → pandas 매핑)
    python eval.py --tag experiment --route-alias sql=pandas

    # 특정 케이스 / 카테고리만
    python eval.py --id TC001
    python eval.py --category pandas_명단

    # 서버 주소 변경
    python eval.py --url http://localhost:8081
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

GOLDSET_PATH = Path(__file__).parent / "goldset.json"
DEFAULT_URL  = "http://localhost:8080"
RESULT_DIR   = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# 채점 로직
# ---------------------------------------------------------------------------

def score_keywords(answer: str, keywords: list[str]) -> tuple[float, list[str], list[str]]:
    """공백 무시 부분 일치로 키워드 재현율 계산."""
    answer_norm = re.sub(r"\s+", "", answer.lower())
    hit, miss = [], []
    for kw in keywords:
        kw_norm = re.sub(r"\s+", "", kw.lower())
        (hit if kw_norm in answer_norm else miss).append(kw)
    recall = len(hit) / len(keywords) if keywords else 1.0
    return recall, hit, miss


def _normalize_text(value: object) -> str:
    """Compare human-facing values without making keyword recall a verdict."""
    return re.sub(r"\s+", "", str(value).casefold())


def _number_forms(value: int | float) -> tuple[str, ...]:
    """Return the Korean display forms that are safe to accept for one number."""
    numeric = int(value) if float(value).is_integer() else value
    return tuple(dict.fromkeys((str(numeric), f"{numeric:,}")))


def _contains_number(answer: str, value: int | float) -> bool:
    normalized = _normalize_text(answer).replace(",", "")
    return any(form.replace(",", "") in normalized for form in _number_forms(value))


def _contains_money(answer: str, value: int | float) -> bool:
    """Accept equivalent Korean money displays such as ``2만원`` and ``20,000원``."""
    expected = float(value)
    if _contains_number(answer, expected):
        return True
    money_pattern = re.compile(
        r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(?:원|천원|만원|억원)(?!\w)"
    )
    for match in money_pattern.finditer(answer):
        text = match.group(0).replace(",", "").replace(" ", "")
        unit = next((candidate for candidate in ("억원", "만원", "천원", "원") if text.endswith(candidate)), "")
        numeric = text[:-len(unit)] if unit else text
        try:
            actual = float(numeric) * {"원": 1, "천원": 1_000, "만원": 10_000, "억원": 100_000_000}[unit]
        except (KeyError, ValueError):
            continue
        if actual is not None and abs(actual - expected) < 1e-9:
            return True
    return False


def _matched_rows_from_evidence(answer: str) -> int | None:
    """Read the deterministic QueryPlan evidence, not arbitrary answer prose."""
    match = re.search(r"조건\s*통과\s*([\d,]+)\s*(?:건|개)", answer)
    return int(match.group(1).replace(",", "")) if match else None


def _unique_people_from_evidence(answer: str) -> int | None:
    match = re.search(r"조건\s*충족\s*고유\s*인원\s*:\s*([\d,]+)\s*명", answer)
    return int(match.group(1).replace(",", "")) if match else None


def _is_no_data_answer(answer: str) -> bool:
    """Recognize the normal user-facing response for an empty filtered result."""
    normalized = _normalize_text(answer)
    return any(
        phrase in normalized
        for phrase in (
            "\uc870\ud68c\ub41c\uae08\uc561\uc774\uc5c6\uc2b5\ub2c8\ub2e4",
            "\uc870\ud68c\ub41c\uc815\ubcf4\uac00\uc5c6\uc2b5\ub2c8\ub2e4",
            "\uc870\ud68c\uacb0\uacfc\uac00\uc5c6\uc2b5\ub2c8\ub2e4",
            "\uc870\ud68c\ub41c\ub370\uc774\ud130\uac00\uc5c6\uc2b5\ub2c8\ub2e4",
        )
    )


def _is_missing_value_answer(answer: str) -> bool:
    """Recognize an existing record whose requested field is blank/null."""
    normalized = _normalize_text(answer)
    return any(
        phrase in normalized
        for phrase in (
            "\uc5c6\uc74c",
            "\ube44\uc5b4\uc788\uc2b5\ub2c8\ub2e4",
            "\ubbf8\ub4f1\ub85d",
            "\uc785\ub825\ub418\uc9c0\uc54a\uc558\uc2b5\ub2c8\ub2e4",
            "\uc815\ubcf4\uac00\uc5c6\uc2b5\ub2c8\ub2e4",
        )
    )


def _is_clarification_answer(answer: str) -> bool:
    normalized = _normalize_text(answer)
    return "여러명" in normalized and "전체이름" in normalized


def _person_totals_from_answer(answer: str) -> list[dict[str, int]]:
    totals = []
    for amount, payments in re.findall(
        r":\s*(\d[\d,]*)원\s*\((\d+)건\)",
        str(answer or ""),
    ):
        totals.append({
            "amount": int(amount.replace(",", "")),
            "payments": int(payments),
        })
    return totals


def score_structured_facts(answer: str, expected: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    """Verify goldset facts against scalar output and deterministic execution evidence.

    ``ground_truth_keywords`` remain a diagnostic aid only.  In particular, a
    name occurring somewhere in an unfiltered table can never make a list
    question pass: its QueryPlan evidence must report the expected row count.
    """
    checks: list[dict[str, Any]] = []

    def add(field: str, expected_value: object, actual: object, ok: bool) -> None:
        checks.append({"field": field, "expected": expected_value, "actual": actual, "ok": ok})

    if "record_count" in expected:
        actual_rows = _matched_rows_from_evidence(answer)
        expected_rows = expected["record_count"]
        empty_result = expected_rows == 0 and _is_no_data_answer(answer)
        add("record_count", expected_rows, 0 if empty_result else actual_rows,
            actual_rows == expected_rows or empty_result)

    if "unique_people" in expected:
        expected_people = expected["unique_people"]
        actual_people = _unique_people_from_evidence(answer)
        add("unique_people", expected_people, actual_people, actual_people == expected_people)

    if "person_totals" in expected:
        expected_totals = sorted(
            (int(item["amount"]), int(item["payments"]))
            for item in expected["person_totals"]
        )
        actual_items = _person_totals_from_answer(answer)
        actual_totals = sorted(
            (item["amount"], item["payments"])
            for item in actual_items
        )
        add("person_totals", expected["person_totals"], actual_items, actual_totals == expected_totals)

    for field in (
        "amount", "min_amount", "max_amount", "mode_amount",
        "name", "matched_name", "major", "email", "phone", "registered_at",
    ):
        if field not in expected:
            continue
        value = expected[field]
        empty_result = expected.get("record_count") == 0 and _is_no_data_answer(answer)
        missing_value = value is None and _is_missing_value_answer(answer)
        ok = empty_result or missing_value if (empty_result or missing_value) else (
            _contains_money(answer, value)
            if field in {"amount", "min_amount", "max_amount", "mode_amount"}
            else _normalize_text(value) in _normalize_text(answer)
        )
        add(field, value, 0 if field == "amount" and empty_result else None, ok)

    if "fee_types" in expected:
        for value in expected["fee_types"]:
            add("fee_types", value, None, _normalize_text(value) in _normalize_text(answer))

    if "exists" in expected:
        # The remaining expected facts (for example amount and record_count)
        # establish existence.  A negative case must explicitly say no data.
        no_data = _is_no_data_answer(answer)
        add("exists", expected["exists"], None, not no_data if expected["exists"] else no_data)

    if "clarification" in expected:
        add("clarification", expected["clarification"], None,
            _is_clarification_answer(answer) == expected["clarification"])

    # Goldset may add explicit exclusions for list questions.  Supporting this
    # now prevents a future return-the-whole-table false positive.
    for value in expected.get("forbidden_values", []):
        add("forbidden_values", value, None, _normalize_text(value) not in _normalize_text(answer))

    return bool(checks) and all(check["ok"] for check in checks), checks


def evaluate_case(
    tc: dict[str, Any],
    base_url: str,
    route_alias: dict[str, str],
    endpoint: str = "/chat",
) -> dict[str, Any]:
    question       = tc["question"]
    expected_route = tc["expected_route"]
    keywords       = tc["ground_truth_keywords"]
    expected       = tc.get("expected", {})

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}{endpoint}",
            json={"question": question},
            timeout=120,
        )
        elapsed = round(time.perf_counter() - start, 2)
        resp.raise_for_status()
        data         = resp.json()
        answer       = data.get("answer", "")
        actual_route = data.get("source", "unknown").lower()
        error        = None
    except Exception as e:
        elapsed      = round(time.perf_counter() - start, 2)
        answer       = ""
        actual_route = "error"
        error        = str(e)

    # route-alias 적용 (ex. sql → pandas)
    normalized_route = route_alias.get(actual_route, actual_route)
    route_ok = normalized_route == expected_route

    keyword_recall, hit, miss = score_keywords(answer, keywords)

    facts_ok, fact_checks = score_structured_facts(answer, expected)
    # Route correctness is necessary but never sufficient.  Keyword recall is
    # intentionally excluded from the verdict because it cannot prove filters,
    # row counts, or scalar values.
    passed = route_ok and facts_ok and error is None

    return {
        "id":              tc["id"],
        "question":        question,
        "category":        tc["category"],
        "difficulty":      tc["difficulty"],
        "expected_route":  expected_route,
        "actual_route":    actual_route,
        "normalized_route": normalized_route,
        "route_ok":        route_ok,
        "keyword_recall":  round(keyword_recall, 3),
        "hit_keywords":    hit,
        "miss_keywords":   miss,
        "fact_checks":     fact_checks,
        "facts_ok":        facts_ok,
        "passed":          passed,
        "elapsed_sec":     elapsed,
        "answer_preview":  answer[:500] if answer else "",
        "error":           error,
    }


# ---------------------------------------------------------------------------
# 콘솔 출력
# ---------------------------------------------------------------------------

def print_result(r: dict[str, Any], verbose: bool = False) -> None:
    status     = "O" if r["passed"] else "X"
    route_mark = "O" if r["route_ok"] else "X"
    alias_note = (
        f"({r['actual_route']}→{r['normalized_route']})"
        if r["actual_route"] != r["normalized_route"] else ""
    )
    print(
        f"[{status}] [{r['id']}] {r['question'][:40]:<42}"
        f"  route:{r['normalized_route']:7}{alias_note}[{route_mark}]"
        f"  kw:{r['keyword_recall']:.0%}"
        f"  {r['elapsed_sec']}s"
    )
    if not r["passed"] or verbose:
        failed_facts = [check for check in r["fact_checks"] if not check["ok"]]
        if failed_facts:
            print(f"     구조화 검증 실패: {failed_facts}")
        if r["miss_keywords"]:
            print(f"     누락 키워드: {r['miss_keywords']}")
        if r["error"]:
            print(f"     오류: {r['error']}")
        if verbose and r["answer_preview"]:
            print(f"     답변: {r['answer_preview']}")


def print_summary(results: list[dict[str, Any]], tag: str) -> None:
    total     = len(results)
    passed    = sum(r["passed"] for r in results)
    route_acc = sum(r["route_ok"] for r in results) / total if total else 0
    avg_recall = sum(r["keyword_recall"] for r in results) / total if total else 0
    avg_time  = sum(r["elapsed_sec"] for r in results) / total if total else 0

    print("\n" + "=" * 60)
    print(f"  [{tag}] 전체 결과: {passed}/{total} 통과  ({passed/total:.0%})")
    print(f"  라우팅 정확도: {route_acc:.0%}")
    print(f"  평균 키워드 재현율: {avg_recall:.0%}")
    print(f"  평균 응답 시간: {avg_time:.1f}s")
    print("=" * 60)

    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    print("\n  카테고리별:")
    for cat, rs in sorted(cats.items()):
        p = sum(x["passed"] for x in rs)
        print(f"    {cat:<20} {p}/{len(rs)} ({p/len(rs):.0%})")

    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"\n  실패 케이스 ({len(failed)}개):")
        for r in failed:
            print(f"    [{r['id']}] {r['question'][:50]}")


# ---------------------------------------------------------------------------
# Excel 저장
# ---------------------------------------------------------------------------

def save_excel(results: list[dict[str, Any]], path: Path) -> None:
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        print("  [경고] pandas 없음 — Excel 저장 건너뜀")
        return

    rows = []
    for r in results:
        rows.append({
            "ID":          r["id"],
            "카테고리":     r["category"],
            "난이도":      r["difficulty"],
            "질문":        r["question"],
            "기대 라우팅":  r["expected_route"],
            "실제 라우팅":  r["actual_route"],
            "라우팅 정규화": r["normalized_route"],
            "라우팅 성공":  "O" if r["route_ok"] else "X",
            "키워드 재현율": f"{r['keyword_recall']:.0%}",
            "적중 키워드":  ", ".join(r["hit_keywords"]),
            "누락 키워드":  ", ".join(r["miss_keywords"]),
            "구조화 검증":  "O" if r["facts_ok"] else "X",
            "구조화 실패":  json.dumps(
                [check for check in r["fact_checks"] if not check["ok"]],
                ensure_ascii=False,
            ),
            "통과":        "O" if r["passed"] else "X",
            "소요시간(s)": r["elapsed_sec"],
            "답변 미리보기": r["answer_preview"],
            "오류":        r["error"] or "",
        })

    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    print(f"  Excel 저장: {path}")


# ---------------------------------------------------------------------------
# Markdown 리포트
# ---------------------------------------------------------------------------

def save_markdown(
    results: list[dict[str, Any]],
    path: Path,
    base_url: str,
    tag: str,
    route_alias: dict[str, str],
) -> None:
    total      = len(results)
    passed     = sum(r["passed"] for r in results)
    route_acc  = sum(r["route_ok"] for r in results) / total if total else 0
    avg_recall = sum(r["keyword_recall"] for r in results) / total if total else 0
    avg_time   = sum(r["elapsed_sec"] for r in results) / total if total else 0
    now        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alias_str = ", ".join(f"{k}→{v}" for k, v in route_alias.items()) if route_alias else "없음"

    lines = [
        f"# RAG 평가 리포트 — `{tag}`",
        "",
        f"**생성 일시**: {now}  ",
        f"**서버**: {base_url}  ",
        f"**브랜치 태그**: `{tag}`  ",
        f"**라우팅 별칭**: {alias_str}  ",
        f"**테스트 케이스**: {total}개",
        "",
        "---",
        "",
        "## 전체 결과",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| 통과 | **{passed}/{total}** ({passed/total:.0%}) |",
        f"| 라우팅 정확도 | {route_acc:.0%} |",
        f"| 평균 키워드 재현율 | {avg_recall:.0%} |",
        f"| 평균 응답 시간 | {avg_time:.1f}s |",
        "",
        "## 카테고리별 결과",
        "",
        "| 카테고리 | 통과 | 전체 | 정확도 |",
        "|----------|:----:|:----:|:------:|",
    ]

    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    for cat, rs in sorted(cats.items()):
        p    = sum(x["passed"] for x in rs)
        icon = "O" if p / len(rs) >= 0.7 else ("△" if p / len(rs) >= 0.4 else "X")
        lines.append(f"| {icon} {cat} | {p} | {len(rs)} | {p/len(rs):.0%} |")

    lines += [
        "",
        "## 케이스별 결과",
        "",
        "| ID | 질문 | 카테고리 | 라우팅(기대→실제) | KW 재현율 | 시간 | 결과 |",
        "|----|------|----------|-----------------|:---------:|:----:|:----:|",
    ]

    for r in results:
        status      = "O" if r["passed"] else "X"
        route_disp  = (
            f"{r['expected_route']}→{r['normalized_route']} (raw:{r['actual_route']})"
            if not r["route_ok"] else r["actual_route"]
        )
        q = r["question"][:38].replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {q} | {r['category']} | {route_disp}"
            f" | {r['keyword_recall']:.0%} | {r['elapsed_sec']}s | {status} |"
        )

    failed = [r for r in results if not r["passed"]]
    if failed:
        lines += ["", f"## 실패 케이스 상세 ({len(failed)}개)", ""]
        for r in failed:
            route_note = "O" if r["route_ok"] else f"X (기대: {r['expected_route']})"
            lines += [
                f"### [{r['id']}] {r['question']}",
                "",
                f"- **카테고리**: {r['category']} · 난이도: {r['difficulty']}",
                f"- **라우팅**: {r['normalized_route']} [{route_note}]",
                f"- **키워드 재현율**: {r['keyword_recall']:.0%}",
                f"- **구조화 검증**: {'O' if r['facts_ok'] else 'X'}",
                f"- **적중**: {', '.join(r['hit_keywords']) or '없음'}",
                f"- **누락**: {', '.join(r['miss_keywords']) or '없음'}",
                f"- **소요 시간**: {r['elapsed_sec']}s",
            ]
            failed_facts = [check for check in r["fact_checks"] if not check["ok"]]
            if failed_facts:
                lines.append(f"- **구조화 실패 항목**: `{json.dumps(failed_facts, ensure_ascii=False)}`")
            if r["answer_preview"]:
                preview = r["answer_preview"].replace("\n", " ")[:200]
                lines.append(f"- **답변 미리보기**: {preview}")
            if r["error"]:
                lines.append(f"- **오류**: `{r['error']}`")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Markdown 저장: {path}")


# ---------------------------------------------------------------------------
# 실패 로그
# ---------------------------------------------------------------------------

def save_failed_log(results: list[dict[str, Any]], path: Path) -> None:
    failed = [r for r in results if not r["passed"]]
    if not failed:
        print("  실패 케이스 없음 - failed.log 미생성")
        return

    lines = [
        f"# 실패 케이스 로그",
        f"생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"총 실패: {len(failed)}개 / {len(results)}개",
        "=" * 60,
        "",
    ]
    for r in failed:
        route_note = "OK" if r["route_ok"] else f"MISMATCH (기대:{r['expected_route']} 실제:{r['actual_route']})"
        lines += [
            f"[{r['id']}] {r['question']}",
            f"  카테고리  : {r['category']} ({r['difficulty']})",
            f"  라우팅    : {route_note}",
            f"  KW 재현율 : {r['keyword_recall']:.0%}",
            f"  구조화 검증: {'O' if r['facts_ok'] else 'X'}",
            f"  적중      : {r['hit_keywords']}",
            f"  누락      : {r['miss_keywords']}",
            f"  소요시간  : {r['elapsed_sec']}s",
            f"  답변      : {r['answer_preview']}",
        ]
        failed_facts = [check for check in r["fact_checks"] if not check["ok"]]
        if failed_facts:
            lines.append(f"  구조화 실패: {json.dumps(failed_facts, ensure_ascii=False)}")
        if r["error"]:
            lines.append(f"  오류      : {r['error']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  실패 로그: {path}")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def parse_route_alias(raw: list[str]) -> dict[str, str]:
    """'sql=pandas' 형태 문자열 목록 → {'sql': 'pandas'} 딕셔너리."""
    alias: dict[str, str] = {}
    for item in raw:
        if "=" in item:
            src, dst = item.split("=", 1)
            alias[src.strip().lower()] = dst.strip().lower()
    return alias


def main() -> None:
    parser = argparse.ArgumentParser(description="골드셋 기반 RAG 평가")
    parser.add_argument("--url",          default=DEFAULT_URL, help="FastAPI 서버 주소")
    parser.add_argument("--tag",          default="default",   help="결과 파일 식별 태그 (브랜치명 권장)")
    parser.add_argument("--endpoint",     default="/chat",     help="평가 대상 엔드포인트 (예: /chat/naive)")
    parser.add_argument("--route-alias",  nargs="*", default=[], metavar="SRC=DST",
                        help="라우팅 레이블 별칭 (예: sql=pandas)")
    parser.add_argument("--id",           help="특정 케이스 ID만 실행 (예: TC001)")
    parser.add_argument("--category",     help="카테고리 필터 (예: pandas_명단)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    route_alias = parse_route_alias(args.route_alias)

    if not GOLDSET_PATH.exists():
        print(f"goldset.json 없음: {GOLDSET_PATH}")
        sys.exit(1)

    goldset    = json.loads(GOLDSET_PATH.read_text(encoding="utf-8"))
    test_cases = goldset["test_cases"]

    if args.id:
        test_cases = [tc for tc in test_cases if tc["id"] == args.id]
    if args.category:
        test_cases = [tc for tc in test_cases if tc["category"] == args.category]

    if not test_cases:
        print("해당 조건의 테스트케이스가 없습니다.")
        sys.exit(1)

    alias_note = f" (alias: {route_alias})" if route_alias else ""
    print(f"서버: {args.url}  엔드포인트: {args.endpoint}  태그: {args.tag}{alias_note}")
    print(f"테스트: {len(test_cases)}개\n" + "-" * 60)

    results = []
    for tc in test_cases:
        r = evaluate_case(tc, args.url, route_alias, endpoint=args.endpoint)
        results.append(r)
        print_result(r, verbose=args.verbose)

    print_summary(results, tag=args.tag)

    # 결과 저장
    RESULT_DIR.mkdir(exist_ok=True)
    now_str = datetime.now().strftime("%m%d_%H%M")
    stem    = f"{now_str}_{args.tag}"

    save_markdown(results, RESULT_DIR / f"{stem}.md", args.url, args.tag, route_alias)
    save_failed_log(results, RESULT_DIR / f"{stem}_failed.log")


if __name__ == "__main__":
    main()
