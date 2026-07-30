from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import requests


DEFAULT_GOLDSET = (
    Path(__file__).parent / "goldsets" / "vector_goldset_choisanghun.json"
)


def normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\s,_·()\[\]{}'\"’‘“”]+", "", text)


def contains(haystack: str, needle: str) -> bool:
    return normalize(needle) in normalize(haystack)


def evaluate_case(case: dict, *, base_url: str, source: str, mode: str) -> dict:
    started = time.perf_counter()
    error = None
    payload: dict[str, Any] = {}
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/chat",
            json={
                "question": case["question"],
                "sources": [source],
                "mode": mode,
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed = round(time.perf_counter() - started, 2)
    answer = str(payload.get("answer") or "")
    route = str(payload.get("source") or "").lower()
    sources = [str(item) for item in payload.get("sources") or []]
    evidence = payload.get("evidence") or []
    section_titles = [
        str(item.get("section_title") or "")
        for item in evidence
        if isinstance(item, dict)
    ]

    checks: list[dict[str, Any]] = []

    def add(kind: str, expected: Any, ok: bool) -> None:
        checks.append({"kind": kind, "expected": expected, "ok": ok})

    add("route", case.get("expected_route", "vector"), route == "vector")
    add("source", source, source in sources)

    for fact in case.get("required_all", []):
        add("required_all", fact, contains(answer, fact))

    for alternatives in case.get("required_any", []):
        add(
            "required_any",
            alternatives,
            any(contains(answer, alternative) for alternative in alternatives),
        )

    ordered_facts = case.get("required_order", [])
    if ordered_facts:
        normalized_answer = normalize(answer)
        cursor = 0
        order_ok = True
        for fact in ordered_facts:
            position = normalized_answer.find(normalize(fact), cursor)
            if position < 0:
                order_ok = False
                break
            cursor = position + len(normalize(fact))
        add("required_order", ordered_facts, order_ok)

    for forbidden in case.get("forbidden", []):
        add("forbidden", forbidden, not contains(answer, forbidden))

    expected_sections = case.get("expected_sections", [])
    if expected_sections:
        add(
            "expected_sections",
            expected_sections,
            any(
                contains(actual, expected) or contains(expected, actual)
                for actual in section_titles
                for expected in expected_sections
                if actual and expected
            ),
        )

    return {
        "id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "passed": error is None and bool(checks) and all(item["ok"] for item in checks),
        "route": route,
        "sources": sources,
        "section_titles": section_titles,
        "checks": checks,
        "elapsed_sec": elapsed,
        "answer": answer,
        "error": error,
    }


def select_cases(cases: list[dict], args: argparse.Namespace) -> list[dict]:
    selected = cases
    if args.id:
        wanted = {item.strip() for item in args.id.split(",") if item.strip()}
        selected = [case for case in selected if case["id"] in wanted]
    if args.difficulty:
        selected = [
            case for case in selected
            if case["difficulty"] == args.difficulty
        ]
    if args.category:
        selected = [
            case for case in selected
            if args.category in case["category"]
        ]
    if args.offset:
        selected = selected[args.offset:]
    if args.limit:
        selected = selected[:args.limit]
    return selected


def print_summary(results: list[dict]) -> None:
    passed = sum(result["passed"] for result in results)
    print(f"\n결과: {passed}/{len(results)} 통과 ({passed / max(len(results), 1):.1%})")
    by_difficulty = Counter(
        (result["difficulty"], result["passed"])
        for result in results
    )
    for difficulty in ("easy", "medium", "hard"):
        total = sum(
            count for (level, _), count in by_difficulty.items()
            if level == difficulty
        )
        success = by_difficulty[(difficulty, True)]
        if total:
            print(f"- {difficulty}: {success}/{total}")
    failed = [result for result in results if not result["passed"]]
    if failed:
        print("\n실패:")
        for result in failed:
            failed_checks = [
                check for check in result["checks"] if not check["ok"]
            ]
            print(
                f"- {result['id']} ({result['elapsed_sec']}s): "
                f"{failed_checks or result['error']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="최상훈장학금 Vector 골드셋 평가")
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--id", help="쉼표로 구분한 테스트 ID")
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"))
    parser.add_argument("--category")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    goldset = json.loads(args.goldset.read_text(encoding="utf-8"))
    cases = select_cases(goldset["test_cases"], args)
    if not cases:
        print("선택된 테스트가 없습니다.")
        return 2

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} {case['question']}")
        result = evaluate_case(
            case,
            base_url=args.url,
            source=goldset["dataset"],
            mode=goldset.get("mode", "natural"),
        )
        results.append(result)
        if args.verbose:
            print(result["answer"])

    report = {
        "dataset": goldset["dataset"],
        "base_url": args.url,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n저장: {args.out}")

    print_summary(results)
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
