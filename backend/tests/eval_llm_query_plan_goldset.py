from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


TEST_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TEST_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from datastore.state import _df_namespace, _df_sources, _load_dataframes
from pandas_engine.plan_validator import validate_query_plan
from pandas_engine.query_executor import execute_query_plan
from rag.deterministic_query_plan import build_auto_schema_grounded_plan
from rag.query_planner import QueryPlannerError, generate_query_plan


DEFAULT_GOLDSET = TEST_DIR / "goldsets" / "llm_query_plan_goldset.json"


def _filter_matches(case: dict[str, Any], plan: Any) -> bool:
    expected_column = case["column"]
    expected_values = {str(value) for value in case["values"]}
    relevant = [
        condition
        for condition in plan.filters
        if condition.column == expected_column
    ]
    if case["operator"] == "contains":
        return any(
            condition.operator == "contains"
            and str(condition.value) in expected_values
            for condition in relevant
        )
    if any(
        condition.operator == "in"
        and {str(value) for value in condition.value} == expected_values
        for condition in relevant
    ):
        return True
    return (
        plan.filter_logic == "any"
        and {
            str(condition.value)
            for condition in relevant
            if condition.operator == "eq"
        }
        == expected_values
    )


async def _evaluate_case(case: dict[str, Any]) -> tuple[bool, list[str], Any | None]:
    failures: list[str] = []
    pre_operation, pre_plan = build_auto_schema_grounded_plan(
        case["question"],
        dataframes=_df_namespace,
    )
    if pre_operation is not None or pre_plan is not None:
        failures.append(
            f"pre-router captured question: operation={pre_operation!r}"
        )
        return False, failures, pre_plan

    try:
        plan = await generate_query_plan(
            case["question"],
            operation_hint="structured_query",
        )
    except QueryPlannerError as error:
        return False, [f"LLM generation failed: {error}"], None

    if plan.status != "ready":
        failures.append(f"status={plan.status!r}")
    if plan.operation != "list":
        failures.append(f"operation={plan.operation!r}, expected='list'")
    if not _filter_matches(case, plan):
        failures.append("expected filter semantics missing")

    validation = validate_query_plan(
        plan,
        question=case["question"],
        dataframes=_df_namespace,
        source_by_alias=_df_sources,
        operation_hint="structured_query",
    )
    if not validation.is_executable:
        failures.append(
            "validation="
            + ",".join(issue.code for issue in validation.issues)
        )
        return False, failures, plan

    execution = execute_query_plan(validation)
    row_count = len(execution.value)
    if row_count != case["expected_rows"]:
        failures.append(
            f"rows={row_count}, expected={case['expected_rows']}"
        )
    return not failures, failures, plan


async def main_async(args: argparse.Namespace) -> int:
    _load_dataframes()
    goldset = json.loads(args.goldset.read_text(encoding="utf-8"))
    passed = 0
    failures: list[tuple[str, list[str]]] = []
    selected_cases = goldset["cases"][args.start - 1:]
    if args.limit is not None:
        selected_cases = selected_cases[:args.limit]
    for case in selected_cases:
        ok, reasons, plan = await _evaluate_case(case)
        if ok:
            passed += 1
            print(
                f"PASS {case['id']} "
                f"operation={plan.operation} filters={len(plan.filters)}",
                flush=True,
            )
        else:
            failures.append((case["id"], reasons))
            print(f"FAIL {case['id']}: {'; '.join(reasons)}", flush=True)
            if args.verbose and plan is not None:
                print(
                    "  PLAN " + plan.model_dump_json(exclude_none=True),
                    flush=True,
                )

    total = len(selected_cases)
    print(f"\nRESULT {passed}/{total} passed ({passed / total * 100:.1f}%)")
    if failures:
        print("FAILURE_IDS " + ", ".join(case_id for case_id, _ in failures))
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    parser.add_argument("--start", type=int, default=1, choices=range(1, 16))
    parser.add_argument("--limit", type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
