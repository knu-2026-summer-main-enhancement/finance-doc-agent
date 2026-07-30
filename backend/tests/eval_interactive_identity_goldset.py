from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DEFAULT_GOLDSET = ROOT / "goldsets" / "interactive_identity_goldset.json"


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = Request(url, data=data, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _record_links(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        segment
        for segment in result.get("inline_segments", [])
        if segment.get("kind") == "record_entity"
    ]


def _detail(base_url: str, reference: str, row_index: int) -> dict[str, Any]:
    return _json_request(
        f"{base_url}/chat/results/{quote(reference, safe='')}/person/{row_index}"
    )


def _attributes(detail: dict[str, Any]) -> dict[str, Any]:
    return {
        str(item.get("column")): item.get("value")
        for item in detail.get("attributes", [])
    }


def _evaluate_field(
    *,
    base_url: str,
    expected: dict[str, Any],
    kind: str,
    result: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    records = result.get("records") or []
    field = "전화번호" if kind == "phone" else "전공"
    if result.get("operation") != "list":
        failures.append(f"operation={result.get('operation')!r}")
    if len(records) != expected["row_count"]:
        failures.append(f"records={len(records)} expected={expected['row_count']}")
    if any(field not in record for record in records):
        failures.append(f"missing {field} key")
    if kind == "phone":
        nonempty = sum(bool(record.get(field)) for record in records)
        if nonempty != expected["nonempty_phone_count"]:
            failures.append(
                f"nonempty phones={nonempty} expected={expected['nonempty_phone_count']}"
            )
    else:
        majors = [record.get(field) for record in records]
        if majors != expected["majors"]:
            failures.append("record major order/value mismatch")

    links = _record_links(result)
    indexes = [link.get("row_index") for link in links]
    expected_indexes = list(range(expected["row_count"]))
    if indexes != expected_indexes:
        failures.append(f"link indexes={indexes} expected={expected_indexes}")
        return failures

    reference = result.get("records_detail_ref")
    if not reference:
        failures.append("missing records_detail_ref")
        return failures
    for index, record in enumerate(records):
        detail = _detail(base_url, reference, index)
        attributes = _attributes(detail)
        if attributes.get("전공") != expected["majors"][index]:
            failures.append(f"card {index} major mismatch")
        if (
            kind == "phone"
            and bool(record.get("전화번호"))
            and not bool(attributes.get("전화번호"))
        ):
            failures.append(f"card {index} lost a known phone value")
    return failures


def _evaluate_amount(
    *,
    base_url: str,
    expected: dict[str, Any],
    result: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    records = result.get("records") or []
    if result.get("operation") != "person_totals":
        failures.append(f"operation={result.get('operation')!r}")
        return failures
    amount_sum = sum(float(record.get("결제_금액") or 0) for record in records)
    if amount_sum != float(expected["amount_total"]):
        failures.append(
            f"amount total={amount_sum:g} expected={expected['amount_total']}"
        )
    links = _record_links(result)
    indexes = [link.get("row_index") for link in links]
    if indexes != list(range(len(records))):
        failures.append("amount-card link indexes are not sequential")
    reference = result.get("records_detail_ref")
    if not reference:
        failures.append("missing records_detail_ref")
        return failures
    allowed_majors = set(expected["majors"])
    for index in range(len(records)):
        attributes = _attributes(_detail(base_url, reference, index))
        if attributes.get("전공") not in allowed_majors:
            failures.append(f"amount card {index} points outside expected people")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--goldset", type=Path, default=DEFAULT_GOLDSET)
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    goldset = json.loads(args.goldset.read_text(encoding="utf-8"))

    passed = 0
    failed: list[tuple[str, list[str]]] = []
    case_number = 0
    for expected in goldset["cases"]:
        for template in goldset["question_templates"]:
            case_number += 1
            case_id = f"I{case_number:03d}"
            kind = template["kind"]
            question = template["text"].format(person=expected["person"])
            response = _json_request(
                f"{base_url}/chat",
                {"question": question, "sources": [], "mode": "auto"},
            )
            result = response.get("result") or {}
            failures = []
            if response.get("source") != "pandas":
                failures.append(f"source={response.get('source')!r}")
            if kind in {"phone", "major"}:
                failures.extend(
                    _evaluate_field(
                        base_url=base_url,
                        expected=expected,
                        kind=kind,
                        result=result,
                    )
                )
            else:
                failures.extend(
                    _evaluate_amount(
                        base_url=base_url,
                        expected=expected,
                        result=result,
                    )
                )
            if failures:
                failed.append((case_id, failures))
                print(f"FAIL {case_id} {kind} {expected['person']}: {'; '.join(failures)}")
            else:
                passed += 1
                print(f"PASS {case_id} {kind} {expected['person']}")

    total = passed + len(failed)
    print(f"\nRESULT {passed}/{total} passed ({passed / total * 100:.1f}%)")
    if failed:
        print("FAILURE_IDS " + ", ".join(case_id for case_id, _ in failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
