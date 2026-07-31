from __future__ import annotations

import re


_TABLE_REQUEST = re.compile(
    r"(?<![가-힣A-Za-z0-9])(?:표|테이블)"
    r"(?=(?:\s|$|[?!.]|로|형태|형식|보기|출력|조회|보여|알려))"
)


def is_explicit_table_request(question: str | None) -> bool:
    """Return whether the user explicitly requested a tabular presentation."""

    return bool(_TABLE_REQUEST.search(str(question or "")))
