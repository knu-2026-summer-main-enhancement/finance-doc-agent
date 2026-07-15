from __future__ import annotations

from rag.guard import GuardResult


def build_guide_response(result: GuardResult) -> str:
    lines = ["질문을 바로 처리하기 어렵습니다."]

    if result.reason:
        lines.append("")
        lines.append(f"이유: {result.reason}")

    if result.suggestions:
        lines.append("")
        lines.append("다시 질문하는 방법:")
        for suggestion in result.suggestions:
            lines.append(f"- {suggestion}")

    return "\n".join(lines)
