from __future__ import annotations

# 사용자 질문 원문을 로그에 남기지 않고 상관관계 확인용 해시와 길이만 만든다.
# 전화번호·이메일 등 원문 복원이 가능한 값을 이 메타데이터에 추가하지 않는다.

from hashlib import sha256


def question_log_metadata(question: object) -> tuple[str, int]:
    """Return correlation metadata without retaining user-entered question text."""

    text = str(question or "")
    return sha256(text.encode("utf-8")).hexdigest()[:12], len(text)
