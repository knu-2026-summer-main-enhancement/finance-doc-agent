from __future__ import annotations

from hashlib import sha256


def question_log_metadata(question: object) -> tuple[str, int]:
    """Return correlation metadata without retaining user-entered question text."""

    text = str(question or "")
    return sha256(text.encode("utf-8")).hexdigest()[:12], len(text)
