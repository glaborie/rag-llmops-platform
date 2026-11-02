"""The golden evaluation set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ragllmops.logging_utils import get_logger

logger = get_logger("evaluate.dataset")


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One labelled question from the golden set."""

    id: str
    question: str
    expected_doc_ids: list[str]
    key_terms: list[str]
    category: str
    difficulty: str

    @classmethod
    def from_dict(cls, payload: dict) -> GoldenCase:
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            expected_doc_ids=[str(d) for d in payload.get("expected_doc_ids", [])],
            key_terms=[str(t) for t in payload.get("key_terms", [])],
            category=str(payload.get("category", "general")),
            difficulty=str(payload.get("difficulty", "medium")),
        )


def load_golden_set(path: Path) -> list[GoldenCase]:
    """Read a JSONL golden set."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"golden set not found: {path}")

    cases: list[GoldenCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        cases.append(GoldenCase.from_dict(payload))

    logger.info("loaded %s golden cases from %s", len(cases), path)
    return cases
