"""The feedback vocabulary.

A verdict is what a user says about an answer. It is stored as a small
integer in the request log's existing ``feedback`` column, so the encoding is
part of the on-disk contract and must not change silently.
"""

from __future__ import annotations

from enum import Enum

#: Label sources. Explicit judgements and implicit signals are never
#: conflated: a training set that mixes them without saying so produces a
#: model nobody can reason about.
EXPLICIT = "explicit"
IMPLICIT = "implicit"
#: Weak negatives drawn from the candidates the retriever did not serve.
#: Inferred from the retriever's own order, never from a user.
SAMPLED = "sampled"


class Verdict(str, Enum):
    """What a user can say about an answer."""

    HELPFUL = "helpful"
    UNHELPFUL = "unhelpful"
    WRONG_CITATION = "wrong_citation"


#: The values the API accepts.
VERDICTS: tuple[str, ...] = tuple(v.value for v in Verdict)

_CODES: dict[Verdict, int] = {
    Verdict.HELPFUL: 1,
    Verdict.UNHELPFUL: -1,
    Verdict.WRONG_CITATION: -2,
}
_BY_CODE: dict[int, Verdict] = {code: verdict for verdict, code in _CODES.items()}


def encode_verdict(verdict: str) -> int:
    """Turn a verdict into the integer stored in the request log."""
    try:
        return _CODES[Verdict(verdict)]
    except ValueError as exc:
        raise ValueError(
            f"unknown verdict {verdict!r}; expected one of {', '.join(VERDICTS)}"
        ) from exc


def decode_verdict(code: int | None) -> Verdict | None:
    """Turn the stored integer back into a verdict."""
    if code is None:
        return None
    return _BY_CODE.get(int(code))


def label_source(verdict: Verdict) -> str:
    """The provenance label written next to every training example."""
    return f"{EXPLICIT}:{verdict.value}"
