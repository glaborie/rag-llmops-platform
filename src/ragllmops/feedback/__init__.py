"""Feedback capture and the judgement set it feeds.

This package is the middle of the learning loop:

``POST /feedback``  ->  the request log's ``feedback`` column
the request log     ->  ``dataset.py`` turns judgements into training triples
the triples         ->  a learned ranker is trained, gated and promoted
"""

from __future__ import annotations

from ragllmops.feedback.labels import (
    EXPLICIT,
    IMPLICIT,
    VERDICTS,
    Verdict,
    decode_verdict,
    encode_verdict,
    label_source,
)

__all__ = [
    "EXPLICIT",
    "IMPLICIT",
    "VERDICTS",
    "Verdict",
    "decode_verdict",
    "encode_verdict",
    "label_source",
]
