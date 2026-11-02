"""Comparing a canary against its champion.

A traffic split is not an experiment. The split decides who *serves*; the
statistics decide whether the challenger is *better*. Without them the canary
is a randomiser with extra steps.

Two tests are implemented, because the platform measures two kinds of thing:

* the **refusal rate** is a proportion per request, so the comparison is a
  two-proportion z-test;
* the **retrieval score** (and anything else continuous) is compared with a
  bootstrap, which makes no assumption about the shape of the distribution.

Both report a p-value, and both refuse to conclude anything from a sample
that is too small to support a conclusion.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ragllmops.logging_utils import get_logger
from ragllmops.models import RequestRecord

logger = get_logger("monitor.experiment")

#: Below this many requests per side the test is not worth running.
MIN_SAMPLES = 30

#: The default significance level.
ALPHA = 0.05


@dataclass(slots=True)
class Comparison:
    """The outcome of one champion-vs-challenger comparison."""

    metric: str
    champion: float
    challenger: float
    delta: float
    p_value: float
    samples: dict[str, int] = field(default_factory=dict)
    significant: bool = False
    verdict: str = ""
    #: Whether a *higher* value is the better one. A lower refusal rate is
    #: an improvement; a higher retrieval score is too.
    higher_is_better: bool = True

    @property
    def favours_challenger(self) -> bool:
        """Whether the challenger is on the good side of this metric."""
        if self.delta == 0:
            return False
        return self.delta > 0 if self.higher_is_better else self.delta < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "champion": round(self.champion, 6),
            "challenger": round(self.challenger, 6),
            "delta": round(self.delta, 6),
            "p_value": round(self.p_value, 6),
            "samples": dict(self.samples),
            "significant": self.significant,
            "favours_challenger": self.favours_challenger,
            "verdict": self.verdict,
        }


# ----------------------------------------------------------------------
# the tests
# ----------------------------------------------------------------------
def two_proportion_p_value(
    successes_a: int, trials_a: int, successes_b: int, trials_b: int
) -> float:
    """Two-sided p-value of two independent proportions.

    Normal approximation with a continuity correction — the sample is large
    enough for it by the time the canary is worth reading.
    """
    if trials_a <= 0 or trials_b <= 0:
        return 1.0
    p_a = successes_a / trials_a
    p_b = successes_b / trials_b
    pooled = (successes_a + successes_b) / (trials_a + trials_b)
    variance = pooled * (1 - pooled) * (1 / trials_a + 1 / trials_b)
    if variance <= 0:
        return 1.0
    difference = abs(p_a - p_b)
    # Continuity correction: without it the test is anti-conservative.
    corrected = max(0.0, difference - 0.5 * (1 / trials_a + 1 / trials_b))
    z = corrected / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


def bootstrap_p_value(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 17,
) -> float:
    """Two-sided p-value of the difference of means, by bootstrap.

    The resampled differences are compared against *zero*: the question is
    whether the observed gap could plausibly have come out of noise.
    """
    if not sample_a or not sample_b:
        return 1.0
    rng = random.Random(seed)

    deltas: list[float] = []
    for _ in range(iterations):
        a = _resample(sample_a, rng)
        b = _resample(sample_b, rng)
        deltas.append(_mean(b) - _mean(a))

    at_least_zero = sum(1 for delta in deltas if delta >= 0)
    below_zero = sum(1 for delta in deltas if delta <= 0)
    p_value = 2 * min(
        (at_least_zero + 1) / (iterations + 1),
        (below_zero + 1) / (iterations + 1),
    )
    return min(1.0, p_value)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _resample(values: Sequence[float], rng: random.Random) -> list[float]:
    size = len(values)
    return [values[rng.randrange(size)] for _ in range(size)]


# ----------------------------------------------------------------------
# the comparisons
# ----------------------------------------------------------------------
def compare_refusal_rates(
    champion: Sequence[RequestRecord],
    challenger: Sequence[RequestRecord],
    *,
    alpha: float = ALPHA,
    min_samples: int = MIN_SAMPLES,
) -> Comparison:
    """Compare how often each side refused to answer."""
    champion_refusals = sum(1 for record in champion if record.refused)
    challenger_refusals = sum(1 for record in challenger if record.refused)

    comparison = Comparison(
        metric="refusal_rate",
        champion=champion_refusals / len(champion) if champion else 0.0,
        challenger=challenger_refusals / len(challenger) if challenger else 0.0,
        delta=0.0,
        p_value=1.0,
        samples={"champion": len(champion), "challenger": len(challenger)},
        higher_is_better=False,
    )
    comparison.delta = comparison.challenger - comparison.champion

    if len(champion) < min_samples or len(challenger) < min_samples:
        comparison.verdict = (
            f"not enough traffic to compare "
            f"(need {min_samples} per side, have {len(champion)}/{len(challenger)})"
        )
        return comparison

    comparison.p_value = two_proportion_p_value(
        champion_refusals,
        len(champion),
        challenger_refusals,
        len(challenger),
    )
    comparison.significant = comparison.p_value < alpha
    comparison.verdict = _verdict(comparison, lower_is_better=True)
    return comparison


def compare_scores(
    champion: Sequence[RequestRecord],
    challenger: Sequence[RequestRecord],
    *,
    alpha: float = ALPHA,
    min_samples: int = MIN_SAMPLES,
    iterations: int = 2000,
    seed: int = 17,
) -> Comparison:
    """Compare the retrieval score of the two sides.

    A higher score means the retriever was more confident; it is a proxy for
    "the answer had better material", not a quality judgement on its own.
    """
    champion_scores = [record.top_score for record in champion]
    challenger_scores = [record.top_score for record in challenger]

    comparison = Comparison(
        metric="top_score",
        champion=_mean(champion_scores),
        challenger=_mean(challenger_scores),
        delta=0.0,
        p_value=1.0,
        samples={"champion": len(champion), "challenger": len(challenger)},
    )
    comparison.delta = comparison.challenger - comparison.champion

    if len(champion_scores) < min_samples or len(challenger_scores) < min_samples:
        comparison.verdict = (
            f"not enough traffic to compare "
            f"(need {min_samples} per side, have {len(champion_scores)}/{len(challenger_scores)})"
        )
        return comparison

    comparison.p_value = bootstrap_p_value(
        champion_scores, challenger_scores, iterations=iterations, seed=seed
    )
    comparison.significant = comparison.p_value < alpha
    comparison.verdict = _verdict(comparison, lower_is_better=False)
    return comparison


def _verdict(comparison: Comparison, *, lower_is_better: bool) -> str:
    """Say what the numbers mean, in words."""
    if not comparison.significant:
        return "no significant difference"
    if comparison.favours_challenger:
        return "the challenger is significantly better"
    return "the challenger is significantly worse"


def split_by_bundle(records: Sequence[RequestRecord]) -> dict[str, list[RequestRecord]]:
    """Group the request log by the bundle that served each request."""
    grouped: dict[str, list[RequestRecord]] = {}
    for record in records:
        grouped.setdefault(record.bundle_version, []).append(record)
    return grouped


def canary_report(
    records: Sequence[RequestRecord],
    champion_version: str | None,
    challenger_version: str | None,
    *,
    alpha: float = ALPHA,
    min_samples: int = MIN_SAMPLES,
) -> dict[str, Any]:
    """Compare the two sides of a canary on the traffic they served."""
    if not champion_version or not challenger_version:
        return {
            "ready": False,
            "reason": "no champion/challenger pair to compare",
            "comparisons": [],
        }

    grouped = split_by_bundle(records)
    champion = grouped.get(champion_version, [])
    challenger = grouped.get(challenger_version, [])

    comparisons = [
        compare_refusal_rates(champion, challenger, alpha=alpha, min_samples=min_samples),
        compare_scores(champion, challenger, alpha=alpha, min_samples=min_samples),
    ]
    significant = [c for c in comparisons if c.significant]

    return {
        "ready": bool(champion) and bool(challenger),
        "champion": champion_version,
        "challenger": challenger_version,
        "champion_requests": len(champion),
        "challenger_requests": len(challenger),
        "comparisons": [comparison.to_dict() for comparison in comparisons],
        "significant": bool(significant),
        # The challenger may only be promoted automatically when it wins on
        # every metric that could be read, and loses on none.
        "promotable": bool(significant) and all(
            comparison.favours_challenger for comparison in significant
        ),
        "alpha": alpha,
    }
