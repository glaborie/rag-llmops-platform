"""Drift detection.

The request log is split into a *reference* window (how the system behaved
when it was healthy) and a *current* window (how it behaves now). Numeric
features are compared with the Population Stability Index; categorical
features with total variation distance.

PSI is implemented here rather than pulled from a dependency so the monitor
always runs. When Evidently is installed, ``render_evidently_report`` adds a
rich HTML report on top of the same numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ragllmops.logging_utils import get_logger
from ragllmops.models import RequestRecord

logger = get_logger("monitor.drift")

DEFAULT_NUMERIC_FEATURES = (
    "latency_ms",
    "top_score",
    "chunk_count",
    "prompt_tokens",
    "completion_tokens",
    "answer_chars",
)

DEFAULT_CATEGORICAL_FEATURES = ("llm_model", "llm_provider", "prompt_id")

_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """Drift measurement for a single numeric feature."""

    feature: str
    psi: float
    drifted: bool
    reference_mean: float
    current_mean: float
    reference_samples: int
    current_samples: int

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 6),
            "drifted": self.drifted,
            "reference_mean": round(self.reference_mean, 6),
            "current_mean": round(self.current_mean, 6),
            "reference_samples": self.reference_samples,
            "current_samples": self.current_samples,
        }


@dataclass(slots=True)
class DriftReport:
    """The full drift picture for a monitoring window."""

    generated_at: str = ""
    reference_size: int = 0
    current_size: int = 0
    features: list[FeatureDrift] = field(default_factory=list)
    categorical: dict[str, dict] = field(default_factory=dict)
    is_drifted: bool = False
    drifted_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "reference_size": self.reference_size,
            "current_size": self.current_size,
            "is_drifted": self.is_drifted,
            "drifted_features": self.drifted_features,
            "features": [feature.to_dict() for feature in self.features],
            "categorical": self.categorical,
        }

    def summary(self) -> str:
        lines = [
            f"generated at   : {self.generated_at}",
            f"reference size : {self.reference_size}",
            f"current size   : {self.current_size}",
        ]
        for feature in self.features:
            marker = "DRIFT" if feature.drifted else "ok   "
            lines.append(
                f"  [{marker}] {feature.feature:<20} psi={feature.psi:.4f} "
                f"mean {feature.reference_mean:.3f} -> {feature.current_mean:.3f}"
            )
        for name, stats in self.categorical.items():
            lines.append(
                f"  [{'DRIFT' if stats.get('drifted') else 'ok   '}] {name:<20} "
                f"tv={stats.get('total_variation', 0):.4f}"
            )
        lines.append(f"drift detected : {self.is_drifted}")
        return "\n".join(lines)


def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    bins: int = 10,
) -> float:
    """Compute the Population Stability Index between two samples."""
    if not reference or not current:
        return 0.0

    edges = _quantile_edges(sorted(reference), bins)
    if len(edges) < 3:
        # The reference barely varies (a constant feature such as
        # `chunk_count`). Quantile bins on the reference would collapse into
        # a single bin and hide a large shift, so fall back to uniform bins
        # across the combined range.
        edges = _uniform_edges([*reference, *current], bins)
    if len(edges) < 2:
        lowest = min(min(reference), min(current))
        highest = max(max(reference), max(current))
        edges = [lowest - 1e-9, highest + 1e-9]

    reference_counts = _bin_counts(reference, edges)
    current_counts = _bin_counts(current, edges)

    psi = 0.0
    for ref_count, cur_count in zip(reference_counts, current_counts, strict=True):
        expected = ref_count / len(reference)
        actual = cur_count / len(current)
        expected = max(expected, _EPSILON)
        actual = max(actual, _EPSILON)
        psi += (actual - expected) * math.log(actual / expected)
    return max(0.0, psi)


def _quantile_edges(ordered: Sequence[float], bins: int) -> list[float]:
    """Quantile bin edges for an already-sorted sample."""
    if not ordered:
        return []
    if len(ordered) == 1:
        return [ordered[0] - 1e-9, ordered[0] + 1e-9]
    quantiles = [index / bins for index in range(bins + 1)]
    edges = [_quantile(ordered, q) for q in quantiles]
    return sorted(set(edges))


def _uniform_edges(values: Sequence[float], bins: int) -> list[float]:
    """Uniform-width bin edges across the range of *values*."""
    if not values:
        return []
    lowest, highest = min(values), max(values)
    if highest <= lowest:
        return [lowest - 1e-9, lowest + 1e-9]
    step = (highest - lowest) / bins
    edges = [lowest + step * index for index in range(bins)]
    edges.append(highest + 1e-9)
    return edges


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[int(position)])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def _bin_counts(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    """Count values per bin defined by *edges*."""
    counts = [0] * (len(edges) - 1)
    for value in values:
        index = _bin_index(float(value), edges)
        counts[index] += 1
    return counts


def _bin_index(value: float, edges: Sequence[float]) -> int:
    """Locate a value in a bin list, clamping to the ends."""
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return len(edges) - 2
    for index in range(len(edges) - 1):
        if edges[index] <= value < edges[index + 1]:
            return index
    return len(edges) - 2


def total_variation_distance(
    reference: Sequence[str], current: Sequence[str]
) -> tuple[float, dict[str, float]]:
    """Total variation distance between two categorical distributions."""
    if not reference or not current:
        return 0.0, {}

    ref_counts: dict[str, int] = {}
    for value in reference:
        ref_counts[value] = ref_counts.get(value, 0) + 1
    cur_counts: dict[str, int] = {}
    for value in current:
        cur_counts[value] = cur_counts.get(value, 0) + 1

    categories = set(ref_counts) | set(cur_counts)
    distance = 0.0
    distribution: dict[str, float] = {}
    for category in sorted(categories):
        ref_share = ref_counts.get(category, 0) / len(reference)
        cur_share = cur_counts.get(category, 0) / len(current)
        distance += abs(ref_share - cur_share)
        distribution[category] = round(cur_share - ref_share, 6)

    return distance / 2, distribution


def compute_drift(
    reference: list[RequestRecord],
    current: list[RequestRecord],
    threshold: float = 0.2,
    numeric_features: Sequence[str] = DEFAULT_NUMERIC_FEATURES,
    categorical_features: Sequence[str] = DEFAULT_CATEGORICAL_FEATURES,
    categorical_threshold: float = 0.2,
) -> DriftReport:
    """Compare a reference window with a current window."""
    report = DriftReport(
        generated_at=datetime.now(UTC).isoformat(),
        reference_size=len(reference),
        current_size=len(current),
    )

    if not reference or not current:
        logger.warning(
            "drift skipped: reference=%s current=%s samples", len(reference), len(current)
        )
        return report

    for feature in numeric_features:
        ref_values = [float(getattr(r, feature, 0) or 0) for r in reference]
        cur_values = [float(getattr(r, feature, 0) or 0) for r in current]
        if not ref_values or not cur_values:
            continue

        psi = population_stability_index(ref_values, cur_values)
        report.features.append(
            FeatureDrift(
                feature=feature,
                psi=psi,
                drifted=psi > threshold,
                reference_mean=sum(ref_values) / len(ref_values),
                current_mean=sum(cur_values) / len(cur_values),
                reference_samples=len(ref_values),
                current_samples=len(cur_values),
            )
        )

    for feature in categorical_features:
        ref_values = [str(getattr(r, feature, "") or "") for r in reference]
        cur_values = [str(getattr(r, feature, "") or "") for r in current]
        distance, distribution = total_variation_distance(ref_values, cur_values)
        report.categorical[feature] = {
            "total_variation": round(distance, 6),
            "drifted": distance > categorical_threshold,
            "shift": distribution,
        }

    report.drifted_features = [f.feature for f in report.features if f.drifted]
    report.drifted_features += [
        name for name, stats in report.categorical.items() if stats.get("drifted")
    ]
    report.is_drifted = bool(report.drifted_features)
    return report


def split_windows(
    records: list[RequestRecord], current_fraction: float = 0.3
) -> tuple[list[RequestRecord], list[RequestRecord]]:
    """Split a chronological record list into reference and current windows."""
    if len(records) < 10:
        return records, records
    split = max(1, int(len(records) * (1 - current_fraction)))
    return records[:split], records[split:]
