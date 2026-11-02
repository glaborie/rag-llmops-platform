"""Prometheus metrics.

A dedicated registry is used instead of the global default one, so that
creating the app twice (in tests, for example) never raises a duplicate
timeseries error.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "ragllmops_requests_total",
    "Total answered requests.",
    labelnames=("bundle_version", "prompt_id", "provider"),
    registry=REGISTRY,
)

LATENCY = Histogram(
    "ragllmops_request_duration_seconds",
    "End-to-end answer latency.",
    labelnames=("bundle_version",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

RETRIEVAL_LATENCY = Histogram(
    "ragllmops_retrieval_duration_seconds",
    "Retrieval latency.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

TOKENS = Counter(
    "ragllmops_tokens_total",
    "Prompt and completion tokens.",
    labelnames=("kind", "model"),
    registry=REGISTRY,
)

RETRIEVED_CHUNKS = Histogram(
    "ragllmops_retrieved_chunks",
    "Number of chunks retrieved per request.",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20),
    registry=REGISTRY,
)

RETRIEVAL_SCORE = Histogram(
    "ragllmops_top_retrieval_score",
    "Score of the best retrieved chunk.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

REFUSALS = Counter(
    "ragllmops_refusals_total",
    "Answers that refused to answer.",
    labelnames=("bundle_version",),
    registry=REGISTRY,
)

FEEDBACK = Counter(
    "ragllmops_feedback_total",
    "User verdicts recorded against served answers.",
    labelnames=("verdict",),
    registry=REGISTRY,
)

DRIFT = Gauge(
    "ragllmops_drift_psi",
    "Latest Population Stability Index per feature.",
    labelnames=("feature",),
    registry=REGISTRY,
)

CANARY_TRAFFIC = Gauge(
    "ragllmops_canary_traffic_pct",
    "Percentage of traffic routed to the challenger bundle.",
    registry=REGISTRY,
)

ALIAS_VERSION = Gauge(
    "ragllmops_alias_version",
    "Version currently pointed at by an alias.",
    labelnames=("alias",),
    registry=REGISTRY,
)
