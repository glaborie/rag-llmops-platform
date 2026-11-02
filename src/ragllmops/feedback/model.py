"""A small learned ranker.

The model is a logistic regression over the features in
:mod:`ragllmops.feedback.features`. A linear model is the right size here:
a few dozen judgements are enough to fit it, training takes milliseconds and
the result stays readable — you can print the weights and see which signal
the ranker actually uses.

The model is content-addressed and stored as JSON, next to the metadata that
makes it auditable: the hash of the training data, the feature schema
version and the metrics it scored on the held-out slice.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ragllmops.feedback.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from ragllmops.logging_utils import get_logger
from ragllmops.models import stable_hash

logger = get_logger("feedback.model")

#: The version of the model format. Bumped when the meaning of the weights
#: changes.
MODEL_VERSION = "logreg-v1"


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One labelled feature vector."""

    features: dict[str, float]
    label: int
    #: The request the example came from — examples of the same query must
    #: stay together in a split.
    request_id: str = ""
    chunk_id: str = ""


@dataclass(frozen=True, slots=True)
class RankerModel:
    """A trained ranker: weights, bias and the normalisation it expects."""

    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    means: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    version: str = MODEL_VERSION
    #: The feature schema the model was trained on.
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    def score(self, features: Mapping[str, float]) -> float:
        """Score one candidate; higher is more relevant."""
        total = self.bias
        for name, weight in zip(self.feature_names, self.weights, strict=True):
            value = float(features.get(name, 0.0))
            total += weight * self._normalise(name, value)
        return _sigmoid(total)

    def score_many(self, features: Iterable[Mapping[str, float]]) -> list[float]:
        return [self.score(item) for item in features]

    def _normalise(self, name: str, value: float) -> float:
        """Z-score a feature with the training statistics."""
        if not self.means or not self.scales:
            return value
        try:
            index = self.feature_names.index(name)
        except ValueError:  # pragma: no cover - defensive
            return value
        scale = self.scales[index] or 1.0
        return (value - self.means[index]) / scale

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "feature_schema_version": self.feature_schema_version,
            "feature_names": list(self.feature_names),
            "weights": [round(w, 8) for w in self.weights],
            "bias": round(self.bias, 8),
            "means": [round(m, 8) for m in self.means],
            "scales": [round(s, 8) for s in self.scales],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RankerModel:
        return cls(
            version=str(payload.get("version", MODEL_VERSION)),
            feature_schema_version=str(
                payload.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
            ),
            feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
            weights=tuple(float(w) for w in payload.get("weights", [])),
            bias=float(payload.get("bias", 0.0)),
            means=tuple(float(m) for m in payload.get("means", [])),
            scales=tuple(float(s) for s in payload.get("scales", [])),
        )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------
def training_data_hash(examples: Sequence[TrainingExample]) -> str:
    """A stable hash of the training data.

    It identifies *what was learned from*, so a model artifact can be traced
    back to the judgements it was fitted on.
    """
    payload = [
        {
            "request_id": example.request_id,
            "chunk_id": example.chunk_id,
            "label": example.label,
            "features": {
                name: round(float(example.features.get(name, 0.0)), 6)
                for name in FEATURE_NAMES
            },
        }
        for example in examples
    ]
    return stable_hash(payload)


def train_ranker(
    examples: Sequence[TrainingExample],
    *,
    iterations: int = 500,
    learning_rate: float = 0.2,
    l2: float = 0.01,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> RankerModel:
    """Fit a logistic regression on the labelled examples.

    Gradient descent on the regularised logistic loss, starting from zero
    weights so the result is deterministic.
    """
    names = tuple(feature_names)
    if not examples:
        raise ValueError("cannot train a ranker without examples")

    matrix = np.array(
        [[float(example.features.get(name, 0.0)) for name in names] for example in examples],
        dtype="float64",
    )
    labels = np.array([float(example.label) for example in examples], dtype="float64")

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-9] = 1.0
    normalised = (matrix - means) / scales

    weights = np.zeros(len(names), dtype="float64")
    bias = 0.0
    n = len(examples)

    for _ in range(iterations):
        predictions = _sigmoid_array(normalised @ weights + bias)
        error = predictions - labels
        weights -= learning_rate * ((normalised.T @ error) / n + l2 * weights)
        bias -= learning_rate * float(error.mean())

    return RankerModel(
        feature_names=names,
        weights=tuple(float(w) for w in weights),
        bias=float(bias),
        means=tuple(float(m) for m in means),
        scales=tuple(float(s) for s in scales),
    )


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def evaluate_model(
    model: RankerModel,
    examples: Sequence[TrainingExample],
) -> dict[str, float]:
    """Score a model on a slice of the judgement set.

    ``pairwise_accuracy`` is the metric that matters for a ranker: over every
    (relevant, irrelevant) pair of the same query, how often does the
    relevant one end up on top? That is exactly what a reranker is judged
    on, and it is the number the gate compares.
    """
    if not examples:
        return {"examples": 0.0, "accuracy": 0.0, "pairwise_accuracy": 0.0, "auc": 0.0}
    scores = [model.score(example.features) for example in examples]
    return _metrics(scores, examples)


def evaluate_baseline(
    examples: Sequence[TrainingExample],
    feature: str = "term_overlap",
) -> dict[str, float]:
    """The metrics of the rule-based baseline on the same examples.

    The incumbent reranker scores a candidate by its term overlap with the
    query. Scoring the held-out slice with that feature gives the number the
    learned ranker has to beat.
    """
    if not examples:
        return {"examples": 0.0, "accuracy": 0.0, "pairwise_accuracy": 0.0, "auc": 0.0}
    scores = [float(example.features.get(feature, 0.0)) for example in examples]
    return _metrics(scores, examples)


def _metrics(scores: Sequence[float], examples: Sequence[TrainingExample]) -> dict[str, float]:
    """Accuracy, pairwise accuracy and AUC for a set of scores."""
    labels = [example.label for example in examples]

    correct = sum(
        1 for score, label in zip(scores, labels, strict=True) if (score >= 0.5) == bool(label)
    )
    accuracy = correct / len(examples)

    by_request: dict[str, list[tuple[float, int]]] = {}
    for example, score in zip(examples, scores, strict=True):
        by_request.setdefault(example.request_id, []).append((score, example.label))

    pairs = 0
    pairs_won = 0
    for entries in by_request.values():
        positives = [score for score, label in entries if label == 1]
        negatives = [score for score, label in entries if label == 0]
        for positive in positives:
            for negative in negatives:
                pairs += 1
                if positive > negative:
                    pairs_won += 1
    pairwise = pairs_won / pairs if pairs else 0.0

    return {
        "examples": float(len(examples)),
        "accuracy": round(accuracy, 6),
        "pairwise_accuracy": round(pairwise, 6),
        "auc": round(_auc(scores, labels), 6),
        "pairs": float(pairs),
    }


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney AUC, computed with ties counted as half a win."""
    positives = [score for score, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [score for score, label in zip(scores, labels, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else (0.5 if positive == negative else 0.0)
    return wins / (len(positives) * len(negatives))


# ----------------------------------------------------------------------
# artifacts
# ----------------------------------------------------------------------
def save_ranker(
    model: RankerModel,
    directory: Path,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Store a trained ranker and return its artifact id.

    The id is the content hash of the artifact, so the same model always has
    the same id and storing it twice is a no-op.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.to_dict(),
        "metadata": dict(metadata or {}),
    }
    blob = json.dumps(payload, sort_keys=True, indent=2)
    artifact_id = stable_hash(payload)
    path = directory / f"{artifact_id}.json"
    if not path.exists():
        path.write_text(blob, encoding="utf-8")
    logger.info("stored ranker artifact %s", artifact_id)
    return artifact_id


def load_ranker(path: Path) -> RankerModel:
    """Load a trained ranker from its artifact file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model_payload = payload.get("model", payload)
    return RankerModel.from_dict(model_payload)


def load_ranker_metadata(path: Path) -> dict[str, Any]:
    """Load the metadata stored next to a ranker artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(payload.get("metadata", {}))


def artifact_path(directory: Path, artifact_id: str) -> Path:
    """The file a ranker artifact lives in."""
    return Path(directory) / f"{artifact_id}.json"


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "MODEL_VERSION",
    "RankerModel",
    "TrainingExample",
    "artifact_path",
    "evaluate_baseline",
    "evaluate_model",
    "load_ranker",
    "load_ranker_metadata",
    "save_ranker",
    "train_ranker",
    "training_data_hash",
]
