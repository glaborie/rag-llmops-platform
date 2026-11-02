"""Turn the request log into a judgement set.

The judgement set is the training material for the learned ranker: labelled
``(query, chunk)`` pairs, each one carrying the provenance of its label.

Two rules shape everything in this module:

1. **Explicit judgements win.** A user who said "this answer was useless" is
   stronger evidence than any signal the platform can infer. The implicit
   fallback is only used when nobody said anything.
2. **The provenance of a label is part of the label.** Every example records
   where it came from, so a training set that mixes human judgements with
   inferred ones can be inspected, weighted or filtered later.

Only the chunks that were actually served are labelled. A chunk that was
retrieved but never shown to the user carries no information about the
answer, and inventing a label for it would teach the model to guess.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ragllmops.features.base import doc_id_from_chunk
from ragllmops.feedback.labels import (
    EXPLICIT,
    IMPLICIT,
    SAMPLED,
    Verdict,
    decode_verdict,
    label_source,
)
from ragllmops.models import RequestRecord, stable_hash

#: The two slices of the judgement set.
TRAIN = "train"
VALIDATION = "validation"

#: Signature of the feature extractor: ``(query, chunk_id) -> features``.
FeatureFn = Callable[[str, str], Mapping[str, float]]

#: Signature of the candidate replayer: ``query -> candidate chunk ids``.
CandidateFn = Callable[[str], Sequence[str]]

#: The citation markers the platform's prompts ask for (``[S1]``, ``[S2]``…).
_CITATION_MARKER = re.compile(r"\[S(\d+)\]")


def cited_chunk_ids(record: RequestRecord) -> list[str]:
    """The chunks the answer actually used, read from its citation markers.

    This is a better signal than "the chunk that was served first": the
    answer records which snippet it drew on, and that is the chunk a
    "this was helpful" verdict is really about. Answers without markers
    (a model that does not cite) fall back to the leading chunk.
    """
    served = list(record.retrieved_ids)
    indices = sorted({int(match) for match in _CITATION_MARKER.findall(record.answer or "")})
    cited = [served[index - 1] for index in indices if 1 <= index <= len(served)]
    return cited or served[:1]


@dataclass(frozen=True, slots=True)
class Judgement:
    """One labelled ``(query, chunk)`` pair."""

    request_id: str
    query: str
    chunk_id: str
    doc_id: str
    #: 1 = relevant, 0 = not relevant.
    label: int
    #: Where the label came from, e.g. ``explicit:helpful``.
    label_source: str
    #: ``train`` or ``validation``.
    split: str
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "label": self.label,
            "label_source": self.label_source,
            "split": self.split,
            "features": dict(self.features),
        }


def derive_labels(
    record: RequestRecord,
    candidates: Sequence[str] = (),
    *,
    max_negatives: int = 3,
) -> dict[str, tuple[int, str]]:
    """Work out the label of every chunk that was served for *record*.

    The rules, in order of precedence:

    * ``helpful``           -> the chunk(s) the answer **cited** are the ones
      that answered the question (label 1); the rest of the served context is
      labelled as not answering it (label 0). The rest of the context is
      labelled rather than dropped on purpose: a reranker is judged on the
      order of a candidate list, and a list with only positives teaches
      nothing about the order.
    * ``unhelpful``         -> none of the served chunks was relevant.
    * ``wrong_citation``    -> the blamed chunk is not relevant; the rest are
      unknown.
    * no explicit feedback, but the answer refused -> none of the served
      chunks was relevant (an implicit signal, labelled as such).

    On top of that, up to ``max_negatives`` candidates the retriever proposed
    but did **not** serve are added as weak negatives. Answer-level feedback
    says "this answer was good", not "these three chunks were worse than the
    first one"; without the sampled negatives every request would be all
    positives or all negatives, and no ranking metric could be computed at
    all. They are labelled ``sampled:candidate`` — inferred from the
    retriever's own order, never from a user.

    A request that produced no signal at all yields no labels.
    """
    served = list(record.retrieved_ids)
    if not served:
        return {}

    verdict = decode_verdict(record.feedback)
    if verdict is Verdict.HELPFUL:
        cited = cited_chunk_ids(record)
        labels = {chunk_id: (1, label_source(verdict)) for chunk_id in cited}
        for chunk_id in served:
            labels.setdefault(chunk_id, (0, f"{EXPLICIT}:context"))
        _add_sampled_negatives(labels, candidates, record, max_negatives=max_negatives)
        return labels
    if verdict is Verdict.UNHELPFUL:
        return {chunk_id: (0, label_source(verdict)) for chunk_id in served}
    if verdict is Verdict.WRONG_CITATION:
        blamed = record.feedback_chunk_id
        if blamed and blamed in served:
            return {blamed: (0, label_source(verdict))}
        return {}

    # No explicit judgement: fall back to what the platform can infer.
    if record.refused:
        return {chunk_id: (0, f"{IMPLICIT}:refusal") for chunk_id in served}
    return {}


def _add_sampled_negatives(
    labels: dict[str, tuple[int, str]],
    candidates: Sequence[str],
    record: RequestRecord,
    *,
    max_negatives: int,
) -> None:
    """Add a few weak negatives drawn from the candidates that were not served."""
    if max_negatives <= 0:
        return
    unseen = [chunk_id for chunk_id in candidates if chunk_id not in labels]
    if not unseen:
        return
    # Deterministic per request: the same log always produces the same
    # training set, which is what makes the artifact reproducible.
    rng = random.Random(stable_hash(record.request_id))
    rng.shuffle(unseen)
    for chunk_id in unseen[:max_negatives]:
        labels[chunk_id] = (0, f"{SAMPLED}:candidate")


def build_judgements(
    records: Sequence[RequestRecord],
    *,
    feature_fn: FeatureFn | None = None,
    candidate_fn: CandidateFn | None = None,
    max_negatives: int = 3,
    validation_fraction: float = 0.2,
    seed: int = 17,
) -> list[Judgement]:
    """Build the judgement set from the request log.

    The split is by *request*, never by judgement: two judgements of the same
    query have to stay on the same side of the split, otherwise the
    validation slice is contaminated by the training slice.
    """
    labelled: list[tuple[RequestRecord, dict[str, tuple[int, str]]]] = []
    for record in records:
        candidates = candidate_fn(record.query) if candidate_fn is not None else []
        labels = derive_labels(record, candidates, max_negatives=max_negatives)
        if labels:
            labelled.append((record, labels))

    split_of = _split_requests(labelled, validation_fraction=validation_fraction, seed=seed)

    judgements: list[Judgement] = []
    for record, labels in labelled:
        split = split_of[record.request_id]
        for chunk_id, (label, source) in labels.items():
            features: dict[str, float] = {}
            if feature_fn is not None:
                features = {k: float(v) for k, v in feature_fn(record.query, chunk_id).items()}
            judgements.append(
                Judgement(
                    request_id=record.request_id,
                    query=record.query,
                    chunk_id=chunk_id,
                    doc_id=doc_id_from_chunk(chunk_id),
                    label=label,
                    label_source=source,
                    split=split,
                    features=features,
                )
            )
    return judgements


def _split_requests(
    labelled: Sequence[tuple[RequestRecord, dict[str, tuple[int, str]]]],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Assign every request to the train or the validation slice."""
    request_ids = sorted({record.request_id for record, _ in labelled})
    if not request_ids:
        return {}

    fraction = min(max(validation_fraction, 0.0), 0.9)
    shuffled = list(request_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = int(round(len(shuffled) * fraction))
    validation = set(shuffled[:validation_count])

    return {
        request_id: (VALIDATION if request_id in validation else TRAIN)
        for request_id in request_ids
    }


def summarise(judgements: Sequence[Judgement]) -> dict[str, object]:
    """Summarise a judgement set: how much of it there is and where it came from."""
    by_source: dict[str, int] = {}
    by_split: dict[str, int] = {}
    positives = 0
    for judgement in judgements:
        by_source[judgement.label_source] = by_source.get(judgement.label_source, 0) + 1
        by_split[judgement.split] = by_split.get(judgement.split, 0) + 1
        positives += judgement.label

    requests = {judgement.request_id for judgement in judgements}
    explicit = sum(
        count for source, count in by_source.items() if source.startswith(EXPLICIT)
    )
    sampled = sum(
        count for source, count in by_source.items() if source.startswith(SAMPLED)
    )
    return {
        "examples": len(judgements),
        "requests": len(requests),
        "positives": positives,
        "negatives": len(judgements) - positives,
        "explicit": explicit,
        "sampled": sampled,
        "implicit": len(judgements) - explicit - sampled,
        "by_source": dict(sorted(by_source.items())),
        "by_split": dict(sorted(by_split.items())),
    }
