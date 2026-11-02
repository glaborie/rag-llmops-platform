"""Embedding backends."""

import math

from ragllmops.embed.hashing import HashingEmbedder


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed_one("autoscaling cooldown")
    second = embedder.embed_one("autoscaling cooldown")
    assert first == second


def test_hashing_embedder_normalises_vectors():
    embedder = HashingEmbedder(dim=64)
    vector = embedder.embed_one("autoscaling cooldown policy")
    norm = math.sqrt(sum(value * value for value in vector))
    assert abs(norm - 1.0) < 1e-6


def test_hashing_embedder_ranks_related_text_closer():
    embedder = HashingEmbedder(dim=256)
    query = embedder.embed_one("autoscaling cooldown period")
    related = embedder.embed_one("the autoscaling cooldown period is 120 seconds")
    unrelated = embedder.embed_one("bake sourdough bread with a starter")
    assert _cosine(query, related) > _cosine(query, unrelated)


def test_hashing_embedder_handles_empty_text():
    embedder = HashingEmbedder(dim=32)
    vector = embedder.embed_one("")
    assert vector == [0.0] * 32


def test_hashing_embedder_rejects_bad_dimension():
    try:
        HashingEmbedder(dim=0)
    except ValueError:
        pass
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a ValueError")


def _cosine(a, b):  # noqa: ANN001, ANN201
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
