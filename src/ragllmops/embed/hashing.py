"""A dependency-free, deterministic hashing embedder.

This is the default backend so that the platform boots, indexes and answers
queries with nothing installed beyond numpy. It is a *hashed bag of words*
projection: not semantically strong, but stable across processes and runs,
which is exactly what the tests and the offline evaluation need.

Switch to ``sentence-transformer`` via ``RAG_EMBEDDER_BACKEND`` for real
semantic retrieval.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence

from ragllmops.artifacts import builtin_artifact_hash
from ragllmops.text import tokenize


class HashingEmbedder:
    """Stable hashed-bag-of-words embeddings with L2 normalisation."""

    def __init__(self, dim: int = 384, name: str = "hashing-v1") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.name = name
        # No files back this embedder, so its identity is the algorithm plus
        # the parameters that change its output.
        self.artifact_hash = builtin_artifact_hash(name, dim=dim)

    def _bucket(self, token: str) -> tuple[int, float]:
        """Map a token to a bucket index and a signed weight.

        The sign trick keeps collisions from systematically inflating the
        vector, which is what plain hashing does.
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            counts = Counter(tokenize(text))
            vector = [0.0] * self.dim
            if not counts:
                vectors.append(vector)
                continue
            total = sum(counts.values())
            for token, count in counts.items():
                index, sign = self._bucket(token)
                # Sub-linear term frequency, as in classic tf-idf weighting.
                vector[index] += sign * (1.0 + math.log(count)) * (count / total)
            norm = math.sqrt(sum(v * v for v in vector))
            if norm > 0:
                vector = [v / norm for v in vector]
            vectors.append(vector)
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
