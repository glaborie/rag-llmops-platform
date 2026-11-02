"""A self-contained BM25 implementation.

Dense retrieval misses exact identifiers (a limit, a product name, an API
header). BM25 catches them. Implementing it here keeps the sparse side of the
hybrid retriever dependency-free and fully testable.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from ragllmops.models import DocumentChunk
from ragllmops.text import tokenize


class Bm25Index:
    """Okapi BM25 over a fixed corpus of chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: dict[str, DocumentChunk] = {}
        self._freqs: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------
    def fit(self, chunks: Sequence[DocumentChunk]) -> Bm25Index:
        """(Re)build the index from a chunk collection."""
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._freqs = {}
        self._lengths = {}
        self._df = Counter()

        for chunk_id, chunk in self._chunks.items():
            tokens = tokenize(f"{chunk.title} {chunk.text}")
            self._freqs[chunk_id] = Counter(tokens)
            self._lengths[chunk_id] = len(tokens)
            for term in set(tokens):
                self._df[term] += 1

        total_length = sum(self._lengths.values())
        self._avgdl = (total_length / len(self._lengths)) if self._lengths else 0.0
        return self

    # ------------------------------------------------------------------
    # querying
    # ------------------------------------------------------------------
    def idf(self, term: str) -> float:
        """Inverse document frequency with the standard BM25 smoothing."""
        n = len(self._chunks)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Return the ``top_k`` best-matching chunks."""
        if not self._chunks:
            return []

        terms = tokenize(query)
        if not terms:
            return []

        scored: list[tuple[DocumentChunk, float]] = []
        for chunk_id, chunk in self._chunks.items():
            score = self.score_terms(terms, chunk_id)
            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def score(self, query: str, chunk_id: str) -> float:
        """Score one chunk against a query.

        Unlike ``search`` this does not need the chunk to be in the top-k,
        which is what a learned ranker needs: it re-scores the candidates it
        was handed, not the ones the index happens to surface first.
        """
        if chunk_id not in self._chunks:
            return 0.0
        return self.score_terms(tokenize(query), chunk_id)

    def score_terms(self, terms: Sequence[str], chunk_id: str) -> float:
        """Score a chunk against pre-tokenised query terms."""
        freqs = self._freqs.get(chunk_id)
        if freqs is None:
            return 0.0
        length = self._lengths.get(chunk_id, 0)
        score = 0.0
        for term in terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (
                1 - self.b + self.b * (length / self._avgdl if self._avgdl else 1.0)
            )
            score += self.idf(term) * (tf * (self.k1 + 1)) / denom
        return score

    @property
    def size(self) -> int:
        return len(self._chunks)

    def chunks(self) -> list[DocumentChunk]:
        return list(self._chunks.values())
