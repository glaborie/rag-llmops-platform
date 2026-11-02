"""Shared text utilities.

Tokenisation lives in one place so that the hashing embedder, the BM25 index
and the extractive answerer all agree on what a "term" is. That agreement is
what keeps retrieval and generation pointing at the same words.
"""

from __future__ import annotations

import re

STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by can do does for from has have how i if in into is it
    its of on or that the their then there these this to was what when where
    which who will with you your our us not no but
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric terms.

    Identifiers are split on their separators, so ``max-poll-interval``,
    ``max.poll.interval`` and ``cost_center`` all reduce to the same plain
    words. That keeps a question phrased as "max poll interval" matchable
    against a document that writes it as ``max.poll.interval``.
    """
    return _WORD_RE.findall(text.lower())


def stem(term: str) -> str:
    """A deliberately tiny suffix stemmer.

    Porter stemming is overkill here and adds a dependency; handling the
    plural/suffix cases that actually show up in product documentation is
    enough to make term matching behave.
    """
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("ses"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    return term


def content_terms(text: str) -> list[str]:
    """Stopword-free, stemmed terms — used for scoring and term overlap."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def term_set(text: str) -> set[str]:
    """A set of content terms, for overlap comparisons."""
    return set(content_terms(text))


def sentence_split(text: str) -> list[str]:
    """Split a block of text into sentences, keeping the punctuation."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]
