"""Consistency checks on the shipped corpus and golden set.

These are the tests that keep the evaluation data honest: every golden case
must point at a document that exists, and every document must be covered.
"""

import json

import pytest

from ragllmops.evaluate.dataset import load_golden_set
from ragllmops.ingest.loader import load_documents
from tests.conftest import REAL_CORPUS, REAL_GOLDEN_SET

pytestmark = pytest.mark.skipif(
    not (REAL_CORPUS.exists() and REAL_GOLDEN_SET.exists()),
    reason="the shipped corpus is not present",
)


@pytest.fixture(scope="module")
def documents():
    return load_documents(REAL_CORPUS)


@pytest.fixture(scope="module")
def cases():
    return load_golden_set(REAL_GOLDEN_SET)


def test_corpus_has_ten_documents(documents):
    assert len(documents) == 10


def test_corpus_documents_are_unique(documents):
    ids = [document.doc_id for document in documents]
    assert len(ids) == len(set(ids))


def test_every_document_has_content(documents):
    for document in documents:
        assert document.word_count >= 200, document.doc_id
        assert document.title
        assert document.category


def test_golden_set_has_thirty_cases(cases):
    assert len(cases) == 30


def test_golden_case_ids_are_unique(cases):
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))


def test_every_case_points_at_a_real_document(cases, documents):
    known = {document.doc_id for document in documents}
    for case in cases:
        assert case.expected_doc_ids, case.id
        for doc_id in case.expected_doc_ids:
            assert doc_id in known, f"{case.id} references unknown document {doc_id}"


def test_every_document_is_covered_by_at_least_two_cases(cases, documents):
    counts: dict[str, int] = {document.doc_id: 0 for document in documents}
    for case in cases:
        for doc_id in case.expected_doc_ids:
            counts[doc_id] += 1
    missing = [doc_id for doc_id, count in counts.items() if count < 2]
    assert not missing, f"documents with fewer than two questions: {missing}"


def test_key_terms_are_grounded_in_the_referenced_documents(cases, documents):
    by_id = {document.doc_id: document for document in documents}
    for case in cases:
        haystack = " ".join(
            by_id[doc_id].body.lower() for doc_id in case.expected_doc_ids
        )
        for term in case.key_terms:
            assert term.lower() in haystack, f"{case.id}: '{term}' is not in {case.expected_doc_ids}"


def test_cases_have_usable_shapes(cases):
    for case in cases:
        assert 1 <= len(case.key_terms) <= 4
        assert len(case.question.split()) >= 8
        assert case.difficulty in {"easy", "medium", "hard"}


def test_golden_set_is_valid_jsonl():
    for line in REAL_GOLDEN_SET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            assert set(payload) == {
                "id",
                "question",
                "expected_doc_ids",
                "key_terms",
                "category",
                "difficulty",
            }
