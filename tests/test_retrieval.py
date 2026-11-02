"""BM25, rank fusion and reranking."""

from ragllmops.models import DocumentChunk
from ragllmops.retrieval.bm25 import Bm25Index
from ragllmops.retrieval.hybrid import reciprocal_rank_fusion
from ragllmops.retrieval.rerank import TermOverlapReranker


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("::")[0],
        title=chunk_id.split("::")[0],
        category="test",
        text=text,
    )


def test_bm25_ranks_the_exact_match_first():
    index = Bm25Index().fit(
        [
            _chunk("a::0", "the autoscaling cooldown is 120 seconds"),
            _chunk("b::0", "object storage retention defaults to 90 days"),
            _chunk("c::0", "network load balancers route traffic to subnets"),
        ]
    )
    results = index.search("autoscaling cooldown")
    assert results
    assert results[0][0].chunk_id == "a::0"


def test_bm25_ignores_documents_without_the_terms():
    index = Bm25Index().fit(
        [
            _chunk("a::0", "the autoscaling cooldown is 120 seconds"),
            _chunk("b::0", "object storage retention defaults to 90 days"),
        ]
    )
    results = index.search("object retention")
    assert [chunk.chunk_id for chunk, _ in results] == ["b::0"]


def test_bm25_on_empty_index():
    assert Bm25Index().search("anything") == []


def test_rrf_prefers_items_ranked_by_both_lists():
    fused = reciprocal_rank_fusion(
        [{"a": 1.0, "b": 0.9}, {"a": 0.8, "c": 0.7}], weights=[1.0, 1.0]
    )
    assert list(fused)[0] == "a"


def test_rrf_weights_change_the_order():
    dense_first = reciprocal_rank_fusion([{"a": 1.0}, {"b": 1.0}], weights=[1.0, 0.1])
    sparse_first = reciprocal_rank_fusion([{"a": 1.0}, {"b": 1.0}], weights=[0.1, 1.0])
    assert list(dense_first)[0] == "a"
    assert list(sparse_first)[0] == "b"


def test_rrf_rejects_mismatched_weights():
    try:
        reciprocal_rank_fusion([{"a": 1.0}], weights=[1.0, 1.0])
    except ValueError:
        pass
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected a ValueError")


def test_term_overlap_reranker_prefers_the_matching_chunk():
    chunks = [
        _chunk("a::0", "the autoscaling cooldown is 120 seconds"),
        _chunk("b::0", "object storage retention defaults to 90 days"),
    ]
    scores = TermOverlapReranker().score("what is the autoscaling cooldown", chunks)
    assert scores[0] > scores[1]
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_term_overlap_reranker_handles_no_query_terms():
    chunks = [_chunk("a::0", "some text")]
    assert TermOverlapReranker().score("the a of", chunks) == [0.0]
