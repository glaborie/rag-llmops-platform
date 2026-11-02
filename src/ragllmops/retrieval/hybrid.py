"""Hybrid fusion of dense and sparse rankings.

Reciprocal Rank Fusion (RRF) is used rather than score blending because dense
cosine scores and BM25 scores are not on comparable scales — normalising them
is arbitrary, while RRF only needs the *order* of each list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Mapping[str, float]],
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Fuse several ranked lists into one.

    Each argument is a mapping of ``id -> score`` in ranked order (highest
    score first). The returned mapping is ``id -> fused score``, also in
    ranked order.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")

    fused: dict[str, float] = {}
    for index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else float(weights[index])
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + weight / (k + rank)

    return dict(sorted(fused.items(), key=lambda item: item[1], reverse=True))


def rank_map(items: Sequence[str]) -> dict[str, float]:
    """Turn an ordered list of ids into a ``id -> rank`` mapping."""
    return {item_id: float(rank) for rank, item_id in enumerate(items, start=1)}
