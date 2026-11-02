"""The answering chain: retrieve -> (feature-boost) -> prompt -> generate."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from ragllmops.logging_utils import get_logger
from ragllmops.models import Citation, RAGConfigBundle, RAGResponse, RetrievedChunk
from ragllmops.rag.prompts import get_prompt, is_refusal

logger = get_logger("rag.chain")


@dataclass(frozen=True, slots=True)
class ChainResult:
    """The chain's internal result, before it is turned into a response."""

    response: RAGResponse
    retrieved: list[RetrievedChunk]
    context: str
    snippets: list[str]


class RagChain:
    """Answers a question with a specific configuration bundle.

    The chain is deliberately stateless: everything that varies between
    deployments (prompt, retrieval depth, LLM identity) comes from the
    bundle, so the same chain can serve a champion and a challenger
    simultaneously.
    """

    def __init__(
        self,
        bundle: RAGConfigBundle,
        retrieval,
        llm,
        feature_provider=None,
        ranker=None,
    ) -> None:
        self.bundle = bundle
        self.retrieval = retrieval
        self.llm = llm
        self.feature_provider = feature_provider
        #: The learned ranker this bundle carries, if any. It replaces the
        #: rule-based reranker for this chain only, which is what lets the
        #: champion and the challenger be compared on the same traffic.
        self.ranker = ranker

    # ------------------------------------------------------------------
    def answer(
        self,
        query: str,
        top_k: int | None = None,
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> ChainResult:
        """Answer *query* end to end."""
        started = _now()
        request_id = request_id or uuid.uuid4().hex

        retrieved = self.retrieval.retrieve(
            query, top_k=top_k or self.bundle.retrieval_top_k, ranker=self.ranker
        )
        retrieved = self._apply_feature_boost(retrieved)

        context, snippets = self._build_context(retrieved)

        prompt = get_prompt(self.bundle.prompt_id)
        system, user = prompt.render(question=query, context=context)

        result = self.llm.complete(
            system=system,
            user=user,
            temperature=self.bundle.temperature,
            max_tokens=self.bundle.max_tokens,
            context=snippets,
        )

        citations = [
            Citation(
                index=index,
                chunk_id=item.chunk.chunk_id,
                doc_id=item.chunk.doc_id,
                title=item.chunk.title,
                score=round(item.rerank_score if item.rerank_score is not None else item.score, 6),
            )
            for index, item in enumerate(retrieved[: len(snippets)], start=1)
        ]

        response = RAGResponse(
            request_id=request_id,
            answer=result.text,
            citations=citations,
            bundle_version=self.bundle.version,
            prompt_id=self.bundle.prompt_id,
            llm_model=result.model,
            llm_provider=result.provider,
            latency_ms=round(_now() - started, 3),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            retrieved_count=len(retrieved),
            refused=is_refusal(result.text),
            query=query,
            user_id=user_id,
        )
        return ChainResult(response=response, retrieved=retrieved, context=context, snippets=snippets)

    # ------------------------------------------------------------------
    def _build_context(self, retrieved: Sequence[RetrievedChunk]) -> tuple[str, list[str]]:
        """Render the numbered context block and the raw snippets."""
        parts: list[str] = []
        snippets: list[str] = []
        used = 0
        limit = self.bundle.max_context_chars

        for index, item in enumerate(retrieved, start=1):
            snippet = f"[S{index}] {item.chunk.title}\n{item.chunk.text}"
            if used + len(snippet) > limit:
                break
            snippets.append(item.chunk.text)
            parts.append(snippet)
            used += len(snippet)

        return "\n\n".join(parts), snippets

    def _apply_feature_boost(self, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Nudge documents that have historically produced better answers.

        This is the online half of the feature store: the offline job
        materialises per-document statistics, the serving path reads them and
        uses them to order otherwise-equivalent candidates.
        """
        if self.feature_provider is None or not retrieved:
            return retrieved

        try:
            features = self.feature_provider.get_online_features(
                {item.chunk.doc_id for item in retrieved}
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("feature lookup failed (%s); skipping boost", exc)
            return retrieved

        if not features:
            return retrieved

        relevances = [
            float(features.get(item.chunk.doc_id, {}).get("avg_relevance", 0.0) or 0.0)
            for item in retrieved
        ]
        ceiling = max(relevances) if relevances else 0.0
        if ceiling <= 0:
            return retrieved

        boosted: list[RetrievedChunk] = []
        for item, relevance in zip(retrieved, relevances, strict=True):
            bonus = 0.05 * (relevance / ceiling)
            base = item.rerank_score if item.rerank_score is not None else item.score
            boosted.append(
                RetrievedChunk(
                    chunk=item.chunk,
                    score=item.score,
                    dense_rank=item.dense_rank,
                    sparse_rank=item.sparse_rank,
                    rerank_score=round(base + bonus, 6),
                )
            )
        boosted.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        return boosted


def _now() -> float:
    import time

    return time.perf_counter()
