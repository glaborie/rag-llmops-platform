"""The platform façade.

``Platform`` owns every backend and exposes the handful of operations the
rest of the system needs: build an index, answer a question, evaluate a
bundle, promote a version. Everything is injectable, so tests can swap in
fakes without touching the filesystem or a network.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragllmops.config import Settings, get_settings
from ragllmops.embed import build_embedder
from ragllmops.evaluate.dataset import load_golden_set
from ragllmops.evaluate.runner import EvalReport, GateThresholds, run_evaluation
from ragllmops.features import (
    build_feature_provider,
    materialize,
    offline_features_path,
    parquet_features_path,
)
from ragllmops.feedback.dataset import (
    TRAIN,
    VALIDATION,
    Judgement,
    build_judgements,
    summarise,
)
from ragllmops.feedback.features import RankerFeatureExtractor
from ragllmops.feedback.labels import Verdict, encode_verdict
from ragllmops.feedback.model import (
    TrainingExample,
    artifact_path,
    evaluate_baseline,
    evaluate_model,
    load_ranker,
    save_ranker,
    train_ranker,
    training_data_hash,
)
from ragllmops.feedback.ranker import LearnedRanker
from ragllmops.ingest.pipeline import IndexResult, build_index, embed_chunks
from ragllmops.llm import build_llm
from ragllmops.logging_utils import get_logger
from ragllmops.models import RAGConfigBundle, RequestRecord
from ragllmops.rag.bundle import BundleBuilder
from ragllmops.rag.chain import RagChain
from ragllmops.registry import build_registry
from ragllmops.retrieval import (
    Bm25Index,
    ChunkCatalog,
    RetrievalService,
    build_reranker,
    build_vector_store,
)
from ragllmops.store import build_log_store

logger = get_logger("platform")


class Platform:
    """Everything the RAG service needs, wired together."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder=None,
        vector_store=None,
        llm=None,
        registry=None,
        log_store=None,
        feature_provider=None,
        reranker=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

        self.embedder = embedder or build_embedder(self.settings)
        self.vector_store = vector_store or build_vector_store(self.settings)
        self.llm = llm or build_llm(self.settings)
        self.registry = registry or build_registry(self.settings)
        self.log_store = log_store or build_log_store(self.settings)
        self.feature_provider = feature_provider or build_feature_provider(self.settings)
        self.reranker = (
            reranker
            if reranker is not None
            else build_reranker(self.settings.rerank_model, self.settings.rerank_enabled)
        )

        self.bm25 = Bm25Index()
        self.catalog = ChunkCatalog()
        self._retrieval: RetrievalService | None = None
        self._chains: dict[str, RagChain] = {}
        self._rankers: dict[str, LearnedRanker] = {}

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    @property
    def catalog_path(self) -> Path:
        return self.settings.data_path / "index" / "catalog.json"

    # ------------------------------------------------------------------
    # index lifecycle
    # ------------------------------------------------------------------
    def reindex(self, reset: bool = True) -> IndexResult:
        """Rebuild the whole index from the corpus."""
        chunks, result = build_index(
            embedder=self.embedder,
            vector_store=self.vector_store,
            corpus_dir=self.settings.corpus_path,
            reset=reset,
        )
        self.catalog = ChunkCatalog(chunks)
        self.catalog.save(self.catalog_path)
        self.bm25 = Bm25Index().fit(chunks)
        self._retrieval = None
        self._rebuild_retrieval()
        self.materialize_features()
        logger.info("reindex complete: %s chunks", result.chunks)
        return result

    def load_index(self) -> int:
        """Load a previously built index from disk.

        If the vector store lost its contents (for example the in-memory
        backend after a restart), the vectors are recomputed from the
        catalogue — the platform heals itself instead of serving nothing.
        """
        self.catalog = ChunkCatalog.load(self.catalog_path)
        if not len(self.catalog):
            logger.warning("no index found at %s; run `ragllmops ingest` first", self.catalog_path)
            self._retrieval = None
            self._rebuild_retrieval()
            return 0

        chunks = self.catalog.all()
        self.bm25 = Bm25Index().fit(chunks)

        try:
            stored = self.vector_store.count()
        except Exception:
            stored = 0
        if stored != len(chunks):
            logger.info(
                "vector store holds %s of %s chunks; re-embedding", stored, len(chunks)
            )
            vectors = embed_chunks(self.embedder, chunks)
            self.vector_store.upsert(chunks, vectors)

        self._retrieval = None
        self._rebuild_retrieval()
        logger.info("loaded %s chunks from %s", len(chunks), self.catalog_path)
        return len(chunks)

    def _rebuild_retrieval(self) -> None:
        """(Re)create the retrieval service from the current index."""
        self._retrieval = RetrievalService(
            embedder=self.embedder,
            dense_store=self.vector_store,
            bm25=self.bm25,
            reranker=self.reranker,
            top_k=self.settings.retrieval_top_k,
            candidates=self.settings.retrieval_candidates,
            dense_weight=self.settings.retrieval_dense_weight,
            sparse_weight=self.settings.retrieval_sparse_weight,
        )

    @property
    def retrieval(self) -> RetrievalService:
        if self._retrieval is None:
            self._rebuild_retrieval()
        return self._retrieval

    # ------------------------------------------------------------------
    # bundles & chains
    # ------------------------------------------------------------------
    def chain_for(self, bundle: RAGConfigBundle) -> RagChain:
        """Build (and cache) a chain for a bundle."""
        key = f"{bundle.version}:{bundle.fingerprint()[:8]}"
        chain = self._chains.get(key)
        if chain is None:
            chain = RagChain(
                bundle=bundle,
                retrieval=self.retrieval,
                llm=self.llm,
                feature_provider=self.feature_provider,
                ranker=self.ranker_for(bundle),
            )
            self._chains[key] = chain
        return chain

    def build_bundle(self, **overrides: Any) -> RAGConfigBundle:
        """Build a bundle that records the backends *this* platform resolved.

        The platform already holds the embedder and the reranker that serve
        traffic, so a bundle built here describes the running system rather
        than the requested configuration.
        """
        return BundleBuilder(self.settings).build(
            embedder=self.embedder,
            reranker=self.reranker,
            **overrides,
        )

    def default_bundle(self) -> RAGConfigBundle:
        """A bundle built from the current settings."""
        return self.build_bundle()

    def champion_bundle(self) -> RAGConfigBundle | None:
        """The bundle currently serving production traffic."""
        return self.registry.get_by_alias(self.settings.champion_alias)

    def challenger_bundle(self) -> RAGConfigBundle | None:
        """The bundle under evaluation."""
        return self.registry.get_by_alias(self.settings.challenger_alias)

    def current_bundle(self) -> RAGConfigBundle:
        """Champion if there is one, otherwise the newest registered bundle."""
        return (
            self.champion_bundle()
            or self.registry.latest()
            or self.default_bundle()
        )

    def register_bundle(self, bundle: RAGConfigBundle) -> str:
        """Register a bundle in the registry."""
        return self.registry.register(bundle)

    def promote(self, alias: str, version: str) -> None:
        """Point an alias at a version."""
        self.registry.set_alias(alias, version)

    # ------------------------------------------------------------------
    # serving
    # ------------------------------------------------------------------
    def answer(
        self,
        query: str,
        bundle: RAGConfigBundle | None = None,
        top_k: int | None = None,
        user_id: str | None = None,
        log: bool = True,
    ):
        """Answer a query with the given (or current) bundle."""
        active = bundle or self.current_bundle()
        chain = self.chain_for(active)
        result = chain.answer(query, top_k=top_k, user_id=user_id)

        if log:
            self.log_response(result)
        return result

    def log_response(self, result) -> None:
        """Write a served request to the request log."""
        response = result.response
        scores = [citation.score for citation in response.citations]
        record = RequestRecord(
            request_id=response.request_id,
            ts=datetime.now(UTC),
            query=response.query,
            answer=response.answer,
            bundle_version=response.bundle_version,
            prompt_id=response.prompt_id,
            llm_model=response.llm_model,
            llm_provider=response.llm_provider,
            latency_ms=response.latency_ms,
            top_score=max(scores) if scores else 0.0,
            chunk_count=response.retrieved_count,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            answer_chars=len(response.answer),
            refused=response.refused,
            retrieved_ids=[citation.chunk_id for citation in response.citations],
            user_id=response.user_id,
        )
        try:
            self.log_store.log(record)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not write to the request log: %s", exc)

    # ------------------------------------------------------------------
    # feedback
    # ------------------------------------------------------------------
    def record_feedback(
        self,
        request_id: str,
        verdict: str,
        chunk_id: str | None = None,
    ) -> RequestRecord:
        """Attach a user's verdict to the request that produced the answer.

        The judgement goes back into the request log, which keeps the log the
        single source of truth about production traffic: the learning loop
        trains on the same rows the monitoring reads.

        Raises ``KeyError`` when the request is unknown and ``ValueError``
        when the verdict or the blamed chunk does not make sense.
        """
        code = encode_verdict(verdict)
        record = self.log_store.get(request_id)
        if record is None:
            raise KeyError(f"no request {request_id!r} in the log")

        if verdict == Verdict.WRONG_CITATION.value and not chunk_id:
            raise ValueError("a wrong_citation verdict needs the chunk_id that was at fault")
        if chunk_id and chunk_id not in record.retrieved_ids:
            raise ValueError(
                f"chunk {chunk_id!r} was not served for request {request_id!r}; "
                f"the request retrieved {record.retrieved_ids}"
            )

        if not self.log_store.set_feedback(request_id, code, chunk_id):
            raise KeyError(f"no request {request_id!r} in the log")

        updated = self.log_store.get(request_id)
        assert updated is not None  # the row was just updated
        logger.info(
            "recorded %s feedback for request %s%s",
            verdict,
            request_id,
            f" (chunk {chunk_id})" if chunk_id else "",
        )
        return updated

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def gate_thresholds(self) -> GateThresholds:
        """Thresholds for the quality gate, taken from the settings."""
        return GateThresholds(
            min_faithfulness=self.settings.eval_gate_min_faithfulness,
            min_context_recall=self.settings.eval_gate_min_context_recall,
            min_key_term_coverage=self.settings.eval_gate_min_key_term_coverage,
            min_judge_grounded_rate=self.settings.eval_gate_min_judge_grounded_rate,
        )

    def evaluate(
        self,
        bundle: RAGConfigBundle | None = None,
        thresholds: GateThresholds | None = None,
        use_judge: bool = False,
    ) -> EvalReport:
        """Run the golden set against a bundle."""
        active = bundle or self.current_bundle()
        cases = load_golden_set(self.settings.eval_file)
        chain = self.chain_for(active)

        judge = None
        if use_judge:
            from ragllmops.evaluate.judge import judge_faithfulness

            def judge(question: str, answer: str, context: str) -> float:  # noqa: ANN001
                return judge_faithfulness(self.llm, question, answer, context)

        return run_evaluation(
            chain=chain,
            cases=cases,
            thresholds=thresholds or self.gate_thresholds(),
            judge=judge,
        )

    # ------------------------------------------------------------------
    # the learning loop
    # ------------------------------------------------------------------
    def feature_extractor(self) -> RankerFeatureExtractor:
        """The feature extractor the learned ranker is trained and served with."""
        return RankerFeatureExtractor(
            bm25=self.bm25,
            dense_store=self.vector_store,
            embedder=self.embedder,
            feature_provider=self.feature_provider,
            candidates=self.settings.retrieval_candidates,
            dense_weight=self.settings.retrieval_dense_weight,
            sparse_weight=self.settings.retrieval_sparse_weight,
        )

    def build_judgements(
        self,
        *,
        validation_fraction: float = 0.2,
        seed: int = 17,
        limit: int = 100000,
    ) -> list[Judgement]:
        """Turn the request log into a judgement set.

        Explicit judgements come first; the implicit ones only fill the gaps.
        Every example keeps the provenance of its label, and the split is by
        request so the validation slice is not contaminated by the training
        slice.
        """
        records = self.log_store.fetch(limit=limit)
        extractor = self.feature_extractor()
        cache: dict[str, tuple[list[str], dict[str, dict[str, float]]]] = {}

        def replay(query: str) -> tuple[list[str], dict[str, dict[str, float]]]:
            """The candidates of a query and their features, computed once."""
            entry = cache.get(query)
            if entry is None:
                chunks = extractor.candidates(query)
                entry = (
                    [chunk.chunk_id for chunk in chunks],
                    extractor.extract(query, chunks),
                )
                cache[query] = entry
            return entry

        def feature_fn(query: str, chunk_id: str) -> dict[str, float]:
            found = replay(query)[1].get(chunk_id)
            if found is None:
                # The chunk is no longer a candidate (the corpus changed);
                # score it on its own rather than dropping the example.
                chunk = self.catalog.get(chunk_id)
                found = extractor.extract(query, [chunk]).get(chunk_id, {}) if chunk else {}
            return found

        def candidate_fn(query: str) -> list[str]:
            return replay(query)[0]

        return build_judgements(
            records,
            feature_fn=feature_fn,
            candidate_fn=candidate_fn,
            validation_fraction=validation_fraction,
            seed=seed,
        )

    def train_ranker_from_feedback(
        self,
        *,
        validation_fraction: float = 0.2,
        seed: int = 17,
        min_examples: int = 20,
    ) -> dict[str, Any]:
        """Train a ranker on the feedback and store it as an artifact.

        Returns a report: the artifact id, the metrics on the held-out slice
        and the metrics of the rule-based baseline on the *same* slice. The
        gate compares those two numbers, so the learned ranker has to beat
        the rule it replaces — not merely exist.
        """
        judgements = self.build_judgements(
            validation_fraction=validation_fraction, seed=seed
        )
        examples = [
            TrainingExample(
                features=judgement.features,
                label=judgement.label,
                request_id=judgement.request_id,
                chunk_id=judgement.chunk_id,
            )
            for judgement in judgements
        ]
        pairs = list(zip(examples, judgements, strict=True))
        train = [example for example, judgement in pairs if judgement.split == TRAIN]
        validation = [example for example, judgement in pairs if judgement.split == VALIDATION]

        report: dict[str, Any] = {
            "examples": len(examples),
            "train_examples": len(train),
            "validation_examples": len(validation),
            "summary": summarise(judgements),
        }
        if len(examples) < min_examples:
            report.update(
                {
                    "trained": False,
                    "reason": (
                        f"only {len(examples)} labelled examples; "
                        f"at least {min_examples} are needed"
                    ),
                }
            )
            return report

        if not train:
            report.update({"trained": False, "reason": "the training slice is empty"})
            return report

        if not validation:
            report.update(
                {
                    "trained": False,
                    "reason": (
                        "the held-out slice is empty; there is no evidence to "
                        "judge the model on"
                    ),
                }
            )
            return report

        labels = {example.label for example in train}
        if len(labels) < 2:
            report.update(
                {
                    "trained": False,
                    "reason": (
                        "the training slice has only one kind of label "
                        f"({sorted(labels)}); a ranker cannot be fitted on it"
                    ),
                }
            )
            return report

        model = train_ranker(train)
        train_metrics = evaluate_model(model, train)
        validation_metrics = evaluate_model(model, validation)
        baseline_metrics = evaluate_baseline(validation)

        if not validation_metrics.get("pairs"):
            # Without a comparable pair the ranking metric is vacuous: the
            # model would be shipped on the strength of nothing.
            report.update(
                {
                    "trained": False,
                    "reason": (
                        "the held-out slice has no comparable (relevant, "
                        "irrelevant) pair; the model cannot be judged"
                    ),
                    "validation_metrics": validation_metrics,
                    "baseline_metrics": baseline_metrics,
                }
            )
            return report

        artifact_id = save_ranker(
            model,
            self.settings.ranker_dir,
            metadata={
                "training_data_hash": training_data_hash(train),
                "feature_schema_version": model.feature_schema_version,
                "feature_names": list(model.feature_names),
                "metrics": {"train": train_metrics, "validation": validation_metrics},
                "baseline_metrics": baseline_metrics,
                "examples": len(examples),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

        report.update(
            {
                "trained": True,
                "artifact_id": artifact_id,
                "model": model.to_dict(),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "baseline_metrics": baseline_metrics,
            }
        )
        logger.info(
            "trained a ranker on %s examples (%s); held-out pairwise accuracy %.3f",
            len(examples),
            artifact_id,
            validation_metrics.get("pairwise_accuracy", 0.0),
        )
        return report

    def ranker_for(self, bundle: RAGConfigBundle) -> LearnedRanker | None:
        """The learned ranker a bundle carries, if it has one."""
        artifact_id = bundle.ranker_artifact_id
        if not artifact_id:
            return None
        cached = self._rankers.get(artifact_id)
        if cached is not None:
            return cached

        path = artifact_path(self.settings.ranker_dir, artifact_id)
        if not path.exists():
            logger.warning("ranker artifact %s is missing at %s", artifact_id, path)
            return None
        try:
            model = load_ranker(path)
            ranker = LearnedRanker(model, self.feature_extractor(), artifact_id=artifact_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not load the ranker artifact %s (%s)", artifact_id, exc)
            return None
        self._rankers[artifact_id] = ranker
        return ranker

    # ------------------------------------------------------------------
    # features
    # ------------------------------------------------------------------
    def materialize_features(self) -> dict[str, dict[str, float]]:
        """Recompute the online features from the request log."""
        features = materialize(
            log_store=self.log_store,
            doc_ids=self.catalog.doc_ids() or None,
            offline_path=offline_features_path(self.settings),
            online_path=self.settings.features_online_file,
            parquet_path=parquet_features_path(self.settings),
        )
        # The online store caches the file in memory; refresh it so serving
        # sees the values that were just materialised.
        refresh = getattr(self.feature_provider, "refresh", None)
        if callable(refresh):
            refresh()
        return features

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """A snapshot of the platform's state, used by ``/health``."""
        try:
            chunks = self.vector_store.count()
        except Exception:
            chunks = -1

        return {
            "status": "ok" if chunks >= 0 else "degraded",
            "time": datetime.now(UTC).isoformat(),
            "index": {
                "chunks": chunks,
                "catalog": len(self.catalog),
                "documents": len(self.catalog.doc_ids()),
                "vector_store": getattr(self.vector_store, "name", "unknown"),
                "configured_vector_store": self.settings.vector_store,
            },
            "registry": {
                "backend": self.settings.registry_backend,
                "aliases": self.registry.aliases(),
                "versions": len(self.registry.list_versions()),
            },
            "retrieval": self.retrieval.stats(),
            "backends": self.backend_summary(),
            "log_store": {
                "backend": self.settings.log_store_backend,
                "records": self.log_store.count(),
            },
            "llm": {
                "provider": self.settings.llm_provider,
                "model": self.settings.llm_model,
            },
        }

    def backend_summary(self) -> dict[str, Any]:
        """What is actually running, next to what was asked for.

        The two are not the same when an optional dependency is missing: the
        hashing embedder silently replaces the requested sentence-transformer,
        and a promotion that records the requested name cannot be reproduced.
        """
        embedder = self.embedder
        reranker = self.reranker
        champion = self.champion_bundle()
        return {
            "embedder": {
                "id": getattr(embedder, "name", "unknown"),
                "dim": int(getattr(embedder, "dim", 0) or 0),
                "requested": self.settings.embedding_model,
                "requested_backend": self.settings.embedder_backend,
                "artifact_hash": getattr(embedder, "artifact_hash", "") or "unpinned",
            },
            "reranker": {
                "id": getattr(reranker, "name", None) or "disabled",
                "requested": self.settings.rerank_model,
                "artifact_hash": getattr(reranker, "artifact_hash", "") or "unpinned",
            },
            "llm": {
                "id": getattr(self.llm, "model", self.settings.llm_model),
                "provider": getattr(self.llm, "provider", self.settings.llm_provider),
            },
            "learned_ranker": {
                "artifact_id": champion.ranker_artifact_id if champion else "",
                "metrics": dict(champion.ranker_metrics) if champion else {},
            },
        }

    def start_time(self) -> float:
        """Monotonic start reference (used for uptime reporting)."""
        return time.perf_counter()
