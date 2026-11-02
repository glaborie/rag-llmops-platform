"""Core domain models shared across the platform.

These are deliberately plain dataclasses: they cross process boundaries (they
are serialised into the registry, the request log and the MLflow artifacts), so
they must stay stable and dependency-free.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> datetime:
    return datetime.now(UTC)


def stable_hash(payload: Any) -> str:
    """Deterministic sha256 over any JSON-serialisable payload."""
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A retrievable unit of text carved out of a source document."""

    chunk_id: str
    doc_id: str
    title: str
    category: str
    text: str
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DocumentChunk:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk plus the scores that got it into the final context."""

    chunk: DocumentChunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "rerank_score": self.rerank_score,
        }


@dataclass(frozen=True, slots=True)
class Citation:
    """A citation returned to the caller."""

    index: int
    chunk_id: str
    doc_id: str
    title: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RAGConfigBundle:
    """The registrable unit of this platform.

    A bundle is *everything* that determines an answer: prompt template,
    embedding model, retrieval parameters, reranking settings and the LLM
    identity. Registering bundles (rather than just model weights) is what
    lets us promote a whole configuration safely.

    The backends are recorded twice on purpose:

    * ``embedder_id`` / ``reranker_id`` are the backends that **actually
      ran** — the resolved identity, not the configured name;
    * ``embedder_requested`` / ``reranker_requested`` are what the settings
      **asked for**, kept for provenance.

    A registry that only stores the requested names claims a bundle was
    evaluated with ``all-MiniLM-L6-v2`` when it was evaluated with the
    hashing embedder, which makes the promotion impossible to reproduce.
    """

    version: str
    prompt_id: str
    prompt_template: str
    llm_provider: str
    llm_model: str
    llm_base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    retrieval_top_k: int = 4
    retrieval_candidates: int = 20
    rerank_enabled: bool = True
    max_context_chars: int = 6000
    # --- resolved provenance -------------------------------------------
    #: Identity of the embedder that ran, e.g. ``hashing-v1``.
    embedder_id: str = ""
    #: Width of the vectors that embedder produces.
    embedding_dim: int = 0
    #: The model name the configuration asked for.
    embedder_requested: str = ""
    #: Identity of the reranker that ran, e.g. ``term-overlap-v1``.
    reranker_id: str = ""
    #: The model name the configuration asked for.
    reranker_requested: str = ""
    #: Content hash of the model files behind the resolved backends.
    model_artifact_hash: str = ""
    #: Artifact id of the learned ranker, when the bundle uses one.
    ranker_artifact_id: str = ""
    #: The learned ranker's metrics on the held-out slice.
    ranker_metrics: dict[str, float] = field(default_factory=dict)
    created_at: str = ""
    notes: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Content hash used for de-duplication and cache keys."""
        payload = {
            "prompt_id": self.prompt_id,
            "prompt_template": self.prompt_template,
            "embedder_id": self.embedder_id,
            "embedding_dim": self.embedding_dim,
            "embedder_requested": self.embedder_requested,
            "reranker_id": self.reranker_id,
            "reranker_requested": self.reranker_requested,
            "model_artifact_hash": self.model_artifact_hash,
            "ranker_artifact_id": self.ranker_artifact_id,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "retrieval_top_k": self.retrieval_top_k,
            "retrieval_candidates": self.retrieval_candidates,
            "rerank_enabled": self.rerank_enabled,
            "max_context_chars": self.max_context_chars,
        }
        return stable_hash(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RAGConfigBundle:
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in payload.items() if k in known}
        # Bundles registered before the provenance fix only carried the
        # *requested* model names. Keep them as the requested names; the
        # resolved identity of those old bundles is simply unknown.
        if not data.get("embedder_requested") and payload.get("embedding_model"):
            data["embedder_requested"] = payload["embedding_model"]
        if not data.get("reranker_requested") and payload.get("rerank_model"):
            data["reranker_requested"] = payload["rerank_model"]
        return cls(**data)

    def backend_summary(self) -> dict[str, str]:
        """What actually served this bundle, for reports and ``/health``."""
        return {
            "embedder": self.embedder_id or "unknown",
            "embedding_dim": str(self.embedding_dim),
            "embedder_requested": self.embedder_requested or "unknown",
            "reranker": self.reranker_id or "unknown",
            "reranker_requested": self.reranker_requested or "unknown",
            "model_artifact_hash": self.model_artifact_hash or "unpinned",
            "ranker": self.ranker_artifact_id or "rule-based",
        }

    def with_metrics(self, metrics: dict[str, float]) -> RAGConfigBundle:
        merged = {**self.metrics, **metrics}
        return RAGConfigBundle(**{**self.to_dict(), "metrics": merged})


@dataclass(frozen=True, slots=True)
class RAGResponse:
    """The answer returned by the serving layer."""

    request_id: str
    answer: str
    citations: list[Citation]
    bundle_version: str
    prompt_id: str
    llm_model: str
    llm_provider: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    retrieved_count: int
    refused: bool
    query: str = ""
    user_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["citations"] = [c.to_dict() for c in self.citations]
        return payload


@dataclass(slots=True)
class RequestRecord:
    """One served request, written to the request log.

    The request log is the backbone of monitoring: it is the "production
    data" that drift detection reads and that future training runs consume.
    """

    request_id: str
    ts: datetime
    query: str
    answer: str
    bundle_version: str
    prompt_id: str
    llm_model: str
    llm_provider: str
    latency_ms: float
    top_score: float
    chunk_count: int
    prompt_tokens: int
    completion_tokens: int
    answer_chars: int
    refused: bool
    retrieved_ids: list[str]
    user_id: str | None = None
    tenant: str | None = None
    #: The user's verdict, encoded by :mod:`ragllmops.feedback.labels`.
    feedback: int | None = None
    #: The chunk the user blamed, when they blamed one.
    feedback_chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ts"] = self.ts.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RequestRecord:
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in payload.items() if k in known}
        if isinstance(data.get("ts"), str):
            data["ts"] = datetime.fromisoformat(data["ts"])
        return cls(**data)
