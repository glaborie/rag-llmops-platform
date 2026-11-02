"""Construction and persistence of registrable configuration bundles."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragllmops.artifacts import combined_artifact_hash
from ragllmops.config import Settings, get_settings
from ragllmops.embed import build_embedder
from ragllmops.logging_utils import get_logger
from ragllmops.models import RAGConfigBundle, stable_hash
from ragllmops.rag.prompts import DEFAULT_PROMPT_ID, get_prompt
from ragllmops.retrieval.rerank import build_reranker

logger = get_logger("rag.bundle")


def load_bundle(payload: dict[str, Any]) -> RAGConfigBundle:
    """Rebuild a bundle from a serialised mapping."""
    return RAGConfigBundle.from_dict(payload)


def with_version(bundle: RAGConfigBundle) -> RAGConfigBundle:
    """Re-derive the version of a bundle after its content changed.

    A bundle is addressed by its fingerprint, so anything that changes what
    the bundle does has to change its version too — otherwise two different
    configurations would share one address in the registry.
    """
    return replace(bundle, version=f"v{bundle.fingerprint()[:10]}")


def dump_bundle(bundle: RAGConfigBundle) -> str:
    """Serialise a bundle to canonical JSON (stable key order)."""
    return json.dumps(bundle.to_dict(), sort_keys=True, indent=2, default=str)


class BundleBuilder:
    """Builds bundles from the ambient settings plus explicit overrides.

    The builder resolves the embedder and the reranker *before* it writes the
    bundle, so what gets registered is what actually ran — not what the
    configuration asked for. A missing optional dependency therefore shows up
    in the registry as the fallback that was really used.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def build(
        self,
        *,
        embedder=None,
        reranker=None,
        prompt_id: str = DEFAULT_PROMPT_ID,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        llm_base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retrieval_top_k: int | None = None,
        retrieval_candidates: int | None = None,
        rerank_enabled: bool | None = None,
        max_context_chars: int | None = None,
        notes: str = "",
        metrics: dict[str, float] | None = None,
    ) -> RAGConfigBundle:
        """Create a bundle, defaulting everything to the current settings.

        ``embedder`` and ``reranker`` are the *resolved* backend objects. When
        they are not supplied the builder resolves them itself, which keeps
        the CLI and the closed loop honest without a running platform.
        """
        s = self.settings
        prompt = get_prompt(prompt_id)
        provider = llm_provider or s.llm_provider
        model = llm_model or s.llm_model
        base_url = llm_base_url if llm_base_url is not None else s.llm_base_url

        embedder = embedder if embedder is not None else build_embedder(s)
        enabled = s.rerank_enabled if rerank_enabled is None else rerank_enabled
        if reranker is None:
            reranker = build_reranker(s.rerank_model, enabled)

        embedder_id = str(getattr(embedder, "name", "") or "unknown")
        embedding_dim = int(getattr(embedder, "dim", 0) or 0)
        reranker_id = str(getattr(reranker, "name", "") or "disabled")
        artifact_hash = combined_artifact_hash(
            str(getattr(embedder, "artifact_hash", "") or ""),
            str(getattr(reranker, "artifact_hash", "") or ""),
        )

        bundle = RAGConfigBundle(
            version="",
            prompt_id=prompt.prompt_id,
            prompt_template=prompt.user,
            embedder_id=embedder_id,
            embedding_dim=embedding_dim,
            embedder_requested=s.embedding_model,
            reranker_id=reranker_id,
            reranker_requested=s.rerank_model,
            model_artifact_hash=artifact_hash,
            llm_provider=provider,
            llm_model=model,
            llm_base_url=base_url,
            temperature=s.temperature if temperature is None else temperature,
            max_tokens=s.max_tokens if max_tokens is None else max_tokens,
            retrieval_top_k=s.retrieval_top_k if retrieval_top_k is None else retrieval_top_k,
            retrieval_candidates=(
                s.retrieval_candidates if retrieval_candidates is None else retrieval_candidates
            ),
            rerank_enabled=enabled,
            max_context_chars=(
                s.max_context_chars if max_context_chars is None else max_context_chars
            ),
            created_at=datetime.now(UTC).isoformat(),
            notes=notes,
            metrics=dict(metrics or {}),
        )
        return self._with_version(bundle)

    @staticmethod
    def _with_version(bundle: RAGConfigBundle) -> RAGConfigBundle:
        """Derive a deterministic version string from the bundle content."""
        short = bundle.fingerprint()[:10]
        return replace(bundle, version=f"v{short}")
    def from_file(self, path: Path) -> RAGConfigBundle:
        """Load a bundle from a JSON file on disk."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        bundle = load_bundle(payload)
        logger.debug("loaded bundle %s from %s", bundle.version, path)
        return bundle


def bundle_diff(old: RAGConfigBundle, new: RAGConfigBundle) -> dict[str, Any]:
    """Field-level diff between two bundles — used in promotion reports."""
    a, b = old.to_dict(), new.to_dict()
    changed: dict[str, Any] = {}
    for key in sorted(set(a) | set(b)):
        if key in {"version", "created_at", "metrics"}:
            continue
        if a.get(key) != b.get(key):
            changed[key] = {"old": a.get(key), "new": b.get(key)}
    return changed


def bundle_hash(bundle: RAGConfigBundle) -> str:
    """Alias for the content fingerprint, for readability at call sites."""
    return stable_hash(bundle.to_dict())
