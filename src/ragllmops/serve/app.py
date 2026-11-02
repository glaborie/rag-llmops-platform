"""The FastAPI serving layer.

Endpoints:

``POST /ask``            answer a question (canary-aware)
``POST /feedback``       record a user's verdict on an answer
``GET  /health``         platform state
``GET  /metrics``        Prometheus exposition
``GET  /admin/registry`` registered versions and aliases
``POST /admin/promote``  move an alias to a version
``POST /admin/ingest``   rebuild the index from the corpus
``POST /admin/evaluate`` run the golden set and the quality gate
``GET  /admin/drift``    drift report over the request log
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from ragllmops.logging_utils import configure_logging, get_logger
from ragllmops.monitor.alerts import build_notifier
from ragllmops.monitor.drift import compute_drift, split_windows
from ragllmops.monitor.experiment import canary_report
from ragllmops.platform import Platform
from ragllmops.serve import metrics as m
from ragllmops.serve.canary import CanaryRouter
from ragllmops.serve.watcher import AliasWatcher

logger = get_logger("serve.app")


# ----------------------------------------------------------------------
# schemas
# ----------------------------------------------------------------------
class AskRequest(BaseModel):
    """A question to answer."""

    query: str = Field(min_length=1, description="The question to answer.")
    top_k: int | None = Field(default=None, ge=1, le=20)
    user_id: str | None = None
    tenant: str | None = None


class PromoteRequest(BaseModel):
    """Move a registry alias."""

    alias: str = Field(min_length=1)
    version: str = Field(min_length=1)


class FeedbackRequest(BaseModel):
    """A user's verdict on an answer."""

    request_id: str = Field(min_length=1, description="The id returned by /ask.")
    verdict: str = Field(
        description="One of: helpful, unhelpful, wrong_citation.",
        pattern="^(helpful|unhelpful|wrong_citation)$",
    )
    chunk_id: str | None = Field(
        default=None,
        description="The chunk that was at fault, when the verdict blames one.",
    )


@dataclass(slots=True)
class AppState:
    """Mutable runtime state shared by the routes."""

    platform: Platform
    router: CanaryRouter
    watcher: AliasWatcher

    def refresh_router(self) -> None:
        """Re-read the aliases and rebuild the canary router."""
        settings = self.platform.settings
        self.router = CanaryRouter(
            champion=self.platform.champion_bundle(),
            challenger=self.platform.challenger_bundle(),
            traffic_pct=settings.canary_traffic_pct,
        )
        m.CANARY_TRAFFIC.set(settings.canary_traffic_pct)
        for alias, version in self.platform.registry.aliases().items():
            m.ALIAS_VERSION.labels(alias).set(_version_number(version))


def _version_number(version: str) -> float:
    """Turn a version string into a number for the gauge (best effort)."""
    try:
        return float(version.lstrip("v").split("_")[0])
    except (ValueError, AttributeError):
        return 0.0


def _ensure_index(platform: Platform) -> None:
    """Build the retrieval index when it is missing.

    A stack that starts from an empty volume has no catalogue, and without
    one there is nothing to retrieve.
    """
    if len(platform.catalog):
        return
    try:
        result = platform.reindex()
        logger.info("no index found; built one from the corpus (%s chunks)", result.chunks)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("could not build the retrieval index (%s)", exc)


def _bootstrap_registry(platform: Platform) -> None:
    """Give a fresh stack something to serve.

    A registry without a champion cannot answer a single question, so the
    default bundle is registered and promoted once. An existing champion is
    never touched — this is a bootstrap, not a takeover.
    """
    alias = platform.settings.champion_alias
    if platform.champion_bundle() is not None:
        return
    try:
        bundle = platform.default_bundle()
        version = platform.register_bundle(bundle)
        platform.promote(alias, version)
        logger.info(
            "no champion in the registry; registered %s as %s", bundle.version, alias
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("could not bootstrap the registry (%s)", exc)


# ----------------------------------------------------------------------
# application factory
# ----------------------------------------------------------------------
def create_app(platform: Platform | None = None) -> FastAPI:
    """Build the FastAPI application."""
    configure_logging()
    active_platform = platform or Platform()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202 - FastAPI API
        active_platform.load_index()
        _ensure_index(active_platform)
        _bootstrap_registry(active_platform)
        state.refresh_router()
        state.watcher.poll()
        task = asyncio.create_task(state.watcher.run())
        logger.info("ragllmops API ready")
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="Northwind RAG API",
        description=(
            "A locally-operated RAG service with experiment tracking, model "
            "registry, data validation, drift monitoring and CI/CD."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    state = AppState(
        platform=active_platform,
        router=CanaryRouter(None, None, active_platform.settings.canary_traffic_pct),
        # A promotion that happens outside this process (the CLI, another
        # replica) still has to reach the traffic: the watcher rebuilds the
        # router as soon as it sees the alias move.
        watcher=AliasWatcher(
            active_platform,
            notifier=build_notifier(active_platform.settings),
            on_change=lambda: state.refresh_router(),
        ),
    )
    app.state.platform = state

    # ------------------------------------------------------------------
    @app.post("/ask", tags=["serving"])
    def ask(payload: AskRequest) -> dict[str, Any]:
        """Answer a question."""
        request_id = _request_id()
        bundle = state.router.select(request_id)
        if bundle is None:
            raise HTTPException(status_code=503, detail="no bundle registered")

        started = time.perf_counter()
        try:
            result = active_platform.answer(
                payload.query, bundle=bundle, top_k=payload.top_k, user_id=payload.user_id
            )
        except Exception as exc:
            logger.exception("answer failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        elapsed = time.perf_counter() - started

        response = result.response
        m.REQUESTS.labels(
            bundle_version=response.bundle_version,
            prompt_id=response.prompt_id,
            provider=response.llm_provider,
        ).inc()
        m.LATENCY.labels(bundle_version=response.bundle_version).observe(elapsed)
        m.TOKENS.labels("prompt", response.llm_model).inc(response.prompt_tokens)
        m.TOKENS.labels("completion", response.llm_model).inc(response.completion_tokens)
        m.RETRIEVED_CHUNKS.observe(response.retrieved_count)
        if response.citations:
            m.RETRIEVAL_SCORE.observe(max(c.score for c in response.citations))
        if response.refused:
            m.REFUSALS.labels(bundle_version=response.bundle_version).inc()

        payload_out = response.to_dict()
        payload_out["route"] = state.router.route_of(request_id)
        return payload_out

    # ------------------------------------------------------------------
    @app.post("/feedback", tags=["serving"])
    def feedback(payload: FeedbackRequest) -> dict[str, Any]:
        """Record what the user thought of an answer.

        The judgement is written back to the request log; the learning loop
        trains on exactly these rows.
        """
        try:
            record = active_platform.record_feedback(
                payload.request_id, payload.verdict, payload.chunk_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        m.FEEDBACK.labels(payload.verdict).inc()
        return {
            "request_id": record.request_id,
            "verdict": payload.verdict,
            "chunk_id": record.feedback_chunk_id,
            "recorded": True,
        }

    # ------------------------------------------------------------------
    @app.get("/health", tags=["serving"])
    def health() -> dict[str, Any]:
        """Platform health snapshot."""
        data = active_platform.health()
        data["canary"] = {
            "traffic_pct": state.router.traffic_pct,
            "active": state.router.active,
            "champion": state.router.champion.version if state.router.champion else None,
            "challenger": state.router.challenger.version if state.router.challenger else None,
        }
        return data

    @app.get("/metrics", tags=["serving"])
    def prometheus_metrics() -> Response:
        """Prometheus metrics."""
        return Response(content=generate_latest(m.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    # ------------------------------------------------------------------
    @app.get("/admin/registry", tags=["admin"])
    def registry_state() -> dict[str, Any]:
        """Registered versions and alias assignments."""
        registry = active_platform.registry
        return {
            "aliases": registry.aliases(),
            "versions": [
                {
                    "version": version,
                    "prompt_id": (registry.get(version).prompt_id if registry.get(version) else None),
                    "metrics": (registry.get(version).metrics if registry.get(version) else {}),
                }
                for version in registry.list_versions()
            ],
        }

    @app.post("/admin/promote", tags=["admin"])
    def promote(payload: PromoteRequest) -> dict[str, Any]:
        """Move an alias to a version."""
        try:
            active_platform.promote(payload.alias, payload.version)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        state.refresh_router()
        state.watcher.poll()
        return {"alias": payload.alias, "version": payload.version, "aliases": (
            active_platform.registry.aliases()
        )}

    @app.post("/admin/ingest", tags=["admin"])
    def ingest() -> dict[str, Any]:
        """Rebuild the index from the corpus."""
        result = active_platform.reindex()
        return result.to_dict()

    @app.post("/admin/evaluate", tags=["admin"])
    def evaluate() -> dict[str, Any]:
        """Run the golden set and the quality gate."""
        report = active_platform.evaluate()
        return report.to_dict()

    @app.get("/admin/drift", tags=["admin"])
    def drift() -> dict[str, Any]:
        """Drift report over the request log."""
        records = active_platform.log_store.fetch(limit=5000)
        reference, current = split_windows(records)
        report = compute_drift(
            reference, current, threshold=active_platform.settings.drift_psi_threshold
        )
        for feature in report.features:
            m.DRIFT.labels(feature.feature).set(feature.psi)
        return report.to_dict()

    @app.get("/admin/canary", tags=["admin"])
    def canary() -> dict[str, Any]:
        """Compare the champion and the challenger on the traffic they served.

        A traffic split on its own says nothing about which side is better;
        this is the part that does.
        """
        records = active_platform.log_store.fetch(limit=50000)
        champion = active_platform.champion_bundle()
        challenger = active_platform.challenger_bundle()
        return canary_report(
            records,
            champion.version if champion else None,
            challenger.version if challenger else None,
            min_samples=active_platform.settings.canary_min_samples,
        )

    return app


def _request_id() -> str:
    import uuid

    return uuid.uuid4().hex


app = None  # populated by `ragllmops serve` / uvicorn factory
