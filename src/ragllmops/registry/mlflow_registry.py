"""MLflow-backed bundle registry (optional).

Configuration bundles are registered as MLflow models so that promotion,
rollback and audit all happen inside the tracking server. The bundle is
stored as a JSON artifact of a ``pyfunc`` model; metrics recorded during
evaluation are attached to the same model version.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any

from ragllmops.logging_utils import get_logger
from ragllmops.models import RAGConfigBundle

logger = get_logger("registry.mlflow")


def _require_mlflow():
    try:
        import mlflow  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "mlflow is not installed. Run: pip install -r requirements-ml.txt "
            "(or set RAG_REGISTRY_BACKEND=local)"
        ) from exc
    return mlflow


def _make_python_model(bundle: RAGConfigBundle):
    """Build a pyfunc model class carrying the bundle."""
    mlflow = _require_mlflow()

    class BundlePythonModel(mlflow.pyfunc.PythonModel):
        """Carries a RAG configuration bundle as an MLflow model."""

        def __init__(self, config: RAGConfigBundle) -> None:
            self.bundle = config

        def load_context(self, context) -> None:  # noqa: ANN001 - MLflow API
            artifact = context.artifacts.get("bundle.json")
            if artifact:
                payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
                self.bundle = RAGConfigBundle.from_dict(payload)

        def predict(self, context, model_input, params=None):  # noqa: ANN001 - MLflow API
            return {
                "version": self.bundle.version,
                "prompt_id": self.bundle.prompt_id,
                "llm_model": self.bundle.llm_model,
            }

    return BundlePythonModel(bundle)


class MlflowBundleRegistry:
    """Registry that stores bundles as MLflow model versions."""

    def __init__(
        self,
        tracking_uri: str,
        registered_model_name: str,
        artifact_root: str | None = None,
    ) -> None:
        mlflow = _require_mlflow()
        mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self.registered_model_name = registered_model_name
        self._mlflow = mlflow
        self._client = mlflow.MlflowClient()
        self._experiment_id: str | None = None
        if artifact_root:
            self._experiment_id = self._open_experiment(artifact_root)

    # ------------------------------------------------------------------
    def _open_experiment(self, artifact_root: str) -> str | None:
        """Keep the bundles in an experiment that points at *artifact_root*.

        The artifact location of an experiment is fixed when the experiment is
        created, so the bundles cannot simply be logged in the default
        experiment with a custom root. Returns the experiment id, or None
        when the default experiment has to be used.
        """
        setter = getattr(self._mlflow, "set_experiment_artifact_location", None)
        if setter is not None:  # pragma: no cover - very old MLflow
            setter(artifact_root)
            return None

        name = f"{self.registered_model_name}-bundles"
        try:
            experiment = self._client.get_experiment_by_name(name)
        except Exception:
            experiment = None
        if experiment is not None:
            # get_experiment_by_name returns an Experiment entity.
            return str(getattr(experiment, "experiment_id", experiment))

        try:
            created = self._client.create_experiment(
                name, artifact_location=artifact_root
            )
        except Exception as exc:
            logger.warning("could not create the MLflow experiment (%s)", exc)
            return None
        # create_experiment returns the experiment id.
        return str(getattr(created, "experiment_id", created))

    # ------------------------------------------------------------------
    def register(self, bundle: RAGConfigBundle) -> str:
        """Register a bundle as a new MLflow model version."""
        with tempfile.TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "bundle.json"
            artifact_path.write_text(
                json.dumps(bundle.to_dict(), indent=2, default=str), encoding="utf-8"
            )
            with self._run_context():
                self._mlflow.pyfunc.log_model(
                    # "artifact_path" is the *name* the model is stored under
                    # in the run, not a local directory: MLflow rejects an
                    # absolute path, and the model files are written by
                    # MLflow itself.
                    artifact_path="model",
                    python_model=_make_python_model(bundle),
                    artifacts={"bundle.json": str(artifact_path)},
                    registered_model_name=self.registered_model_name,
                )

        versions = self.list_versions()
        logger.info("registered bundle %s in MLflow (%s)", bundle.version, self.registered_model_name)
        return versions[0] if versions else bundle.version

    def _run_context(self):
        """Log the model inside the experiment that owns the artifacts."""
        if self._experiment_id is None:
            return contextlib.nullcontext()
        return self._mlflow.start_run(experiment_id=self._experiment_id)

    def get(self, version: str) -> RAGConfigBundle | None:
        """Fetch a bundle by MLflow model version or by bundle version."""
        if version.isdigit():
            return self._get_model_version(version)
        model_version = self._resolve_model_version(version)
        return self._get_model_version(model_version) if model_version else None

    def _get_model_version(self, model_version: str) -> RAGConfigBundle | None:
        """Load the bundle stored in an MLflow model version."""
        try:
            model = self._mlflow.pyfunc.load_model(
                f"models:/{self.registered_model_name}/{model_version}"
            )
        except Exception as exc:
            logger.warning("could not load MLflow version %s (%s)", model_version, exc)
            return None
        return _extract_bundle(model)

    def list_versions(self) -> list[str]:
        try:
            found = self._client.search_model_versions(
                f"name='{self.registered_model_name}'"
            )
        except Exception as exc:
            logger.warning("could not list MLflow versions (%s)", exc)
            return []
        # The model version is a number, but the search API does not sort on
        # it — sort here so that the newest version comes first.
        return sorted(
            (str(item.version) for item in found),
            key=lambda version: int(version) if version.isdigit() else 0,
            reverse=True,
        )

    def set_alias(self, alias: str, version: str) -> None:
        """Point *alias* at a bundle.

        The platform addresses bundles by their content fingerprint
        (``v<hash>``) while MLflow addresses them by a model version number.
        Both are accepted so the rest of the platform does not have to know
        which registry is in use.
        """
        model_version = self._resolve_model_version(version)
        self._client.set_registered_model_alias(self.registered_model_name, alias, model_version)
        logger.info("alias %s -> %s (MLflow model version %s)", alias, version, model_version)

    def _resolve_model_version(self, version: str) -> str:
        """Map a bundle version onto the MLflow model version."""
        if version.isdigit():
            return version
        for candidate in self.list_versions():
            bundle = self._get_model_version(candidate)
            if bundle is not None and bundle.version == version:
                return candidate
        raise KeyError(f"no MLflow model version holds the bundle {version}")

    def get_by_alias(self, alias: str) -> RAGConfigBundle | None:
        try:
            model_version = self._client.get_model_version_by_alias(
                self.registered_model_name, alias
            )
        except Exception:
            return None
        return self.get(str(model_version.version))

    def aliases(self) -> dict[str, str]:
        try:
            model = self._client.get_registered_model(self.registered_model_name)
        except Exception:
            return {}
        raw = getattr(model, "aliases", None)
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
        return {}

    def latest(self) -> RAGConfigBundle | None:
        versions = self.list_versions()
        return self.get(versions[0]) if versions else None

    def log_metrics(self, version: str, metrics: dict[str, float]) -> None:
        """Attach evaluation metrics to a registered model version."""
        try:
            self._client.log_model_version(
                name=self.registered_model_name, version=version, metrics=metrics
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not log metrics for %s (%s)", version, exc)


def _extract_bundle(model: Any) -> RAGConfigBundle | None:
    """Pull the bundle out of a loaded pyfunc model."""
    try:
        python_model = model.unwrap_python_model()
    except Exception:
        python_model = getattr(model, "_model_impl", None)
        python_model = getattr(python_model, "python_model", None)
    bundle = getattr(python_model, "bundle", None)
    return bundle if isinstance(bundle, RAGConfigBundle) else None
