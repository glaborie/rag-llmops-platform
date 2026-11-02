"""Environment-driven configuration.

Every setting can be overridden with a ``RAG_``-prefixed environment variable
(see ``.env.example``). Nothing here imports an optional dependency, so this
module is always safe to import.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the whole platform."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = PROJECT_ROOT

    # --- paths ------------------------------------------------------------
    data_dir: Path = Path("data")
    corpus_dir: Path = Path("data/corpus")
    eval_path: Path = Path("data/eval/golden_set.jsonl")
    reports_dir: Path = Path("data/reports")
    models_dir: Path = Path("data/models")

    # --- LLM --------------------------------------------------------------
    llm_provider: str = "extractive"  # extractive | openai | ollama
    llm_model: str = "local-extractive-v1"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 512

    # --- embeddings -------------------------------------------------------
    embedder_backend: str = "hashing"  # hashing | sentence-transformer
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # --- retrieval --------------------------------------------------------
    vector_store: str = "qdrant"  # qdrant | memory
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "ragllmops_chunks"
    #: Number of chunks put in front of the model. Tuned against the golden
    #: set: k=1 maximises precision (0.93) but loses recall (0.87), k=4
    #: maximises recall (0.97) but drops precision to 0.43. k=2 sits at the
    #: knee of the curve (precision 0.68, recall 0.95).
    retrieval_top_k: int = 2
    retrieval_candidates: int = 20
    #: Relative weight of the dense ranking in the reciprocal-rank fusion.
    #: Tuned against the golden set (see `docs/TUNING.md`).
    retrieval_dense_weight: float = 0.35
    retrieval_sparse_weight: float = 0.65
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_context_chars: int = 6000

    # --- registry ---------------------------------------------------------
    registry_backend: str = "local"  # local | mlflow
    registry_path: Path = Path("data/registry/registry.json")
    registry_name: str = "ragllmops-bundle"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_root: str = "./mlruns"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"

    # --- request log ------------------------------------------------------
    log_store_backend: str = "sqlite"  # sqlite | postgres
    sqlite_path: Path = Path("data/logs/requests.db")
    postgres_dsn: str = "postgresql://rag:rag@localhost:5432/rag"

    # --- features ---------------------------------------------------------
    feature_backend: str = "local"  # local | feast
    feast_repo: Path = Path("data/features/feature_repo")
    features_online_path: Path = Path("data/features/online_store.json")

    # --- serving ----------------------------------------------------------
    canary_traffic_pct: int = 0
    canary_min_samples: int = 30
    watcher_interval_seconds: int = 15

    # --- gates ------------------------------------------------------------
    eval_gate_min_faithfulness: float = 0.5
    eval_gate_min_context_recall: float = 0.6
    eval_gate_min_key_term_coverage: float = 0.5
    #: How many answers the LLM judge has to call grounded. Only applies
    #: when a judge actually returned verdicts.
    eval_gate_min_judge_grounded_rate: float = 0.5

    # --- monitoring -------------------------------------------------------
    drift_psi_threshold: float = 0.2
    drift_min_samples: int = 50
    alert_webhook_url: str | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def path(self, value: Path) -> Path:
        """Resolve *value* relative to the project root when it is relative."""
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (self.project_root / candidate)

    @property
    def corpus_path(self) -> Path:
        return self.path(self.corpus_dir)

    @property
    def eval_file(self) -> Path:
        return self.path(self.eval_path)

    @property
    def reports_path(self) -> Path:
        return self.path(self.reports_dir)

    @property
    def models_path(self) -> Path:
        """Where trained artifacts (the learned ranker) are stored."""
        return self.path(self.models_dir)

    @property
    def ranker_dir(self) -> Path:
        """Where the learned ranker artifacts live."""
        return self.models_path / "ranker"

    @property
    def registry_file(self) -> Path:
        return self.path(self.registry_path)

    @property
    def sqlite_file(self) -> Path:
        return self.path(self.sqlite_path)

    @property
    def features_online_file(self) -> Path:
        return self.path(self.features_online_path)

    @property
    def data_path(self) -> Path:
        return self.path(self.data_dir)

    def ensure_dirs(self) -> None:
        """Create the writable directories the platform needs."""
        for directory in (
            self.data_path,
            self.reports_path,
            self.registry_file.parent,
            self.sqlite_file.parent,
            self.features_online_file.parent,
            self.data_path / "cache",
            self.data_path / "runs",
            self.models_path,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (cached)."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache — used by tests that patch the environment."""
    get_settings.cache_clear()
