"""Shared fixtures.

Every test runs against a temporary directory with dependency-free backends,
so the suite needs no Docker, no network and no API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragllmops.config import Settings
from ragllmops.embed.hashing import HashingEmbedder
from ragllmops.features.local_provider import LocalFeatureProvider
from ragllmops.llm.extractive import ExtractiveLLM
from ragllmops.platform import Platform
from ragllmops.registry.local_registry import LocalBundleRegistry
from ragllmops.retrieval.dense import InMemoryVectorStore
from ragllmops.store.sqlite_log_store import SQLiteLogStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CORPUS = PROJECT_ROOT / "data" / "corpus"
REAL_GOLDEN_SET = PROJECT_ROOT / "data" / "eval" / "golden_set.jsonl"

SAMPLE_DOCS = {
    "compute.md": """---
doc_id: compute
title: Compute Instances
category: compute
---

## Instance families

The nw-standard family targets 65% CPU utilisation. Autoscaling uses a 120 second
cooldown and a 300 second scale-down delay.

## Storage

Root volumes are 10 GB by default.
""",
    "storage.md": """---
doc_id: storage
title: Object Storage
category: storage
---

## Retention

Objects are retained for 90 days by default. Lifecycle policies expire objects
after 30 days in the archive tier.

## Replication

Cross-region replication uses a 15 minute sync window.
""",
}


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A small synthetic corpus."""
    directory = tmp_path / "corpus"
    directory.mkdir()
    for name, content in SAMPLE_DOCS.items():
        (directory / name).write_text(content, encoding="utf-8")
    return directory


@pytest.fixture
def golden_set_path(tmp_path: Path) -> Path:
    """A small synthetic golden set matching the synthetic corpus."""
    path = tmp_path / "eval" / "golden.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        '{"id": "q001", "question": "What CPU target does the autoscaler use?",'
        ' "expected_doc_ids": ["compute"], "key_terms": ["65%", "120 second"],'
        ' "category": "compute", "difficulty": "easy"}',
        '{"id": "q002", "question": "What is the default object retention period?",'
        ' "expected_doc_ids": ["storage"], "key_terms": ["90 days"],'
        ' "category": "storage", "difficulty": "easy"}',
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def settings(tmp_path: Path, corpus_dir: Path, golden_set_path: Path) -> Settings:
    """Settings pointing at a temporary workspace."""
    config = Settings(
        project_root=tmp_path,
        data_dir=Path("data"),
        corpus_dir=corpus_dir,
        eval_path=golden_set_path,
        reports_dir=Path("data/reports"),
        registry_path=Path("data/registry/registry.json"),
        sqlite_path=Path("data/logs/requests.db"),
        features_online_path=Path("data/features/online_store.json"),
        embedder_backend="hashing",
        embedding_dim=128,
        vector_store="memory",
        registry_backend="local",
        log_store_backend="sqlite",
        feature_backend="local",
        llm_provider="extractive",
        llm_model="local-extractive-v1",
        rerank_enabled=True,
        retrieval_top_k=2,
    )
    config.ensure_dirs()
    return config


@pytest.fixture
def platform(settings: Settings) -> Platform:
    """A platform with dependency-free backends."""
    instance = Platform(
        settings=settings,
        embedder=HashingEmbedder(dim=settings.embedding_dim),
        vector_store=InMemoryVectorStore(dim=settings.embedding_dim),
        llm=ExtractiveLLM(),
        registry=LocalBundleRegistry(settings.registry_file),
        log_store=SQLiteLogStore(settings.sqlite_file),
        feature_provider=LocalFeatureProvider(settings.features_online_file),
    )
    instance.reindex()
    return instance


@pytest.fixture
def real_corpus_available() -> bool:
    """Whether the shipped corpus is present."""
    return REAL_CORPUS.exists() and REAL_GOLDEN_SET.exists()
