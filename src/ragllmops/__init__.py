"""Local, end-to-end LLMOps platform for a RAG service.

The package is organised as a set of small, swappable backends behind a single
:class:`ragllmops.platform.Platform` façade:

``rag``        prompts, chain, registrable configuration bundles
``embed``      embedding backends (hashing / sentence-transformers)
``retrieval``  dense store, BM25, hybrid fusion, reranking
``llm``        LLM providers (extractive / openai-compatible / ollama)
``registry``   model registry (local JSON / MLflow)
``store``      request log (SQLite / Postgres)
``features``   online feature store (local / Feast)
``evaluate``   offline metrics and quality gates
``monitor``    drift detection and alerting
``orchestration``  a small DAG engine with caching, retries and lineage

Every heavyweight dependency (MLflow, Feast, sentence-transformers, Evidently)
is optional: the platform boots and self-tests without them.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
