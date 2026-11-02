# AGENTS.md

Working agreement for coding agents (and humans) in `rag-llmops-platform`.

A local, end-to-end LLMOps platform for a RAG service: versioning, retrieval,
evaluation gates, drift monitoring and a closed retraining loop. Python
3.10+, FastAPI, ~7k lines under `src/ragllmops/`, ~1.3k lines of tests.

Read `README.md` for what it does, `ARCHITECTURE.md` for how it fits
together, `FIXES.md` for what was recently broken and why.

---

## 1. Layout

```
src/ragllmops/
  platform.py        the façade: build an index, answer, evaluate, promote
  config.py          Settings (pydantic-settings, env prefix RAG_)
  models.py          RAGConfigBundle, DocumentChunk, RequestRecord
  artifacts.py       content hashing for model files (the provenance anchor)
  cli.py             `python -m ragllmops.cli <cmd>`
  ingest/            corpus -> chunks -> embeddings -> vector store
  retrieval/         dense + BM25 + RRF + rerank, chunk catalogue
  rag/               bundle builder, prompts, the answering chain
  llm/               extractive (default), openai, ollama
  embed/             hashing (default), sentence-transformer
  registry/          local (default) and mlflow backends
  store/             request log: sqlite (default) and postgres
  features/          offline/online feature store, local and feast
  feedback/          verdicts, judgement set, features, trained ranker
  evaluate/          golden set, metrics, LLM judge
  monitor/           PSI drift, canary statistics, alert notifiers
  orchestration/     DAG engine, scheduler, closed loop
  serve/             FastAPI app, canary router, alias watcher, metrics
tests/               177 tests, no Docker/network required
data/corpus/         10 markdown docs (fictional cloud provider)
data/eval/           golden_set.jsonl — 30 labelled questions
data/models/ranker/  trained ranker artifacts (content-addressed)
```

The dependency direction is one-way: `serve → platform → backends`. Nothing
in `retrieval/`, `ingest/` or `registry/` may import from `serve/`.

---

## 2. Environment

This checkout has a uv-managed `.venv` (Python 3.13) with the core
dependencies installed. It has **no `pip`**:

```bash
VIRTUAL_ENV=.venv uv pip install "pytest>=8" "pytest-cov>=5" "ruff>=0.4"
```

`make install` / `make install-ml` assume a system `python` with `pip`; they
are fine on a clean machine, not needed here.

---

## 3. Commands

```bash
# tests
.venv/bin/python -m pytest tests -q                       # 177 passed
.venv/bin/python -m pytest tests -q -m integration         # needs qdrant+postgres
make test PYTHON=.venv/bin/python                          # same thing

# lint (CI runs exactly this)
.venv/bin/python -m ruff check --no-cache src tests
make lint PYTHON=.venv/bin/python                          # same thing

# the CLI
.venv/bin/python -m ragllmops.cli doctor      # what is wired up, what is missing
.venv/bin/python -m ragllmops.cli ingest      # build the retrieval index
.venv/bin/python -m ragllmops.cli evaluate    # golden set + quality gate (exit 1 on failure)
.venv/bin/python -m ragllmops.cli serve       # API on :8000
.venv/bin/python -m ragllmops.cli loop        # the full closed loop
.venv/bin/python -m ragllmops.cli feedback    # the judgement set from the request log
.venv/bin/python -m ragllmops.cli train-ranker  # train a ranker on the feedback
.venv/bin/python -m ragllmops.cli canary      # champion vs challenger, with a p-value
.venv/bin/python -m ragllmops.cli registry list|show|promote|register
```

`--no-cache` on ruff is required: `.ruff_cache/` in this checkout is
read-only and ruff fails with `Permission denied` writing to it. The
`lint` and `fmt` Makefile targets pass `--no-cache` for that reason; when
you call ruff directly, pass it too.

**Do not run `ruff format` across the repo.** 17 of 79 files are not
format-clean under ruff 0.17, and CI only enforces `ruff check`. Formatting
them produces a huge spurious diff. Format only the files you touched, and
only if the surrounding file is already formatted.

### The quality gate

`ragllmops.cli evaluate` is the project's own quality gate: it runs the
30-case golden set and fails when faithfulness / context recall / key-term
coverage drop below the thresholds in `config.py`. A change that alters
retrieval or prompting must keep it green. `.github/workflows/ci.yml` runs
it as a build gate, so a regression blocks the merge.

---

## 4. Docker stack

```bash
docker compose up -d --build        # api :8000, mlflow :5000, qdrant :6333, postgres :5433, prometheus :9090
docker compose logs -f api
docker compose exec api python -m ragllmops.cli registry list
docker compose down -v               # also wipe the volumes
```

The API image is built with `INSTALL_SERVING_BACKENDS=true`
(`requirements-serving-ml.txt`: mlflow + psycopg, no model weights). The
MLflow service is built from `deploy/mlflow.Dockerfile` (adds the Postgres
driver the upstream image lacks). On a fresh volume the API builds its
index and registers a default `@champion` by itself.

The build is slow (large wheels, slow disk): budget ~10 minutes and do not
kill it mid-pip, or the layer cache is lost.

---

## 5. Conventions

- **Every backend is a Protocol + a dependency-free implementation + an
  optional heavyweight one + a factory that degrades gracefully.**
  `build_llm`, `build_embedder`, `build_vector_store`, `build_registry`,
  `build_log_store`, `build_reranker`, `build_feature_provider`. When you
  add a backend, follow the pattern and add a branch to the factory.
- **Never import an optional dependency at module scope.** The core must
  import and run with only `requirements.txt` installed. Optional imports
  go inside the function that needs them, and the `ImportError` is caught
  there.
- **No network at import time and no network in the unit tests.** The suite
  must pass with no Docker, no API key and no model download.
- `from __future__ import annotations` at the top of every module.
- Type hints on public functions; `str | None` rather than `Optional[str]`.
- Docstrings on every module and every public function — say *why*, not
  *what*.
- Logging via `ragllmops.logging_utils.get_logger("namespace")`; never
  `print` in library code (the CLI is the only place `print` belongs).
- Line length 100; ruff rules `E,F,I,UP,B`.

---

## 6. Invariants — do not break these

These are the things the recent bugs in `FIXES.md` violated. Breaking one
of them fails in production but still passes the unit tests, because the
tests use fakes.

1. **The registry must be able to answer "who is the champion?"** without
   the process restarting. A promotion made anywhere — the CLI, another
   replica, the closed loop — must reach the traffic. The alias watcher
   detects the change *and* rebuilds the canary router.
2. **A backend that was explicitly asked for must not silently degrade.**
   The graceful fallbacks are for *missing extras*, not for misconfigured
   services. If `RAG_REGISTRY_BACKEND=mlflow` cannot reach MLflow, that is
   an error, not a reason to serve from an empty local file.
3. **A bundle is addressed by its content fingerprint** (`v<hash>`), and the
   registry is responsible for translating that to whatever the backend
   uses. Do not let a backend-specific identifier leak into
   `platform.py` or `closed_loop.py`.
4. **The retrieval catalogue is the source of truth for chunk text.** If
   `catalog.json` is missing, BM25 is empty and citations lose their
   text — the API rebuilds the index on startup for that reason.
5. **The default `retrieval_top_k` is 2, and the fusion weights are
   0.35 dense / 0.65 sparse.** They were tuned against the golden set
   (`README.md` has the precision/recall sweep). Changing them changes the
   measured curve; re-run the sweep and update the table if you do.
6. **A registered bundle is immutable content.** The fingerprint is the
   identity; re-registering the same configuration is a no-op.
7. **A bundle records what ran, not what was configured.** `embedder_id`,
   `embedding_dim` and `reranker_id` come from the resolved backend objects;
   the configured names live next to them as `*_requested`. A bundle built
   under the hashing embedder must never claim `all-MiniLM-L6-v2`.
8. **A label's provenance is part of the label.** An explicit user judgement,
   an inferred one (a refusal) and a weak negative sampled from the retriever's
   own order are never mixed silently: every example in the judgement set
   records where it came from. The train/validation split is by *request*, so
   the same query never lands on both sides.
9. **The learned ranker is judged on the held-out feedback slice**, never on
   the golden set — the golden set was used to tune the retriever, so its
   metrics are optimistic by construction. A ranker that does not beat the
   rule-based baseline on the held-out slice does not go into a bundle.

---

## 7. Traps in this checkout

- **This is not a git repository.** There is no `git diff` to review and no
   `git checkout` to fall back on. Read a file before you edit it, and keep
   edits surgical.
- **These directories are read-only**: `data/index/`, `data/registry/`,
   `data/logs/`, `data/cache/`, `data/runs/`, `data/reports/`. Commands
   that write state — `ingest`, `evaluate` (writes a report), `reset`,
   `train-ranker` — fail with `Permission denied` on the host. Use the
   Docker stack, or point `RAG_DATA_DIR` at a writable path (and
   `RAG_REGISTRY_PATH` / `RAG_SQLITE_PATH` with it: those resolve against
   the project root, not against `RAG_DATA_DIR`).
- `data/corpus/`, `data/eval/` and `data/features/` are writable; treat the
   corpus and the golden set as read-only inputs anyway.
- `.ruff_cache/` and `.pytest_cache/` are read-only — pass `--no-cache`
   (ruff) or ignore the pytest warning.
- The host already runs a Postgres on **5432** (another project), which is
  why this stack publishes **5433**. Inside the compose network Postgres is
  always `postgres:5432`; only the published host port differs.
- `qdrant-client` resolves to 1.19.x while the server image is 1.12.4 and
  warns about the mismatch. Harmless, but do not "fix" it by bumping the
  server image without checking the client API.

---

## 8. Before you call it done

```bash
.venv/bin/python -m pytest tests -q                    # all green
.venv/bin/python -m ruff check --no-cache src tests    # clean
.venv/bin/python -m ragllmops.cli evaluate             # gate still passes
```

If the change touches serving, the registry or retrieval, also verify
against the running stack:

```bash
docker compose up -d --build
curl -sS -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"query":"What is the default object retention period?"}'
curl -sS localhost:8000/health
```

The answer must be grounded and carry citations; `health` must show a
champion, a non-empty index and the backends you intended to use.
