# FIXES.md

What was broken, why, and what changed.

The trigger was the smoke test in `docker-compose.yml`:

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"query":"What is the default object retention period?"}'
```

which returned:

```json
{"detail": "no bundle registered"}
```

HTTP 503.

**Status: fixed.** The command returns HTTP 200 with a grounded, cited
answer. Verified from a wiped stack (`docker compose down -v`).

---

## 1. The failure chain

The 503 was the last symptom of five independent faults stacked on top of
each other. Fixing any one of them alone would still have produced a
broken stack.

```
POST /ask
  └─ state.router.select(request_id)  -> None
       └─ champion_bundle()           -> None
            └─ registry.get_by_alias("champion")
                 └─ registry has 0 versions, 0 aliases
                      ├─ cause A: the API image had no MLflow client, so the
                      │           registry silently fell back to an empty file
                      ├─ cause B: the MLflow server could not reach Postgres
                      │           (wrong port), so it never started at all
                      └─ cause C: the MLflow image had no postgres driver
```

Plus two more that only bite *after* the endpoint answers:

| # | Symptom | Cause |
|---|---------|-------|
| D | `/ask` returns an answer with no citations | the API volume had no `index/catalog.json`; BM25 was empty (`chunks_sparse: 0`) |
| E | a promotion from another process never reaches the traffic | the alias watcher detected the change but never rebuilt the canary router — and, worse, swallowed the *first* promotion entirely |

---

## 2. Root causes and fixes

### A. The API image had no MLflow client — silent fallback to an empty registry

`docker-compose.yml` configures the real backends
(`RAG_REGISTRY_BACKEND=mlflow`, `RAG_LOG_STORE_BACKEND=postgres`), but the
`Dockerfile` installs only `requirements.txt` unless `INSTALL_ML=true` is
passed, and the compose file passed no build arg at all.

`mlflow` and `psycopg` were therefore missing, and both fallbacks are
*silent*:

```python
# src/ragllmops/registry/__init__.py
except ImportError as exc:
    logger.warning("MLflow registry unavailable (%s); using the local registry", exc)
    return LocalBundleRegistry(s.registry_file)      # -> data/registry/registry.json
```

The container's `/app/data` is the `api-state` volume, which had no
registry file in it, so the fallback registry was empty: 0 versions, 0
aliases, no champion. The container logs said so, but only if you read
past the first two lines:

```
WARNING ragllmops.registry  MLflow registry unavailable (mlflow is not
installed. Run: pip install -r requirements-ml.txt ...); using the local registry
WARNING ragllmops.store     psycopg unavailable (psycopg is not installed.
...); using the SQLite request log
```

**Fix.** A third install mode that adds only what the *serving* stack
needs — the two client libraries, no model weights (no torch):

- `requirements-serving-ml.txt` (new): `mlflow>=2.16,<2.17`, `psycopg[binary]>=3.1`
- `Dockerfile`: new `ARG INSTALL_SERVING_BACKENDS=false` and the branch that
  installs `requirements.txt` + `requirements-serving-ml.txt`
- `docker-compose.yml`: the `api` service builds with
  `INSTALL_SERVING_BACKENDS: "true"`

The embedder (`hashing`) and the reranker keep their dependency-free
fallbacks, so the image stays ~950 MB instead of pulling a multi-GB torch.

### B. Postgres was addressed on the wrong port — MLflow never started

The `postgres` image listens on **5432** inside the container. Compose
published and addressed it as **5433**:

```yaml
ports:
  - "5433:5433"                                        # nothing listens on 5433
environment:
  RAG_POSTGRES_DSN: postgresql://rag:rag@postgres:5433/rag
command: >
  mlflow server ... --backend-store-uri postgresql://rag:rag@postgres:5433/rag
```

Inside the compose network, `postgres:5433` is *connection refused*. The MLflow
server therefore crashed on boot and the registry was unreachable:

```
(psycopg2.OperationalError) connection to server at "postgres" (192.168.112.3),
port 5433 failed: Connection refused
```

The host-side `localhost:5433` still *appeared* to open, because Docker's
userland proxy accepts the TCP connection and then fails to forward it — a
misleading signal when debugging from the host.

**Fix.** Keep 5433 as the *published* port (the host already runs a
Postgres on 5432) and map it to the container's real port:

```yaml
ports:
  - "5433:5432"
```

Both DSNs now use `postgres:5432` — the port that is correct inside the
compose network.

### C. The MLflow image ships no Postgres driver

With the port fixed, the server failed one step later:

```
ModuleNotFoundError: No module named 'psycopg2'
```

`ghcr.io/mlflow/mlflow` does not bundle a Postgres driver, so it cannot use
a Postgres backend store out of the box.

**Fix.** `deploy/mlflow.Dockerfile` (new):

```dockerfile
FROM ghcr.io/mlflow/mlflow:v2.16.2
RUN pip install --no-cache-dir psycopg2-binary
```

The compose `mlflow` service now builds from it. The server is pinned to
2.16.2 (the version the compose file already referenced in a comment) and
the client in `requirements-serving-ml.txt` is pinned to the same minor
version, so the two cannot drift apart.

### D. Artifacts have to be served by the server, not written by the client

A remote MLflow client cannot write to the server's filesystem. With
`--default-artifact-root /mlflow/artifacts` the server hands the client a
*local path* as the artifact URI, and the client tries to write the model
to its own disk where the server never sees it.

**Fix.** The server proxies the artifacts and the client leaves the
artifact location alone:

```yaml
command: >
  mlflow server ...
  --serve-artifacts
  --artifacts-destination /mlflow/artifacts
environment:
  RAG_MLFLOW_ARTIFACT_ROOT: ""     # empty -> the server decides
```

### E. The API volume had no index

The `api-state` volume had no `index/catalog.json`, so `load_index()` found
nothing, BM25 was empty (`chunks_sparse: 0`), and any answer would have
had no citations to ground it.

**Fix.** The serving layer now builds the index when it is missing
(`_ensure_index` in `src/ragllmops/serve/app.py`), mirroring the
self-healing that `Platform.load_index()` already does for a lost vector
store.

### F. A registry without a champion cannot answer anything

Even with a healthy MLflow, a fresh stack has no bundle, so `/ask` returns
503 again.

**Fix.** `_bootstrap_registry` in `src/ragllmops/serve/app.py` registers the
default bundle and promotes it to `@champion` — but **only** when no
champion exists. An existing champion is never touched: this is a
bootstrap, not a takeover.

---

## 3. Latent code bugs (the ones the unit tests could not catch)

The test suite uses fakes for the registry, so every one of these passed
CI while the real stack was broken.

### 3.1 `mlflow.set_experiment_artifact_location` does not exist

```python
if artifact_root:
    mlflow.set_experiment_artifact_location(artifact_root)   # AttributeError
```

The function is absent from MLflow 2.16 *and* 2.22 *and* 3.x (it was
removed years ago). With the default `RAG_MLFLOW_ARTIFACT_ROOT=./mlruns`
this raised `AttributeError` inside `Platform.__init__` — and because the
fallback in `build_registry` only catches `ImportError`, it was **not**
caught: the API would have died at startup with the default settings.

**Fix** (`src/ragllmops/registry/mlflow_registry.py`): the legacy setter is
used when it exists; otherwise the bundles are logged in a dedicated
experiment (`<model-name>-bundles`) created with the requested
`artifact_location`. `create_experiment` returns an id string while
`get_experiment_by_name` returns an entity, so both are normalised.

### 3.2 `list_versions()` used an order-by the server rejects

```python
self._client.search_model_versions(f"name='{name}'", order_by=["version DESC"])
# INVALID_PARAMETER_VALUE: Invalid order_by clause 'version DESC'.
```

The exception was swallowed, so the registry always reported **0 versions** —
which is why `/admin/registry` was empty and `latest()` returned `None`.

**Fix**: drop the `order_by` and sort the model versions numerically in
Python (they are numbers, and the search API does not sort on them).

### 3.3 `log_model()` was given a path instead of a name

```python
mlflow.pyfunc.log_model(artifact_path=Path(tmp) / "model", ...)
```

Two errors in one line:

1. a `PosixPath` reaches the YAML dump of the model metadata →
   `yaml.representer.RepresenterError: ('cannot represent an object', PosixPath(...))`
2. `artifact_path` is the *name* the model is stored under in the run, not a
   local directory → `MlflowException: Invalid artifact path: '/tmp/.../model'`

**Fix**: `artifact_path="model"`. The model files are written by MLflow
itself; only the `artifacts={"bundle.json": ...}` entry is a local file.

### 3.4 Aliases could not be pointed at a bundle

The platform addresses bundles by their content fingerprint (`v<hash>`);
MLflow addresses them by a model version number (`1`, `2`, …). The closed
loop promotes by `champion.version` (the fingerprint), which the MLflow
backend rejected.

**Fix**: `set_alias()` and `get()` now accept both and resolve a bundle
version onto the model version that holds it. The rest of the platform
does not have to know which registry is in use.

### 3.5 The alias watcher swallowed the first promotion

```python
if not self._last_aliases:
    # First poll: record the current state without reporting a change.
    self._last_aliases = dict(aliases)
    return False
```

"First poll" was detected by the alias mapping being *empty*. A registry
that legitimately has no aliases yet therefore re-took the first-poll branch
on **every** poll and returned `False` forever — the first promotion was
silently swallowed, and the champion never appeared.

**Fix**: an explicit `_primed` flag.

### 3.6 A promotion never reached the traffic

Even when the watcher did detect a change, it only recorded it. The canary
router was rebuilt in exactly two places: application startup and
`POST /admin/promote`. A promotion made by the CLI, by another replica, or
by the closed loop was logged and then ignored — the README's "the serving
layer picks that up through the alias watcher: **no restart**" did not
hold.

**Fix**: `AliasWatcher` takes an `on_change` callback, and the app wires it
to `state.refresh_router()`. Verified: a promotion from another process
swaps the traffic within `RAG_WATCHER_INTERVAL_SECONDS` (15s), no restart.

---

## 4. Files changed

**Infrastructure**

| File | Change |
|------|--------|
| `docker-compose.yml` | `postgres` published as `5433:5432`; both DSNs → `postgres:5432`; `api` builds with `INSTALL_SERVING_BACKENDS=true`; `RAG_MLFLOW_ARTIFACT_ROOT: ""`; `mlflow` built from `deploy/mlflow.Dockerfile` with `--serve-artifacts` |
| `Dockerfile` | new `INSTALL_SERVING_BACKENDS` build arg |
| `requirements-serving-ml.txt` | **new** — mlflow + psycopg for the serving stack, no model weights |
| `deploy/mlflow.Dockerfile` | **new** — MLflow 2.16.2 + `psycopg2-binary` |
| `.env.example` | documents when to leave the artifact root empty |

**Code**

| File | Change |
|------|--------|
| `src/ragllmops/registry/mlflow_registry.py` | version-tolerant artifact root (§3.1), version listing (§3.2), `log_model` name (§3.3), bundle-version aliases (§3.4) |
| `src/ragllmops/serve/watcher.py` | `_primed` flag instead of the empty-dict check (§3.5); `on_change` callback (§3.6) |
| `src/ragllmops/serve/app.py` | wires `on_change` → `refresh_router`; `_ensure_index` (E); `_bootstrap_registry` (F) |

**Tests / docs**

| File | Change |
|------|--------|
| `tests/test_api.py` | 3 regression tests: first alias of an empty registry, `on_change` notification, fresh-stack bootstrap |
| `README.md` | the full-stack section now documents the build arg, the bootstrap and the promotion-without-restart flow |

---

## 5. Verification

```bash
docker compose down -v          # wipe every volume
docker compose up -d --build    # fresh stack
```

| Check | Result |
|-------|--------|
| `POST /ask` | HTTP 200, answer cites `storage::1` (0.85) and `data_ingestion::0` (0.70) |
| `GET /health` | `chunks: 40`, `chunks_sparse: 40`, `registry.backend: mlflow`, `log_store.backend: postgres`, `canary.champion: vc7101f67df` |
| `POST /admin/evaluate` | gate **passed** — faithfulness 1.00, context recall 0.883, key-term coverage 0.806 over the 30-case golden set |
| `GET /admin/drift` | PSI report over the Postgres request log |
| `GET /metrics` | Prometheus exposition, `ragllmops_requests_total` increments |
| external promotion | `registry promote champion --target N` → traffic swaps within 15s, no restart |
| `pytest tests -q` | 115 passed (112 before, 3 new) |
| `ruff check src tests` | clean |

---

## 6. Follow-ups (not fixed here)

1. **Qdrant client/server mismatch.** The client resolves to 1.19.1 while
   the server is pinned to 1.12.4:
   `UserWarning: Qdrant client version 1.19.1 is incompatible with server
   version 1.12.4`. Harmless today, but pin `qdrant-client` to match the
   server, or bump the server image.
2. **Silent backend fallbacks.** `build_registry` and `build_log_store`
   downgrade to the local/SQLite backends on an `ImportError` and only log a
   warning. That is exactly what hid cause A. Consider making the fallback
   opt-in, or at least failing loudly when the deployment explicitly asks
   for a backend it cannot load.
3. **Qdrant client timeout.** The default 5s REST timeout was hit while the
   disk was saturated by a concurrent image build. Consider a larger
   timeout for bulk upserts.
4. **`RAG_MLFLOW_ARTIFACT_ROOT` semantics.** It now means "the artifact
   location of the experiment that holds the bundles". A local path only
   works when the client and the server share a filesystem; a remote server
   needs it empty (or a URI it understands).
