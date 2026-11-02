# TODO

The build plan for `rag-llmops-platform`, ordered by what actually unblocks the
next capability.

Last updated: 2026-10-20.

---

## Where the project stands

Working end to end, verified locally:

| | |
| --- | --- |
| Index | 10 documents → 40 chunks (heading-aware chunking) |
| Quality gate | **PASS** — faithfulness 1.000, recall 0.950, precision 0.533, key-term 0.828 |
| Closed loop | 10 steps: bootstrap → baseline → traffic shift → drift (5 features) → train-reranker → candidate → evaluate → gate → promote → incident report |
| Learning loop | `POST /feedback` → judgement set → trained ranker → gated promotion, end to end |
| Tests | 177 passing, ruff clean |
| Promotion | champion `v777de639eb` → `vf62da99190`, no restart (alias watcher) |

**Currently running on the defaults** — no `.env`, so:

- embedder: `hashing` (deterministic bag-of-words projection, not semantic)
- LLM: `extractive` (quotes the retrieved context, no generation)
- reranker: `term-overlap` (no cross-encoder)
- traffic: synthetic (golden-set questions + off-topic questions)

Everything below assumes those defaults until you swap them.

---

## The gap worth closing

> The closed loop can only promote a change **a human already thought of**
> (`retrieval_top_k=3` instead of `1`). Nothing in the platform *discovers* an
> improvement on its own.

That is the line between an operated platform and a learning platform. The
next build crosses it.

**Status: the machinery is in place; the evidence is not.** The feedback
endpoint, the judgement set, the trained ranker, the gate and the canary
statistics all exist and run end to end. What is missing is *real traffic*
against a real corpus — the part that cannot be built, only collected. See
"the prerequisite you cannot skip" below.

---

## P0 — Fix the provenance bug (do this first, ~20 min)

**Problem.** The registry records the *configured* backend names, not the ones
that actually ran. The current registry says:

```
"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
"rerank_model":    "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

…while `/health` reports `hashing-v1` and `term-overlap-v1`. A promotion whose
provenance is wrong is not reproducible, which defeats the point of versioning.

**Fix.** Resolve the backend at bundle-build time and store the *actual*
resolved identifiers.

- [x] `src/ragllmops/rag/bundle.py` — `BundleBuilder.build()` takes the
      resolved embedder/reranker identity (`embedder.name`, `embedder.dim`,
      `reranker.name`), not `settings.embedding_model` / `settings.rerank_model`.
- [x] `RAGConfigBundle` carries `embedder_id`, `embedding_dim`,
      `reranker_id` next to `embedder_requested` / `reranker_requested`.
- [x] `src/ragllmops/platform.py` — `Platform.build_bundle()` passes the real
      backends into the builder; `Platform.backend_summary()` exposes them.
- [x] `model_artifact_hash` (content hash of the model files) so a promoted
      version pins the exact weights, not just the model name. See
      `src/ragllmops/artifacts.py`.
- [x] Test: a bundle built under `embedder_backend=hashing` must not claim
      `all-MiniLM-L6-v2` — `tests/test_provenance.py`.

---

## P1 — The learning loop *(the recommended next build)*

**Production feedback → trained reranker → gated promotion.**

Roughly 3–4 days. Reuses ~90% of what already exists: bundle, registry, gate,
canary, DAG engine, request log.

`RequestRecord.feedback` already exists in the schema and is **never written**.
That is the hole in the middle of the system.

### M1 — Capture judgements *(~1 h)*

- [x] `POST /feedback` endpoint: `request_id` + verdict (`helpful` /
      `unhelpful` / `wrong_citation`) + optional `chunk_id` that was at fault.
- [x] Written to the existing `feedback` column in `src/ragllmops/store/`
      (plus a `feedback_chunk_id` column for the blamed chunk, with a
      migration for logs written by an earlier version).
- [x] `POST /ask` response includes the `request_id` (it already did).

### M2 — Build the judgement set *(~½ day)*

- [x] New module `src/ragllmops/feedback/dataset.py`: from the request log,
      emit `(query, chunk, label)` triples.
- [x] Labels come from explicit feedback first; fall back to an implicit
      signal (the answer refused) when no explicit feedback exists — the
      provenance of each label is recorded (`explicit:helpful`,
      `explicit:context`, `explicit:unhelpful`, `explicit:wrong_citation`,
      `implicit:refusal`) so the two are never conflated.
- [x] Held-out split (train/validation), by *request* so the same query never
      lands on both sides — the validation slice is what the gate compares
      against.

### M3 — Train a small ranker *(~½ day)*

- [x] Feature extraction over the features the platform already computes:
      BM25 score, dense score, term overlap, doc `avg_relevance` +
      `retrieval_count` (the feature store), candidate position, chunk length
      — `src/ragllmops/feedback/features.py`.
- [x] Model: logistic regression in numpy (no new dependency). ~100 labelled
      examples is enough; training takes milliseconds.
- [x] Persisted as a content-addressed artifact with the training-data hash,
      the feature schema version and the metrics — `src/ragllmops/feedback/model.py`.

### M4 — Register it in the bundle and gate it *(~1 day)*

- [x] `RAGConfigBundle` carries `ranker_artifact_id` + `ranker_metrics`.
- [x] New DAG asset `train_reranker` in
      `src/ragllmops/orchestration/pipeline.py`, between `evaluate` and `gate`.
- [x] `gate` compares the candidate ranker against the rule-based baseline
      (and the incumbent's ranker, if it has one) on the held-out slice —
      the existing `improved` comparison logic.
- [x] Promotion moves the alias; the previous champion stays reachable at
      `@challenger` for rollback.

### M5 — Canary the learned ranker *(~½ day)*

- [x] Traffic split between the rule-based reranker and the learned one via
      the existing `CanaryRouter` — the learned ranker is part of the bundle,
      so the champion and the challenger are served side by side.
- [x] Significance testing on the canary results (see P3):
      `src/ragllmops/monitor/experiment.py` — a two-proportion test on the
      refusal rate and a bootstrap on the retrieval score, with a floor of
      30 requests per side. Promotion stays manual.

### The prerequisite you cannot skip: real traffic

The current log holds only synthetic golden-set questions. To train anything
worth training you need real judgements.

- [ ] Point the platform at a corpus you actually care about — your own
      notes, a docs site, a chat export — and use it for a week.
- [ ] Target: 50–100 judgements. A linear ranker needs no more.
- [ ] If you have to simulate: build the machinery against simulated feedback
      so the plumbing is real, and **say so plainly in the README**. Do not
      let simulated feedback read as production evidence.

---

## P2 — Runners-up, in the order I'd do them

| # | Build | Do it when | Effort |
| --- | --- | --- | --- |
| 1 | **Incremental ingestion + corpus sensor** | As soon as you have a real corpus. `ingest` rebuilds and re-embeds everything today; you want content hashing so only changed documents are re-embedded, plus a sensor that fires when the corpus changes (not only when traffic does). | small |
| 2 | **Statistically sound canary** | **Done** — the comparison lives in `monitor/experiment.py` and is exposed by `cli canary` and `GET /admin/canary`. What is left is *volume*: the test needs 30+ requests per side, which needs real traffic. | small |
| 3 | **Real LLM + activate the LLM judge** | **Half done.** The judge now runs (`cli evaluate --judge`), its verdict is recorded per case, aggregated as `judge_grounded_rate` and gated on. What is left is a real model: the extractive provider cannot grade, so the metric is empty. With `RAG_LLM_PROVIDER=ollama` the golden set becomes a *model* comparison. | ~1 day |
| 4 | **Multi-tenant router, second corpus** | Later. Big surface area, mostly plumbing — but it is the step that makes the platform genuinely multi-product. | big |

---

## P3 — Detail worth designing for

**Canary significance testing.** **Done** — `monitor/experiment.py` compares
the two sides with a two-proportion test on the refusal rate and a bootstrap
on the retrieval score, and refuses to conclude anything below 30 requests
per side. Auto-promotion is deliberately *not* implemented: the report says
whether the challenger is promotable, and a human moves the alias. A
two-proportion test on the refusal rate is a weak instrument, and auto-
promoting on it would be worse than not promoting at all.

**Held-out discipline.** The golden set is used to *tune* the retriever, so
its metrics are optimistic. The learned ranker in M4 must be judged on the
held-out feedback slice, not on the golden set, or the gate is grading itself.

---

## Things not to do yet

- **Don't swap the DAG engine for Dagster/Prefect.** The current engine is
  ~200 readable lines and you haven't hit the problems it solves (dynamic
  partitioning, backfills, remote execution, a UI). Swapping now trades
  understanding for tooling — the opposite of what makes a platform project
  impressive. The mapping table is in `ARCHITECTURE.md` when you need it.
- **Don't add a vector database you don't need.** Qdrant via compose is
  enough; a dedicated cluster is an ops project, not an MLOps one.
- **Don't re-tune the defaults before the embedder is real.** The shipped
  `retrieval_top_k=2` and the dense/sparse weights were swept against the
  golden set using the *hashing* embedder. Install a real embedding model
  first, then re-run the sweep — the curve will move.

---

## Suggested execution order

1. [x] P0 provenance fix — makes the registry trustworthy.
2. [ ] P2.3 real LLM — the judge is wired into the gate, but the extractive
       provider cannot grade it; a real model makes the golden set meaningful.
3. [x] P1 M1 + M2 — feedback capture and judgement set.
4. [ ] **Start real traffic now**, in parallel — it takes a week of wall-clock
       and nothing else on this list is blocked by it.
5. [x] P1 M3 + M4 — train, register, gate.
6. [x] P1 M5 + P2.2 — canary with significance testing.
7. [ ] P2.1 incremental ingestion — before the corpus gets big.
8. [ ] Re-run the retrieval sweep with the real embedder; update the defaults
       and the README table.
