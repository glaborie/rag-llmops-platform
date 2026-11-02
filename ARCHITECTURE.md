# Architecture

How the pieces fit together, why they are shaped this way, and where to
extend them.

---

## 1. The registrable unit: a bundle

Most MLOps platforms version *models*. In a RAG system the model is only one
of several things that determine the answer. This platform versions the whole
answer configuration:

```python
@dataclass(frozen=True)
class RAGConfigBundle:
    version: str                 # content-addressed, e.g. "vc7101f67df"
    prompt_id: str               # "v1" | "v2"
    prompt_template: str
    embedder_id: str             # what RAN, e.g. "hashing-v1"
    embedding_dim: int
    embedder_requested: str      # what was ASKED for
    reranker_id: str             # what RAN, e.g. "term-overlap-v1"
    reranker_requested: str      # what was ASKED for
    model_artifact_hash: str     # content hash of the model files
    ranker_artifact_id: str      # the learned ranker, when there is one
    ranker_metrics: dict[str, float]
    llm_provider: str
    llm_model: str
    temperature: float
    max_tokens: int
    retrieval_top_k: int
    retrieval_candidates: int
    rerank_enabled: bool
    max_context_chars: int
    metrics: dict[str, float]    # the evaluation that justified promoting it
```

`version` is derived from a hash of the content fields, so two identical
configurations share a version and a changed prompt produces a new one. That
makes the registry idempotent and the promotion history readable.

**Why this matters.** "Promote the new model" is ambiguous in a RAG system:
was the prompt changed? the chunking? the top-k? the model? By versioning the
bundle, the question disappears — a version *is* the answer configuration.

**Why the backends are recorded twice.** A bundle has to say what *ran*, not
what was configured. `RAG_EMBEDDER_BACKEND=sentence-transformer` on a machine
without `sentence-transformers` installed silently falls back to the hashing
embedder; a registry that recorded the configured name would claim the bundle
was evaluated with `all-MiniLM-L6-v2` when it was not, and the promotion could
not be reproduced. The requested names are kept next to the resolved ones so
the fallback is visible rather than hidden.

The bundle is stored in the registry (a JSON file by default, an MLflow model
version when `RAG_REGISTRY_BACKEND=mlflow`), and aliases (`champion`,
`challenger`) point at versions.

---

## 2. Retrieval

```
query
  │
  ├─► dense search   (Qdrant / in-memory, cosine over L2-normalised vectors) ─┐
  │                                                                            ├─► RRF fusion
  └─► BM25 search    (exact terms, Okapi BM25, implemented in-tree) ──────────┘        │
                                                                                      ▼
                                                                              reranker (cross-encoder
                                                                              or term-overlap)
                                                                                      │
                                                                                      ▼
                                                                              top-k chunks
```

**Why hybrid.** Dense retrieval fails on identifiers: `max.poll.interval`,
`600 requests`, `nw-standard`. BM25 fails on paraphrase. A documentation
question usually contains both ("what cooldown does the autoscaler use when
scaling my instance pool"), so you need both.

**Why RRF and not score blending.** Dense cosine scores and BM25 scores are
not on comparable scales; normalising them is an arbitrary choice. RRF only
needs the *order* of each list, which is stable across scoring changes.

**Why the weights are configurable.** The dense/sparse balance depends on the
embedder. With a real semantic embedder the dense side deserves more weight;
with the hashing embedder the sparse side clearly does. Both are settings,
and both were chosen by sweeping the golden set.

**Chunking** is heading-aware: each `##` section of a document becomes at
least one chunk, and oversized sections are split into overlapping windows on
paragraph boundaries. A chunk that ends mid-sentence loses the context that
makes it retrievable.

---

## 3. The chain

```python
class RagChain:
    def answer(self, query, top_k=None, user_id=None) -> ChainResult:
        retrieved = self.retrieval.retrieve(query, top_k=...)
        retrieved = self._apply_feature_boost(retrieved)   # online features
        context, snippets = self._build_context(retrieved)  # numbered [S1] [S2]
        system, user = get_prompt(self.bundle.prompt_id).render(query, context)
        result = self.llm.complete(system, user, temperature=..., max_tokens=...,
                                   context=snippets)
        return ChainResult(response=..., retrieved=..., context=..., snippets=...)
```

The chain is stateless and takes everything variable from the bundle. That is
what lets one process serve a champion and a challenger simultaneously, and
what makes the canary router a three-line class.

**Feature boost.** The online feature store provides per-document statistics
(`avg_relevance`, `retrieval_count`, `refusal_rate`) computed offline from the
request log. At serving time they add a small bonus to equally-ranked
candidates. The offline/online parity check (`ragllmops.cli features`) proves
the value served is the value computed.

---

## 4. Evaluation and the gate

The golden set (`data/eval/golden_set.jsonl`) is 30 labelled questions:

```json
{"id": "q013", "question": "What CPU target and cooldown timings does the autoscaler use...",
 "expected_doc_ids": ["compute"], "key_terms": ["65%", "120 seconds", "300 seconds"],
 "category": "compute", "difficulty": "medium"}
```

Metrics (all deterministic, so CI behaves the same on every machine):

| Metric | Definition | Why |
| --- | --- | --- |
| `context_precision` | relevant retrieved documents ÷ retrieved documents | are we wasting context? |
| `context_recall` | expected documents retrieved ÷ expected documents | are we missing the document? |
| `faithfulness` | answer sentences supported by the context ÷ answer sentences | is the answer grounded? |
| `key_term_coverage` | required key terms present in the answer ÷ required key terms | is the answer *correct*? |
| `answer_relevancy` | question terms present in the answer ÷ question terms | is the answer on topic? |
| `citation_hit_rate` | citations pointing at an expected document ÷ citations | do the citations point at the source? |

The gate (`GateThresholds`) checks aggregates, not individual cases:

```python
failures = check_gate(aggregates, thresholds)
```

`key_term_coverage` is the metric that catches "fluent but wrong" — a model
can be perfectly faithful to the wrong document.

**The gate is a build gate.** `.github/workflows/ci.yml` runs
`ragllmops.cli evaluate` and fails the build when the golden set regresses.
That is the RAG equivalent of a regression test on model quality.

**Tuning uses the same data.** Sweeping `retrieval_top_k` and the fusion
weights against the golden set is how the defaults were chosen (see the
README table). The gate then locks the tuned configuration in.

---

## 5. Drift

The request log is the platform's production data. Drift is computed by
splitting it into a reference window and a current window and comparing them.

**Numeric features** (`latency_ms`, `top_score`, `chunk_count`,
`prompt_tokens`, `completion_tokens`, `answer_chars`) use the Population
Stability Index:

```
PSI = Σ (actual% − expected%) × ln(actual% / expected%)
```

PSI < 0.1 is stable, 0.1–0.25 is a moderate shift, > 0.25 is significant.
The platform's threshold is 0.2.

Bins come from the reference quantiles. When the reference barely varies (a
constant feature such as `chunk_count`, which is common in production), the
quantile edges collapse into one bin and would hide a large shift — so the
implementation falls back to uniform bins across the combined range.

**Categorical features** (`llm_model`, `llm_provider`, `prompt_id`) use total
variation distance between the two distributions.

**What drift means for a RAG service.** A drop in `top_score` means the
retriever is confidently returning worse material — usually because the
questions changed, or because the corpus stopped covering what people ask
about. A rise in `latency_ms` and `answer_chars` is the signature of a
different (or larger) context being assembled.

---

## 6. Orchestration

The DAG engine (`src/ragllmops/orchestration/`) provides:

- **dependency order** — assets run in topological order;
- **idempotence** — a cache key derived from the asset's code fingerprint, its
  dependencies' fingerprints and its parameters; an unchanged asset is not
  re-run;
- **retries** — exponential backoff;
- **containment** — a failed dependency skips its dependents rather than
  running them on stale inputs;
- **lineage** — every run and every dependency edge is recorded to JSONL.

The platform's own pipeline:

```
ingest ──► materialize_features ──► evaluate ──► train_reranker ──► gate ──► promote
monitor ──────────────────────────────────────────────────────────────► alert
```

`gate` compares the candidate with the incumbent's recorded metrics and
`promote` moves the `@champion` alias only when the candidate both clears the
quality floor and beats the incumbent. The previous champion is kept reachable
at `@challenger`, so a rollback is one alias move.

---

## 6b. The learning loop

`src/ragllmops/feedback/` is the middle of the system: it turns production
traffic into a trained model.

```
POST /feedback ──► request log (the `feedback` column)
                        │
                        ▼
              feedback/dataset.py       (query, chunk, label) triples
              label + provenance        explicit:helpful / explicit:context /
                                        explicit:unhelpful / explicit:wrong_citation /
                                        implicit:refusal
                        │
                        ▼
              feedback/features.py      BM25, dense, term overlap, doc stats,
                                        position, chunk length
                        │
                        ▼
              feedback/model.py         logistic regression (numpy only)
                                        → content-addressed artifact
                        │
                        ▼
              bundle.ranker_artifact_id → the gate compares it with the
                                        rule-based baseline on the held-out slice
```

Two rules make this honest:

- **The provenance of a label is part of the label.** An explicit judgement
  from a user, an inferred one (a refusal) and a weak negative sampled from
  the retriever's own order are never mixed silently; every example records
  where it came from.
- **The split is by request.** Two judgements of the same query must not end
  up on opposite sides of the train/validation split, or the validation score
  is measuring the training data.

**Why the held-out slice and not the golden set.** The golden set was used to
tune the retriever, so its metrics are optimistic by construction. A learned
ranker judged on the golden set would be graded by the data it was built
against. The gate therefore compares the learned ranker with the rule-based
one on the feedback the platform has never trained on.

**The canary.** `src/ragllmops/monitor/experiment.py` compares the two sides
of a traffic split: a two-proportion test on the refusal rate and a bootstrap
on the retrieval score, both refusing to conclude anything below 30 requests
per side. The report says whether the challenger is *promotable*; promoting
stays a manual decision.

### Mapping onto a real orchestrator

| This project | Dagster | Prefect | Airflow |
| --- | --- | --- | --- |
| `Asset` | `@asset` | `@task` | PythonOperator |
| `deps` | `AssetDep` | implicit by argument | `>>` |
| cache key | `AutoMaterialize` policy | `cache_key_fn` | not native |
| retries | `RetryPolicy` | `retries` | `retries` |
| lineage | automatic | automatic | automatic |
| schedule | `@schedule` | `@flow(schedule=)` | `schedule_interval` |
| sensor | `@sensor` | `flow` polling | `Sensor` |

The concepts transfer one-to-one; the reason to write it out is to see
exactly what your pipeline guarantees.

---

## 7. Serving

```
POST /ask
  │
  ├─► CanaryRouter.select(request_id)      deterministic hash split
  │      champion ─┐
  │      challenger┘ traffic_pct % of requests
  │
  ├─► RagChain.answer(...)                  per bundle
  │      └─► the bundle's learned ranker, when it has one
  ├─► request log                            (Postgres / SQLite)
  └─► Prometheus metrics

POST /feedback
  │
  └─► request log (the `feedback` column)   the judgement that trains the ranker
```

**Alias watcher.** A background task polls the registry; when an alias moves
it rebuilds the router. Promoting a model does not require restarting the
API — the difference between a deployment and a promotion.

**Metrics exposed** (`/metrics`): request count, latency histogram,
retrieval latency, token counters, retrieved-chunk histogram, top-score
histogram, refusal counter, drift PSI gauge, canary traffic gauge and alias
version gauge.

---

## 8. Extension points

| To add | Do this |
| --- | --- |
| A new LLM provider | implement `LLMProvider` in `src/ragllmops/llm/`, add a branch in `build_llm` |
| A new vector store | implement the `search/upsert/count/reset` contract, add a branch in `build_vector_store` |
| A new prompt | add a `PromptTemplate` to `PROMPT_REGISTRY` in `src/ragllmops/rag/prompts.py` |
| A new metric | add it to `src/ragllmops/evaluate/metrics.py` and to `aggregate()` |
| A new pipeline step | add an `Asset` to `build_pipeline_dag()` |
| A new alert channel | implement `Notifier`, add it to `build_notifier` |
| A different corpus | point `RAG_CORPUS_DIR` at it and regenerate the golden set |
| A new feature for the ranker | add it to `FEATURE_NAMES` in `src/ragllmops/feedback/features.py` and bump `FEATURE_SCHEMA_VERSION` (old artifacts will then refuse to load) |
| A new kind of judgement | add a verdict to `src/ragllmops/feedback/labels.py` and a rule to `derive_labels()` |

Every backend follows the same pattern: a Protocol, a dependency-free
implementation, an optional heavyweight implementation, and a factory that
degrades gracefully when the heavyweight dependency is missing.
