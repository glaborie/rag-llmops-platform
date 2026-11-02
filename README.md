# rag-llmops-platform

A local, end-to-end **LLMOps platform for a RAG service** — the same "last mile"
machinery you build for a tabular model (experiment tracking, model registry,
feature store, data validation, drift monitoring, CI/CD), applied to
retrieval-augmented generation.

It is the follow-up to
[How to Build an End-to-End ML Platform Locally](https://www.freecodecamp.org/news/build-end-to-end-ml-platform-locally-from-experiment-tracking-to-cicd/).
That tutorial builds the platform for fraud detection; this one builds it for
a RAG service, and — more importantly — **closes the loop** the tutorial
leaves manual.

Everything runs on your machine. No cloud account, no API key, no GPU.

---

## The idea

A RAG service has the same failure modes as any ML system, and the same
defences:

| Pillar | In the fraud-detection tutorial | Here, for RAG |
| --- | --- | --- |
| Experiment tracking | MLflow runs | MLflow runs **and** the retrievability sweep below |
| Model registry | `@champion` alias on a model | `@champion` alias on a **configuration bundle** (prompt + retrieval + LLM) |
| Feature store | Feast (merchant stats) | Feast (per-document relevance/popularity) |
| Data validation | Great Expectations on inputs | Structural validation of the corpus + golden-set gate |
| Drift monitoring | Evidently on features | PSI over the request log (latency, top score, tokens) |
| CI/CD | GitHub Actions | GitHub Actions **with a quality gate that blocks the merge** |
| Retraining loop | described in prose | **implemented, and demonstrable in one command** |

The unit that gets versioned, promoted and rolled back is not a model — it is
a **bundle**: the prompt template, the embedding model, the retrieval
parameters, the reranker and the LLM identity, together. Promoting a bundle
promotes the whole answer configuration, which is what actually ships.

---

## Quickstart

The core runs with **no services at all** — no Docker, no Qdrant, no
MLflow, no API key.

```bash
git clone <your-fork> rag-llmops-platform && cd rag-llmops-platform

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps -e .

python -m ragllmops.cli ingest      # build the index from the corpus
python -m ragllmops.cli evaluate    # run the golden set + quality gate
python -m ragllmops.cli serve       # http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"query":"What is the default object retention period?"}'
```

```json
{
  "answer": "Objects are retained for 90 days by default [S1]. ...",
  "citations": [{"index": 1, "chunk_id": "storage::1", "doc_id": "storage", "score": 0.83}],
  "bundle_version": "vc7101f67df",
  "prompt_id": "v2",
  "retrieved_count": 2,
  "route": "champion"
}
```

Nothing is downloaded and no model is fetched: the default embedder is a
deterministic hashing projection and the default "LLM" is an extractive
answerer that quotes the retrieved context. That is on purpose — it makes the
whole platform **reproducible and CI-testable offline**. Point it at a real
model and the plumbing is unchanged.

### With the full stack

```bash
docker compose up -d --build
```

brings up the API, Qdrant, Postgres, Redis, MLflow and Prometheus. The API
image then uses the real backends: Qdrant for vectors, Postgres for the
request log, MLflow for the registry. The image is built with
`INSTALL_SERVING_BACKENDS=true`, which adds the two clients the API needs
(MLflow, psycopg) without the model weights.

On a fresh volume the API has no index and no champion, so it builds the
index from the corpus and registers the default bundle as `@champion` on
startup. The smoke test works straight away:

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"query":"What is the default object retention period?"}'
```

A promotion made from another process — the CLI, a second replica — is
picked up by the alias watcher within `RAG_WATCHER_INTERVAL_SECONDS`, with
no restart:

```bash
docker compose exec api python -m ragllmops.cli registry register
docker compose exec api python -m ragllmops.cli registry promote champion --target 2
```

---

## The closed loop

This is the part the tutorial leaves in prose. One command:

```bash
python -m ragllmops.cli loop
```

```
closed loop (10 steps)
  09:38:18  ingest: built the retrieval index
  09:38:18  bootstrap: registered v777de639eb as champion (recall=0.867, faithfulness=1.000)
  09:38:19  baseline: served 90 requests
  09:38:19  traffic-shift: served 15 requests
  09:38:19  monitor: drift=detected features=['latency_ms', 'top_score', 'prompt_tokens',
                                             'completion_tokens', 'answer_chars']
  09:38:19  train-reranker: no ranker trained (only 4 labelled examples; at least 20 are needed)
  09:38:19  candidate: registered vf62da99190 as challenger
  09:38:20  evaluate: gate=PASS faithfulness=1.000 recall=0.950
  09:38:20  gate: promote=True
  09:38:20  promote: champion v777de639eb -> vf62da99190
```

What actually happened:

1. **Baseline.** 90 in-domain requests were served and logged. Their feature
   distributions became the reference window.
2. **Traffic shift.** 15 off-topic questions were served. The users' behaviour
   changed; the platform did not.
3. **Detection.** The PSI of `top_score`, `latency_ms`, `prompt_tokens`,
   `completion_tokens` and `answer_chars` all crossed the threshold. The
   retrieval scores collapsed — the system was confidently retrieving
   irrelevant chunks.
4. **Learning.** The loop tried to train a ranker on the feedback it had
   collected. With synthetic traffic there are four labelled examples and no
   model is trained — the loop says so instead of pretending.
5. **Candidate.** A wider retrieval configuration was registered as
   `@challenger`.
6. **Evaluation.** The candidate was scored against the 30-question golden
   set. Faithfulness 1.000, context recall 0.950 — it cleared the gate.
7. **Promotion.** The candidate was better than the incumbent, so the
   `@champion` alias moved. The serving layer picks that up through the alias
   watcher: **no restart**.

The run writes an incident report to `data/reports/incident-<timestamp>.md`
with the timeline, the drift table, the gate comparison and the next steps.
That report is the artefact worth putting in a portfolio.

---

## The learning loop: feedback → trained reranker → gated promotion

The closed loop above promotes a change a human already thought of. The
learning loop closes the gap: production feedback becomes a trained ranker,
and the ranker is gated and promoted like anything else.

### 1. Capture the judgement

`POST /ask` returns a `request_id`. Send it back with a verdict:

```bash
curl -sS -X POST localhost:8000/ask \
     -H 'Content-Type: application/json' \
     -d '{"query":"What is the default object retention period?"}' \
  | jq -r .request_id

curl -sS -X POST localhost:8000/feedback \
     -H 'Content-Type: application/json' \
     -d '{"request_id":"<id>","verdict":"helpful"}'

# blame a specific citation
curl -sS -X POST localhost:8000/feedback \
     -H 'Content-Type: application/json' \
     -d '{"request_id":"<id>","verdict":"wrong_citation","chunk_id":"storage::3"}'
```

The verdicts are `helpful`, `unhelpful` and `wrong_citation`. The judgement is
written back into the request log's `feedback` column — the log stays the
single source of truth about production traffic.

### 2. Build the judgement set

```bash
python -m ragllmops.cli feedback
```

```
== judgement set ==
examples           : 20
requests           : 6
positives/negatives: 6/14
explicit/sampled/inferred: 11/9/0
by source:
  explicit:context          0
  explicit:helpful          6
  explicit:unhelpful        4
  explicit:wrong_citation   1
  sampled:candidate         9
by split:
  train                     15
  validation                5
```

Every example carries the provenance of its label. `explicit:*` came from a
user; `implicit:refusal` was inferred from a refusal; `sampled:candidate` is a
weak negative inferred from the retriever's own order. They are never
conflated, and the split is by *request*, so the validation slice is not
contaminated by the training slice.

The labelling rules are conservative and documented in
`src/ragllmops/feedback/dataset.py`:

| verdict | label |
| --- | --- |
| `helpful` | the chunk(s) the answer **cited** are relevant (1); the rest of the served context is not (0) |
| `unhelpful` | none of the served chunks was relevant (0) |
| `wrong_citation` | the blamed chunk was not relevant (0) |
| *(no verdict, but the answer refused)* | none of the served chunks was relevant (0), labelled `implicit:refusal` |
| *(candidates the retriever proposed but did not serve)* | weak negatives (0), labelled `sampled:candidate` |

The weak negatives exist for a reason. Answer-level feedback says "this
answer was good", not "these three chunks were worse than the first one".
Without them every request would be all positives or all negatives, and no
*ranking* metric could be computed at all. They are the weakest signal in
the set and they are labelled as such — but they do carry a bias: they are
drawn from the chunks the retriever already ranked lower, so a model fitted
on them partly learns to reproduce the current order.

### 3. Train a ranker

```bash
python -m ragllmops.cli train-ranker
```

```
== trained ranker ==
artifact      : 477c36bdff8d3347dc23da3d90ec221741363b4626a2ed11d06c1c6640bf5d97
examples      : 11 (train 9, validation 2)
held-out pairwise accuracy : 1.000
rule-based baseline        : 0.000
```

The model is a logistic regression over the features the platform already
computes — BM25 score, dense score, term overlap, the document's
`avg_relevance` and `retrieval_count`, the candidate position and the chunk
length. It is numpy-only, trains in milliseconds, and the weights are
readable. The artifact is content-addressed and stores the hash of its
training data, the feature schema version and the metrics.

### 4. Gate and promote

The `train_reranker` asset sits between `evaluate` and `gate` in the DAG:

```
ingest → materialize_features → evaluate → train_reranker → gate → promote
```

The gate compares the learned ranker with the rule it replaces **on the
held-out feedback slice** — never on the golden set, which the retriever was
already tuned on. A ranker that does not beat the rule-based one is not put
into the bundle.

**The feedback in this repository is simulated.** The shipped corpus is
fiction, the traffic is synthetic, and the numbers above come from a
hand-written script, not from users. The machinery is real; the *evidence*
is not. See "Limitations, honestly" below.

---

## The golden set is also a tuning tool

`data/eval/golden_set.jsonl` holds 30 labelled questions with the documents
each answer must be grounded in and the key terms it must contain. It is used
two ways: as a **gate** (CI blocks a merge when quality regresses) and as a
**dataset you tune retrieval against**.

Sweeping the retrieval depth over the golden set gives the classic
precision/recall curve:

| `retrieval_top_k` | context precision | context recall | key-term coverage |
| --- | --- | --- | --- |
| 1 | 0.933 | 0.867 | 0.828 |
| **2** | **0.683** | **0.950** | **0.828** |
| 3 | 0.517 | 0.950 | 0.839 |
| 4 | 0.436 | 0.967 | 0.814 |
| 5 | 0.351 | 0.983 | 0.803 |
| 6 | 0.296 | 0.983 | 0.803 |

`k=1` is precise but misses multi-part questions; `k=4+` recalls more but
starts putting irrelevant documents in front of the model. **k=2 is the knee**,
and that is the shipped default. The fusion weights (dense 0.35 / sparse 0.65)
were tuned the same way.

This is the workflow the platform exists for: change the retriever, re-run
the golden set, keep the change only if the numbers improve.

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │            ORCHESTRATION                │
                        │  DAG engine: caching, retries, lineage  │
                        │  ingest → features → evaluate → train →  │
                        │            gate → promote  monitor → alert│
                        └──────────────────────────────────────────┘
                             │            │            │
   corpus/*.md ──► chunk ──► index        │            ▼
                              │            │     drift report
                              ▼            ▼            │
                     ┌──────────────┐  ┌──────────────────┴───┐
                     │   RETRIEVAL  │  │   FEATURE STORE     │
                     │ dense + BM25 │  │ per-doc relevance   │
                     │ → RRF fusion │  │ Feast / local JSON  │
                     │ → rerank     │  └──────────────────────┘
                     └──────┬───────┘
                            │ top-k chunks
                            ▼
   ┌────────────────────────────────────────────────────────────┐
   │  SERVING         bundle = prompt + retrieval + LLM        │
   │  FastAPI ──► canary router ──► chain ──► answer + cites   │
   │               ▲                                  │        │
   │               │ alias watcher                    ▼        │
   │  registry ◄───┴────────────────────────  request log       │
   │  (MLflow / local)                       (Postgres / SQLite)│
   └────────────────────────────────────────────────────────────┘
                            ▲                │
                            │                ▼
                     ┌──────────────┐  ┌─────────────────────────┐
                     │   FEEDBACK   │  │   LEARNED RANKER       │
                     │ POST /feedback│ │ logistic regression,   │
                     │ → request log │ │ artifact + held-out    │
                     │ → judgements  │ │ metrics → the gate     │
                     └──────────────┘  └─────────────────────────┘
```

- **Retrieval** is hybrid: dense vectors and BM25 propose candidates, they
  are fused with Reciprocal Rank Fusion, then a reranker re-scores them
  against the query. Dense catches paraphrase; BM25 catches exact
  identifiers (`max.poll.interval`, `600 requests`). Both matter, and the
  weights are tunable against the golden set.
- **The chain** is stateless. Everything that varies between deployments
  comes from the bundle, so the same chain serves a champion and a challenger
  at the same time.
- **The request log** is the platform's production data: drift monitoring
  reads it, the feature store is computed from it, and the learning loop
  trains on it.
- **The DAG engine** is small and readable on purpose: dependency order,
  content-addressed caching, retries with backoff, and lineage, in ~200
  lines you can read. The README maps every concept onto Dagster/Prefect.
- **The feedback loop** writes the user's verdict back into the request log,
  turns the log into a labelled judgement set (keeping the provenance of
  every label), fits a ranker on it and gates that ranker against the
  rule-based one on the held-out slice. The champion and the challenger can
  then serve the same traffic, and the canary report says which one wins.

---

## CLI

```
python -m ragllmops.cli doctor        # what is wired up, what is missing
python -m ragllmops.cli ingest        # build the retrieval index
python -m ragllmops.cli evaluate      # golden set + quality gate (exit 1 on failure)
python -m ragllmops.cli serve         # the API
python -m ragllmops.cli monitor       # drift report over the request log
python -m ragllmops.cli loop          # the closed loop
python -m ragllmops.cli pipeline run  # the platform DAG once
python -m ragllmops.cli pipeline describe
python -m ragllmops.cli registry list|show|promote|register
python -m ragllmops.cli features       # materialise features, check offline/online parity
python -m ragllmops.cli feedback       # the judgement set from the request log
python -m ragllmops.cli train-ranker  # train a ranker on the feedback
python -m ragllmops.cli canary         # champion vs challenger, with significance
python -m ragllmops.cli reset          # clear the local state
```

## HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/ask` | answer a question (canary-aware) |
| `POST` | `/feedback` | record a user's verdict on an answer |
| `GET` | `/health` | platform state |
| `GET` | `/metrics` | Prometheus exposition |
| `GET` | `/admin/registry` | versions and aliases |
| `POST` | `/admin/promote` | move an alias |
| `POST` | `/admin/ingest` | rebuild the index |
| `POST` | `/admin/evaluate` | run the golden set |
| `GET` | `/admin/drift` | drift report |
| `GET` | `/admin/canary` | champion vs challenger, with a significance test |

---

## Testing

```bash
python -m pytest tests -q      # 177 tests, no services required
```

The suite covers the chunker, the BM25 index, rank fusion, the reranker, the
registry, the metrics, the drift maths, the DAG engine (caching, retries,
skip-on-failure), the feature materialisation and parity, the API, the
feedback capture, the judgement set, the trained ranker and the canary
statistics, and the consistency of the shipped corpus and golden set.

CI runs four gates: lint, unit tests, the **quality gate** (the golden set
must pass or the build fails), the closed loop as a smoke test, and an
integration job against real Qdrant and Postgres.

---

## Using a real model

```bash
export RAG_LLM_PROVIDER=openai
export RAG_LLM_BASE_URL=https://api.openai.com/v1
export RAG_LLM_API_KEY=sk-...
export RAG_LLM_MODEL=gpt-4o-mini
export RAG_EMBEDDER_BACKEND=sentence-transformer
export RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Anything that speaks `/chat/completions` works — OpenAI, Ollama
(`http://localhost:11434/v1`), vLLM, LM Studio. For Ollama:

```bash
ollama serve
ollama pull llama3.2
export RAG_LLM_PROVIDER=ollama
export RAG_LLM_BASE_URL=http://localhost:11434/v1
export RAG_LLM_MODEL=llama3.2
```

After switching the embedder, **re-tune `retrieval_top_k` and the fusion
weights** against the golden set — the precision/recall curve above was
measured with the hashing embedder, and a real semantic embedder moves it.

### The LLM judge

`evaluate/judge.py` grades whether an answer is grounded in its context. It
is now part of the gate, not a side experiment:

```bash
python -m ragllmops.cli evaluate --judge
```

```
judge grounded  : 0.867 (26 cases judged)
gate            : PASS
```

The judge's verdicts are recorded per case (`judge_score`), aggregated as
`judge_grounded_rate`, and gated on by `RAG_EVAL_GATE_MIN_JUDGE_GROUNDED_RATE`
(default 0.5). A judge that could not be reached — the extractive provider
cannot grade — produces **no** metric and does not fail the gate; the CLI
says so explicitly instead of pretending.

With a real model the golden set stops being a fixed regression suite and
becomes a *model* comparison: the same gate then decides whether the new
model is allowed to serve.

---

## Limitations, honestly

- **The default embedder is not semantic.** It is a deterministic hashing
  projection. It works, it is fast, and it is good enough for the golden set
  — but a real embedding model will change the retrieval curve, and the
  defaults should be re-tuned.
- **The default "LLM" is extractive.** It selects sentences from the context
  rather than generating prose. It cannot paraphrase and it cannot reason
  across chunks. It exists so the platform is runnable and testable offline.
- **Drift detection is statistical, not semantic.** PSI tells you the
  distribution moved; it does not tell you *why*. The incident report ends
  with the questions a human should ask next.
- **The DAG engine is not a scheduler.** There is no UI, no SLA handling, no
  dynamic partitioning. It is the smallest thing that makes "the platform
  does this on a schedule" true.
- **The canary now has a verdict — but it needs volume.** The comparison
  (`GET /admin/canary`, `cli canary`) runs a two-proportion test on the
  refusal rate and a bootstrap on the retrieval score, and refuses to
  conclude anything from fewer than 30 requests per side. With the
  synthetic traffic that ships here it correctly says "not enough traffic
  to compare".
- **The feedback in this repository is simulated.** The learning loop runs
  end to end, but the judgements it trains on come from a script, not from
  users. A learned ranker fitted on a handful of simulated labels is a
  demonstration of the plumbing, **not** evidence that the platform learned
  anything. The numbers only become meaningful with real traffic (the
  target is 50–100 judgements) and a real corpus.
- **Promotion stays manual.** The canary report says whether the challenger
  is promotable, but nothing promotes it automatically. A two-proportion test
  on the refusal rate is a weak instrument; auto-promoting on it would be
  worse than not promoting at all.

## Where to take it next

The short version is in [`TODO.md`](TODO.md) — the full plan, with the
reasoning, the milestones and the order.

1. ~~**Make the registry trustworthy.**~~ **Done.** A bundle now records the
   backends that *ran* (`embedder_id`, `embedding_dim`, `reranker_id`) next
   to the ones that were *requested*, plus a content hash of the model
   files. A bundle built under the hashing embedder no longer claims
   `all-MiniLM-L6-v2`.
2. **Plug in a real LLM and activate the judge.** `evaluate/judge.py` exists
   but is never called. With a real model the golden set becomes a *model*
   comparison, and you can promote between models through the same gate.
3. ~~**Close the loop on feedback.**~~ **The machinery is done**: capture,
   judgement set, training, gating, canary. What is missing is *evidence* —
   real traffic against a real corpus. That is a week of wall-clock, not a
   week of code.
4. **Make ingestion incremental**, and add a sensor that fires when the
   corpus changes rather than only when the traffic does.
5. **Add a second corpus and a router**, once the single-corpus path is
   solid.

One thing deliberately *not* on that list: replacing the in-house DAG engine
with Dagster or Prefect. It is ~200 readable lines and you have not hit the
problems it solves yet; the mapping onto Dagster/Prefect/Airflow is in
[`ARCHITECTURE.md`](ARCHITECTURE.md) for when you do.

## Licence

MIT. The corpus and the golden set are original fiction about a fictional
cloud provider, written for this project.

## Further reading

- [`TODO.md`](TODO.md) — the build plan: what to do next, in what order
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — how the pieces fit together
- [`FIXES.md`](FIXES.md) — what was broken in the Docker stack, why, and what changed
- [`AGENTS.md`](AGENTS.md) — the working agreement for coding agents
