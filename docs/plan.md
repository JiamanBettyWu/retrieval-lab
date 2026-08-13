# Retrieval Lab — plan

> The working design doc for the repo. Read this before proposing
> architectural changes; `TODO.md` carries current state and open decisions.

A **retrieval-quality lab**: build a two-stage RAG retriever, LoRA-fine-tune its
encoder, and **prove** each improvement on a labeled benchmark (BEIR) with
metrics traced in Weave — then demo the validated pipeline as an "ask my second
brain" front door over a personal wiki.

## The pitch (why this project)

These concepts are studied but never built: **RAG, reranking, learning-to-rank,
embeddings, LoRA, fine-tuning.** This project applies **all six at once**, and
does the one thing most RAG projects skip: it **measures** whether the fancy
parts (reranker, fine-tuned encoder) actually help. That measurement is the
career story — *classic-ML rigor (ranking metrics, headroom, generalization)
applied to a GenAI stack*, exactly the classic-ML→GenAI pivot. Portfolio-
shareable: the benchmark half is public by construction, and no work data is
involved anywhere. **How much of the Phase 3 wiki demo is shareable is still
open** — the wiki is private, and a RAG demo answers by quoting its corpus, so
"who may read the wiki" is the real question. See `TODO.md` D6.

## Concepts exercised

**Will apply:** RAG (retrieve → rerank → generate), embeddings (the bi-encoder
retriever), reranking (bi-encoder → cross-encoder, *measured* for lift),
learning-to-rank (NDCG@k / MRR as the eval lens), LoRA / fine-tuning
(LoRA-fine-tune the encoder on MS MARCO triples), W&B Weave (tracing +
`weave.Evaluation`), deliberate-practice (build-to-learn).

**Stretch:** LLM-evaluation — Phase 4 grades the *generation* half
(faithfulness/groundedness) via LLM-as-judge.

## Core design decisions (locked)

1. **Fine-tune, never train from scratch.** Start from a pre-trained encoder
   (`all-MiniLM-L6-v2` or a BGE base); LoRA-adapt it (nudge low-rank adapters,
   not all weights).
2. **Evaluate ≠ demo.** *Evaluate* (compute NDCG/MRR) on a **labeled BEIR set**
   — needs qrels + headroom. *Demo* on the **wiki** — no labels needed; it's a
   qualitative zero-shot-generalization showcase.
3. **Zero-shot generalization is the headline claim.** Fine-tune on MS MARCO
   (general), evaluate on an *unseen* BEIR domain. "A retriever that generalizes"
   beats "I improved my own test split."
4. **Every phase ships a number and a working artifact.** The deliverable is a
   growing ablation table with *movement* in it.

## Pipeline

```
query
  │
  ▼
[bi-encoder]  embed query + corpus → cosine top-100        ← Phase 0 (+ Phase 2 fine-tunes this)
  │
  ▼
[cross-encoder rerank]  re-score top-100 jointly → top-10   ← Phase 1
  │
  ▼
[generate]  Claude answers from top-k with citations        ← Phase 3 (wiki demo only)
```

### Models — the bi-encoder ≠ cross-encoder trap

They share the `MiniLM-L6` **backbone** but are different model *types* and are
**not interchangeable** (the name collision is the classic trap):

| Role | Model | Type | Outputs |
|---|---|---|---|
| **Phase 0 retriever** | `sentence-transformers/all-MiniLM-L6-v2` | **bi-encoder** | an embedding per text (index the corpus once, cosine top-k) |
| **Phase 1 reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **cross-encoder** | one relevance score per (query, doc) pair — cannot do first-stage retrieval |
| *(optional reference)* | `sentence-transformers/msmarco-MiniLM-L6-cos-v5` | bi-encoder | an MS-MARCO-tuned "upper-bound" comparison |

**Baseline = the *general* `all-MiniLM-L6-v2`, deliberately not the MS-MARCO-tuned
bi-encoder** — Phase 2 fine-tunes on MS MARCO, so starting from an
already-MS-MARCO-specialized model would flatten the ablation. General baseline
= headroom for the LoRA fine-tune to visibly help. (A stronger modern baseline
like `BGE-small-en` also works — just less headroom.)

## Milestones (each independently shippable, each de-risks the next)

**Phase 0 — Baseline retriever + eval harness (the foundation).**
- Pick one small BEIR set with qrels — **NFCorpus** (~3.6K docs, medical, hard)
  or **SciFact** (~5K, scientific claims). Load via the `beir` library.
- Off-the-shelf bi-encoder → embed corpus + queries → cosine top-k.
- Compute **NDCG@10, MRR, Recall@100** (BEIR's `EvaluateRetrieval` /
  `pytrec_eval`). Trace in Weave.
- **Deliverable:** the baseline number *and* the headroom check — confirm
  NDCG@10 is *not* already ~1.0 (else pick a harder set).

**Phase 1 — Cross-encoder reranker (first lift).**
- Retrieve top-100 (Phase 0), rerank with a pre-trained cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) → top-10.
- Re-measure. **Deliverable:** before/after table (bi-encoder vs +rerank).

**Phase 2 — LoRA-fine-tune the bi-encoder (the depth move).**
- Fine-tune the encoder on **MS MARCO** `(query, positive, negative)` triples,
  contrastively (sentence-transformers `MultipleNegativesRankingLoss`, in-batch
  + optional hard negatives), with a **LoRA adapter** on the backbone (`peft`).
- Evaluate **zero-shot** on the *same untouched* BEIR set.
- **Deliverable:** the full ablation — off-the-shelf → +rerank → LoRA-encoder →
  LoRA-encoder+rerank. Zero-shot generalization, proven.
- **Detailed design: [Phase 2 design](#phase-2-design--lora-fine-tune-the-bi-encoder)
  below** (written 2026-08-12). It extends this milestone; nothing here is
  reversed.

**Phase 3 — The wiki demo ("ask my second brain").**
- Point the validated pipeline at the wiki's markdown (chunk pages, embed,
  retrieve → rerank → Claude answers with page citations).
- Thin UI: **Gradio/Streamlit** for speed (or a minimal FastAPI + page).
- **Deliverable:** qualitative zero-shot showcase — a MS-MARCO-trained retriever
  answering over a personal domain. The README front door.

**Phase 4 — Generation eval (stretch).**
- Answer-quality scorers: retrieval-grounded faithfulness + relevance, via
  **LLM-as-judge** (cross-model to dodge self-bias), calibrated against a small
  hand-labeled set (report agreement). In `weave.Evaluation`.

## Phase 2 design — LoRA fine-tune the bi-encoder

> Written 2026-08-12, before any Phase 2 code. Extends the Phase 2 milestone
> above. Every library API named here was verified against the *installed*
> versions (`sentence-transformers` 5.7.0, `transformers` 5.14.1, `torch` 2.13.0)
> and the live HF hub, not from recall — the versions are in `uv.lock`.

### What Phase 2 has to prove

Phase 1 proved a *pre-trained* reranker helps. Phase 2's claim is stronger and
easier to fake: **a retriever fine-tuned on MS MARCO retrieves better on a
domain it never saw.** The number that supports it is NDCG@10 on NFCorpus — the
same set Phases 0 and 1 used, which means NFCorpus is the *test* set and must
stay untouched by every training decision, not just by the training data.

That is the whole difficulty. The training data never touches NFCorpus
(design decision 3 already covers it), but *hyperparameter choice* is a second,
quieter channel from the test set into the model. The rules below close it.

### The measurement rules (pre-committed, before any number exists)

These are Phase 2's equivalent of "an oracle may only reorder what retrieval
found, never add to it." They constrain the experiment, not the code.

- **R1 — No hyperparameter is chosen by looking at NFCorpus.** LoRA rank, alpha,
  learning rate, batch size, epochs and checkpoint step are all selected on a
  held-out **MS MARCO dev slice** only. If NFCorpus NDCG picks the config, the
  result stops being zero-shot and becomes a test-set fit, and the headline
  claim dies quietly — nothing raises, the table just gets a number it hasn't
  earned. Same failure shape as a leaked `corpus_id`: well-formed and wrong.
- **R2 — Every NFCorpus number obtained gets reported.** No running three
  configs and tabling the best. If more than one config is ever evaluated on
  NFCorpus, all of them appear in `LEARNINGS.md`, with the R1-selected one
  marked as the config the table quotes.
- **R3 — The ceiling moves, so it gets re-measured.** A fine-tuned encoder
  returns a *different top-100*, so the 0.6263 oracle ceiling no longer applies
  to it. `oracle.py` is re-run on the LoRA candidate set and **both ceilings are
  reported.** Reusing the old one would compare Phase 2's gain against a bound
  computed for someone else's candidates.
- **R4 — The fine-tune is allowed to lose.** MS MARCO tuning making an
  out-of-domain medical set *worse* is a real and publishable outcome (it is
  what "zero-shot generalization" is testing for). The row ships either way; a
  negative result gets the same README treatment Phase 1's dense-query
  regression got.

### Training data

**`sentence-transformers/msmarco-bm25`, config `triplet`** — 502,931 rows,
~0.23 GB, columns exactly `query` / `positive` / `negative` (verified via the
HF datasets-server; the negatives are BM25-mined, i.e. hard-ish rather than
random). This is the shape `MultipleNegativesRankingLoss` consumes directly:
column order is (anchor, positive, negative), no collation code of our own.

- A `triplet-hard` config exists at 19.1M rows / 3.2 GB. **Out of scope** — it
  is a bigger download than the whole rest of the project and Phase 2 is not a
  "how good can MS MARCO training get" study.
- **How many triples we actually train on is not decided here**, deliberately.
  It is a wall-clock question about MPS throughput, and a guessed number in a
  plan is worth less than a measured one. The *procedure*: the smoke run below
  reports steps/sec, and the subsample is sized to a **training budget of
  roughly 30–60 minutes**, recorded in `LEARNINGS.md` with the throughput that
  justified it.
- **Dev slice for R1:** a held-out shard of the same `triplet` config, never
  trained on, scored with an in-training ST evaluator. It is the *only* signal
  allowed to choose a config.

### The LoRA setup

Confirmed present in the installed `sentence-transformers` 5.7.0:

- `SentenceTransformer.add_adapter(...)` — the PEFT entry point, via
  `base/peft_mixin.py`, which delegates to the transformers
  `PeftAdapterMixin`. Takes a `peft.LoraConfig`. Sibling methods
  `set_adapter` / `disable_adapters` / `enable_adapters` exist and matter for
  testing (see below).
- `SentenceTransformerTrainer` + `SentenceTransformerTrainingArguments`
  (exported from the package root) — these sit on the HF `Trainer`, which is
  why `accelerate` becomes a real dependency.
- `MultipleNegativesRankingLoss` at
  `sentence_transformers.sentence_transformer.losses`.
  `CachedMultipleNegativesRankingLoss` is also available and is the fallback if
  MPS memory caps the batch size — MNRL's signal quality scales with batch size
  (more in-batch negatives), so a small batch is a real quality cost, and the
  cached variant buys a large effective batch back.

**Adapt the backbone, not the pooling.** LoRA targets the transformer's
attention projections; the pooling layer is parameter-free and unchanged. Rank
and target modules are R1-governed — chosen on the dev slice.

### Checkpoint naming, and the cache-key trap wearing a new hat

`cache.py` keys on `(dataset, model_name, top_k)` and slugs `/` to `__`, so a
local checkpoint path works mechanically. But **two LoRA runs with different
hyperparameters written to the same directory produce the same cache key** —
the second run silently reads the first run's retrieval results. This is
exactly the "cache key does not cover `retrieve.py`" trap in new clothing, and
it is worse here because the stale value looks like a legitimate metric.

**Convention:** checkpoints live at `models/lora-<tag>/` where `<tag>` encodes
the config that was varied (e.g. `lora-r16-lr2e5-v1`), and that path is what is
passed as `model_name`. A config change means a new directory, therefore a new
cache key, therefore no silent reuse. `models/` is gitignored (added 2026-08-12
alongside this design) — adapters are rebuildable, and the lockfile plus the
recorded config is what makes them reproducible.

**Why the checkpoint name is a sufficient fix, checked rather than assumed:**
retrieval is the *only* cached stage. `rerank.py` and `oracle.py` both
recompute from the `results` they are handed on every run
(`rerank.py:356` re-scores unconditionally), so there is no second stale-value
channel downstream — get the retrieval key right and the whole chain is honest.
Two consequences: reranking the LoRA candidates costs full cross-encoder
inference every time, which is a wall-clock cost to expect but not a
correctness risk; and the stray `cache/reranked_nfcorpus.json` on disk is an
**orphan from an earlier iteration that nothing reads or writes** — delete it
rather than reasoning about it later.

### The ablation rows Phase 2 lands

| Row | What it isolates |
|---|---|
| Phase 2 · LoRA encoder | the fine-tune alone, vs Phase 0's `0.3159` |
| Phase 2 · LoRA + rerank | the full stack, vs Phase 1's `0.3412` |
| _reference_ · `msmarco-MiniLM-L6-cos-v5` | a **fully** MS-MARCO-specialized bi-encoder, off the shelf, zero training cost |
| _ceiling_ · oracle over LoRA candidates | R3 — the bound that actually applies to the new candidate set |

The reference row is the cheapest high-value row available and it is already
anticipated in the model table above. Without it, Phase 2 can only say "LoRA
moved the number." With it, Phase 2 says **"LoRA captured X% of what full
MS MARCO specialization achieves, at adapter cost"** — which is the actual
question anyone reading a LoRA result has. Verified on the hub as
`sentence-transformers/msmarco-MiniLM-L6-cos-v5`.

Both Phase 2 rows must be reported, not just the better one: the fine-tune and
the reranker can overlap (both were trained toward MS MARCO relevance), so
`LoRA + rerank` may well be *less* than the sum of the two gains. That
interaction is a finding, not a disappointment.

### Two separate checks, deliberately not conflated

A single NFCorpus number cannot distinguish "the training loop is broken" from
"training worked and did not transfer." So Phase 2 reports both:

1. **Did training work?** Falling train loss + improving MS MARCO dev-slice
   score. In-domain, expected to move, and diagnostic only.
2. **Did it generalize?** NFCorpus NDCG@10. The headline. Out-of-domain, and
   under R4 allowed to go either way.

Check 1 passing while check 2 fails is the *interesting* result. Check 1
failing means there is a bug, and no NFCorpus number should be quoted at all.

### D5 (dense-query blending) as an ablation axis

`TODO.md` D5 — the cross-encoder subtracts 1.06 NDCG points on dense queries —
carries into Phase 2 as a **blend weight between bi-encoder and cross-encoder
scores**, and it applies over *both* candidate sets, base and LoRA. It is
independent of the fine-tune and needs no training, so it can land before,
during or after. Worth noting it may partly dissolve on its own: a better
first-stage encoder changes which queries are dense in the top-100.

### What pins Phase 2 in tests

Mirroring how `retrieve.py` and `rerank.py` are pinned — fixture-only, no
download, no training:

- **Triple shape** — the training rows reach the loss as (anchor, positive,
  negative) in that column order. A silent column swap trains the model to
  rank negatives up, and the loss curve still looks fine.
- **The adapter actually changes embeddings** — encode a fixture sentence with
  adapters disabled and enabled; the vectors must differ. Guards the failure
  where LoRA attaches to nothing, training "succeeds," and the fine-tuned model
  is byte-identical to the baseline.
- **Checkpoint path → distinct cache key** — two tags must not collide, per the
  trap above.

### Build order — smoke run before any config is committed

1. **Smoke run first.** A few hundred steps on a tiny triple subset on MPS,
   just to prove the ST-trainer-on-`accelerate`-on-MPS path runs and to
   *measure* throughput. MNRL on MPS is exactly the kind of thing that fails in
   ways a plan cannot predict, and the subsample size depends on the number it
   produces.
2. Size the real run from that throughput (30–60 min budget), train, save to
   `models/lora-<tag>/`.
3. Retrieve on NFCorpus with the checkpoint → `evaluate.py` (new cache key,
   so no `--refresh` needed — but check the log says MISS).
4. Re-run `oracle.py` on the new candidates (R3), and `rerank.py` for the
   LoRA + rerank row.
5. Add the reference-model row — no training, just a second `evaluate.py` run.
6. README table + `LEARNINGS.md`, including every number R2 requires.

`finetune.py` is the new module (already anticipated in the layout below);
`evaluate.py`, `oracle.py` and `rerank.py` need a `--model` argument rather
than any structural change, since all three already take `results` and key off
a model name.

### Dependencies and the lockfile

Phase 2 adds **`peft`**, **`accelerate`**, and **`datasets`** — the last is
currently installed only transitively, which Phase 2 must stop relying on and
declare. They go in a **`train` extra**, so an eval-only install stays light and
the training stack is opt-in like `tracing` already is.

Because the repo's position is that its numbers are version-dependent, the
re-lock lands in **a commit that touches nothing else**, so any metric drift
afterwards is attributable to the dependency change rather than tangled with
code. Re-run `pytest` and re-check the Phase 0/1 numbers against the table
immediately after locking; if they moved, that is the finding to record before
any training starts.

## Repo layout (actual — installable `src/` package)

```
retrieval-lab/
├── README.md              # pitch + the ablation table (the money shot)
├── LEARNINGS.md           # devlog — findings recorded while they're fresh
├── pyproject.toml         # deps + editable install (`uv pip install -e '.[dev]'`)
├── docs/plan.md           # this file
├── src/retrieval_lab/
│   ├── observability.py   # weave init + @op shim (from the mise pattern)
│   ├── data.py            # BEIR load (ships the qrels labels)
│   ├── retrieve.py        # Phase 0 · bi-encoder embed + top-k
│   ├── cache.py           # on-disk retrieval cache (pins candidates across phases)
│   ├── evaluate.py        # Phase 0 · entrypoint: load → retrieve → metrics → headroom
│   └── oracle.py          # Phase 1 · entrypoint: the perfect-rerank ceiling
├── tests/                 # fixture suites (pytest, no download)
├── data/                  # BEIR downloads (git-ignored)
├── cache/                 # cached retrieval runs (git-ignored)
└── reports/               # dated ablation tables
```

**Amends the original flat-layout decision** (see SESSIONS.md 2026-07-15, which
chose root-level files to dodge package-import traps). Reversed 2026-08-07 once
the module count reached seven: `pyproject.toml` + an editable install makes
imports unambiguous rather than fragile, which was the actual concern. Cost is
one setup step and `python -m retrieval_lab.<entrypoint>` invocations.

Planned additions as phases land: `rerank.py` (Phase 1), `finetune.py` (Phase 2),
`app/` (Phase 3).

## Tech stack

- **`sentence-transformers`** — bi-encoder, cross-encoder, MS MARCO training
  losses (MNRL). Recent versions integrate `peft` for LoRA adapters.
- **`beir`** — datasets (with qrels) + `EvaluateRetrieval` (NDCG/MAP/Recall/
  Precision via `pytrec_eval`).
- **`peft`** — the LoRA adapter on the encoder backbone.
- **`weave`** — tracing + `weave.Evaluation` (reuse the `observability.py` shim
  from the mise project).
- **Anthropic Claude** — the Phase 3 answer generator.
- Vector search: brute-force in-memory cosine is fine at BEIR-small scale; add
  FAISS only if a corpus grows.

## Scope discipline — explicitly OUT (v1)

- No training from scratch; **LoRA only**, never full fine-tuning.
- No full-scale MS MARCO *evaluation* — small BEIR set for eval, MS MARCO only
  as *training* data.
- No production/multi-user app — this is a lab + demo.
- No chunking-strategy rabbit hole in Phase 0–2 (BEIR docs are pre-chunked);
  revisit only for the Phase 3 wiki demo.

## Open decisions

- **Eval dataset:** NFCorpus (harder, more qrels/query) vs SciFact (cleaner,
  claim-style). Leaning **NFCorpus** to guarantee headroom.
- **Demo UI:** Gradio/Streamlit (fast) vs FastAPI+React (a web rep). Leaning
  **Gradio** — the science is the star.

## Learning resources to line up

- Sentence-BERT paper — the bi/cross-encoder foundation.
- `sentence-transformers` MS MARCO training + `beir` eval docs.
- W&B "LLM Apps Evaluation" course — for Phase 4.
