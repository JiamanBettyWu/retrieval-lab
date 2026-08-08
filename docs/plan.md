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
shareable: public benchmark + public wiki, no work data.

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
