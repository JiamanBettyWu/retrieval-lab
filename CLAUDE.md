# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & commands

```bash
uv venv && uv pip install -e '.[dev]'      # extras: dev, tracing (Weave), train (Phase 2 LoRA)
export WANDB_API_KEY=...                   # optional — tracing is inert without it

uv sync --locked --all-extras              # reproduce the ablation table's exact versions

pytest                                     # both fixture suites, ~6s, no download
pytest tests/test_oracle.py -q             # one file
pytest -k "keys_are_real" -q               # one test

python -m retrieval_lab.evaluate --dataset nfcorpus   # baseline metrics + headroom check
python -m retrieval_lab.oracle   --dataset nfcorpus   # the ceiling a reranker could reach
python -m retrieval_lab.rerank   --dataset nfcorpus   # Phase 1 cross-encoder rerank
python -m retrieval_lab.rerank   --dataset nfcorpus --breakdown   # + the per-bucket analysis

python -m retrieval_lab.hotpot_pool                               # Phase 4: build hotpotqa-distractor-pool (~27MB, once)
python -m retrieval_lab.generate --n-queries 50                   # Phase 4a: generate answers (needs `ollama serve`)
python -m retrieval_lab.fixture --n-queries 30 --seed 1           # Phase 4c.1: write the hand-labelling sheet
python -m retrieval_lab.fixture --n-queries 30 --seed 1 --label   # ... label it (resumable, one row at a time)
python -m retrieval_lab.fixture --n-queries 30 --seed 1 --check   # ... verify the labels against the batch

python -m retrieval_lab.finetune --smoke                          # Phase 2: throughput + path check
python -m retrieval_lab.finetune --tag r16-a32-lr2e5-100k --triples 100000 --k 10000
```

**Always run from the repo root.** `data/` (BEIR downloads) and `cache/`
(retrieval runs) resolve against the current working directory, not the package
root. Both are gitignored and rebuildable.

**`uv pip install` and `uv sync` are not interchangeable here.** `uv pip install`
resolves fresh and only ever adds; it ignores `uv.lock` entirely. `uv sync` reads
the lock and makes the venv *exactly* match it, uninstalling everything else —
`--extra dev` on its own removes Weave, so pass `--all-extras`. Use `uv sync
--locked --all-extras` when a number in the ablation table needs reproducing,
`uv pip install` for everyday work. The lockfile is committed because the repo's
claims are version-dependent: metrics shift with model and tokenizer releases,
and an unpinned environment makes "a real regression" and "a newer torch"
indistinguishable.

`--refresh` on either entrypoint recomputes retrieval instead of reading the
cache. **The cache key covers dataset + model + top_k but NOT the contents of
`retrieve.py`** — after editing that file, a stale cache will silently serve old
results. Pass `--refresh`.

## What this project is

A retrieval-quality lab. It builds a two-stage RAG retriever, LoRA-fine-tunes
its encoder, and — the part that distinguishes it — **proves each improvement on
a labeled benchmark instead of asserting it.** The deliverable is a growing
ablation table with real movement in it, not a working demo. `docs/plan.md` is
the multi-phase design doc; read it before proposing architectural changes.

Phases: **0** baseline bi-encoder (done) · **1** cross-encoder rerank (done) ·
**2** LoRA-fine-tune the encoder on MS MARCO, evaluate zero-shot on the untouched
BEIR set · **3** wiki demo UI · **4** LLM-as-judge generation eval.

## Architecture

```
query → [bi-encoder] cosine top-100 → [cross-encoder] rerank to top-10 → [generate]
         Phase 0, fine-tuned in P2      Phase 1                           Phase 3
```

`data.py` loads BEIR (corpus, queries, **qrels** — the ground-truth labels).
`retrieve.py` embeds and ranks. `cache.py` persists the result.
`evaluate.py`, `oracle.py` and `rerank.py` are the entrypoints;
`observability.py` shims Weave's `@op` so every phase runs with weave absent or
uninitialized.

**`rerank.py --breakdown` is where Phase 1's analysis lives**, not in a notebook
— every per-bucket figure quoted in `README.md` is printed by that flag, so the
claims stay reproducible. Its statistics (`spearman_permutation`, `_partial`,
`n_relevant_in_top_k`) are pinned by `tests/test_rerank.py` for the same reason
the retrieval code is: a correlation that silently depends on input order is the
same class of well-formed-but-wrong output as a leaked `corpus_id`.

**`cache.py` is load-bearing beyond speed.** Every later phase re-scores the
*same* first-stage candidates — Phase 1 reranks them, `oracle.py` bounds what
that reranking could achieve. Caching makes "both saw identical candidates" a
fact rather than an assumption about determinism. Don't bypass it when adding a
phase; take `results` as an argument like `oracle_rerank` does.

### The measurement discipline

This is the repo's actual thesis, and it should shape any change:

- **Measure the ceiling before building the thing.** Phase 0's job was never
  "build a retriever" — it was confirming the dataset can even *show* a reranker
  helping. `oracle.py` applies the same logic one level down: it reads the qrels
  and orders each query's retrieved candidates perfectly, so its NDCG@10 is the
  hard upper bound on any real reranker over those candidates. Phase 1's result
  gets read against that number, not against 1.0.
- **An oracle may only reorder what retrieval found, never add to it.** Drawing
  candidates from `qrels` instead of from `results` would measure a perfect
  *retriever* and silently inflate the ceiling. `tests/test_oracle.py` pins this.
- **Prove optimality, not just improvement.** `oracle >= baseline` passes for any
  improvement, including one leaving gain on the table. `arithmetic_ceiling()`
  derives the answer independently of the sort under test; the two must agree.

### Failure mode to recognize

**`NDCG@10 = 0.0000` almost always means wrong doc ids, not a weak model.**
`semantic_search` returns `corpus_id` — a positional index into the encode order,
not a BEIR doc id. Leak those through and nothing raises; `pytrec_eval` just
finds zero overlap with the qrels. Run `pytest` before trusting any number from
`evaluate.py`: the fixtures are five docs with known-correct answers, and the
load-bearing one feeds them through BEIR's real evaluator, which must score
exactly `NDCG@10 = 1.0`. Fixture first, full run second — run the real eval first
and a low number has four possible causes.

### Metric facts worth not re-deriving

- qrels values are **relevance grades** (NFCorpus uses 1 and 2, higher = more
  relevant), not rank positions. Many docs share a grade.
- `pytrec_eval` computes NDCG with **linear gain** (`gain = grade`), not the
  exponential `2^grade − 1` common in papers.
- NFCorpus qrels are dense (median 16 relevant docs/query). Consequences:
  **Recall@10 caps at 0.615**, not 1.0; and `NDCG@100 < NDCG@10` is expected, not
  a bug — the ideal-DCG denominator grows faster than retrieved gain.

### The bi-encoder / cross-encoder name trap

`all-MiniLM-L6-v2` (bi-encoder, emits embeddings, Phase 0) and
`cross-encoder/ms-marco-MiniLM-L-6-v2` (emits one score per query-doc pair,
Phase 1) share a backbone but are not interchangeable — the cross-encoder cannot
do first-stage retrieval. The baseline is deliberately the *general* MiniLM, not
the MS-MARCO-tuned bi-encoder: Phase 2 fine-tunes on MS MARCO, so starting from
an already-specialized model would flatten the ablation.

## Repo conventions

- **`TODO.md`** = forward-looking state (current state, next actions),
  overwritten each session. **`SESSIONS.md`** = append-only dated journal; never
  edit past entries. Both are refreshed via `/baton:handoff`, which is
  user-invocable only — ask the user to run it.
- **Concrete work lives in GitHub issues, not in `TODO.md`** (adopted
  2026-08-23). The dividing line: **an issue is a unit of work with a definition
  of done; an "Open decision" is a question not yet shaped enough to have one.**
  A decision that has costed options and a recommendation has already become an
  issue — file it. `TODO.md` keeps only "Current state", "Needs attention",
  "Pick up here", and the genuinely unshaped decisions, because that file is
  injected whole at every session start and must earn its length.
  - **"Pick up here" is the entry point.** It names the next concrete actions
    and links the issue where a step has one, so a session can orient from
    `TODO.md` alone. Read the linked issue when you reach that step, and
    `gh issue list` whenever the backlog is what you actually need.
  - **D-numbers are historical.** D1–D16 are referenced by number in
    `docs/plan.md` and `SESSIONS.md`, so migrated decisions keep theirs in the
    issue title (`D13: …`). New issues are referred to by issue number only. If
    a genuinely unshaped decision ever needs a D-number again, continue the
    sequence at **D17** — never restart it, since collisions with history are
    silent.
- **Substantive code changes go through a PR; bookkeeping goes straight to
  `main`** (adopted 2026-08-23). A new module, a phase, a prompt-version bump,
  anything that invalidates cached artifacts or changes a published number:
  branch, PR, `/code-review`, merge. Handoffs, `SESSIONS.md`, `LEARNINGS.md`,
  `TODO.md` and doc-only edits: commit to `main` directly. **The test is whether
  a reviewer would have anything to say** — a PR whose entire diff is a journal
  entry is ceremony, and this repo does not do things it cannot justify.
  Branches are named `feat/issue-<N>-<slug>` — e.g. `feat/issue-2-judge-bakeoff`
  — so the branch, its PR and its issue are one thread. Other prefixes (`fix/`,
  `docs/`) take the same shape; work with no issue behind it drops the
  `issue-<N>` segment.
- **`LEARNINGS.md`** = the project devlog. Append findings while they're fresh,
  in the file's existing voice; no need to mark which entries Claude wrote.
- **`README.md` carries the ablation table** — it's the portfolio front door.
  Update it when a phase lands a number.
- Decisions that reverse an earlier recorded one should say so explicitly (see
  the layout-amendment note in `docs/plan.md`) rather than overwrite it silently.
