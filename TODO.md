# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-12 (latest session):** **Phase 2 is designed but not built.**
`docs/plan.md` "Phase 2 design" is the spec — read it first, especially rules
**R1–R4**; R1 (no hyperparameter may be selected by looking at NFCorpus) is the
one that constrains every choice. Training deps (`peft`, `accelerate`,
`datasets`) landed in a `train` extra with **verified zero drift** — all Phase
0/1 numbers reproduce, candidate set byte-identical. `finetune.py` is scaffolded
with plumbing written and the ML left to hand-write; `load_triples` is done.
Detail in [SESSIONS.md](SESSIONS.md).

```bash
uv pip install -e '.[dev,train]'                      # from the repo root, always
pytest                                                # 40 tests, ~10s, no download
python -m retrieval_lab.evaluate --dataset nfcorpus   # baseline    0.3159
python -m retrieval_lab.oracle   --dataset nfcorpus   # ceiling     0.6263
python -m retrieval_lab.rerank   --dataset nfcorpus   # Phase 1     0.3412
python -m retrieval_lab.finetune --smoke              # NOT YET RUNNABLE (2 stubs left)
```

## How Phase 2 is being built (working mode — carry this forward)

Betty is hand-writing the ML as a learning exercise. **Do not fill in the
stubs.** The split: plumbing is written (argparse, logging, paths, the
trainable-params sanity check, the throughput→triple-count arithmetic); the ML
is `NotImplementedError` with docstrings that explain the reasoning and name the
traps. Coaching mode: **explain the concept first, then she writes it**, then
review. Two stubs remain in `src/retrieval_lab/finetune.py`:

- `build_lora_model()` — **next.** Concept to explain before she writes: LoRA
  rank vs alpha (effective scale is `alpha/r`), and why the pooling layer is
  not a target. The trap is live and already in the docstring: PEFT matches
  `target_modules` **by name suffix**, and this backbone has
  `attention.output.dense`, `intermediate.dense` *and* `output.dense`, so
  `["dense"]` silently hits three things of two different shapes. Real module
  names are listed in the docstring; backbone is 22,713,216 params.
- `train()` — trainer args, MNRL, save, return steps/sec. Concepts: batch size
  as a *quality* knob (in-batch negatives), `fp16` being CUDA-only, and what
  `report_to` must be so the file runs without weave/wandb.

## Open decisions

### D2: Phase 3 demo UI (not urgent — Phase 3 is far off)
- **Options:** A) **Gradio/Streamlit** — fast, minimal B) **FastAPI + React** —
  a web-eng rep, more work
- **Recommendation:** Gradio — the science (the ablation table) is the star.
- **Blocked on:** Phase 2 landing first.
- **If A:** thin `app/` with Gradio. **If B:** reuse the mise FastAPI+React
  pattern.

### D5: How should Phase 2 handle dense queries, where reranking actively hurts?
- **Context:** on the 86 queries with 11+ relevant docs in the top-100, the
  cross-encoder *subtracts* 1.06 NDCG points (13% of everything the other
  buckets earned) while still improving MRR — it promotes one good doc to rank 1
  and evicts others from the top ten. See the MRR-vs-NDCG table in `README.md`.
- **Options:** A) **Score blending** — combine cross-encoder and bi-encoder
  scores so rank 1 improves without the eviction; one code path B) **Routing** —
  skip reranking dense queries; needs a detector, and "dense" isn't knowable at
  query time without the qrels C) **Do nothing** — let the fine-tune address it
- **Recommendation:** A. The only option needing no query-time signal the system
  lacks. Ceiling on the win is ~`+0.0033` NDCG@10 (`0.3412 → 0.3445`).
- **Blocked on:** nothing — needs no training, so it can land before, during or
  after the fine-tune, over either candidate set.
- **If A:** the blend weight is an ablation axis and earns its own README row.
  **If B:** the detector is the hard part — a "dense" rule and a "baseline was
  already good" rule select nearly the same queries (rho 0.57). **If C:** record
  it as a known regression so Phase 2's numbers stay readable.
- **Note:** may partly dissolve on its own — a better first-stage encoder
  changes which queries are dense in the top-100.

### D6: Is the Phase 3 demo hosted, and over what corpus?
- **Context:** a RAG demo's output *is* passages quoted from the corpus, so
  anyone who can query it reads the notes a chunk at a time — server-side does
  not mean private. The question is who may read the wiki.
- **Options:** A) **Local-only + README screencast** — private, ~zero work
  B) **Curated public subset** — a live link; curation is the cost C) **A public
  corpus** — reproducible, fits the thesis, drops the "second brain" framing
  D) **Auth** — private, demos to nobody
- **Recommendation:** A. C if a live link matters more than the framing.
- **Blocked on:** Betty — not urgent, Phase 3 is two phases out.
- **If B/C:** `docs/plan.md` Phase 3 needs rewriting and D2 interacts (hosting
  argues for Gradio on HF Spaces).

## Needs attention

- ⚠️ **`src/retrieval_lab/finetune.py` is untracked** — deliberately left out of
  the handoff commit (a handoff stages only its own files). It contains
  hand-written work plus two intentional `NotImplementedError` stubs. Commit it
  as WIP or leave it in the working tree, but don't lose it.
- ⚠️ **`uv sync --all-extras` uninstalled an ad-hoc ipykernel tree** this
  session. Documented behaviour (sync *matches* the lock rather than adding),
  but if `scratch.ipynb` is in use: `uv pip install ipykernel`.
- ⚠️ **The retrieval cache key still does not cover `retrieve.py`'s contents.**
  Phase 2 does *not* need to edit that file (the checkpoint path in the cache
  key does the work), so the trap stays dormant — but it is one edit away.
- ⚠️ **`data/` and `cache/` resolve against the working directory**, not the
  package root, so entrypoints must be run from the repo root.
- ⚠️ **The qrel-density explanation is supported but not proven.** If Phase 2
  quotes it, keep the README's hedge.

## Pick up here

1. **`build_lora_model()`** — explain rank/alpha and the suffix-matching trap,
   then let Betty write it. The trainable-params check in `main()` will tell her
   immediately whether the adapter attached (`trainable == total` means it
   didn't; `0` means nothing trains and looks exactly like a bad learning rate).
2. **`train()`**, then run `--smoke` — its real output is **steps/sec**, which
   is what sizes the real triple subsample against a 30–60 min budget. Record
   the throughput and the chosen size in `LEARNINGS.md`.
3. **Then the real run, and R3:** re-run `oracle.py` on the LoRA candidate set —
   the 0.6263 ceiling does not carry over. Both ceilings get reported, and both
   Phase 2 rows (LoRA, LoRA+rerank) plus the zero-cost
   `msmarco-MiniLM-L6-cos-v5` reference row go in the README table.
