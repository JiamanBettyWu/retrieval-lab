# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-14 (latest session):** **Phase 2 trains.** `finetune.py` is
complete and committed (`8a64055`, pushed) — LoRA r=16/alpha=32 on
`["query","value"]`, 147,456 trainable params (0.65%), MNRL over MS MARCO
triples, dev slice held out per R1. **A 100k-triple run was launched overnight
in Betty's own terminal** (~31 min, log at
`cache/train-r16-a32-lr2e5-100k.log`); its result is unknown at handoff time.
**No NFCorpus number exists yet, deliberately.** Detail in
[SESSIONS.md](SESSIONS.md); the findings are in `LEARNINGS.md` (2026-08-14).

```bash
uv pip install -e '.[dev,train]'                      # from the repo root, always
pytest                                                # 40 tests, ~8s, no download
python -m retrieval_lab.evaluate --dataset nfcorpus   # baseline    0.3159
python -m retrieval_lab.oracle   --dataset nfcorpus   # ceiling     0.6263
python -m retrieval_lab.rerank   --dataset nfcorpus   # Phase 1     0.3412
python -m retrieval_lab.finetune --tag r16-a32-lr2e5-100k --triples 100000 --k 10000
```

## How Phase 2 is being built (working mode — carry this forward)

Betty hand-writes the ML as a learning exercise; Claude writes plumbing
(argparse, logging, paths, sanity checks, arithmetic) and **coaches: explain
the concept first, then she writes it**, then review. Verifying her code and
demonstrating bugs empirically is wanted — silently fixing them is not. Both
`finetune.py` stubs are now done, so the next ML-shaped work (the `--model`
wiring below is plumbing; a `tests/test_finetune.py` is arguably hers) should
follow the same split.

## Open decisions

### D2: Phase 3 demo UI (not urgent — Phase 3 is far off)
- **Options:** A) **Gradio/Streamlit** — fast, minimal B) **FastAPI + React** —
  a web-eng rep, more work
- **Recommendation:** Gradio — the science (the ablation table) is the star.
- **Blocked on:** Phase 2 landing first.
- **If A:** thin `app/` with Gradio. **If B:** reuse the mise FastAPI+React
  pattern.

### D5: How should Phase 2 handle dense queries, where reranking actively hurts?
- **Context:** on the 86 queries with 11+ relevant docs, the cross-encoder
  *subtracts* 1.06 NDCG points while improving MRR — it promotes one good doc
  to rank 1 and evicts others. See `README.md`.
- **Options:** A) **Score blending** — one code path, no query-time signal
  needed B) **Routing** — needs a detector; "dense" isn't knowable without the
  qrels C) **Do nothing** — let the fine-tune address it
- **Recommendation:** A. Ceiling on the win is ~`+0.0033` NDCG@10.
- **Blocked on:** nothing — needs no training, lands over either candidate set.
- **If A:** the blend weight is an ablation axis and earns a README row.
  **If C:** record it as a known regression. **Note:** may partly dissolve — a
  better encoder changes which queries are dense.

### D6: Is the Phase 3 demo hosted, and over what corpus?
- **Context:** a RAG demo's output *is* passages from the corpus, so
  server-side does not mean private.
- **Options:** A) **Local-only + screencast** — private, ~zero work
  B) **Curated public subset** C) **A public corpus** — reproducible, drops the
  "second brain" framing D) **Auth** — private, demos to nobody
- **Recommendation:** A. C if a live link matters more than the framing.
- **Blocked on:** Betty — not urgent.
- **If B/C:** `docs/plan.md` Phase 3 needs rewriting; D2 interacts.

### D7: Is `--lr 2e-5` right for LoRA, and does a second config get run?
- **Context:** `2e-5` is a *full fine-tuning* learning rate; LoRA is
  conventionally trained at 1e-4–5e-4 because it moves few, freshly-initialised
  parameters. The default was Claude's and may simply be wrong. Note `--lr` sets
  the **peak** — the scheduler decays linearly to ~0 over `max_steps`.
- **Options:** A) **Run 1e-4 and 5e-4 too**, pick on dev loss — ~30 min each
  B) **Ship the 2e-5 run alone** — fastest, but leaves an obvious question
  C) **Vary `target_modules` instead** (q-v vs q-k-v) — a different axis
- **Recommendation:** A. It is the highest-suspicion knob, and R1 compliance is
  already built (dev slice + `load_best_model_at_end`).
- **Blocked on:** the overnight run's `eval_loss` curve.
- **If A:** fresh adapter per config, new `--tag` each (the tag is identity —
  it must flow into the retrieval cache key), **NFCorpus stays untouched until
  the winner is picked**, then per R2 every number obtained gets reported.
  **If B:** say plainly in the README that lr was not tuned.

## Needs attention

- ⚠️ **`evaluate.py`, `oracle.py` and `rerank.py` have no `--model` flag** —
  they hardcode `BI_ENCODER` (`evaluate.py:31`, `oracle.py:99`,
  `rerank.py:344`). Nothing can evaluate the adapter until this is wired. The
  plumbing is ready: `cache.py:22` keys on `model_name` and
  `load_encoder("models/smoke")` was verified to load a checkpoint (22,860,672
  params = backbone + adapter). **This is the blocker for every Phase 2 number.**
- ⚠️ **`tests/test_finetune.py` does not exist** but is cited in
  `load_triples`'s docstring. The disjointness invariant (train ∩ dev = ∅ at
  realistic `n`, and the `n + k` guard) is exactly what a fixture test should
  pin — it is the bug that shipped today, and it is invisible at smoke scale.
- ⚠️ **The overnight run's outcome is unknown.** If the terminal died, restart;
  checkpoints land every 200 steps in `models/lora-r16-a32-lr2e5-100k/`, so an
  interruption costs minutes. A stale `checkpoint-200` from a killed run was
  cleared before launch.
- ⚠️ **`logging_steps=200` is hardcoded** in `train()`. Fine at 3,125 steps (15
  points); degenerate at `--smoke`'s 200 steps, where it yields one reading and
  no loss trajectory.
- ⚠️ Carried forward: the retrieval cache key still does not cover
  `retrieve.py`'s contents; `data/` and `cache/` resolve against the working
  directory; the qrel-density explanation is supported but not proven.

## Pick up here

1. **Read the overnight `eval_loss` curve** (15 points, in
   `cache/train-r16-a32-lr2e5-100k.log`). Falling then flattening = it trained;
   falling then rising = `load_best_model_at_end` earned its keep. Train loss
   near `ln(64) ≈ 4.16` would mean nothing learned. Record throughput and the
   curve's shape in `LEARNINGS.md`.
2. **Wire `--model` through the three entrypoints** (see "Needs attention") —
   this is the blocker. Then evaluate the adapter on NFCorpus **once**.
3. **Then R3:** re-run `oracle.py` over the *new* candidate set — the 0.6263
   ceiling does not carry over. Both ceilings get reported, and both Phase 2
   rows (LoRA, LoRA+rerank) plus the zero-cost `msmarco-MiniLM-L6-cos-v5`
   reference row go in the README table.
