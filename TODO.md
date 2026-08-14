# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-14 (latest session):** **Phase 2 is complete and it lost.** The
LoRA fine-tune (r=16, α=32, lr 1e-3, picked on MS MARCO dev loss per R1) scores
NFCorpus NDCG@10 **0.2725** against baseline **0.3159** — but the off-the-shelf
`msmarco-MiniLM-L6-cos-v5` control scores **0.2584**, so the loss is the cost of
MS MARCO specialisation rather than a broken recipe, and **the cross-encoder
absorbs 99% of it** (0.3407 vs Phase 1's 0.3412). Seven NFCorpus looks were
spent and all seven are in the README table per R2. Findings in `LEARNINGS.md`
(2026-08-14); narrative and decisions in [SESSIONS.md](SESSIONS.md).

```bash
pytest                                                # 40 tests, ~8s, no download
python -m retrieval_lab.evaluate --dataset nfcorpus   # baseline 0.3159 (Phase 1: rerank, 0.3412)
python -m retrieval_lab.evaluate --dataset nfcorpus --model models/lora-r16-a32-lr1e3-100k
python -m retrieval_lab.finetune --tag <tag> --triples 100000 --k 10000 --lr 1e-3
```

## Working mode (carry this forward)

Betty hand-writes the ML; Claude writes plumbing (argparse, logging, paths,
sanity checks) and **coaches: concept first, then she writes it**, then review.
Demonstrating bugs empirically is wanted; silently fixing them is not.

## Open decisions

### D2: Phase 3 demo UI (now next in line — Phase 2 has landed)
- **Options:** A) **Gradio/Streamlit** — fast, minimal B) **FastAPI + React** —
  a web-eng rep, more work
- **Recommendation:** Gradio — the science (the ablation table) is the star.
- **Blocked on:** Betty. No longer blocked by Phase 2.
- **If A:** thin `app/` with Gradio. **If B:** reuse the mise FastAPI+React
  pattern. D6 interacts either way.

### D5: How to handle dense queries, where reranking actively hurts?
- **Context:** on the 86 queries with 11+ relevant docs the cross-encoder
  *subtracts* 1.06 NDCG points while improving MRR. **Phase 2 did not dissolve
  this** — its damage was diffuse across all density buckets (−0.053 / −0.023 /
  −0.048), so a different encoder did not change which queries are dense.
- **Options:** A) **Score blending** — one code path B) **Routing** — needs a
  detector; "dense" isn't knowable without the qrels C) **Do nothing**
- **Recommendation:** A. Ceiling on the win is ~`+0.0033` NDCG@10.
- **Blocked on:** nothing — needs no training, lands over either candidate set.
- **If A:** the blend weight is an ablation axis and earns a README row.

### D6: Is the Phase 3 demo hosted, and over what corpus?
- **Context:** a RAG demo's output *is* passages from the corpus, so
  server-side does not mean private.
- **Options:** A) **Local-only + screencast** — private, ~zero work
  B) **Curated public subset** C) **A public corpus** — reproducible, drops the
  "second brain" framing D) **Auth** — private, demos to nobody
- **Recommendation:** A. C if a live link matters more than the framing.
- **Blocked on:** Betty. Now urgent, since D2 is next. **If B/C:**
  `docs/plan.md` Phase 3 needs rewriting.

### D8: Is there a Phase 2.5 capacity ablation, or does Phase 2 close here?
- **Context:** all four learning rates converged to ~0.272 while the
  specialisation axis runs to the control's 0.258. The leading explanation is
  the **capacity ceiling** of 147,456 params on `query`+`value`, exhausted
  identically by every lr — currently an inference, not a measurement.
- **Options:** A) **Run r=64 and/or add `key`** — directly tests it; ~60 min +
  1 NFCorpus look each B) **Close Phase 2**, move to D2/Phase 3 C) **Test the
  fix instead** — fewer total steps, or mixed-domain replay
- **Recommendation:** B for the portfolio — the Phase 2 story is complete
  without it. A if the mechanism is the interesting part. C is the only option
  that could turn Phase 2 into a win, but it is a new experiment, not an ablation.
- **Blocked on:** Betty.
- **If A:** new `--tag` per config (the tag keys the retrieval cache), select on
  dev loss only (R1), report every number obtained (R2), settle D9 first.

### D9: Does the ablation table get error bars?
- **Context:** no configuration was ever run twice, so every number in the
  README is n=1 with **no estimate of run-to-run variance**. The early lr gaps
  (0.052, 0.026) dwarf any plausible noise; the `5e-4 -> 1e-3` gap of `0.0051`
  that decided the winner does not.
- **Options:** A) **Seed replicates on 2–3 configs** (~60 min each) — a noise
  floor for the whole table B) **State n=1 and move on** — already done, free
  C) **Replicate only the winner** — cheapest real answer
- **Recommendation:** C if D8-A happens (more close calls need a noise floor),
  otherwise B. A repo whose thesis is "measured, not asserted" is exposed here.
- **Blocked on:** Betty; interacts with D8.
- **If A/C:** `--tag <config>-seed<N>`; the spread goes in the table, not prose.

## Needs attention

- ⚠️ **`README.md` and `LEARNINGS.md` are committed but unpushed**, as is this
  handoff. Two commits sit ahead of `origin/main`.
- ⚠️ **`tests/test_finetune.py` still does not exist** but is cited in
  `load_triples`'s docstring. It should pin train ∩ dev disjointness and the
  `n + k` guard — that bug shipped once and is invisible at smoke scale.
- ⚠️ **The selector is a proxy, and Phase 2 measured how far it can diverge.**
  Hyperparameters were chosen on MS MARCO dev *loss*; a held-out MS MARCO dev
  **NDCG@10** stays R1-legal and is far closer to the target metric. Fix before
  any further search (D8/D9).
- ⚠️ **The damage curve between step 200 and one epoch is unmeasured** — "before
  one epoch" is supported, "at step N" is not. 3 more looks would settle it.
- ⚠️ **`logging_steps=20` / `eval_steps=200` are hardcoded** (`finetune.py:214,218`).
  Fine at 3,125 steps; degenerate at `--smoke`'s 200.
- ⚠️ Carried forward: the retrieval cache key does not cover `retrieve.py`'s
  contents (`--refresh` after editing); `data/`/`cache/` resolve against cwd.

## Pick up here

1. **Push.** Two commits are ahead of `origin/main` — the Phase 2 write-up and
   this handoff.
2. **Answer D8** — close Phase 2, or spend ~2 hours testing the capacity-ceiling
   explanation. Everything else in this file waits on that fork.
3. **If Phase 2 closes: start D2/D6 together** (Phase 3 demo UI and whether it
   is hosted) — they are the same decision from two angles, and `docs/plan.md`
   Phase 3 needs rewriting if the corpus changes.
