# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-17 (latest session):** **Phase 4 has a dataset and a corpus.**
D12 moved Phase 4 off NFCorpus — whose queries turned out not to be questions —
onto **`hotpotqa-distractor-pool`**, a 66,581-doc corpus pooled from HotpotQA's
dev distractor paragraphs and built by `src/retrieval_lab/hotpot_pool.py`.
Correctness scoring is back in scope (HotpotQA ships gold short answers, so
nothing is invented) but **does not replace the ~50 hand labels** — those
calibrate the judge on *faithfulness*, which no dataset labels.
`src/retrieval_lab/generate.py` is scaffolded with `build_prompt`,
`parse_answer` and `should_refuse` left as stubs. 69 tests green; nothing has
been generated yet, so no generation cache exists. Detail in
[SESSIONS.md](SESSIONS.md); findings in `LEARNINGS.md` (2026-08-17).

```bash
pytest                                                # 69 tests, ~8s, no download
python -m retrieval_lab.hotpot_pool                   # build the Phase 4 corpus (once)
python -m retrieval_lab.generate --n-queries 50       # Phase 4a — needs `ollama serve` + the stubs
python -m retrieval_lab.evaluate --dataset nfcorpus   # baseline 0.3159 (Phase 1 rerank: 0.3412)
```

## Working mode (carry this forward)

Betty hand-writes the ML; Claude writes plumbing (argparse, logging, paths,
sanity checks) and **coaches: concept first, then she writes it**, then review.
Demonstrating bugs empirically is wanted; silently fixing them is not.

## Open decisions

### D5: How to handle dense queries, where reranking actively hurts? (parked)
- **Context:** on the 86 queries with 11+ relevant docs the cross-encoder
  *subtracts* 1.06 NDCG points while improving MRR. Phase 2 did not dissolve it.
- **Options:** A) **Score blending** B) **Routing** — needs a detector; "dense"
  isn't knowable without the qrels C) **Do nothing**
- **Recommendation:** A (the blend weight earns a README row), but **only if you
  return to retrieval** — Phase 4 is the direction and this is Phase 1 cleanup.
- **Blocked on:** nothing; needs no training.

### D10: Which generator, and which judge? (Phase 4) — **generator settled, judges still tentative**
- **Settled:** generator `qwen3:8b`, local via Ollama. Two judges, not one, so
  their agreement becomes a result reported next to human κ.
- **Still tentative:** **primary judge Claude Sonnet 5** (`claude-sonnet-5`) and
  **second judge `Qwen3.8-27B`** (local). Neither has cleared the 4c.1
  grounded/ungrounded fixture, which is the gate — a judge swapped after the
  fact voids every generation already scored under it.
- **Constraint that survives any swap:** the local judge must not be a qwen3
  model. The generator is `qwen3:8b`, and a shared pretraining corpus and
  post-training recipe is exactly the self-bias D10 exists to dodge; a different
  parameter count does not buy the independence a different family does.
- **Amended 2026-08-17 (cache key):** one key became two — generations key on
  `(dataset, retriever, top_k, n_context, generator, prompt_version)`,
  judgements on that plus `(judge, rubric_version)`. Folding judge IDs into the
  generation key would discard every generation whenever only a judge changed,
  and this decision *expects* the judges to change. Pinned by
  `tests/test_generate.py::test_swapping_a_judge_does_not_invalidate_generations`.
- **Blocked on:** the fixture existing (see D13 and "Pick up here").

### D13: Does the local judge need to be 27B, or is a smaller/different model better suited?
- **Context (raised 2026-08-17, Betty):** `Qwen3.8-27B` was picked for D10's
  second-judge slot on "bigger is safer", never on a measurement. `TODO.md` has
  carried "unverified whether a dense 27B (~16–17GB resident at Q4) runs at a
  usable rate on this machine" since it was chosen.
- **The principle to test:** judging is a *classification* task against a
  rubric, not a generation task. Parameter count mostly buys world knowledge,
  and the judge needs little — everything it rules on is supplied in the prompt.
  What it needs is instruction-following and calibration, which smaller
  instruct-tuned models may have plenty of.
- **Options:** A) **Keep `Qwen3.8-27B`** — as recorded, but unmeasured and
  possibly too slow to run on the full sample B) **Bake off 4–5 candidates
  against the 4c.1 fixture** — the fixture is small, so testing five costs about
  what testing one does C) **Drop the local judge**, run Sonnet 5 alone — cheapest,
  but forfeits the open-weight-agreement result and the "could an open-weight
  judge stand alone?" question a later phase wants answered.
- **Recommendation:** B. It is the same discipline as the rest of the repo —
  decide on the fixture's evidence, not on a spec sheet — and it costs an
  evening at most.
- **Blocked on:** the 4c.1 fixture, which does not exist yet. Judges cannot be
  compared before there is something to compare them on.
- **If B:** score each candidate on fixture separation *and* throughput on this
  machine, then re-open D10's second-judge slot with numbers. **If C:** D10's
  "two judges" half is withdrawn and the κ section loses its agreement column.

## Needs attention

- ⚠️ **The 4c.1 fixture does not exist** (`data/labels/` is empty) and it gates
  both D10 and D13 — no judge can be validated, and no generation scored under
  an unvalidated judge counts. It is ground truth, so it is Betty's to author;
  `scratchpad/probe.md` has seed material, including a genuine ungrounded case
  (hotpot Q2 cites [1] for a claim [1] never makes).
- ⚠️ **The contamination refutation is generator-specific and n=5.** If
  `qwen3:8b` is ever swapped for a larger generator, re-run the no-context probe
  before trusting `hotpotqa-distractor-pool` — a generator that answers from
  memory flattens every per-config difference. Gold answers make this a script
  run, not a reading session.
- ⚠️ **`SESSIONS.md` is 530+ lines**, past the ~500 rotation threshold. Not
  rotated tonight because that moves files; say the word and it archives to
  `sessions/` with a fresh journal started.
- ⚠️ **`README.md` has no Phase 4 row yet** and still describes the project as
  three-dataset. Left alone deliberately — Phase 4 has produced no number, and
  the ablation table is for landed results.
- ⚠️ Carried forward: **`tests/test_finetune.py` still does not exist** but is
  cited in `load_triples`'s docstring — should pin train ∩ dev disjointness, the
  `n + k` guard, and the `--dev-loss` k/batch-size defaults.
- ⚠️ Carried forward: the retrieval cache key does not cover `retrieve.py`'s
  contents (`--refresh` after editing); `data/`/`cache/` resolve against cwd.

## Pick up here

1. **Author the 4c.1 grounded/ungrounded fixture.** It gates D10 and D13, and
   nothing downstream is trustworthy without it. Seed from `scratchpad/probe.md`.
2. **Bake off local judge candidates against it** (D13) — fixture separation and
   throughput on this machine, not spec sheets. Not a qwen3 model.
3. **Fill `generate.py`'s three stubs** — `build_prompt` first, since the output
   format it establishes is what `parse_answer`, token-F1 and the judge rubric
   all have to agree with. First run spends minutes on a retrieval cache miss
   (66K docs) before reaching the stubs; that is not a hang.
