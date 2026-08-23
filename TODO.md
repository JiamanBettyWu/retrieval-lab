# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-23 (latest session):** **Phase 4a is done and has its first
result with an interval around it.** A 100-query batch on real retrieved context
refuses 34.0% of the time `[25.5%, 43.7%]`, and splitting that by how much gold
retrieval delivered gives **9.8% / 53.7% / 87.5%** for 2 / 1 / 0 gold docs in
context — monotonic, with the first two intervals non-overlapping. Format
adherence is 115/115 across both batches and parse misses are 0/100. All three
stubs are written and pinned; 87 tests green, working tree clean. Two generation
batches sit on disk (`n15__seed0`, `n100__seed0`). Findings in `LEARNINGS.md`
(2026-08-23); narrative in [SESSIONS.md](SESSIONS.md).

**Settled this session:** D14 (correctness is reported over non-refused queries,
refusal rate alongside — so any README correctness figure must state its n), and
4b's context is padded to `n_context` rather than gold-only, amended into
`docs/plan.md` (2026-08-23).

```bash
pytest                                                       # 87 tests, ~8s, no download
python -m retrieval_lab.generate --n-queries 100             # Phase 4a — cached (n and seed are in the key)
python -m retrieval_lab.generate --n-queries 30 --seed 1     # the held-out fixture draw
```

## Working mode (carry this forward)

Betty hand-writes the ML; Claude writes plumbing (argparse, logging, paths,
sanity checks) and **coaches: concept first, then she writes it**, then review.
Demonstrating bugs empirically is wanted; silently fixing them is not.

## Open decisions

### D5: How to handle dense queries, where reranking actively hurts? (parked)
- **Context:** on the 86 queries with 11+ relevant docs the cross-encoder
  *subtracts* 1.06 NDCG points while improving MRR. Phase 2 did not dissolve it.
- **Options:** A) score blending B) routing — needs a detector, and "dense" is
  not knowable without the qrels C) do nothing
- **Recommendation:** A (the blend weight earns a README row), but **only if you
  return to retrieval** — Phase 4 is the direction and this is Phase 1 cleanup.
- **Blocked on:** nothing; needs no training.

### D10: Which generator, and which judge? (Phase 4) — **generator settled, judges tentative**
- **Settled:** generator `qwen3:8b`, local via Ollama. Two judges, not one, so
  their agreement becomes a result reported next to human κ.
- **Tentative:** primary judge Claude Sonnet 5 (`claude-sonnet-5`), second judge
  `Qwen3.8-27B` (local). Neither has cleared the 4c.1 fixture, which is the gate
  — a judge swapped after the fact voids every generation scored under it.
- **Constraint that survives any swap:** the local judge must not be a qwen3
  model. The generator is `qwen3:8b`, and a shared pretraining corpus plus
  post-training recipe is exactly the self-bias D10 exists to dodge.
- **Amended 2026-08-17:** generations and judgements key separately, so swapping
  a judge does not discard generations — pinned by
  `tests/test_generate.py::test_swapping_a_judge_does_not_invalidate_generations`.
- **Blocked on:** the 4c.1 fixture (see D13).

### D13: Does the local judge need to be 27B?
- **Context:** `Qwen3.8-27B` was picked on "bigger is safer", never measured, and
  it is unverified whether a dense 27B (~16–17GB at Q4) runs usably here. The
  principle to test: judging is *classification* against a rubric and everything
  it rules on is in the prompt, so parameter count may buy little.
- **Options:** A) keep `Qwen3.8-27B` as recorded, unmeasured B) **bake off 4–5
  candidates against the 4c.1 fixture** — the fixture is small, so five cost
  about what one does C) drop the local judge, run Sonnet 5 alone
- **Recommendation:** B — decide on the fixture's evidence, not a spec sheet.
- **Blocked on:** the 4c.1 fixture. Note `ollama list` holds only `qwen3:8b`,
  excluded as a judge; every candidate needs a multi-GB pull, worth starting early.
- **If B:** score separation *and* throughput, then re-open D10's second-judge
  slot with numbers. **If C:** D10's "two judges" half is withdrawn and the κ
  section loses its agreement column.

### D15: Does the README report refusal rate stratified by gold-passage presence?
- **Context (measured 2026-08-23, n=100):** the flat 34.0% splits into 9.8% /
  53.7% / 87.5% for 2 / 1 / 0 gold docs in context, first two CIs non-overlapping.
  The split is the finding; the flat number hides it.
- **Caveat that must ship with it:** those are *different queries*, so retrieval
  quality is confounded with question difficulty. The split is descriptive. The
  causal claim belongs to the paired multi-config run (4b + the four configs over
  one fixed sample), where difficulty is held constant by construction.
- **Options:** A) flat rate only B) stratified in the table C) **flat in the
  table, stratified in prose behind a `--breakdown`-style flag**
- **Recommendation:** C — the ablation table is the portfolio front door and a
  two-part cell breaks it; `rerank.py --breakdown` is the precedent.
- **Note:** stratifying reads `qrels`, analysis-only — the licence `oracle.py`
  has. It must not reach the generation path.
- **Blocked on:** nothing.

## Needs attention

- ⚠️ **The 4c.1 fixture still does not exist** (`data/labels/` is empty) and it
  gates D10 and D13. Betty's to author, ~30 hand labels. **Draw it held-out**
  (`--seed 1`) so the fixture does not select a judge that then scores those same
  items. Strata to expect, from n=100: 46 answered on full gold, **19 on partial**
  (where faithfulness is genuinely at stake — the model committed while holding
  half the chain), 1 on none, 5 over-refusals, 29 calibrated refusals. The
  answered-with-nothing profile is rare *because* the model refuses 87.5% of the
  time when it has nothing; build around that rather than wishing for it.
- ⚠️ **`LEARNINGS.md`'s `mean token-F1 0.711` predates D14** — it averages over 15
  gold-context questions including a refusal. Under the settled convention that
  probe reads **0.7619 over 14, refusal rate 6.7%**. Nothing to revert (a probe
  result, not a published claim), but quote it the new way, and write the scorer
  so refusals leave the denominator rather than scoring 0.
- ⚠️ **The correctness scorer's normaliser is load-bearing** — standard HotpotQA
  normalisation is worth 0.600 → 0.711 on the gold-context probe. Test it; don't
  eyeball it (see `LEARNINGS.md` 2026-08-19 for what it does and does not fix).
- ⚠️ **The probe scripts live in gitignored `scratchpad/`** (`probe.md`,
  `probe_datasets.py`, `prompt_probe.py`, `sentinel_probe.py`). Safe from a
  reboot, absent from git, and `LEARNINGS.md` cites their numbers — decide whether
  the ones behind published figures belong in the repo.
- ⚠️ **No judge candidate has been pulled** — `ollama list` holds only
  `qwen3:8b`, which D10 excludes as a judge on self-bias grounds. Multi-GB
  download, pure wall-clock, and it blocks D13 and therefore D10. Start it first.
- ⚠️ **`README.md` has no Phase 4 row** and still describes the project as
  three-dataset. Phase 4a now *has* publishable numbers (the refusal curve), but
  no config comparison yet, so there is still nothing for the ablation table.
  What the README says about refusal rate is D15.
- ⚠️ Carried: **`tests/test_finetune.py` does not exist** but is cited in
  `load_triples`'s docstring — should pin train ∩ dev disjointness, the `n + k`
  guard, and the `--dev-loss` k/batch-size defaults. Also, the contamination
  refutation is generator-specific at n=5 — re-run the no-context probe before
  trusting the corpus under a larger generator.

## Pick up here

1. **Start a judge-candidate pull in the background** (D13) — non-qwen3, 12–32B
   (`gemma3:27b`, `mistral-small`). Longest pole, blocks everything downstream.
2. **Draw the held-out fixture batch** — `--n-queries 30 --seed 1` — and
   hand-label it. It coexists with `n100__seed0` now that the key covers seed.
3. **Then D13's bake-off** against the fixture, scoring separation *and*
   throughput; the throughput number is what should choose the final n for the
   full config sweep, not a placeholder.
