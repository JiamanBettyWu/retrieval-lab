# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-22 (latest session):** **Phase 4a is complete and has produced
its first real numbers.** All three stubs are written and pinned — `build_prompt`
at `PROMPT_VERSION = "v1"`, `parse_answer`, `should_refuse` — and a 15-query
batch on real retrieved context held format 15/15 at ten passages, fired the
yes/no rule 0 times, and refused 33.3% of the time, of which **4 of 5 refusals
were correctly calibrated to gold passages retrieval had missed**. 85 tests
green, working tree clean. Findings in `LEARNINGS.md` (2026-08-22); narrative in
[SESSIONS.md](SESSIONS.md).

**D14 is decided (2026-08-22): correctness is reported over non-refused queries,
with refusal rate published alongside.** The denominator therefore varies per
config, so any README row quoting correctness must state its n.

```bash
pytest                                                       # 85 tests, ~8s, no download
python -m retrieval_lab.generate --n-queries 15              # Phase 4a — cached; --refresh to redo
python -m retrieval_lab.evaluate --dataset nfcorpus          # baseline 0.3159 (Phase 1 rerank: 0.3412)
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

### D15: Does the README report refusal rate stratified by gold-passage presence? (new, 2026-08-22)
- **Context:** the trial batch's 33.3% refusal rate splits 4 correctly-calibrated
  (gold missing from context) / 1 over-refusal (both gold present, ranks 2 and 9,
  model failed the multi-hop link). The flat number is honest and uninformative;
  the split is the actual finding.
- **Options:** A) **Flat rate only** — one number per config, no `qrels` needed
  B) **Stratified** — refusal rate given gold-present vs gold-missing, which
  separates calibration from multi-hop failure C) **Flat in the table, stratified
  in prose** — keeps the ablation table one-number-per-cell
- **Recommendation:** C. The table is a portfolio front door and a two-part cell
  breaks it; the split earns a paragraph and a `--breakdown`-style flag, which is
  where Phase 1's per-bucket analysis already lives.
- **Note:** stratifying reads `qrels`, which is analysis-only — the same licence
  `oracle.py` has. It must not reach the generation path.
- **Blocked on:** nothing, but pointless before n is large enough to split.

## Needs attention

- ⚠️ **The 4c.1 fixture still does not exist** (`data/labels/` is empty) and it
  gates D10 and D13. Betty's to author — the material now exists: the 15-query
  batch is naturally stratified (5 answered on full gold, 4 partial, 1 none, plus
  4 calibrated refusals and 1 over-refusal). The Vienna case is the best
  faithfulness item — fluent, cited `[1]`/`[7]`, and wrong.
- ⚠️ **`LEARNINGS.md`'s `mean token-F1 0.711` predates D14** — it averages over 15
  gold-context questions including a refusal. Under the settled convention that
  probe reads **0.7619 over 14, refusal rate 6.7%**. Nothing to revert (a probe
  result, not a published claim), but quote it the new way, and write the scorer
  so refusals leave the denominator rather than scoring 0.
- ⚠️ **The correctness scorer's normaliser is load-bearing** — standard HotpotQA
  normalisation is worth 0.600 → 0.711 on the gold-context probe. Test it; don't
  eyeball it (see `LEARNINGS.md` 2026-08-19 for what it does and does not fix).
- ⚠️ **The trial batch's cache file collides with any later `--n-queries`** — the
  key omits `n_queries` and `seed`, so an n=50 run reads the n=15 file, trips the
  post-hoc count check, and needs `--refresh` (regenerating all 50). Loud and
  working as designed; the trial's 15 generations do not carry forward.
- ⚠️ **The probe scripts live in gitignored `scratchpad/`** (`probe.md`,
  `probe_datasets.py`, `prompt_probe.py`, `sentinel_probe.py`). Safe from a
  reboot, absent from git, and `LEARNINGS.md` cites their numbers — decide whether
  the ones behind published figures belong in the repo.
- ⚠️ **`README.md` has no Phase 4 row** and still describes the project as
  three-dataset. Left alone deliberately: 4a's numbers are diagnostics, not
  ablation-table results.
- ⚠️ Carried: **`tests/test_finetune.py` does not exist** but is cited in
  `load_triples`'s docstring — should pin train ∩ dev disjointness, the `n + k`
  guard, and the `--dev-loss` k/batch-size defaults. Also, the contamination
  refutation is generator-specific at n=5 — re-run the no-context probe before
  trusting the corpus under a larger generator.

## Pick up here

1. **Author the 4c.1 fixture** from the 15-query batch in
   `cache/gen__…__ctx10__qwen3-8b__v1.json`, stratified across the profiles it
   already contains (grounded-and-correct, grounded-and-wrong, ungrounded,
   calibrated refusal, over-refusal). This is the gate on D10 and D13 — no judge
   may be chosen without it.
2. **Start the judge-candidate pulls in the background** (D13). `ollama list`
   still holds only `qwen3:8b`, which is excluded as a judge; every candidate is
   a multi-GB download and the bake-off cannot start until one lands.
3. **Then D13's bake-off** against the fixture — separation *and* throughput —
   which re-opens D10's second-judge slot with numbers instead of a spec sheet.
