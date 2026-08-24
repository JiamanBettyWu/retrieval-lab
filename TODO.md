# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-23 (latest session):** **The 4c.1 fixture exists and is ready to
label, at prompt `v2`.** Two held-out draws sit on disk as blank worksheets —
`seed 1` and `seed 2`, 60 rows pooled, **33 answered / 27 refused** — written by
the new `src/retrieval_lab/fixture.py` (115 tests green). The rubric is settled
and written into the module: two binary axes, `grounded` (answered rows) and
`refusal_ok` (refused rows), with the inapplicable axis pre-filled `"n/a"` so
each row is one decision. `mistral-small` is pulled, so D13 is unblocked.
Labelling restarts from scratch under v2 — both sheets sit blank at 30/30
awaiting decisions, and everything is committed and pushed. Findings in
`LEARNINGS.md` (2026-08-23); narrative in [SESSIONS.md](SESSIONS.md).

```bash
pytest                                                          # 114 tests, ~8s, no download
python -m retrieval_lab.fixture --n-queries 30 --seed 1 --label # resumable, one row at a time
python -m retrieval_lab.fixture --n-queries 30 --seed 1 --check # verify before trusting
```

## Working mode (carry this forward)

Betty hand-writes the ML; Claude writes plumbing (argparse, logging, paths,
sanity checks) and **coaches: concept first, then she writes it**, then review.
Demonstrating bugs empirically is wanted; silently fixing them is not.

## Open decisions

### D5: How to handle dense queries, where reranking actively hurts? (parked)
- **Context:** on the 86 queries with 11+ relevant docs the cross-encoder
  *subtracts* 1.06 NDCG points while improving MRR. Phase 2 did not dissolve it.
- **Options:** A) score blending B) routing (needs a detector) C) do nothing.
  **Recommend A** — but only if you return to retrieval; this is Phase 1 cleanup.

### D10: Which generator, and which judge? (Phase 4) — **generator settled, judges tentative**
- **Settled:** generator `qwen3:8b`. Two judges, not one, so their agreement
  becomes a result reported next to human κ.
- **Tentative:** Sonnet 5 primary, one local. Neither has cleared the fixture,
  which is the gate — a judge swapped after the fact voids every generation
  scored under it. The local judge must not be qwen3 (self-bias);
  `mistral-small` satisfies that.
- **Amended 2026-08-17:** generations and judgements key separately, so a judge
  swap does not discard generations — pinned in `tests/test_generate.py`.
- **Blocked on:** the labelled fixture (D16), then D13's bake-off.

### D13: Does the local judge need to be 27B?
- **Context:** `Qwen3.8-27B` was picked on "bigger is safer", never measured.
  The principle to test: judging is *classification* against a rubric and all it
  rules on is in the prompt, so parameter count may buy little.
- **Unblocked 2026-08-23:** `mistral-small` (14GB) pulled; disk is not a
  constraint (SESSIONS.md 2026-08-23).
- **Options:** A) keep a 27B, unmeasured B) **bake off 3–5 candidates against
  the fixture** (it is small, so five cost about what one does) C) drop the
  local judge. **Recommend B** — decide on evidence, not a spec sheet.
- **Blocked on:** labelled fixture rows (D16).
- **If B:** score separation *and* throughput; throughput should choose the final
  n for the config sweep. **If C:** D10 loses its second judge and the κ section
  loses its agreement column.

### D15: Does the README report refusal rate stratified by gold-passage presence?
- **Context (measured 2026-08-23, n=100, prompt v1):** the flat 34.0% splits into
  9.8% / 53.7% / 87.5% for 2 / 1 / 0 gold docs in context, first two CIs
  non-overlapping. **These are v1 numbers** — see the prompt-version flag below.
- **Caveat that must ship with it:** those are *different queries*, so retrieval
  quality is confounded with question difficulty. The split is descriptive; the
  causal claim belongs to the paired multi-config run (4b).
- **Options:** A) flat only B) stratified in the table C) **flat in the table,
  stratified in prose behind a `--breakdown`-style flag. Recommend C** — a
  two-part cell breaks the ablation table; `rerank.py --breakdown` is precedent.

### D16: How many of the 60 fixture rows get hand-labelled?
- **Context (v2 draws):** `seed 1` gives 16 answered rows, only 3 on partial
  gold. `seed 2` adds 17 answered including 8 partial and 2 on zero gold.
  Pooled: 33 answered / 27 refused, 11 answered-on-partial.
- **Cost asymmetry:** `refusal_ok` is the *slower* axis — confirming an answer
  was not derivable means reading all ten passages; `grounded` needs only the
  passages actually cited.
- **Options:** A) all 60 B) **label `seed 1` fully, then decide from its
  `grounded` split** C) all 33 answered rows + a randomly subsampled (recorded
  seed) share of refusals. **Recommend B** — deciding fixture size on measured
  class balance is the same discipline as measuring the ceiling before building
  the reranker.
- **Blocked on:** labelling `seed 1`.
- **Mechanics:** to skip rows, delete those lines — `validate_labels` raises on
  a blank field but only warns on a short sheet, deliberately.

## Needs attention

- ⚠️ **The fixture is prompt `v2`; the scored batches are still `v1`.**
  `build_prompt` leaked its own format placeholder — "Cite passages as [1], [2]"
  appeared verbatim in ~11% of rationales (19/175 across all batches), injecting
  citation-looking tokens into rationales that cited nothing, which is a
  contaminant on exactly the axis 4c measures. Fixed and bumped to `v2`; the
  fixture draws were regenerated (leak 12 → 0 over the two seeds). **`n100
  seed0` and `n15 seed0` were deliberately NOT re-run**, so every refusal-curve
  number in `LEARNINGS.md` and D15 below is a **v1** measurement while the
  fixture that will certify a judge is v2. Reversible — `prompt_version` is in
  the cache key, the v1 files are intact, and re-running `n100` under v2 is ~10
  unattended minutes. **Do not publish a κ from v2 labels next to refusal
  numbers from v1 generations without re-running that batch first.**
- ⚠️ **Commit the sheets as you label.** `data/labels/*.jsonl` is tracked now
  (re-included by `.gitignore` on purpose — hand labels are the one input that
  cannot be regenerated). A labelling session that ends uncommitted is the one
  way this project can lose work it cannot rebuild.
- ⚠️ **`refusal_ok` yields a LOWER BOUND on over-refusal**, and refusals have
  (at least) two mechanisms behind them — missing context, and *misreading
  present context*, which the gold-stratified curve reads as conservatism
  (`LEARNINGS.md` 2026-08-23). Any README number from this axis must say
  "lower bound". Rubric of record: the comment block above `LABEL_VALUES` in
  `fixture.py`.
- ⚠️ **Carried, the correctness scorer:** `LEARNINGS.md`'s `mean token-F1 0.711`
  predates D14 and reads **0.7619 over 14, refusal rate 6.7%** under the settled
  convention — quote it that way, and write the scorer so refusals leave the
  denominator. Its normaliser is load-bearing (0.600 → 0.711 on the gold-context
  probe): test it, don't eyeball it (`LEARNINGS.md` 2026-08-19).
- ⚠️ **`README.md` has no Phase 4 row** and still describes the project as
  three-dataset. Phase 4a has numbers (the refusal curve) but no config
  comparison, so the ablation table has nothing to gain yet. What the README
  says about refusal rate is D15.
- ⚠️ Carried: **the probe scripts live in gitignored `scratchpad/`** while
  `LEARNINGS.md` cites their numbers — decide whether the ones behind published
  figures belong in the repo. And **`tests/test_finetune.py` does not exist**
  despite being cited in `load_triples`'s docstring (should pin train ∩ dev
  disjointness, the `n + k` guard, the `--dev-loss` defaults).
- Minor: `--label` on a sheet that does not exist yet raises a raw
  `FileNotFoundError` from `read_text` rather than `load_generations`'s
  "draw it first" message.

## Pick up here

1. **Label `seed 1`** — `--label`, then `--check`. Commit the sheet when you
   stop. Read the `grounded` split before deciding whether `seed 2` is needed
   (D16).
2. **Then D13's bake-off** against the fixture, scoring separation *and*
   throughput; the throughput number should choose the final n for the config
   sweep, not a placeholder.
3. **Re-run `n100 seed0` under v2** before any κ is published beside a refusal
   number (~10 unattended minutes; see the prompt-version flag above).
