# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-23 (latest session):** **The 4c.1 fixture exists and is ready to
label.** Two held-out draws sit on disk as blank worksheets — `seed 1` and
`seed 2`, 60 rows pooled, **31 answered / 29 refused** — written by the new
`src/retrieval_lab/fixture.py` (114 tests green). The rubric is settled and
written into the module: two binary axes, `grounded` (answered rows) and
`refusal_ok` (refused rows), with the inapplicable axis pre-filled `"n/a"` so
each row is one decision. `mistral-small` is pulled, so D13 is unblocked.
Labelling is in progress; `fixture.py`, `tests/test_fixture.py` and
`data/labels/` are all still **untracked**. Narrative in [SESSIONS.md](SESSIONS.md).

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
- **Options:** A) score blending B) routing (needs a detector; "dense" is not
  knowable without the qrels) C) do nothing
- **Recommendation:** A (the blend weight earns a README row) — but **only if
  you return to retrieval**. Phase 4 is the direction; this is Phase 1 cleanup.

### D10: Which generator, and which judge? (Phase 4) — **generator settled, judges tentative**
- **Settled:** generator `qwen3:8b`, local via Ollama. Two judges, not one, so
  their agreement becomes a result reported next to human κ.
- **Tentative:** primary judge Claude Sonnet 5, second judge local. Neither has
  cleared the fixture, which is the gate — a judge swapped after the fact voids
  every generation scored under it.
- **Constraint that survives any swap:** the local judge must not be a qwen3
  model (self-bias vs the `qwen3:8b` generator). `mistral-small` satisfies it.
- **Amended 2026-08-17:** generations and judgements key separately, so a judge
  swap does not discard generations — pinned in `tests/test_generate.py`.
- **Blocked on:** the labelled fixture (D16), then D13's bake-off.

### D13: Does the local judge need to be 27B?
- **Context:** `Qwen3.8-27B` was picked on "bigger is safer" and never measured.
  The principle to test: judging is *classification* against a rubric and all it
  rules on is in the prompt, so parameter count may buy little.
- **Unblocked 2026-08-23:** `mistral-small` (14GB) pulled; disk is not a
  constraint (see SESSIONS.md 2026-08-23).
- **Options:** A) keep a 27B, unmeasured B) **bake off 3–5 candidates against
  the fixture** — it is small, so five cost about what one does C) drop the
  local judge, run Sonnet 5 alone
- **Recommendation:** B — decide on the fixture's evidence, not a spec sheet.
- **Blocked on:** labelled fixture rows (D16).
- **If B:** score separation *and* throughput; the throughput number is what
  should choose the final n for the full config sweep. **If C:** D10's "two
  judges" half is withdrawn and the κ section loses its agreement column.

### D15: Does the README report refusal rate stratified by gold-passage presence?
- **Context (measured 2026-08-23, n=100):** the flat 34.0% splits into 9.8% /
  53.7% / 87.5% for 2 / 1 / 0 gold docs in context, first two CIs non-overlapping.
- **Caveat that must ship with it:** those are *different queries*, so retrieval
  quality is confounded with question difficulty. The split is descriptive; the
  causal claim belongs to the paired multi-config run (4b).
- **Options:** A) flat only B) stratified in the table C) **flat in the table,
  stratified in prose behind a `--breakdown`-style flag**
- **Recommendation:** C — a two-part cell breaks the ablation table, which is
  the portfolio front door; `rerank.py --breakdown` is the precedent.

### D16: How many of the 60 fixture rows get hand-labelled?
- **Context:** `seed 1` alone gives 14 answered rows, only 3 on partial gold.
  `seed 2` adds 17 answered including 8 partial and 2 on zero gold. Pooled:
  31 answered / 29 refused.
- **Cost asymmetry:** `refusal_ok` is the *slower* axis — confirming an answer
  was not derivable means reading all ten passages, whereas `grounded` only
  needs the passages actually cited.
- **Options:** A) all 60 B) **label `seed 1` fully, then decide from its
  `grounded` split** C) all 31 answered rows + a randomly subsampled (recorded
  seed) share of refusals
- **Recommendation:** B — if `grounded` comes back near-balanced, seed 2's
  refusals are surplus; if it is 28–2, label seed 2's answered rows next.
  Deciding fixture size on measured class balance is the same discipline as
  measuring the ceiling before building the reranker.
- **Blocked on:** labelling `seed 1`.
- **Mechanics:** to skip rows, delete those lines — `validate_labels` raises on
  a blank field but only warns on a short sheet, deliberately.

## Needs attention

- ⚠️ **Uncommitted work:** `src/retrieval_lab/fixture.py` and
  `tests/test_fixture.py` are untracked (this handoff commits only `TODO.md`,
  `SESSIONS.md`, `CLAUDE.md`). `data/labels/*.jsonl` is also untracked and is
  being edited right now — the sheets are re-included by `.gitignore` on
  purpose, since hand labels are the one input that cannot be regenerated.
  **They should be committed once labelling finishes.**
- ⚠️ **`refusal_ok` yields a LOWER BOUND on over-refusal.** The rubric counts an
  over-refusal only when the rationale itself names the answer; a refusal that
  never names it scores `true` even if the answer was derivable. Any README
  number from this axis must say so. Rubric of record: `fixture.py`, the comment
  block above `LABEL_VALUES`.
- ⚠️ **`LEARNINGS.md` has no entry for this session** — the seed-to-seed refusal
  spread (53.3% / 43.3% / 34.0% on the same config, differing only in which
  questions were drawn) is a finding in that file's voice and worth appending
  while fresh.
- ⚠️ **Carried, the correctness scorer:** `LEARNINGS.md`'s `mean token-F1 0.711`
  predates D14 and reads **0.7619 over 14, refusal rate 6.7%** under the settled
  convention — quote it that way, and write the scorer so refusals leave the
  denominator. Its normaliser is load-bearing (0.600 → 0.711 on the gold-context
  probe): test it, don't eyeball it (`LEARNINGS.md` 2026-08-19).
- ⚠️ **`README.md` has no Phase 4 row** and still describes the project as
  three-dataset. Phase 4a has publishable numbers (the refusal curve) but no
  config comparison, so the ablation table still has nothing to gain. What the
  README says about refusal rate is D15.
- ⚠️ Carried: **the probe scripts live in gitignored `scratchpad/`** while
  `LEARNINGS.md` cites their numbers — decide whether the ones behind published
  figures belong in the repo. And **`tests/test_finetune.py` does not exist**
  despite being cited in `load_triples`'s docstring (should pin train ∩ dev
  disjointness, the `n + k` guard, the `--dev-loss` defaults).
- Minor: `--label` on a sheet that does not exist yet raises a raw
  `FileNotFoundError` from `read_text` rather than `load_generations`'s
  "draw it first" message.

## Pick up here

1. **Label `seed 1`** — `--label`, then `--check`. Read the `grounded` split
   before deciding whether `seed 2` is needed (D16).
2. **Commit `fixture.py`, `tests/test_fixture.py` and the labelled sheets.**
3. **Then D13's bake-off** against the fixture, scoring separation *and*
   throughput; the throughput number should choose the final n for the config
   sweep, not a placeholder.
