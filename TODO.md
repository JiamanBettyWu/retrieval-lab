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

Only questions **not yet shaped enough to be an issue** live here — see
`CLAUDE.md`, "Repo conventions". Concrete work is tracked in
[GitHub issues](https://github.com/JiamanBettyWu/retrieval-lab/issues):
[#1 (D5) dense queries](https://github.com/JiamanBettyWu/retrieval-lab/issues/1) ·
[#2 (D13) judge size](https://github.com/JiamanBettyWu/retrieval-lab/issues/2) ·
[#3 (D15) README stratification](https://github.com/JiamanBettyWu/retrieval-lab/issues/3)

### D10: Which generator, and which judge? (Phase 4) — **generator settled, judges tentative**
- **Settled:** generator `qwen3:8b`. Two judges, not one, so their agreement
  becomes a result reported next to human κ.
- **Tentative:** Sonnet 5 primary, one local. Neither has cleared the fixture,
  which is the gate — a judge swapped after the fact voids every generation
  scored under it. The local judge must not be qwen3 (self-bias);
  `mistral-small` satisfies that.
- **Amended 2026-08-17:** generations and judgements key separately, so a judge
  swap does not discard generations — pinned in `tests/test_generate.py`.
- **Why this is still a decision and not an issue:** "two judges or one" has no
  definition of done until the bake-off produces evidence. Shape it into an
  issue once #2 lands.
- **Blocked on:** the labelled fixture, then #2.

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
   stop. **Then read the `grounded` split and decide how far to go:** if it comes
   back near-balanced, `seed 2`'s rows are surplus; if it is lopsided (say 28-2),
   label `seed 2`'s 17 answered rows next. Deciding fixture size on measured
   class balance is the same discipline as measuring the ceiling before building
   the reranker. Note `refusal_ok` is the *slower* axis — confirming an answer
   was not derivable means reading all ten passages, while `grounded` needs only
   the ones actually cited.
2. **Then the judge bake-off** — issue #2.
3. **Re-run `n100 seed0` under v2** before any κ is published beside a refusal
   number (~10 unattended minutes; see the prompt-version flag above, and #3).
