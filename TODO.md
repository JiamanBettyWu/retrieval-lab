# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-28 (latest session):** **The 4c.2 bake-off harness is finished
and merged** ([#4](https://github.com/JiamanBettyWu/retrieval-lab/pull/4),
`1a6dbf7`) — all four of `judge.py`'s functions are implemented, 160 tests green,
and both candidates now follow the format (`--smoke`: `mistral-small` 4/4,
`gemma3:4b` 4/4). The judge prompt is at `RUBRIC_VERSION v2` after `/code-review`
caught v1 missing on **100% of real calls**; nothing was ever cached under v1.
**The bake-off itself has not run yet** — that is the next command. Narrative in
[SESSIONS.md](SESSIONS.md); findings in `LEARNINGS.md` (2026-08-28).

```bash
pytest                                          # 160 tests, ~8s, no download
python -m retrieval_lab.judge --smoke           # 4 rows/candidate against live models
python -m retrieval_lab.judge --bakeoff         # the real run, ~25-30 min unattended
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
  scored under it. The local judge must not be qwen3 (self-bias).
- **Unblocked as of 2026-08-28:** the harness exists and both local candidates
  clear the format gate. Sonnet 5 still needs an API branch in `judge_one`;
  the fixture, rubric and scoring are transport-agnostic.
- **Why still a decision and not an issue:** "two judges or one" has no
  definition of done until the bake-off produces evidence. Shape it into an
  issue once #2 lands.
- **Blocked on:** the `--bakeoff` run.

## Needs attention

- ⚠️ **`fixture.py` contains TWO `refusal_ok` rubrics and position no longer
  tells you which is current.** The block *above* `LABEL_VALUES` (2026-08-23,
  rationale-based) is **superseded** by "THE BOUNDARIES" *below* it (2026-08-26,
  passages-based) plus the `AXIS_QUESTIONS` amendment in `ce989db`. v1 of the
  judge prompt was built from the superseded one. Both blocks stay by
  convention; `judge.py`'s docstring now names which is current. **Consider
  marking the older block SUPERSEDED in place** — the next reader has no reason
  to expect the lower block wins.
- ⚠️ **`grounded` got stricter in `b4e6b45`** — principle-only wording → the
  three named failure modes (uncited assertion / inference past the passage /
  unsupported negative). That is design decision #1 in `build_judge_prompt`'s
  docstring, it changes what the judge measures, and **the README must say which
  version produced the κ**. One edit reverts it.
- ⚠️ **The fixture is prompt `v2`; the scored batches are still `v1`.**
  (Generator `PROMPT_VERSION`, unrelated to the rubric bump above.) `n100 seed0`
  and `n15 seed0` were deliberately NOT re-run, so every refusal-curve number in
  `LEARNINGS.md` and #3 is a **v1** measurement. **Do not publish a κ from v2
  labels next to refusal numbers from v1 generations without re-running that
  batch first** (~10 unattended minutes).
- ⚠️ **`refusal_ok` yields a LOWER BOUND on over-refusal**, and refusals have
  (at least) two mechanisms — missing context, and *misreading present context*,
  which the gold-stratified curve reads as conservatism (`LEARNINGS.md`
  2026-08-23). Any README number from this axis must say "lower bound".
- ⚠️ **Carried, the correctness scorer:** `LEARNINGS.md`'s `mean token-F1 0.711`
  predates D14 and reads **0.7619 over 14, refusal rate 6.7%** under the settled
  convention — quote it that way, and write the scorer so refusals leave the
  denominator. Its normaliser is load-bearing (0.600 → 0.711 on the gold-context
  probe): test it, don't eyeball it (`LEARNINGS.md` 2026-08-19).
- ⚠️ **`README.md` has no Phase 4 row** and still describes the project as
  three-dataset. Phase 4a has numbers but no config comparison, so the ablation
  table has nothing to gain until the bake-off lands.
- ⚠️ Carried: **the probe scripts live in gitignored `scratchpad/`** while
  `LEARNINGS.md` cites their numbers — decide whether the ones behind published
  figures belong in the repo. And **`tests/test_finetune.py` does not exist**
  despite being cited in `load_triples`'s docstring.
- Minor: `--label` on a sheet that does not exist yet raises a raw
  `FileNotFoundError` from `read_text` rather than `load_generations`'s
  "draw it first" message.

## Pick up here

1. **Run the bake-off** — `--smoke` first out of habit (~4 min), then
   `--bakeoff` (~25-30 min unattended). Note `RUBRIC_VERSION v2` locks in at the
   first `--bakeoff`: any prompt edit after that needs a bump, or the cache
   serves rulings made under different wording.
2. **Close [#2](https://github.com/JiamanBettyWu/retrieval-lab/issues/2) with the
   result** — `rank_candidates` prints the winner *and* its reason, so the issue
   gets a measured answer to D13, not an assertion. Record the winning model's
   **digest**; the losers get `ollama rm`'d and a tag is not a pin.
3. **Then re-run `n100 seed0` under generator prompt v2** (~10 unattended
   minutes) before any κ is published beside a refusal number — see the
   prompt-version flag above and
   [#3](https://github.com/JiamanBettyWu/retrieval-lab/issues/3).
