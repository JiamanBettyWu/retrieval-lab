# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-26 (latest session):** **`seed 1` is labelled (30/30, `--check`
green) and the judge bake-off has a harness.** Splits: `grounded` **10/6** over
16 answered rows, `refusal_ok` **12/2** over 14 refused. On that measured
balance, **labelling is done — `seed 2` stays blank**: it would buy ~2 more
minority cases on the slow axis, which cannot certify anything. The rubric of
record is now written in full above `LABEL_VALUES` in `fixture.py`, and
`AXIS_QUESTIONS` was amended to match it (`refusal_ok` rules on the *passages*,
not the rationale). `src/retrieval_lab/judge.py` is scaffolded on
`feat/issue-2-judge-bakeoff` — plumbing runs end to end and stops at the first
of four stubs. `gemma3:4b` is pulled. 115 tests green. Narrative in
[SESSIONS.md](SESSIONS.md).

```bash
pytest                                                  # 115 tests, ~8s, no download
git checkout feat/issue-2-judge-bakeoff
python -m retrieval_lab.judge --bakeoff                 # runs to the first stub today
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
- **Amended 2026-08-17:** generations and judgements key separately, so a judge
  swap does not discard generations — pinned in `tests/test_generate.py`.
- **Note 2026-08-26:** `judge.py` is Ollama-only by design; certifying Sonnet 5
  against this same fixture needs an API branch in `judge_one`. The fixture,
  rubric and scoring are transport-agnostic.
- **Why this is still a decision and not an issue:** "two judges or one" has no
  definition of done until the bake-off produces evidence. Shape it into an
  issue once #2 lands.
- **Blocked on:** #2.

## Needs attention

- ⚠️ **`main`'s `fixture.py` currently holds two disagreeing copies of the
  `refusal_ok` rubric.** The prose above `LABEL_VALUES` is the amended
  passages-based rule; `AXIS_QUESTIONS` on `main` is still the old
  rationale-based wording. The fix (`ce989db`) is on
  `feat/issue-2-judge-bakeoff` because it belongs in that PR. **Do not run
  `--label` off `main` until the PR merges** — the on-screen text would be the
  superseded rule.
- ⚠️ **The fixture is prompt `v2`; the scored batches are still `v1`.**
  `n100 seed0` and `n15 seed0` were deliberately NOT re-run under v2, so every
  refusal-curve number in `LEARNINGS.md` and #3 is a **v1** measurement while
  the fixture that will certify a judge is v2. Reversible — `prompt_version` is
  in the cache key and the v1 files are intact; re-running `n100` is ~10
  unattended minutes. **Do not publish a κ from v2 labels next to refusal
  numbers from v1 generations without re-running that batch first.**
- ⚠️ **`refusal_ok` yields a LOWER BOUND on over-refusal, and now with a named
  mechanism:** the axis rules on what the passages support, so a refusal whose
  rationale never exposes the answer can still be over-refusal and score `true`.
  Seed 1 was labelled under the *old* rationale-based wording; 10 rows sit in
  the divergence zone and were deliberately not re-verified (2026-08-26) since
  the axis is descriptive-only. Any README number from this axis must say
  "lower bound". **Rater noise also caps κ** — worth one README sentence.
- ⚠️ **After the bake-off, do not revise `seed 1` labels because a judge
  disagreed.** That tunes the reference to the thing being measured and voids
  the certification. Diagnose from the disagreements, or adjudicate on `seed 2`
  (blank, never held out for anything).
- ⚠️ **Carried, the correctness scorer:** `LEARNINGS.md`'s `mean token-F1 0.711`
  predates D14 and reads **0.7619 over 14, refusal rate 6.7%** under the settled
  convention. Write the scorer so refusals leave the denominator, and test the
  normaliser rather than eyeballing it — it is load-bearing (0.600 → 0.711 on
  the gold-context probe, `LEARNINGS.md` 2026-08-19).
- ⚠️ **`README.md` has no Phase 4 row** and still describes the project as
  three-dataset. `CLAUDE.md`'s command list has no `judge.py` entry either —
  add it when the PR merges, not before.
- ⚠️ Carried: **the probe scripts live in gitignored `scratchpad/`** while
  `LEARNINGS.md` cites their numbers. And **`tests/test_finetune.py` does not
  exist** despite being cited in `load_triples`'s docstring.
- Minor: `--label` on a sheet that does not exist yet raises a raw
  `FileNotFoundError` rather than `load_generations`'s "draw it first" message.

## Pick up here

1. **Fill the four stubs in `judge.py`** on `feat/issue-2-judge-bakeoff`, each
   with a CONCEPT block already written: `build_judge_prompt` (mirror the
   *amended* passages-based rubric), `parse_judgement` (return `None`, never a
   default), `cohen_kappa`, `rank_candidates`. For κ, build the 2x2 matrix by
   hand first — the test that matters is the constant rater, which scores 62%
   raw accuracy on this split and must land at κ ≈ 0.
2. **Write `tests/test_judge.py`**, then open the PR and `/code-review`. A
   stubs-only PR gives a reviewer nothing to say; this one should carry the
   `AXIS_QUESTIONS` amendment too.
3. **Run the bake-off** — `mistral-small` (~24B ceiling probe, already on disk)
   vs `gemma3:4b` (floor). Decision rule fixed in advance: a κ gap under ~0.2 on
   n=16 is a tie, broken on throughput. Tie → close
   [#2](https://github.com/JiamanBettyWu/retrieval-lab/issues/2) with the
   finding and record the winner's digest; gap → pull the 8B/14B middle; both
   fail to separate → escalate to a 27B or D13's option C. `ollama rm` the
   losers afterwards.
4. **Re-run `n100 seed0` under v2** (~10 unattended minutes) before any κ is
   published beside a refusal number ([#3](https://github.com/JiamanBettyWu/retrieval-lab/issues/3)).
