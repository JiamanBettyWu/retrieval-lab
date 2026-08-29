# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-28 (latest session):** **Phase 4c.2 is measured and the local
judge is chosen** — `mistral-small` (digest `8039dd90c113`) at κ 0.586 on
`grounded`, against `gemma3:4b` at κ −0.257 with a 0% parse-miss rate on both.
`n100 seed0` was regenerated under generator prompt v2, so every Phase 4 number
now sits on one generator. Manual follow-ups: [#2] is still open and both losing
models are still installed. Narrative in [SESSIONS.md](SESSIONS.md); numbers in
`LEARNINGS.md` (2026-08-28).

```bash
pytest                                          # 160 tests, ~8s, no download
python -m retrieval_lab.judge --bakeoff         # cache HIT now; --refresh to re-judge
```

## Working mode (carry this forward)

Betty hand-writes the ML; Claude writes plumbing (argparse, logging, paths,
sanity checks) and **coaches: concept first, then she writes it**, then review.
Demonstrating bugs empirically is wanted; silently fixing them is not.

## Open decisions

Only questions **not yet shaped enough to be an issue** live here — see
`CLAUDE.md`, "Repo conventions". Concrete work is in
[GitHub issues](https://github.com/JiamanBettyWu/retrieval-lab/issues):
[#1 (D5) dense queries](https://github.com/JiamanBettyWu/retrieval-lab/issues/1) ·
[#2 (D13) judge size](https://github.com/JiamanBettyWu/retrieval-lab/issues/2) ·
[#3 (D15) README stratification](https://github.com/JiamanBettyWu/retrieval-lab/issues/3)

### D10: Two judges, or one? (Phase 4) — **generator and local judge settled**
- **Settled:** generator `qwen3:8b`; local judge `mistral-small`, pinned by
  digest, cleared on the fixture rather than on a spec sheet.
- **Still open:** whether to add Sonnet 5 as a second judge so their agreement
  becomes a result reported next to human κ. Needs an API branch in `judge_one`;
  the fixture, rubric and scoring are already transport-agnostic.
- **Why still a decision:** "two judges or one" has no definition of done until
  someone decides what the agreement column is *for* — a robustness check, or a
  claim. Shape it into an issue once that is answered, then it leaves this file.

## Needs attention

- ⚠️ **`fixture.py`'s over-refusal type case is factually stale, not just
  superseded.** The block above `LABEL_VALUES` names seed-1 row 1 (Edmund
  Mortimer) as the type case; under generator v2 **that row answers**. The
  current type case is the Hund's-rule row (`5ae24b16…`, gold 2/2). Same block is
  also the *superseded* rationale-based `refusal_ok` rubric — "THE BOUNDARIES"
  below it is current. **Mark it SUPERSEDED in place and fix the row pointer**;
  position no longer tells the next reader which wins.
- ⚠️ **Do not publish the bake-off as "size matters".** The candidates differ in
  vendor, architecture and training corpus as well as parameter count (only
  quantization is matched). It is a *selection* result. A size ablation needs two
  sizes in one family (`gemma3:4b` vs `gemma3:27b`) and was not run.
- ⚠️ **`grounded` got stricter in `b4e6b45`** — principle-only wording → three
  named failure modes. It changes what the judge measures, and **the README must
  say which version produced the κ**. One edit reverts it.
- ⚠️ **`refusal_ok` yields a LOWER BOUND on over-refusal**, and refusals have at
  least two mechanisms — missing context, and *misreading present context*, which
  the gold-stratified curve reads as conservatism. Any README number from this
  axis must say "lower bound".
- ⚠️ **`n15 seed0` is still a prompt-v1 batch** (`n100 seed0` was regenerated
  2026-08-28). It is the trial batch and carries no published claim, but do not
  quote it beside a v2 number without re-running it.
- ⚠️ **Carried, the correctness scorer:** `LEARNINGS.md`'s `mean token-F1 0.711`
  predates D14 and reads **0.7619 over 14, refusal rate 6.7%** under the settled
  convention — quote it that way, and write the scorer so refusals leave the
  denominator. Its normaliser is load-bearing (0.600 → 0.711): test it, don't
  eyeball it (`LEARNINGS.md` 2026-08-19).
- ⚠️ **`README.md` has no Phase 4 row** and still describes the project as
  three-dataset. This is now unblocked — the bake-off gives the table something
  to say.
- ⚠️ Carried: **probe scripts live in gitignored `scratchpad/`** while
  `LEARNINGS.md` cites their numbers — decide whether the ones behind published
  figures belong in the repo. And **`tests/test_finetune.py` does not exist**
  despite being cited in `load_triples`'s docstring.
- Minor: `--label` on a sheet that does not exist yet raises a raw
  `FileNotFoundError` rather than `load_generations`'s "draw it first" message.

## Pick up here

1. **Close [#2](https://github.com/JiamanBettyWu/retrieval-lab/issues/2)** — the
   measured answer is already posted as a comment; it just needs closing. Then
   `ollama rm gemma3:4b` (and any other losing candidate) if the disk is wanted
   back — the winner's digest is recorded in the comment and in `LEARNINGS.md`.
2. **Add the Phase 4 row to `README.md`** — first Phase 4 number with a
   comparison behind it. Say which `grounded` wording produced the κ, and mark
   the over-refusal figure a lower bound (both flagged above).
3. **Fix `fixture.py`'s stale type case and mark the superseded rubric block** —
   smallest change on this list and the one most likely to mislead a reader.
