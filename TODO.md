# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-29 (latest session):** No code changed. A verification pass over
the Phase 4c.2 bake-off numbers found one wrong and corrected it in `f75bacd` —
`gemma3:4b` ruled `false` on **8 of the 10** human-`true` rows, not 8 of 11.
Every other Phase 4 figure re-verified clean against the artifacts. Phase 4's
`4b` (the gold-padded ceiling) is still the next actual work. Detail in
[SESSIONS.md](SESSIONS.md).

## Working mode (carry this forward)

Betty hand-writes the ML; Claude writes plumbing (argparse, logging, paths,
sanity checks) and **coaches: concept first, then she writes it**, then review.
Demonstrating bugs empirically is wanted; silently fixing them is not.

## Tracked work

Concrete work lives in
[GitHub issues](https://github.com/JiamanBettyWu/retrieval-lab/issues):
[#1 (D5) dense queries](https://github.com/JiamanBettyWu/retrieval-lab/issues/1) ·
[#3 (D15) README stratification](https://github.com/JiamanBettyWu/retrieval-lab/issues/3) ·
[#5 token-F1 scorer](https://github.com/JiamanBettyWu/retrieval-lab/issues/5).
**No open decisions.** Phase 4's remaining design is 4b then 4d, both specified
in [`docs/plan.md`](docs/plan.md).

## Needs attention

- ⚠️ **Numbers in prose are being written from summaries, not artifacts.** This
  session's error travelled *into* this repo from an external note and sat in
  `LEARNINGS.md` directly contradicting the matrix two lines below it. Worth a
  habit: any confusion-matrix claim gets checked against
  `data/labels/fixture__*.jsonl`, and the cross-judge invariant (same rows,
  same labels ⇒ same human-true count) is free.
- ⚠️ **The external write-up describes `refusal_ok` using the SUPERSEDED rule.**
  It says the rubric counts "only over-refusals the model's own text convicts it
  of" — that is the rationale-based copy, retired 2026-08-26. The rule of record
  tests the PASSAGES (`fixture.py`, THE BOUNDARIES), under which a refusal counts
  whether or not the rationale exposes it. **No number moves** — seed 1's
  divergence region is empty, so every label is identical either way — but it is
  a self-description error in the paragraph explaining the labelling procedure.
  Fix at that document's next revision; nothing to change in this repo, which is
  self-consistent as of `e5e4ee7`.
- ⚠️ **Do not publish the bake-off as "size matters".** The candidates differ in
  vendor, architecture and training corpus as well as parameter count (only
  quantization is matched). It is a *selection* result. A size ablation needs
  two sizes in one family (`gemma3:4b` vs `gemma3:27b`) and was not run.
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
- ⚠️ **Carried:** `LEARNINGS.md`'s `mean token-F1 0.711` predates D14 and reads
  **0.7619 over 14, refusal rate 6.7%** under the settled convention — quote it
  that way. (How to *write* the scorer is now design, not a flag: `docs/plan.md`
  4b amendment 2026-08-29.)
- ⚠️ Carried: **probe scripts live in gitignored `scratchpad/`** while
  `LEARNINGS.md` cites their numbers — decide whether the ones behind published
  figures belong in the repo. And **`tests/test_finetune.py` does not exist**
  despite being cited in `load_triples`'s docstring.
- Minor: `--label` on a sheet that does not exist yet raises a raw
  `FileNotFoundError` rather than `load_generations`'s "draw it first" message.

## Pick up here

1. **Write and test the `token_f1` correctness scorer** —
   [#5](https://github.com/JiamanBettyWu/retrieval-lab/issues/5), a hard
   prerequisite for 4b (`docs/plan.md`, 4b amendment 2026-08-29). Read the issue
   for the full definition of done; the short version is that it does not exist,
   refusals leave the denominator per D14, and the normaliser needs tests rather
   than an eyeball. Branch `feat/issue-5-token-f1-scorer`.
2. **Run 4b — the gold-padded ceiling** (`docs/plan.md` 4b/4d). Read it *before*
   running the config comparison: if gold context barely beats baseline context,
   retrieval was never the bottleneck and 4d's expected effect is smaller still.
   Hold `n_context` fixed; 4b is two runs (gold-padded, gold-only) and now
   reports three measurements (refusal rate, grounded rate, correctness).

Standing note: **widening the label set past n=16 is the highest-value spend on
judge trust** — the CI `[+0.09, +1.00]`, not the judge count, is what limits
every Phase 4 claim.
