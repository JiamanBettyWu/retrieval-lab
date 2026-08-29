# Session log

Reverse-chronological journal — one dated entry per working session: what got
done, what was decided and why. History only; for what to do next see
[TODO.md](TODO.md).

Older entries archived in [`sessions/`](sessions/): 2026-07-15 through
2026-08-08 (Phase 0 scaffold through Phase 1 landing) in
[`sessions/2026-07-15--2026-08-08.md`](sessions/2026-07-15--2026-08-08.md);
2026-08-12 through 2026-08-17 (Phase 2 design through the Phase 4 dataset move)
in
[`sessions/2026-08-12--2026-08-17.md`](sessions/2026-08-12--2026-08-17.md);
2026-08-19 through 2026-08-28 (the output contract through the 4c.2 bake-off
harness) in
[`sessions/2026-08-19--2026-08-28.md`](sessions/2026-08-19--2026-08-28.md).

---

## 2026-08-28 (the bake-off runs, and the small judge fails in a way no format check can see)

**Phase 4c.2 is measured.** `--smoke` first out of habit (`mistral-small` 4/4,
`gemma3:4b` 4/4), then `--bakeoff` against the labelled seed-1 fixture under
rubric v2. `mistral-small` takes it at **κ 0.586** on `grounded` (n=16, 95% CI
[+0.09, +1.00], 13/16 raw agreement, 42.45 s/call); `gemma3:4b` lands at
**κ −0.257** (5/16, 6.77 s/call). The 0.84 gap is far outside the tie band that
was pre-registered before any number existed, so throughput never got to decide.
Numbers and confusion matrices in `LEARNINGS.md` (2026-08-28); the decision and
its reasoning are posted on
[#2](https://github.com/JiamanBettyWu/retrieval-lab/issues/2) rather than
repeated here.

**The finding is not "bigger is better", and the entry says so twice because it
is the thing most likely to decay on retelling.** `gemma3:4b` emitted
well-formed, parseable rulings on 30/30 rows and still landed below chance — it
ruled `false` on 8 of the 11 rows labelled `true`, so it has the *shape* of a
strict groundedness judgement without the discrimination. Both candidates had
passed `--smoke` 4/4 hours earlier. **Parse rate is the cheap monitor one would
actually automate, and it cannot distinguish these two models.** Only labels can.
That is the same class as the 0.0000 NDCG and the leaked format template.

**The bake-off does not isolate size.** Surfaced by a question about what the
result actually shows, and worth recording as a design lesson rather than a
footnote: the two candidates differ in vendor, architecture family and training
corpus as well as parameter count (`gemma3:4b`, Google, 4.3B vs `mistral-small`,
llama-family, 23.6B; only quantization is matched at Q4_K_M). Two points
differing on four axes support a *selection* result and not a causal one. A size
ablation needs two sizes inside one family (`gemma3:4b` vs `gemma3:27b`) and was
not run. D13's title asks whether the judge needs to be 27B; the honest answer is
"no, and this design cannot tell you why."

**The v1/v2 generator split is closed.** `n100 seed0` was still a prompt-v1
batch while the fixture had been v2 since seed 1, so every refusal-curve number
sat on a different generator than the κ. Regenerated (~10 min, cache-additive,
no relabelling — the curve strata come from `gold_in_context`, not from labels).
The curve stays monotonic and the 2/2 and 1/2 intervals still do not overlap,
which is the property the separation argument needs; only the partial-gold bucket
moved (53.7% → 63.4%), and its intervals overlap heavily, so **the shift is not a
claim**. Recomputing the v1 batch with the same script reproduces 2026-08-23
exactly, which is what makes the v2 numbers trustworthy rather than merely new.

**A worked example dissolved, and only the anchored half noticed.** Seed-1 row 1
(Edmund Mortimer) is named as the type case for over-refusal in `fixture.py` and
in `LEARNINGS.md` 2026-08-23. Under v2 that row *answers* — v1 named the answer
and then talked itself out of it. The `raw_sha` anchoring did its job and the
labels invalidated loudly; the prose citing that row went stale silently and is
still stale. The current type case is the Hund's-rule row (`5ae24b16…`, gold 2/2,
`refusal_ok=false`), whose rationale names Friedrich Hund and then refuses on the
question's exact phrasing. Anchoring artefacts is cheap and standard; anchoring
the prose that discusses them is neither.
