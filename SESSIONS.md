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

## 2026-08-28, later (the docs catch up, and the plan's Phase 4 was still an NFCorpus experiment)

Second half of the same session, after the first handoff (`b4e2d05`). No code
changed; the whole stretch was making the written record match what shipped.

**`README.md` got its Phase 4 section** (`fb6f336`) — judge validation table,
the refusal curve by `gold_in_context`, and the layout block updated for the
four Phase 4 modules. **Phase 4 deliberately did not get a row in the ablation
table**, and the table now says why: its columns are retrieval metrics, and a κ
sitting in a column headed `NDCG@10` implies a comparability that does not
exist. The three carried caveats are stated in place rather than left to
`TODO.md` — which `grounded` wording produced the κ, that the over-refusal
figure is a lower bound, and that the bake-off is a selection result and not a
size result. Also noted that `data/labels/` is *not* rebuildable, since it sits
beside a paragraph telling readers everything is safe to delete.

**Issue #2 closed** with a scope note kept next to the decision: the title's
question is answered in the practical sense (no, a 27B judge was not needed) and
the mechanism is unidentified, because the candidates differ in vendor,
architecture and corpus as well as size.

**`docs/plan.md` was swept for drift, in two passes** (`8e7f702`, `c7688fd`).
The first pass caught the diagram naming Claude as the generator, the tech-stack
entry still specifying Anthropic models, the module list saying Phase 4 adds two
modules when it added four, and all three "Live for Phase 4" questions being
settled. The second pass was prompted by Betty spotting that **the Phase 4
design block still said NFCorpus** — the 2026-08-17 dataset move (D12) amended
the milestone above it and missed the design bullets entirely, so the Scope
bullet had been describing an experiment on the wrong dataset for eleven days.
Amending it turned up two more: `Scored axes` was aspirational (it lists answer
relevance, never built and now marked dropped, and correctness, reinstated
2026-08-17 and still not built), and "cheap on a small fast model" did not
survive contact — the judge that passed the fixture runs at 42.45 s/call, so the
config sweep is a ~5.9 h overnight run.

Everything went in as dated amendments rather than edits, per `CLAUDE.md`. The
one worth re-reading is on **Known risks**, because it scores the plan against
itself: both named risks fired, and the second fired through a mechanism the
plan did not anticipate. The bullet trusted an ungrounded fixture to catch a too-
coarse judge before the spend; that fixture was never built, and it would not
have mattered — `gemma3:4b` parsed 30/30 rows cleanly. Hand labels caught it.
**The risk register was right about what could go wrong and wrong about what
would notice.**

**4d exists now** (`ae7fe11`). The config comparison — the milestone the whole
phase exists for — had never been designed; it was implied by the Scope and
Deliverable bullets and nothing else. It specifies running **4b before 4d and
reading it first**: if gold-padded context barely beats baseline context, the
expected effect across four configs spanning `0.2725`–`0.3412` is smaller still,
because those configs differ by less than the gap to perfect evidence. Cheaper
to learn that from two runs than five.

**D10's last half resolved — the second judge is deferred**, and the reasoning is
in `docs/plan.md` under 4d rather than here, since that is where it will be read.
Short version: judge–judge agreement is not evidence of correctness, and this
repo holds the demonstration — `gemma3:4b` was perfectly consistent and
anti-correlated with the truth, so two such judges would have produced a
reassuring agreement column and no information. The constraint on any future
revival is recorded with it: use disagreement as a diagnostic, never filter
contested rows out, because disagreement rate varies with config difficulty and
filtering would drop more rows from the worse configs and flatter them. With no
decisions left, `TODO.md` lost its "Open decisions" section entirely.

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
