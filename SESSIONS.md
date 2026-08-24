# Session log

Reverse-chronological journal — one dated entry per working session: what got
done, what was decided and why. History only; for what to do next see
[TODO.md](TODO.md).

Older entries archived in [`sessions/`](sessions/): 2026-07-15 through
2026-08-08 (Phase 0 scaffold through Phase 1 landing) in
[`sessions/2026-07-15--2026-08-08.md`](sessions/2026-07-15--2026-08-08.md);
2026-08-12 through 2026-08-17 (Phase 2 design through the Phase 4 dataset move)
in
[`sessions/2026-08-12--2026-08-17.md`](sessions/2026-08-12--2026-08-17.md).

---

## 2026-08-23 (labelling starts, and the prompt turns out to be leaking)

The stretch after the handoff commit `def922d` below — kept as a separate entry
because that one is already history and this is what happened next. Betty
started labelling and the fixture immediately paid for itself: two findings came
out of reading rows by hand that no script was going to surface. Both are
recorded in `LEARNINGS.md` (2026-08-23); the narrative is here.

**Row 1 taught the polarity lesson the tooling was built to prevent, and then
taught a better one.** `refusal_ok` was misread as grading the rationale rather
than the refusal decision — despite the on-screen polarity line existing for
exactly that reason. Renaming to `refusal_justified` was offered and declined;
the referent was clear once stated. Worth noting the near-miss: this is the one
corruption channel no checker can close, since inverted labels are
in-vocabulary, internally consistent, and pass `--check` cleanly.

The row itself turned out to be the more interesting artifact. It is a refusal
on *full* gold context where the model misread the passage that answered the
question, and the reasoning behind calling that a distinct failure mode — not
conservatism, a comprehension error — is in the LEARNINGS entry. It also sits
right on the boundary of the over-refusal rule, which is why the rule's
lower-bound caveat is now written into `fixture.py` rather than remembered.

**The rubric gained a clause it had to be argued into.** "Names the answer
correctly" needed a referent: correct *by the passages*, not by HotpotQA's gold.
The two come apart precisely on zero-gold rows, where grading against gold would
mark the model down for information it never had, and the sheet's blindness to
the gold answer is what keeps a labeller on the right question. Written into the
`LABEL_VALUES` block.

**The prompt leak, and the decision it forced.** `Cite passages as [1], [2]` was
appearing verbatim in rationales; a count across all four batches on disk put it
at 19/175, concentrated in rows that otherwise cited nothing. Betty fixed
`build_prompt` (inert template placeholder, citation rule moved into `Rules:`)
and `PROMPT_VERSION` went to `v2`.

The decision was scope. A version bump invalidates every cached generation and
therefore every hand label anchored to it, so the choice was: fix now and lose
three labels, or label an evening under a contaminated prompt and select a judge
on a generation distribution that was about to be abandoned. Fixing won on the
grounds that labels are the one artifact in this repo that cannot be
regenerated, so they should be drawn from the prompt version that will ship.
**Only `seed 1` and `seed 2` were re-run** — Betty's call, and a good one:
`n100 seed0` stays v1, its files are intact, `prompt_version` is in the cache
key, and re-running it later is ten unattended minutes. The standing constraint
is recorded in `TODO.md`: do not publish a κ from v2 labels beside refusal
numbers from v1 generations.

Post-fix the leak is 0/60 across both regenerated seeds. Refusal moved 53.3% →
46.7% on seed 1 and held at 43.3% on seed 2 — noise plus whatever the rewording
bought, not a result. Pooled strata are 33 answered / 27 refused with 11
answered-on-partial-gold, so the class balance that motivated the second draw
survived the regeneration.

**One bug the regeneration exposed.** With v1 and v2 files for the same draw
both on disk, `overlap_with` started warning that the fixture "overlaps earlier
batches — held-out is not clean", naming the v1 twin. False positive:
`sample_queries` keys only on `(n_queries, seed)`, so a prompt bump re-asks the
same questions by construction. Fixed with `same_draw()`, which compares
filenames with the `__v<N>__` token removed. The reason it was worth fixing
immediately rather than noting: a warning that fires on a correct state is one
you train yourself to scroll past, and the next real overlap scrolls past with
it.

Also this stretch: `INAPPLICABLE_AXIS` pre-fills each row's non-applicable axis
from the `refused` flag, so the sheet asks exactly one question per row (60
decisions rather than 60 decisions plus 60 hand-typed `"n/a"`s), and
`AXIS_QUESTIONS` renders each axis's full polarity sentence above the prompt.
Suite is 115 tests. Both sheets sit blank at 30/30 awaiting decisions.

---

## 2026-08-23 (the fixture gets a rubric, and labelling gets a tool)

A later stretch of the same day as the entry below. Phase 4c.1 went from "does
not exist" to "two sheets on disk, blank, with the rubric written down" — plus
`src/retrieval_lab/fixture.py` (new, untracked at time of writing) and
`tests/test_fixture.py` (26 tests; suite now 114).

**`mistral-small` is pulled, so D13 is unblocked.** The pick was made on a disk
reading that turned out wrong: `df` reported 18GB free mid-session, which ruled
out a 27B at Q4 (~17GB) and made mistral-small (14GB) look like it would leave
4GB. After the pull finished the same command reported 48GB free — the low
readings were taken while the ollama blob cache held partial layers plus temp
copies. Disk is not a constraint on the bake-off; the recorded reasoning for
preferring a smaller candidate is, and it still stands on D13's own terms
(judging is rubric classification, so parameter count may buy little).

**Two draws, because the first one was thin where it mattered.** `--seed 1`
(n=30) refused 16/30 = 53.3%, against 34.0% at n=100 — not a contradiction, the
CIs overlap, but it left only **3** answered-on-partial-gold rows, the stratum
where faithfulness is genuinely at stake. `--seed 2` was drawn for that reason
and landed 8 more, plus the first two answered-on-*zero*-gold rows the fixture
has seen. Pooled over 60 rows: 31 answered / 29 refused, which is close to ideal
class balance for a κ. The refusal rate across three draws of the same config —
53.3% / 43.3% / 34.0% — is itself worth remembering when reading any single
small batch. `overlap_with` confirmed no query id is shared with any other
cached batch, so held-out is a checked fact rather than a property assumed from
the seed.

**The labels get anchored to the exact text they describe.** Generation is
nondeterministic even at temperature 0, so a `--refresh` on a batch would leave
hand labels silently pointing at answers nobody read — well-formed, wrong
meaning, nothing raising, the same shape as a leaked `corpus_id`. Every sheet
row therefore carries `raw_sha`, and `load_labels` raises on a mismatch. The
writer refuses to clobber an existing sheet, and builds every row *before*
opening the file: the first version truncated on open and left a 0-byte sheet
when `blank_labels()` raised, which is exactly how `--force` would destroy an
evening's work and then fail.

**The rubric (D14-adjacent, settled, not a live decision).** Two axes, binary
both: `grounded` for answered rows, `refusal_ok` for refused ones. Refusals get
their own axis because a refusal makes no claims and is therefore trivially
"grounded" — scoring it there would have turned 29 of 60 rows into free
agreement that measures nothing. Each row's inapplicable axis is pre-filled
`"n/a"` from the `refused` flag, so there is exactly one decision per row and
those rows leave that axis's κ denominator rather than counting as agreement.

The boundaries were written *before* row 1, because one invented at row 40 makes
rows 1–39 a different rubric and the κ silently mixes two scales. `grounded` is
strict: every clause supported by the passages it cites, so both probe cases in
LEARNINGS.md are `false` — the 7th Sea decorative citation, and the harder
Vienna case (reasoned from Gluck's biography page, genuinely in context, toward
a conclusion that passage does not establish). `refusal_ok` is `false` only when
the rationale itself names the answer correctly and the model refused anyway.
**That test is deliberately narrow and makes the axis conservative** — a refusal
whose rationale never names the answer scores `true` even if the answer was
derivable — so any over-refusal rate it produces is a LOWER BOUND and the README
must say so. It buys rater agreement ("does the rationale name it" is checkable)
at the cost of undercounting.

**Labelling got a tool, for a boring reason and a non-boring one.** Boring:
hand-editing 6,000-character JSONL lines sixty times is how a row gets corrupted
at 11pm. `--label` renders one row wrapped, asks the single question that row
needs, and writes back atomically after *every* answer, skipping rows already
decided so it resumes. Non-boring: the prompt renders the full polarity sentence
above the y/n, because inverted labels are the one corruption channel no checker
can close — they are in-vocabulary and internally consistent, so `--check`
passes, κ computes, and the number is simply wrong. `render_item` also withholds
`gold_in_context` until after the answer: seeing a 0 while judging `refusal_ok`
is anchoring bait that would make the label a function of the qrels rather than
of the passages.

**Two mistakes worth recording.** The demo of `--label` was run twice, and
because the tool resumes, the second run labelled row 2 rather than repeating
row 1 — both trial labels were swept back to blank and the sheets verified at
30/30 awaiting decisions, but it is a reminder that the tool writes on every
keystroke by design. And `--label` on a sheet that does not exist yet dies with
a raw `FileNotFoundError` from `read_text` instead of the "draw it first"
message `load_generations` gives.

---

## 2026-08-23 (the refusal gate turns out to be calibrated)

The same working session as the 2026-08-22 entry below — the clock rolled past
midnight partway through, and the cache-key change recorded there belongs to
this stretch. Read the two together.

**The n=100 batch replicated everything and resolved the number that looked
alarming.** Refusal rate 33.3% → 34.0%, parse misses 0/100, yes/no 7/100 against
a 6.2% base rate. Format adherence at ten passages now stands at 115/115 across
both batches. The trial's `0/15` yes/no had read like the rule under-firing and
was simply small-n. The 95% CI narrowed from 43 points to 18, as the sizing
arithmetic said it would.

**Splitting refusals by gold-passage presence is the actual result** — 9.8% /
53.7% / 87.5% as retrieval delivers 2, 1 or 0 of HotpotQA's two gold docs, with
the first two intervals non-overlapping. As retrieval degrades this generator
declines rather than invents, which is the safe half of the failure space and
precisely the question D11 kept the refusal branch to answer. Numbers and the
fixture-supply consequence are in `LEARNINGS.md` (2026-08-23), commit `37b9147`.

**Betty caught a real flaw in how that was framed, and it changed a design.** The
stratified split was offered here with a confounding caveat: those three buckets
hold *different* queries, so retrieval quality is entangled with question
difficulty. Her reply was that generating over the same query sample under an
oracle-retrieval config and a real one answers the causal question directly —
which is right, is what `docs/plan.md` already specifies (four retrieval configs
plus the ceiling over one fixed seeded sample), and makes the split a
*descriptive* diagnostic rather than the evidence.

Pushing on that surfaced a trap in 4b that would have spent the paired design's
whole advantage. Read literally, "generate from the qrel-perfect context" hands
the model HotpotQA's 2 gold docs while every real config gets 10 retrieved
passages — varying evidence presence, context length and distractor count
together, so "refuses less with gold context" would be indistinguishable from
"refuses less with a short, clean prompt". The trial batch's single over-refusal
is direct reason to care: both gold passages were present, at ranks 2 and 9 with
seven distractors between them, and the model failed to link them. **4b now runs
two context builders** — gold-padded to `n_context` with non-gold docs from that
query's retrieved list (the controlled contrast, and what the config comparison
is read against) and gold-only (the true ceiling, never quoted as the
counterfactual). Recorded as a dated amendment on 4b in `docs/plan.md` rather
than repeated here; commit `85a5db0`.

**Sizing the run was itself decided on numbers rather than habit.** `--n-queries
50` was a placeholder written before anything had been measured. At n=15 refusal
rate carries a CI 43 points wide and two fixture strata hold one example each;
paired config comparisons need roughly 90–120 queries to resolve a 15-point gap
against ~263 per arm unpaired, because pairing controls query difficulty. Hence
100. The generation cache key change that made a held-out fixture draw possible
is described in the 2026-08-22 entry.

**Not started, deliberately:** the judge-candidate pulls. They are pure
wall-clock, they block D13 and therefore D10, and `ollama list` still holds only
`qwen3:8b` — which D10 excludes as a judge on self-bias grounds. Worth starting
before anything else next session.

## 2026-08-22 (the last three stubs close, and the first real batch runs)

**D14 settled: refusals do not enter the token-F1 denominator.** Option B —
correctness is reported over non-refused queries, with **refusal rate published
alongside it** as a per-config number. The reasoning is that the two quantities
answer different questions and averaging them together destroys both: refusal
rate is already a headline figure (it is the whole reason D11 kept the refusal
branch — does worse retrieval make the model *decline* or make it *hallucinate*,
because only one of those is safe), and a correctness number that silently
absorbs declines cannot distinguish a model that got the answer wrong from one
that correctly said it did not know. Option A's single figure was defensible and
was rejected on the grounds that it conflates two failure profiles the project
exists to separate. The cost is accepted explicitly: **the denominator now varies
per configuration**, so any README row quoting correctness must also state the n
it was computed over, or the column is not comparable across rows.

**The convention has a retrospective consequence, and it is small.** The
gold-context probe's `mean token-F1 0.711` was computed over 15 questions, one of
which is the Alceste refusal scoring 0. Under B that query leaves the numerator
and the denominator both: **0.7619 over 14, with a 6.7% refusal rate reported
next to it.** That figure lives in `LEARNINGS.md` as a probe result rather than
as a published claim, so nothing needs reverting — but it is the first number the
new convention touches, and it should be quoted the new way from here on.

**`parse_answer` landed, and the prompt went to v1.** The sentinel was briefly
changed to `__REFUSED__` and changed back: `INSUFFICIENT_CONTEXT` was measured at
8/8 inside its tags against 5/8, and keeping the model's word distinct from the
pipeline's `REFUSAL` is what keeps "the model declined" separable from "my rule
classified this as a decline". The prompt now pins both things it had left
ambiguous — the sentinel goes *inside* `<answer>`, and a refusal still emits a
rationale, which is the only reason the Alceste case could be diagnosed as
question-phrasing rather than missing evidence.

**The parser's real design is its miss contract, and one refactor of it leaked.**
`("", raw)` exists so `should_refuse` can see text the parser did not recognise —
a sentinel emitted without tags, for instance. Splitting the answer and rationale
lookups into independent `try` blocks fixed a good answer being discarded by a
missing rationale, and simultaneously opened a narrower hole: a rationale that
*parses* while the answer does not would travel back with the parsed rationale
instead of `raw`, dropping the sentinel before `should_refuse` ever saw it, and
`validate()` would then book a correct refusal as parser drift. Closed with an
early return, and pinned by `test_a_partial_parse_still_returns_raw`.

**A test docstring claimed the wrong regression and mutation-testing caught it.**
That test was documented as guarding against re-merging the two `try` blocks. Run
against an actually-merged implementation, it passes — the answer lookup throws
first, so the merged version returns `("", raw)` correctly. What it really guards
is the split-without-early-return. The claim was corrected against the run rather
than against intuition. The suite does catch both bad refactors, but via two
different tests. Nine tests added, 69 → 78 green.

**`should_refuse` closed the last stub, and its signature shrank first.** It was
declared as `(question, docs, parsed)`, with the first two reserved for a
pipeline-side ablation — a gate on retrieval scores that refuses before
generating. Those arguments cannot support it: `context_for` sorts by
`results[query_id]` and then hands back only `{title, text}`, so the scores never
arrive. Trimmed to `(parsed)` with a docstring note that the ablation belongs in
`generate_one`/`main` where `results` is still in scope. A signature advertising
a capability it does not have is the same genre of problem as a docstring citing
a test that does not exist.

**The gate's one real bug was the direction D14 had just made expensive.** The
first version consulted the rationale unconditionally. The prompt hands the model
the sentinel string, and models narrate the rules they are given — so a clean
`<answer>Nelson County</answer>` alongside a rationale merely *mentioning*
INSUFFICIENT_CONTEXT returned True. That overwrites a parsed answer with REFUSAL,
exempts the row from `validate()`'s parse-miss gate, inflates refusal rate, and
(as of this morning's D14) drops the query from the correctness denominator: two
headline numbers move and neither moves informatively. Fixed by consulting the
rationale only when the answer is empty — which is exactly when `parse_answer`
guarantees the second slot holds `raw`. SENTINEL became a shared constant so the
prompt and the detector cannot drift; rewording one without the other would take
refusal rate to 0.0%, which reads as a result. The rendered prompt was checked
byte-for-byte after the refactor, so PROMPT_VERSION stayed at v1 — the version
tracks what the model reads, not what the file says. Commits `34029bc`, `dbe48d8`.

**Mutation-testing earned its keep twice, once against a claim of my own.** A
docstring asserted that `test_a_partial_parse_still_returns_raw` guards against
re-merging the two `try` blocks. Run against an actually-merged implementation,
it passes — the answer lookup throws first, so the merged version returns
`("", raw)` correctly. What it really guards is the split-without-early-return.
The claim was corrected against the run rather than against intuition. Both
`should_refuse` tests were then checked the same way before being believed.
69 → 85 tests.

**The first real batch ran, and the headline number was misleading on its own.**
Fifteen questions, ten retrieved passages, prompt v1: format held 15/15 at ~1,200
tokens (the 45/45 was measured at two passages), yes/no fired 0 times against a
6.2% base rate, and refusal rate came back **33.3%** against 6.7% on gold
context. Splitting the five refusals on how much gold retrieval actually
delivered says four were correctly calibrated to missing evidence — the first
direct evidence for the thing the refusal branch exists to measure. The fifth had
both gold passages present at ranks 2 and 9 and failed to link them, so refusal
rate is not a pure retrieval signal; reading it as one folds multi-hop reasoning
failures into a retrieval number. The same batch produced the opposite failure —
zero gold passages retrieved, a fluent cited answer drawn from an adjacent page,
wrong — which is the case a correctness-only metric misreads as a weak generator.
Numbers, the two case studies and the 4c population table are in `LEARNINGS.md`
(2026-08-22), recorded there rather than repeated here. Commit `7e39f40`.

**The generation cache key gained `n_queries` and `seed`, decided after the
handoff.** Sizing the next run raised the question of how big it should be, and
the numbers settled it: at n=15 refusal rate carries a 95% CI 43 points wide, and
two of the fixture strata hold a single example each. Paired config comparisons —
which is what Phase 4 does, on identical queries, the thing `cache.py` exists to
guarantee — need roughly 90–120 queries to resolve a 15-point gap, against ~263
per arm unpaired. So the target became ~100 rather than 50, and a held-out
fixture draw became worth having. That was impossible under the old key: it
omitted both `n_queries` and `seed`, so one config owned one filename and a
second draw overwrote the first. The alternatives considered were accepting the
overlap between fixture and scored set (mildly circular — the fixture selects a
judge that then scores those same items) and copying files aside by hand
(provenance by `cp`, which is what the caching discipline exists to prevent).
The key change retires the collision hazard filed in the handoff instead of
documenting it. The existing 15-answer file was migrated to the new name rather
than regenerated, and re-running the same config confirms a cache HIT. Two tests
added; 85 → 87.

**Raised by that batch and carried forward as D15:** whether the README reports
refusal rate stratified by gold-passage presence. The 33.3% figure is honest and
uninformative; the split is informative but reads `qrels`, which is fine for
analysis the way `oracle.py` does it and not fine inside the pipeline.

## 2026-08-19 (the output contract gets designed, argued with, and measured)

Phase 4a's `build_prompt` landed — hand-written by Betty, reviewed and probed
here — and with it the output contract that `parse_answer`, `should_refuse`,
token-F1 and the 4c judge rubric all have to agree with. Nothing has been
generated at scale yet; what this session bought is a contract that was
*measured* rather than asserted. Findings are recorded in `LEARNINGS.md`
(two entries dated 2026-08-19); this entry keeps the narrative.

**The pick-up order got amended before any work started.** `TODO.md` had the
next steps as fixture → judge bake-off → stubs. That ordering can't hold: the
4c.1 fixture is a set of hand-labelled `(question, passages, answer)` triples,
and authoring it before the output format exists means labelling answers shaped
differently from what the pipeline will emit — a judge validated on the wrong
distribution. `TODO.md` item 3 already conceded the dependency in its own text
("`build_prompt` first, since the output format it establishes is what
`parse_answer`, token-F1 and the judge rubric all have to agree with"), so this
resolves an internal tension rather than reversing the handoff. The stratifica-
tion argument points the same way: drawing the ~50 labels across correct and
incorrect answers requires generations to stratify over. New order — output
contract → stubs → small generation batch → fixture authored from real output →
D13 bake-off. The fixture is still the gate; it just cannot be built first.

**`scratchpad/probe.md` was not in the repo.** `TODO.md` cited it as the
fixture's seed material. It turned out to survive only in a *previous session's*
ephemeral scratchpad under `/private/tmp/claude-501/…/eae99233-…/scratchpad/`,
along with `probe_datasets.py`. Both were copied into this session's scratchpad,
which is equally ephemeral. Still homeless — see `TODO.md`.

**`think: True` was proposed as a free rationale and rejected on evidence.** The
idea: qwen3 is a reasoning model, so let it think and read the reasoning as the
rationale rather than asking for one. Three findings killed it — Ollama returns
thinking in a separate `thinking` field that never reaches `raw`; the content is
meta-commentary about the task rather than evidence, and carries no citation
markers, so the ungrounded-citation failure mode would be invisible in it; and
chain-of-thought is not a faithful record of what produced an answer, so judging
it would measure whether the narration is grounded rather than the answer. It
survives as a possible correctness ablation, which would need `think` added to
`generation_cache_path` first — it is absent today, so running it as a config
would silently collide with the `think: False` file.

**The refusal sentinel was decided by probe, not by taste.** The natural move
was to have the model emit `REFUSAL = "__REFUSED__"` directly, one constant end
to end. Eight unanswerable questions × three candidate sentinels showed
`__REFUSED__` dropping the `<answer>` tags 3 times in 8 while the natural-
language sentinels went 8/8 — and a dropped tag routes a *correct refusal* into
`validate()`'s parse-miss counter. Resolved as two strings: the model emits
`INSUFFICIENT_CONTEXT`, `should_refuse` recognises it, the pipeline normalises
to `REFUSAL`. That keeps the split the dataclass already pays for — `raw` is
what the model said, `answer` is what the pipeline concluded — without which
"the model declined" and "my rule classified this as a decline" are the same
bytes.

**Contract settled:** tagged blocks, rationale-first then answer, `[n]`
citations 1-indexed, titles rendered into each passage, question last, answers
"a few words, or yes/no", `INSUFFICIENT_CONTEXT` inside `<answer>`. Zero-indexed
citations were considered and rejected: every citation convention in the model's
training data is 1-indexed, and a model emitting `[1]` for the first passage
against code mapping `[1] → docs[1]` is an off-by-one that surfaces as
*hallucination in the metrics* rather than as an error. `enumerate(docs, 1)` in
the renderer and `docs[n-1]` in the future `judge.py` are the two halves.

**Two bugs in the draft, both silent.** `f"…{doc["title"]}…"` — nested same-type
quotes, which is PEP 701 and therefore Python 3.12+ only, while `pyproject.toml`
declares `requires-python = ">=3.10"`; the local venv is 3.12.13 so it ran fine
here and would be a `SyntaxError` anywhere else. And then `{doc['title'],}` — a
stray trailing comma making it a tuple, rendering every passage as
`[1] ('Ida (sword)',). …`. No error, valid Python, wrong prompt — and it lands
on the field where gold answers frequently live (`Animorphs`, `YG
Entertainment`), so the model could echo the parens into `<answer>` and crater
token-F1 while reading as a weak generator. Both fixed by Betty.

**The grounding wording was tested and the objection refuted.** The concern
raised here was that *"Only answer if the passages fully support it"* would
break on HotpotQA, where the answer is supported by two gold passages jointly
and by neither alone. `scratchpad/prompt_probe.py` turned that into a
measurement — gold-only context, so retrieval is perfect by construction and any
refusal is caused by wording alone — across three wordings differing in one rule
line. False refusals came back 1/15, 1/15, 0/15. The model combines across
passages without being told it may; the wording stands as written. Numbers and
the inspection of the single refusal are in `LEARNINGS.md`.

**The result that matters was incidental to the test.** Of six answers missing
exact match on gold context, four were not wrong — an article, a possessive, a
pluralisation, and gold answers carrying literal punctuation (`'"Alceste"'`).
The correctness scorer's normaliser is therefore load-bearing rather than a
detail, and the gap reads as a weak generator. This constrains code nobody has
written yet. **Amended after the handoff:** the first draft of this claim put
normalisation's value at 60% → ~87%; implementing the metric and running it gives
**exact-match 0.600 → mean token-F1 0.711**. Only the article case dissolves
fully, the possessive earns partial credit, and `plant genera` vs `genus of
plants` scores zero because token-F1 is surface overlap. Worked numbers in
`LEARNINGS.md` (2026-08-19).

**Left open deliberately:** whether refusals enter the token-F1 denominator.
Raised twice, deferred twice, now recorded as D14 so it stops evaporating — it
changes what the README's correctness number means and should be settled before
any number is published rather than after.

Tests stayed green throughout (69 passed). `LEARNINGS.md` grew from 604 to 737
lines across two entries. `SESSIONS.md` was rotated to
`sessions/2026-08-12--2026-08-17.md` at 642 lines.
