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
