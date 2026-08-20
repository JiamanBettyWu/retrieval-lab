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
