# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-19 (latest session):** **Phase 4a has an output contract, and it
was measured rather than asserted.** `build_prompt` is hand-written and probed —
tagged blocks, rationale-first, `[n]` citations 1-indexed, titles rendered,
`INSUFFICIENT_CONTEXT` inside `<answer>`; the grounding wording survived a
false-refusal test on gold context (1/15) and the format held 45/45.
`parse_answer` and `should_refuse` remain stubs but are now unblocked. Nothing
has been generated at scale, so no generation cache exists. 69 tests green;
`src/retrieval_lab/generate.py` is **uncommitted**. Detail in
[SESSIONS.md](SESSIONS.md); findings in `LEARNINGS.md` (2026-08-19 ×2).

```bash
pytest                                                # 69 tests, ~9s, no download
python -m retrieval_lab.generate --n-queries 50       # Phase 4a — needs `ollama serve` + the two remaining stubs
python -m retrieval_lab.evaluate --dataset nfcorpus   # baseline 0.3159 (Phase 1 rerank: 0.3412)
```

## Working mode (carry this forward)

Betty hand-writes the ML; Claude writes plumbing (argparse, logging, paths,
sanity checks) and **coaches: concept first, then she writes it**, then review.
Demonstrating bugs empirically is wanted; silently fixing them is not.

## Open decisions

### D5: How to handle dense queries, where reranking actively hurts? (parked)
- **Context:** on the 86 queries with 11+ relevant docs the cross-encoder
  *subtracts* 1.06 NDCG points while improving MRR. Phase 2 did not dissolve it.
- **Options:** A) **Score blending** B) **Routing** — needs a detector; "dense"
  isn't knowable without the qrels C) **Do nothing**
- **Recommendation:** A (the blend weight earns a README row), but **only if you
  return to retrieval** — Phase 4 is the direction and this is Phase 1 cleanup.
- **Blocked on:** nothing; needs no training.

### D10: Which generator, and which judge? (Phase 4) — **generator settled, judges still tentative**
- **Settled:** generator `qwen3:8b`, local via Ollama. Two judges, not one, so
  their agreement becomes a result reported next to human κ.
- **Still tentative:** **primary judge Claude Sonnet 5** (`claude-sonnet-5`) and
  **second judge `Qwen3.8-27B`** (local). Neither has cleared the 4c.1
  grounded/ungrounded fixture, which is the gate — a judge swapped after the
  fact voids every generation already scored under it.
- **Constraint that survives any swap:** the local judge must not be a qwen3
  model. The generator is `qwen3:8b`, and a shared pretraining corpus and
  post-training recipe is exactly the self-bias D10 exists to dodge.
- **Amended 2026-08-17 (cache key):** generations and judgements key
  separately, so swapping a judge does not discard generations — pinned by
  `tests/test_generate.py::test_swapping_a_judge_does_not_invalidate_generations`.
- **Blocked on:** the fixture existing (see D13 and "Pick up here").

### D13: Does the local judge need to be 27B, or is a smaller/different model better suited?
- **Context:** `Qwen3.8-27B` was picked on "bigger is safer", never measured,
  and it is unverified whether a dense 27B (~16–17GB at Q4) runs usably here.
  The principle to test: judging is *classification* against a rubric, and
  everything it rules on is in the prompt — so parameter count may buy little.
- **Options:** A) **Keep `Qwen3.8-27B`** — as recorded, unmeasured B) **Bake off
  4–5 candidates against the 4c.1 fixture** — the fixture is small, so five cost
  about what one does C) **Drop the local judge**, run Sonnet 5 alone
- **Recommendation:** B — decide on the fixture's evidence, not a spec sheet.
- **Blocked on:** the 4c.1 fixture, which does not exist yet. Also note
  `ollama list` currently holds only `qwen3:8b`, which is excluded as a judge —
  every candidate needs a multi-GB pull, worth starting in the background early.
- **If B:** score each on fixture separation *and* throughput, then re-open
  D10's second-judge slot with numbers. **If C:** D10's "two judges" half is
  withdrawn and the κ section loses its agreement column.

### D14: Do refusals enter the token-F1 denominator? (new, 2026-08-19)
- **Context:** a refusal against gold `Animorphs` scores 0, which reads as
  "wrong" when it may be "correctly declined". Raised twice this session and
  deferred twice; it changes what the README's correctness number *means*.
- **Options:** A) **Correctness over all queries** — refusals count as wrong;
  one number, but it conflates two failure profiles B) **Correctness over
  non-refused queries, refusal rate reported alongside** — separates
  "hallucinates" from "declines", which is the distinction D11 kept the refusal
  branch to measure, at the cost of a denominator that varies per config
- **Recommendation:** B, precisely because the per-config refusal rate is
  already a headline number and entangling it with correctness makes both
  unreadable — but A is defensible if the README wants one figure.
- **Blocked on:** nothing. Settle it before any correctness number is published.

## Needs attention

- ⚠️ **`src/retrieval_lab/generate.py` is uncommitted** — this session's
  `build_prompt` lives only in the working tree. Left out of the handoff commit
  deliberately so it lands under Betty's own message.
- ⚠️ **The scratchpad probe material has no durable home.** `probe.md` (cited by
  the old TODO but never in the repo) was recovered from a prior session's
  `/private/tmp/claude-501/…` dir; it plus tonight's `prompt_probe.py` and
  `sentinel_probe.py` sit in an equally ephemeral scratchpad. A reboot loses them.
- ⚠️ **The 4c.1 fixture still does not exist** (`data/labels/` is empty) and it
  gates D10 and D13. Betty's to author, but now **from real pipeline
  generations** rather than the old probe, so the labels match the shipped format.
- ⚠️ **The correctness scorer's normaliser is load-bearing.** Standard HotpotQA
  normalisation (lowercase, strip articles, strip punctuation) moved gold-context
  accuracy from 9/15 to ~13/15 — omit it and you report ~60% where the truth is
  ~87%. Test it; don't eyeball it.
- ⚠️ **`README.md` has no Phase 4 row yet** and still describes the project as
  three-dataset. Left alone deliberately — Phase 4 has still produced no landed
  number, and the ablation table is for results.
- ⚠️ Carried forward: **`tests/test_finetune.py` still does not exist** but is
  cited in `load_triples`'s docstring — should pin train ∩ dev disjointness, the
  `n + k` guard, and the `--dev-loss` k/batch-size defaults.
- ⚠️ Carried forward: the retrieval cache key does not cover `retrieve.py`'s
  contents (`--refresh` after editing); `data/`/`cache/` resolve against cwd;
  and the contamination refutation is generator-specific at n=5, so re-run the
  no-context probe before trusting the corpus under a larger generator.

## Pick up here

1. **Fill `parse_answer` and `should_refuse`** — the contract they must agree
   with is now measured, not assumed: `<rationale>…</rationale>` then
   `<answer>…</answer>`, refusal as `INSUFFICIENT_CONTEXT` *inside* `<answer>`,
   normalised to `REFUSAL` by the pipeline so `raw` keeps the model's own word.
2. **Run a small generation batch (n≈10–20)** on real retrieved context. First
   run spends minutes on a retrieval cache miss (66K docs) before reaching the
   generator; that is not a hang. Watch two things the probes could not measure:
   format adherence at ten passages (~1,200 tokens, vs the two-passage prompts
   tested) and the yes/no rate against its 6.2% base rate — the probe caught one
   comparison-shaped question answered `yes` when gold was a name.
3. **Author the 4c.1 fixture from that batch**, stratified across correct and
   incorrect answers — then D13's bake-off has something to bake off against.
