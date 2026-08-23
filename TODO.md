# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-22 (latest session):** **Phase 4a parses, and D14 is settled.**
`build_prompt` is hand-written, probed, and now at **`PROMPT_VERSION = "v1"`** —
the refusal rule pins the sentinel *inside* `<answer>` and requires a rationale
on refusals too, closing an ambiguity the v0 wording left open.
`parse_answer` is written and pinned by nine tests; its miss contract (`("", raw)`
whenever the answer is empty, however it got that way) is what lets
`should_refuse` see a sentinel the model emitted without tags. **`should_refuse`
is the last stub.** Nothing has been generated at scale, so no generation cache
exists. 78 tests green. Detail in [SESSIONS.md](SESSIONS.md); findings in
`LEARNINGS.md` (2026-08-19 ×2).

**D14 is decided (2026-08-22): correctness is reported over non-refused queries,
with refusal rate published alongside.** Consequence to carry into every
correctness number: the denominator varies per config, so a README row quoting
correctness must state its n.

```bash
pytest                                                # 78 tests, ~8s, no download
python -m retrieval_lab.generate --n-queries 50       # Phase 4a — needs `ollama serve` + should_refuse
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

## Needs attention

- ⚠️ **`LEARNINGS.md`'s `mean token-F1 0.711` predates D14.** It averages over
  15 gold-context questions including the Alceste refusal. Under the settled
  convention the same probe reads **0.7619 over 14, refusal rate 6.7%**. Nothing
  to revert — it is a probe result, not a published claim — but quote it the new
  way from here, and write the scorer so refusals leave the denominator rather
  than scoring 0.
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

1. **Fill `should_refuse`** — the last stub. Model-side: recognise
   `INSUFFICIENT_CONTEXT` in the parsed answer and let the pipeline normalise it
   to `REFUSAL`, so `raw` keeps the model's own word. The one real decision left
   is what it returns when `answer == ""`, which is exactly the case
   `parse_answer`'s `("", raw)` contract was built to feed — the sentinel may be
   sitting in the rationale slot untagged.
2. **Run a small generation batch (n≈10–20)** on real retrieved context. First
   run spends minutes on a retrieval cache miss (66K docs) before reaching the
   generator; that is not a hang. Watch two things the probes could not measure:
   format adherence at ten passages (~1,200 tokens, vs the two-passage prompts
   tested) and the yes/no rate against its 6.2% base rate — the probe caught one
   comparison-shaped question answered `yes` when gold was a name.
3. **Author the 4c.1 fixture from that batch**, stratified across correct and
   incorrect answers — then D13's bake-off has something to bake off against.
