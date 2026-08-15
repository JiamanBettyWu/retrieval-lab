# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-14 (latest session):** **Phase 2 is closed and the direction
changed.** The fine-tune's loss replicated on three datasets (NFCorpus, SciFact,
FiQA) with identical ordering, a prediction committed *before* the run
(`66e798c`) was refuted, and the mechanism is now **training-distribution
breadth**, not domain gap. **Phase 3 is retired** — its deliverable was the
retriever Phase 2 measured to be worse — and **Phase 4 (generation +
LLM-as-judge) is the next phase**, amended into `docs/plan.md`. **D8 resolved B**
(no capacity ablation; Phase 2 stays closed) and **D11 resolved C** (no
LangGraph — the skill is covered elsewhere; the refusal gate ships as a plain
conditional). One thing is scaffolded and unwritten: `dev_loss()`. Detail in
[SESSIONS.md](SESSIONS.md); findings in `LEARNINGS.md` (2026-08-14).

```bash
pytest                                                # 40 tests, ~8s, no download
python -m retrieval_lab.evaluate --dataset nfcorpus   # baseline 0.3159 (Phase 1 rerank: 0.3412)
python -m retrieval_lab.evaluate --dataset nfcorpus --model models/lora-r16-a32-lr1e3-100k
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

### D9: Does the ablation table get error bars?
- **Context:** no config was ever run twice — every number is n=1 with no
  run-to-run variance estimate. The `5e-4 -> 1e-3` gap of `0.0051` that picked
  the winner now has **two** reasons to distrust it: no noise floor, and an
  eval batch size of 8 that compressed the loss scale.
- **Options:** A) **Seed replicates on 2–3 configs** (~60 min each)
  B) **State n=1 and move on** — already done, free C) **Replicate the winner
  only** — cheapest real answer
- **Recommendation:** **B.** D8 resolved to keep Phase 2 closed, so no further
  close calls are coming and a noise floor would inform nothing. Revisit only if
  Phase 2 reopens.
- **Blocked on:** Betty. **If A/C:** `--tag <config>-seed<N>`; the spread goes in
  the table, not prose.

### D10: Which generator, and which judge? (Phase 4, blocking)
- **Context:** they must differ, to dodge self-bias. The judge must pass the 4b
  grounded/ungrounded fixture or the phase cannot proceed.
- **Options:** A) **Small-fast generator, stronger judge** — the judge is the
  measuring instrument B) **Same tier both** — cheaper, risks a judge too coarse
  C) **Two judges**, report their agreement alongside human κ
- **Recommendation:** A, with C as a cheap add-on on a sample.
- **Blocked on:** Betty; needed before any Phase 4 code. **If A:** both model IDs
  go in the generation cache key alongside `prompt_version` — swapping either
  silently invalidates results.

## Needs attention

- ⚠️ **`dev_loss()` is scaffolded and unwritten** (`finetune.py`,
  `NotImplementedError`). Until it runs, "Phase 2 traded out-of-domain quality
  for in-domain gain" is **unverified** — the base model's dev loss was never
  measured. Must use `batch_size=8`. The `--dev-loss` flag is unwired plumbing
  waiting on it.
- ⚠️ **`README.md` still carries the narrower Phase 2 framing** — "what moving
  toward MS MARCO costs on biomedical text", one dataset, a domain-match
  mechanism. Three datasets and a refuted prediction later that is stale, and
  **this is the portfolio front door.** The numbers are right; the mechanism
  paragraph and the ablation table are not. Highest-priority stale doc.
- ⚠️ **`data/` is gitignored wholesale**, but Phase 4's hand labels
  (`data/labels/`) are the one input that **cannot be regenerated**. Needs a
  `.gitignore` exception before any labelling starts, or the κ is
  unreproducible.
- ⚠️ **`tests/test_finetune.py` still does not exist** but is cited in
  `load_triples`'s docstring — should pin train ∩ dev disjointness and the
  `n + k` guard.
- ⚠️ Carried forward: the retrieval cache key does not cover `retrieve.py`'s
  contents (`--refresh` after editing); `data/`/`cache/` resolve against cwd.

## Pick up here

1. **Write `dev_loss()`** (yours; scaffold + trap are in the docstring) and run
   it on `all-MiniLM-L6-v2` at `batch_size=8`. One number decides whether Phase
   2 was a trade or loss in both directions — and the README's wording depends
   on the answer.
2. **Update `README.md`** with the three-dataset table, the refuted prediction,
   and the breadth mechanism. Do it after step 1 so both corrections land once.
3. **Start Phase 4 by reading, not coding:** generate ~10 NFCorpus answers by
   hand and read them. Tells you whether biomedical abstracts make faithfulness
   trivial — and whether the refusal gate would ever fire — before you spend 500
   generations building around either assumption.
