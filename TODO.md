# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-16 (latest session):** **Phase 2 is closed, with nothing left
open in it.** The missing baseline was measured — `all-MiniLM-L6-v2` scores
**0.6597** MS MARCO dev loss against the fine-tunes' `0.3960 / 0.3443 / 0.3180 /
0.3129`, so Phase 2 bought real in-domain gain and "traded out-of-domain quality
for in-domain gain" is **verified**. `README.md` has caught up in the same pass:
three-dataset table, the refuted FiQA prediction kept as a refuted prediction,
the breadth mechanism, and Phase 3's retirement. **Phase 4 (generation +
LLM-as-judge) is next and is blocked on D10.** Detail in
[SESSIONS.md](SESSIONS.md); findings in `LEARNINGS.md` (2026-08-16).

```bash
pytest                                                # 40 tests, ~8s, no download
python -m retrieval_lab.finetune --dev-loss           # base model dev loss: 0.6597
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

- ⚠️ **`data/` is gitignored wholesale**, but Phase 4's hand labels
  (`data/labels/`) are the one input that **cannot be regenerated**. Needs a
  `.gitignore` exception before any labelling starts, or the κ is
  unreproducible.
- ⚠️ **`tests/test_finetune.py` still does not exist** but is cited in
  `load_triples`'s docstring — should pin train ∩ dev disjointness, the `n + k`
  guard, and now the `--dev-loss` k/batch-size defaults.
- ⚠️ Carried forward: the retrieval cache key does not cover `retrieve.py`'s
  contents (`--refresh` after editing); `data/`/`cache/` resolve against cwd.

## Pick up here

1. **Resolve D10** — nothing in Phase 4 can be written before it, and both model
   IDs belong in the generation cache key from the first commit.
2. **Start Phase 4 by reading, not coding:** generate ~10 NFCorpus answers by
   hand and read them. Tells you whether biomedical abstracts make faithfulness
   trivial — and whether the refusal gate would ever fire — before you spend 500
   generations building around either assumption.
3. **Add the `data/labels/` `.gitignore` exception** before the first label is
   written, not after.
