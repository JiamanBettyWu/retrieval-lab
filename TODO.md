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
LLM-as-judge) is next**; D10 has a tentative answer (qwen3:8b generator,
Sonnet 5 primary judge, Qwen3.8-27B second judge) so nothing blocks starting,
but neither judge counts until it clears the 4b fixture. Detail in
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

### D10: Which generator, and which judge? (Phase 4) — **tentatively A + C**
- **Context:** they must differ, to dodge self-bias. The judge must pass the 4b
  grounded/ungrounded fixture or the phase cannot proceed.
- **Options:** A) **Small-fast generator, stronger judge** — the judge is the
  measuring instrument B) **Same tier both** — cheaper, risks a judge too coarse
  C) **Two judges**, report their agreement alongside human κ
- **Tentative pick (2026-08-16, Betty): A + C.** Generator `qwen3:8b` (local,
  via Ollama). **Primary judge Claude Sonnet 5** (`claude-sonnet-5`), **second
  judge `Qwen3.8-27B`** (open weights, local) on a sample.
- **Why two rather than the open-weight judge alone:** `Qwen3.8-27B` judging
  `qwen3:8b` output is the *same family* scoring itself — shared pretraining
  corpus and post-training recipe, so it carries exactly the self-bias D10
  exists to dodge. A different parameter count does not buy the independence a
  different family does. Sonnet 5 stays the measuring instrument; the
  open-weight judge rides alongside, and **their agreement becomes a result** —
  reported next to human κ, it also answers whether an open-weight judge could
  stand alone in a later phase.
- **Marked tentative:** neither judge has cleared the 4b grounded/ungrounded
  fixture yet, and Sonnet 5 is revisitable if it proves too coarse or too costly
  across 500 generations.
- **Blocked on:** nothing to start — but the fixture is the gate. A judge that
  fails it changes, and any generations already scored under it are void.
- **Carries with it:** **all three** model IDs go in the generation cache key
  alongside `prompt_version` — swapping any one silently invalidates results,
  and a *tentative* choice is exactly the case where that will happen. Wire the
  key before the first generation, not after.
- **Unverified:** whether a dense 27B (~16–17GB resident at Q4) runs at a usable
  rate on this machine. Check before committing the second judge to the full
  sample — the fallback is to score fewer items with it, not to drop the κ.

## Needs attention

- ⚠️ **`tests/test_finetune.py` still does not exist** but is cited in
  `load_triples`'s docstring — should pin train ∩ dev disjointness, the `n + k`
  guard, and now the `--dev-loss` k/batch-size defaults.
- ⚠️ Carried forward: the retrieval cache key does not cover `retrieve.py`'s
  contents (`--refresh` after editing); `data/`/`cache/` resolve against cwd.

## Pick up here

1. **Start Phase 4 by reading, not coding:** generate ~10 NFCorpus answers by
   hand with `qwen3:8b` and read them. Tells you whether biomedical abstracts
   make faithfulness trivial — and whether the refusal gate would ever fire —
   before you spend 500 generations building around either assumption.
2. **Run both judges against the 4b grounded/ungrounded fixture** before either
   scores anything real. D10 is tentative until they pass; a judge swapped after
   the fact voids every generation already scored under it.
3. **Wire the generation cache key** (all three model IDs + `prompt_version`) before
   the first cached generation — a tentative model choice is precisely the case
   where a silent stale-cache hit will bite.
