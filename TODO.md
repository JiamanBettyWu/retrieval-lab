# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-08 (latest session):** Phase 0 is green and **D3 is resolved —
build Phase 1**: the oracle puts the ceiling at **NDCG@10 0.6263** vs a 0.3159
baseline, so +0.3104 of headroom is available to a cross-encoder. Phase 1 is now
**scaffolded with `rerank()` left as a `TODO(human)` for Betty to implement**
(`847fb53`) — see "Pick up here". All work is pushed (`main` == `origin/main`);
the repo is still **private** — see D4. Detail in [SESSIONS.md](SESSIONS.md).

> ⚠️ **`pytest` is currently RED on purpose** — 22 pass, 10 error with
> `NotImplementedError` from the unimplemented `rerank()`. Nothing is broken.
> It goes fully green when that function lands.

Setup and run commands (the repo became an installable package on 2026-08-07):

```bash
uv pip install -e '.[dev]'                            # from the repo root, always
pytest
python -m retrieval_lab.evaluate --dataset nfcorpus   # baseline    0.3159
python -m retrieval_lab.oracle   --dataset nfcorpus   # ceiling     0.6263
python -m retrieval_lab.rerank   --dataset nfcorpus   # Phase 1 — needs rerank()
```

## Open decisions

### D2: Phase 3 demo UI (not urgent — Phase 3 is far off)
- **Options:** A) **Gradio/Streamlit** — fast, minimal B) **FastAPI + React** —
  a web-eng rep, more work
- **Recommendation:** Gradio — the science (the ablation table) is the star, not
  the UI.
- **Blocked on:** Phases 1–2 landing first.
- **If A:** thin `app/` with Gradio. **If B:** reuse the mise FastAPI+React
  pattern.

### D4: Flip the repo public now that Phase 0 is presentable?
- **Context:** the 2026-07-15 entry set "private for now, public once Phase 0 is
  presentable" as the plan. Phase 0 now has a baseline, a measured ceiling, a
  test suite, and a README that leads with the ablation table.
- **Options:** A) **Flip now** — the headroom story (measuring the ceiling
  before building) already stands on its own as the portfolio point B) **Wait
  for Phase 1** — a before/after row makes the table show *movement*, which was
  the stated deliverable
- **Recommendation:** B, narrowly. The current table has one real row plus a
  ceiling; one more row turns it from a baseline into an ablation. Phase 1 is
  the next task anyway, so the wait is short.
- **Blocked on:** Betty's call — also worth a skim of `LEARNINGS.md` and
  `SESSIONS.md` for anything not meant to be public.
- **If A:** `GH_HOST=github.com gh repo edit --visibility public`, push first.
- **If B:** revisit once the Phase 1 row lands; nothing to revert.

## Needs attention

- ⚠️ **`data/` and `cache/` resolve against the current working directory**, not
  the package root. Entrypoints must be run from the repo root or BEIR
  re-downloads elsewhere. Deliberate (anchoring to a computed repo root gets
  fragile once installed) but it is an assumption baked into `data.py` and
  `cache.py`.
- ⚠️ **The retrieval cache key does not cover `retrieve.py`'s contents.** Edit
  that file without `--refresh` and stale results are served silently — the
  exact class of failure Phase 0 already lost time to.

## Pick up here

1. **Implement `rerank()`** in `src/retrieval_lab/rerank.py` — scaffolded with a
   `TODO(human)` docstring (four steps) and a `raise NotImplementedError` to
   delete. Betty is writing this one herself. `tests/test_rerank.py` is already
   written and currently red (10 errors, all `NotImplementedError`); it goes
   green when the function is correct. Then:

   ```bash
   pytest tests/test_rerank.py                          # fixture first, always
   python -m retrieval_lab.rerank --dataset nfcorpus --limit 20   # smoke test
   python -m retrieval_lab.rerank --dataset nfcorpus     # the real number
   ```

   The two traps the tests exist to catch: build pairs as `(query, doc)` not
   `(doc, query)`, and **return all ~100 candidates reordered — never truncate
   to 10** (that collapses Recall@100 and breaks comparability with Phase 0).
2. **Fill the Phase 1 ablation row** in `README.md`, and read the result against
   the **0.6263 oracle ceiling, not 1.0**. `rerank.py`'s output prints the
   "% of available headroom captured" for exactly this reason.
3. **Settle D4** — flip the repo public once that row lands.
