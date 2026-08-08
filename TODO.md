# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-07 (latest session):** **Phase 0 is green and complete** —
`retrieve()`'s id-mapping bug fixed, a fast fixture test added
(`test_retrieve.py`), and the baseline measured on BEIR/NFCorpus: **NDCG@10
0.3159 · MRR@10 0.5046 · Recall@100 0.3115** (commit `5af1e0d`). The headroom
check passes and **D1 is resolved in favor of NFCorpus** on empirical grounds.
Next up is Phase 1's reranker, but measure its ceiling first (D3). Detail in
[SESSIONS.md](SESSIONS.md).

## Open decisions

### D2: Phase 3 demo UI (not urgent — Phase 3 is far off)
- **Options:** A) **Gradio/Streamlit** — fast, minimal B) **FastAPI + React** —
  a web-eng rep, more work
- **Recommendation:** Gradio — the science (the ablation table) is the star, not
  the UI.
- **Blocked on:** Phases 1–2 landing first.
- **If A:** thin `app/` with Gradio. **If B:** reuse the mise FastAPI+React
  pattern.

### D3: Does Phase 1 (cross-encoder rerank) clear its own headroom bar?
- **The question:** the reranker can only reorder what retrieval already found,
  and Recall@100 is 0.311. An **oracle rerank** (sort the existing top-100 by
  qrels, compute NDCG@10) is a pure sort with no model inference and gives the
  hard upper bound on everything Phase 1 can deliver.
- **Options:** A) **Oracle first, then build** — one cheap check, and it earns a
  third ablation column (off-the-shelf / +rerank / oracle) B) **Build the
  cross-encoder directly** — skips a step, but the resulting lift has no ceiling
  to interpret it against.
- **Recommendation:** A. It mirrors the Phase 0 discipline already recorded in
  `LEARNINGS.md` ("the cheap experiment that could kill the plan runs first"),
  and reporting the ceiling alongside the lift is the stronger portfolio artifact.
- **Blocked on:** nothing — resolvable empirically in one short run.
- **If oracle NDCG@10 is high (~0.7+):** relevant docs are in the top-100, just
  badly ordered → build Phase 1's cross-encoder as planned.
- **If oracle NDCG@10 is low (~0.4):** reordering can't save it; recall is the
  binding constraint → deprioritize Phase 1, jump to Phase 2's LoRA fine-tune
  (which improves retrieval itself), and note the reranker's cap in the README.

## Needs attention

- ⚠️ **`LEARNINGS.md`'s two new entries were written by Claude**, in the file's
  existing voice. It's a personal devlog — worth a read-through and a rewrite in
  Betty's own words, or a cut, before this repo goes public.
- ⚠️ **Two commits are unpushed** (`5af1e0d` + this handoff). Repo is still
  **private**; per the 2026-07-15 entry the plan is to flip it public once
  Phase 0 is presentable — which it now is. Push is a manual step. Remember
  `GH_HOST=github.com` for any `gh` operation (see SESSIONS.md 2026-07-15).

## Pick up here

1. **Oracle-rerank ceiling** (D3) — reorder the existing top-100 by qrels,
   compute NDCG@10, add it to the README ablation table as a third row.
2. **Cache retrieval results to disk** — `evaluate.py` re-encodes the corpus
   every run (~1 min). Phase 1 must rerank the *same* top-100 Phase 0 produced,
   so this is likely a prerequisite, not just a speedup.
3. **Phase 1 proper** — rerank top-100 with
   `cross-encoder/ms-marco-MiniLM-L-6-v2` → top-10, re-measure, fill the
   before/after ablation row. Re-run `python test_retrieve.py` after any change
   to `retrieve()`.
