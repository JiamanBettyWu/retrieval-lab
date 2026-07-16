# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-07-15 (latest session):** Phase 0 scaffold shipped (`3b4afee`) and
pushed to a **private** GitHub repo (github.com/JiamanBettyWu/retrieval-lab);
plan copied to `docs/plan.md` (`41a2b17`). The one open code task is
`retrieve()` in `retrieve.py` — Betty started it (id/text lists done) and will
finish the encode → semantic_search → id-map steps. Detail in
[SESSIONS.md](SESSIONS.md).

## Open decisions

### D1: Which BEIR eval dataset for Phase 0?
- **Options:** A) **NFCorpus** (~3.6K docs, medical, hard — more qrels/query)
  B) **SciFact** (~5K docs, scientific claims, cleaner)
- **Recommendation:** NFCorpus — its difficulty best guarantees the *headroom*
  the reranker/fine-tune need to show measurable lift.
- **Blocked on:** nothing — confirmable empirically by the Phase 0 headroom
  check (is NDCG@10 well below 1.0?).
- **If A:** proceed on NFCorpus (the scaffold defaults to it). **If B:** pass
  `--dataset scifact`; no code change needed.

### D2: Phase 3 demo UI (not urgent — Phase 3 is far off)
- **Options:** A) **Gradio/Streamlit** — fast, minimal B) **FastAPI + React** —
  a web-eng rep, more work
- **Recommendation:** Gradio — the science (the ablation table) is the star, not
  the UI.
- **Blocked on:** Phases 0–2 landing first.
- **If A:** thin `app/` with Gradio. **If B:** reuse the mise FastAPI+React
  pattern.

## Pick up here

1. Finish `retrieve()` in `retrieve.py` — encode (`normalize_embeddings=True`)
   → `st_util.semantic_search(top_k)` → map `corpus_id` back to real `doc_id`
   → return `{query_id: {doc_id: score}}`. Remove the `raise NotImplementedError`.
2. `uv venv && uv pip install -r requirements.txt` (optional: `export WANDB_API_KEY=...`).
3. `python evaluate.py --dataset nfcorpus` → read the baseline + headroom
   verdict together, then commit Phase 0 green.
