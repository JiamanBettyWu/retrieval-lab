# TODO

Forward-looking state. Session history lives in [SESSIONS.md](SESSIONS.md).

## Current state

**As of 2026-08-08 (latest session):** **Phase 1 is done and the repo is
public** — https://github.com/JiamanBettyWu/retrieval-lab (D4 resolved). The
cross-encoder took NDCG@10 from 0.3159 to **0.3412** — `+0.0253`, or **8.1% of
the 0.3104 headroom** the oracle measured. `pytest` is **fully green (40
tests)**. Explaining the modest gain took four attempts and the honest answer is
in the README; the actionable residue is that the reranker *loses* 1.06 NDCG
points on dense queries, which is a free Phase 2 lever (D5). Detail in
[SESSIONS.md](SESSIONS.md).

```bash
uv pip install -e '.[dev]'                            # from the repo root, always
pytest                                                # 40 tests, ~8s, no download
python -m retrieval_lab.evaluate --dataset nfcorpus   # baseline    0.3159
python -m retrieval_lab.oracle   --dataset nfcorpus   # ceiling     0.6263
python -m retrieval_lab.rerank   --dataset nfcorpus   # Phase 1     0.3412
python -m retrieval_lab.rerank   --dataset nfcorpus --breakdown   # the analysis
```

## Open decisions

### D2: Phase 3 demo UI (not urgent — Phase 3 is far off)
- **Options:** A) **Gradio/Streamlit** — fast, minimal B) **FastAPI + React** —
  a web-eng rep, more work
- **Recommendation:** Gradio — the science (the ablation table) is the star, not
  the UI.
- **Blocked on:** Phase 2 landing first.
- **If A:** thin `app/` with Gradio. **If B:** reuse the mise FastAPI+React
  pattern.

### D5: How should Phase 2 handle dense queries, where reranking actively hurts?
- **Context:** on the 86 queries with 11+ relevant docs in the top-100, the
  cross-encoder *subtracts* 1.06 NDCG points (13% of everything the other
  buckets earned) while still improving MRR — it promotes one good doc to rank 1
  and evicts others from the top ten. See the MRR-vs-NDCG table in `README.md`.
- **Options:** A) **Score blending** — combine cross-encoder and bi-encoder
  scores (e.g. weighted sum or reciprocal-rank fusion) so rank 1 improves
  without the eviction; keeps one code path B) **Routing** — detect dense
  queries and skip reranking them; simpler, but needs a detector and a
  threshold, and "dense" isn't knowable at query time without the qrels
  C) **Do nothing** — accept the loss, let the LoRA fine-tune address it
- **Recommendation:** A. It is the only option that needs no query-time signal
  the system doesn't have, and the MRR result says the cross-encoder's rank-1
  judgement is worth keeping even where its full ordering isn't. Ceiling on the
  win is ~`+0.0033` NDCG@10 (`0.3412 → 0.3445`) from stopping the damage alone.
- **Blocked on:** nothing — actionable whenever Phase 2 starts.
- **If A:** add a blend weight as an ablation axis; it earns its own README row.
  **If B:** the detector is the hard part, and note that a "dense" rule and a
  "baseline was already good" rule select nearly the same queries (they
  correlate at 0.57), so it would work without confirming *why*.
  **If C:** record it as a known regression so Phase 2's numbers stay readable.

## Needs attention

- ⚠️ **The retrieval cache key still does not cover `retrieve.py`'s contents.**
  Edit that file without `--refresh` and stale results are served silently.
  Unchanged from previous sessions; called out because Phase 2 will touch
  retrieval directly and this is the exact trap Phase 0 lost time to.
- ⚠️ **`data/` and `cache/` resolve against the working directory**, not the
  package root, so entrypoints must be run from the repo root. Deliberate, but
  it is an assumption baked into `data.py` and `cache.py`.
- ⚠️ **The qrel-density explanation is supported but not proven.** The README
  says so explicitly. If Phase 2 quotes it, keep the hedge — the dense bucket is
  also the high-baseline bucket, and separating them needs a reranker trained on
  dense qrels.

## Pick up here

1. **Start Phase 2** (LoRA fine-tune on MS MARCO, evaluate zero-shot on the
   untouched BEIR set), carrying D5's blending idea in as an ablation axis.
2. When Phase 2 lands a number, add its row to the README table and re-read it
   against the **0.6263 ceiling** — though note Phase 2 raises that ceiling by
   changing the candidate set, so `oracle.py` must be re-run, not reused.
3. Optional, now that the repo is public: pin it to the GitHub profile
   (Settings → Profile → pinned repositories) if it's meant to be front-and-
   centre for the GenAI pivot.
