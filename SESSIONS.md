# Session log

Reverse-chronological journal — one dated entry per working session: what got
done, what was decided and why. History only; for what to do next see
[TODO.md](TODO.md).

---

## 2026-07-15 (Phase 0 scaffold + repo created)

Kicked off **retrieval-lab**, a build-to-learn project chosen off the `llm-wiki`
gap-map (concepts studied but never built: RAG, reranking, learning-to-rank,
embeddings, LoRA, fine-tuning). Full plan in `docs/plan.md` (mirrors the vault
note `notes/retrieval-lab`).

**Shipped:**
- Phase 0 scaffold — commit `3b4afee`. Flat layout at repo root: `data.py`
  (BEIR/NFCorpus load with qrels), `evaluate.py` (main: load → retrieve → BEIR
  metrics NDCG@10/MRR@10/Recall@100 → headroom check), `observability.py`
  (Weave init + `@op` import-shim lifted from the mise pattern), `retrieve.py`
  (Phase 0 bi-encoder — **left as a `TODO(human)` for Betty to implement**).
- `docs/plan.md` — commit `41a2b17`. Repo-adapted copy of the vault plan
  (wiki-links flattened, repo-layout section corrected to the actual flat tree).
- GitHub repo created **private**: https://github.com/JiamanBettyWu/retrieval-lab
  (`origin`, `main` tracks `origin/main`).

**WIP handed to Betty:** `retrieve()` in `retrieve.py`. She did step 1 (built
`doc_ids`/`doc_texts`/`query_ids`/`query_texts`) + a bare `model.encode()` stub;
the `raise NotImplementedError` marker is still in place. Remaining: encode with
`normalize_embeddings=True` → `st_util.semantic_search(top_k)` → map each hit's
`corpus_id` (a row index) back to the real BEIR `doc_id`, returning
`{query_id: {doc_id: score}}`. She'll finish it later today.

**Decisions with reasoning (baked into the scaffold):**
- **Evaluate on a labeled BEIR benchmark, not the wiki.** The wiki is too small
  and clean to *score* — near-perfect base retrieval = no headroom to show a
  reranker/fine-tune helping, and no qrels to compute NDCG. BEIR ships qrels and
  is hard enough to leave headroom. The wiki is reserved for the Phase 3 *demo*
  (qualitative, no labels). ("Evaluate ≠ demo.")
- **Baseline encoder = general `all-MiniLM-L6-v2`, deliberately NOT the
  MS-MARCO-tuned bi-encoder.** Phase 2 LoRA-fine-tunes on MS MARCO; starting from
  an already-MS-MARCO-specialized model would flatten the ablation. General
  baseline = headroom for the fine-tune to visibly help.
- **bi-encoder ≠ cross-encoder** (the name-collision trap): `all-MiniLM-L6-v2`
  (retriever, emits embeddings) and `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (Phase 1 reranker, emits a score) share the MiniLM-L6 backbone but are
  different model *types* — the reranker cannot do first-stage retrieval.
- **Zero-shot generalization is the headline claim** (Phase 2): fine-tune on
  MS MARCO, evaluate on an *unseen* BEIR domain — a stronger portfolio line than
  improving one's own test split.
- **Flat repo layout** (files at root, not `src/`) — small lab; also sidesteps
  the package-import traps mise hit. Weave is optional via the shim, so Phase 0
  runs with no weave and no `WANDB_API_KEY`.
- **Private repo for now**, flip public when Phase 0 is presentable (mise's path).

**Non-obvious / quirks:**
- **Two `gh` hosts are both "active"** (`github.com` = JiamanBettyWu,
  `github.gatech.edu` = jwu809). Repo was created with `GH_HOST=github.com`
  forced, to avoid landing on the GT Enterprise host. Use that env var for any
  future `gh` op on this repo.
- **The plan lives in two places** — `docs/plan.md` here and `notes/retrieval-lab`
  in the vault. Per the vault's code-project convention ("the repo is the
  source"), once this graduates to a `wiki/projects/` page the **repo's copy
  becomes canonical** and the vault harvests from it. Watch drift direction.
- `data/` (BEIR downloads) and `.venv/` are gitignored.
