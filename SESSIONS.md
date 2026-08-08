# Session log

Reverse-chronological journal — one dated entry per working session: what got
done, what was decided and why. History only; for what to do next see
[TODO.md](TODO.md).

---

## 2026-08-07 (Phase 0 green — retrieve() fixed, baseline measured, D1 resolved)

Picked up the `retrieve()` WIP handed over on 2026-07-15 and closed out Phase 0.
Betty had finished the implementation but hadn't run it — the session started
from "I know there are probably bugs, but I'm not sure how to test it," so the
work was as much about *building the test* as fixing the code. Everything below
shipped in commit `5af1e0d`.

**The bug (and why it needed a test to find).** `retrieve.py:59` returned
`[{corpus_id: score}, ...]` — a list of single-key dicts, keyed by
`semantic_search`'s `corpus_id`, which is a positional index into the encode
order rather than a BEIR `doc_id`. Two errors in one line: wrong container, and
the step-4 `doc_ids[...]` lookup skipped entirely. The container error would have
surfaced quickly; the id error would not have. Nothing raises — `pytrec_eval`
simply finds zero overlap with the qrels and prints `NDCG@10: 0.0000`, which
reads as "MiniLM is bad at medical retrieval," not "my ids are wrong." A bug that
is invisible *because* its output is well-formed. Fixed to
`{doc_ids[hit["corpus_id"]]: float(hit["score"]) for hit in hits[i]}` — the
`float()` matters, pytrec_eval wants Python floats, not torch scalars. The stale
`TODO(human)` block in the docstring was replaced with a note on why
normalization makes cosine == dot product and why the id remap is needed.

**The test, and the reasoning behind its shape** (`test_retrieve.py`, new). A
five-document fixture — two medical docs matching the two queries, three
obvious distractors (sourdough, Iceland, cricket) — with hand-written qrels. It
runs in ~5s with no download and no pytest dependency (bare asserts; pytest
isn't in `requirements.txt` and wasn't worth adding to get a test running). Doc
ids are deliberately non-sequential strings (`MED-001`…) so a leaked row index
cannot accidentally pass. Each assertion targets one failure mode so a red test
names its own cause: dict-not-list, keys-exist-in-corpus, values-are-float,
top_k honored, top_k > corpus clamps, and rank-1 correctness (which would catch
transposed args to `semantic_search`). The load-bearing one is the last: it runs
BEIR's real `EvaluateRetrieval` over the fixture, which must score **exactly
`NDCG@10 = 1.0`**. Asserting on your own dict's shape is not the same as the
downstream consumer accepting it, and perfect-or-zero is maximally diagnostic —
a mid-range number would have said nothing. Deliberately *not* a stub model: a
stub returning numpy would have masked any mismatch between `convert_to_tensor=True`
and what `semantic_search` expects, and MiniLM on five docs is instant anyway.
Ran it against the broken code first to confirm it actually fails (`got list`),
then fixed. Sequencing is the point — fixture first, full BEIR run second; run
the real eval first and a low number has four possible causes, run it second and
it has one interpretation.

**Baseline (BEIR/nfcorpus, `all-MiniLM-L6-v2`, top-100):** NDCG@10 **0.3159**,
MRR@10 **0.5046**, Recall@100 **0.3115**. Weave tracing was live for the run
(`wandb.ai/bettyjiamanwu/retrieval-lab`, call `019fdf0a-4e1e-7d92-b4ed-f4bda58f3010`).
Recorded in the README ablation table.

**The recall-ceiling check (a near-miss worth remembering).** `evaluate.py`'s
headroom logic printed "Recall@100 0.311 is low — raising retrieval recall is the
lever," and that verdict was very nearly taken at face value. But NFCorpus was
picked *for* dense qrels, and dense qrels can cap recall arithmetically — a query
with 200 relevant docs caps Recall@100 at 0.50 no matter how good the retriever
is. Two hints pointed that way: MRR@10 (0.505) sitting far above NDCG@10 (0.316),
the signature of "finds one good doc fast, misses the tail," and NDCG@100 (0.295)
sitting *below* NDCG@10. So the ceiling was computed directly from the qrels
(`sum(min(n,100)/n)/|Q|`, counting only rel > 0): **0.9647**. Median 16 relevant
docs/query, mean 38, max 475, but only 22 of 323 queries exceed 100 — the top-100
cutoff barely binds. **The verdict stands as written**: 0.311 against a 0.965
ceiling is genuinely low retrieval, ~32% efficiency, and raising recall is a real
Phase 2 lever. The hypothesis was wrong, but the check was cheap and the byproduct
is useful — **Recall@10 caps at 0.6146**, not 1.0, which matters before that
column ever lands in an ablation table. The NDCG@100 < NDCG@10 inversion is
likewise expected with dense qrels (ideal-DCG denominator grows faster than
retrieved gain), not a bug.

**D1 resolved — NFCorpus, on empirical grounds.** The scaffold defaulted to it
and the headroom check confirmed it: NDCG@10 0.316 is far from 1.0, and the
top-100 holds roughly 2× the relevant docs the top-10 surfaces (Recall@10 0.155 →
Recall@100 0.311), so there is real material for Phase 1's reranker to reorder.
No reason to switch to SciFact.

**Next move decided, not yet done: measure Phase 1's ceiling before building it**
(now D3). This applies the Phase 0 discipline already recorded in `LEARNINGS.md`
one level down — an *oracle rerank*: take the top-100 the bi-encoder already
returns, reorder it perfectly using qrels as ground truth, compute NDCG@10. Pure
sort, no model inference, and it hard-bounds everything a real cross-encoder can
deliver. It also earns a third ablation column (off-the-shelf / +rerank / oracle),
which is a stronger portfolio artifact than the lift alone — most RAG projects
report the improvement and never the ceiling.

**Also updated:** `README.md` — Phase 0 ablation row filled in, a
headroom-verdict section added with the ceiling numbers and the two
reading-notes caveats, and `test_retrieve.py` documented in the run instructions
with *why* the fixture must precede the real eval. `LEARNINGS.md` — two new
devlog entries (the silent-0.0 bug and its test design; the check-the-ceiling
lesson). Both were written by Claude in the file's existing voice; Betty may want
to rewrite them in her own.

**Non-obvious:** `evaluate.py` recomputes retrieval from scratch on every run
(~1 min of encoding on this machine). Phase 1 must rerank the *same* top-100
Phase 0 produced, so caching retrieval results to disk is likely the first
refactor — otherwise every ablation row repays the encode cost and there's no
guarantee the reranker saw an identical candidate set. Also: with
`WANDB_API_KEY` set, `@op` on `retrieve` serializes the full corpus arg and all
result entries — it completed fine and wasn't slow, but it's noisy to debug
through.

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
