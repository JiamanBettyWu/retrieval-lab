# retrieval-lab

A retrieval-quality lab: build a two-stage RAG retriever, LoRA-fine-tune its
encoder, and **prove** each improvement on a labeled benchmark (BEIR) with
metrics traced in Weave. Then demo the validated pipeline as an "ask my second
brain" front door over a personal wiki.

> Full plan lives in the `llm-wiki` vault: `notes/retrieval-lab`.

## Phase 0 — baseline retriever + eval harness

An off-the-shelf **bi-encoder** (`all-MiniLM-L6-v2`) over a small BEIR set
(NFCorpus), scored against BEIR's `qrels`. Produces two things:

1. a **baseline** (NDCG@10, MRR@10) to improve on in later phases, and
2. the **headroom check** — is NDCG@10 well below 1.0 (room for a reranker to
   help), and is Recall@100 high enough that the reranker even has the right
   docs to reorder?

### Setup

```bash
uv venv && uv pip install -r requirements.txt   # or: pip install -r requirements.txt
# optional, for Weave tracing:
export WANDB_API_KEY=...
```

### Run

```bash
python test_retrieve.py                 # ~5s, no download — verifies retrieve()
python test_oracle.py                   # instant — verifies oracle_rerank()
python evaluate.py --dataset nfcorpus   # the baseline
python oracle.py   --dataset nfcorpus   # the ceiling a reranker could reach
```

Retrieval results are cached to `cache/` per (dataset, model, top_k), so only
the first run pays the ~1 min encode — and every later stage provably reranks
the *same* candidate set rather than a re-encoded one. The cache key does not
cover `retrieve.py`'s contents: pass `--refresh` after editing it.

`test_retrieve.py` is a five-document fixture with a known-correct answer. It
checks the shape of `retrieve()`'s output *and* runs BEIR's own evaluator over
it, which must score exactly `NDCG@10 = 1.0`. That last check is the one that
catches the failure mode the full run can't diagnose: if `corpus_id` row indices
leak through unmapped, `pytrec_eval` finds no overlap with the qrels and reports
`NDCG@10 = 0.0` — which reads as "weak model," not "wrong ids." Run the fixture
before trusting any number from `evaluate.py`.

## Ablation (fills in as phases land)

| Stage | Model | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|---|
| Phase 0 · bi-encoder | `all-MiniLM-L6-v2` | **0.3159** | **0.5046** | **0.3115** |
| Phase 1 · + rerank | `+ cross-encoder/ms-marco-MiniLM-L-6-v2` | | | |
| Phase 2 · LoRA encoder | fine-tuned on MS MARCO | | | |
| _ceiling_ · **oracle rerank** | _perfect reorder of the same top-100_ | _0.6263_ | — | _0.3115_ |

The last row is not a system — it's the **measuring stick**. `oracle.py` reads
the qrels and orders each query's retrieved candidates perfectly, but may only
*reorder* what retrieval found, never add to it. So `0.6263` is the hard upper
bound on anything a real cross-encoder can reach over these candidates, and the
Phase 1 row should be read against it rather than against 1.0.

**Phase 1 is worth building (D3 resolved).** Available headroom is **+0.3104 —
a 98% relative gain over baseline** — meaning the bi-encoder's top-100 already
*contains* roughly the right documents and is simply ordering them badly. That
is precisely the failure a cross-encoder repairs. The remaining `1 − 0.6263 =
0.3737` is unreachable by any reranker; it's recall, and it belongs to Phase 2.

Recall@10 says it more plainly: reordering alone lifts it **0.155 → 0.276**,
nearly to Recall@100's 0.3115. Almost every relevant document the retriever
found is *already in the top-100 but buried below rank 10*; perfect ordering
pulls essentially all of them into the top 10.

`oracle.py` asserts two invariants on every run: that the oracle beats the
baseline on each query individually, and — the stronger claim — that its score
equals an **arithmetic ceiling derived independently of the sort under test**
(`0.626292`, matching to floating point). The first alone would pass for any
improvement, including one leaving gain on the table; only the second
establishes that `0.6263` is genuinely the maximum.

### Phase 0 headroom verdict — proceed

- **NDCG@10 0.316** — far below 1.0, so a reranker has room to show lift.
- **Recall@100 0.311** against a **0.965 achievable ceiling** (NFCorpus qrels are
  dense: median 16 relevant docs/query, but only 22 of 323 queries exceed 100, so
  the top-100 cutoff barely binds). Retrieval recall is genuinely low, not a
  denominator artifact — which makes raising recall a real Phase 2 lever.
- **Top-100 holds ~2× the relevant docs the top-10 surfaces** (Recall@10 0.155 →
  Recall@100 0.311), so there is material for Phase 1's reranker to reorder.

Two reading notes so these aren't mistaken for bugs: `NDCG@100` (0.295) sits
*below* `NDCG@10` (0.316) because dense qrels grow the ideal-DCG denominator
faster than retrieved gain; and **Recall@10 caps at 0.615**, not 1.0, since the
median query has 16 relevant docs and only 10 slots.

## Files

- `data.py` — BEIR download + load (ships the `qrels` labels).
- `retrieve.py` — the bi-encoder retrieval (Phase 0 core).
- `test_retrieve.py` — fixture test for `retrieve()` (fast, no download).
- `evaluate.py` — main: load → retrieve → BEIR metrics → headroom check.
- `observability.py` — Weave init + `@op` shim (works with or without weave).
