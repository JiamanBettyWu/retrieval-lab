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
python evaluate.py --dataset nfcorpus
```

## Ablation (fills in as phases land)

| Stage | Model | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|---|
| Phase 0 · bi-encoder | `all-MiniLM-L6-v2` | _tbd_ | _tbd_ | _tbd_ |
| Phase 1 · + rerank | `+ cross-encoder/ms-marco-MiniLM-L-6-v2` | | | |
| Phase 2 · LoRA encoder | fine-tuned on MS MARCO | | | |

## Files

- `data.py` — BEIR download + load (ships the `qrels` labels).
- `retrieve.py` — the bi-encoder retrieval (Phase 0 core).
- `evaluate.py` — main: load → retrieve → BEIR metrics → headroom check.
- `observability.py` — Weave init + `@op` shim (works with or without weave).
