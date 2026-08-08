# LEARNINGS — retrieval-lab

Append-only devlog, written while it's fresh. Harvested into the `llm-wiki`
project page once this graduates from plan to built.

## 2026-07 — Phase 0: the headroom check comes before any reranker
Phase 0's real job isn't "build a retriever" — it's to confirm the dataset can
even *show* a reranker helping. Two gates: NDCG@10 must be well below 1.0 (room
to improve), and Recall@100 must be high enough that the reranker has the right
docs to reorder — it can never recover a relevant doc the retriever left out of
the top-100. Cheap experiment that could kill the plan runs first.

## 2026-08-07 — a retrieval bug that scores 0.0 instead of crashing
`retrieve()` returned `{row_index: score}` instead of `{doc_id: score}` —
`semantic_search` hands back `corpus_id`, a positional index into the encode
order, not a BEIR id. Nothing raises. `pytrec_eval` just finds zero overlap with
the qrels and prints `NDCG@10: 0.0000`, which reads as "the model is bad on this
dataset." The bug is invisible *because* the output is well-formed.

The lesson generalizes past this repo: **asserting on your own output's shape is
not the same as the downstream consumer accepting it.** The check that actually
discriminates is a five-doc fixture with hand-written qrels run through the real
`EvaluateRetrieval` — with one relevant doc per query ranked first, it must score
exactly 1.0. Perfect-or-zero is maximally diagnostic; a mid-range number would
have told me nothing. Sequencing matters as much as the assertions: fixture
first, full BEIR run second. Run the real eval first and a low number has four
possible causes; run it second and it has one interpretation.

## 2026-08-07 — check the metric's ceiling before reading it as a failure
`Recall@100 = 0.311` looked like weak retrieval, but on a dataset picked for
*dense* qrels the cap could just as easily have been arithmetic — if a query has
200 relevant docs, top-100 caps recall at 0.50 no matter how good the retriever
is. Computed the actual ceiling from the qrels (`sum(min(n,100)/n)/|Q|`): **0.965**.
So the low recall is real, and "raise recall" is a genuine lever. But the same
calculation shows **Recall@10 caps at 0.615** — worth knowing before reporting it
in an ablation table against an implied 1.0. Any metric with k in its name has a
denominator worth checking before you interpret it.
