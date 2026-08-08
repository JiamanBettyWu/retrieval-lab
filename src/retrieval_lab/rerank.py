"""Phase 1 reranking: a CROSS-ENCODER over the bi-encoder's candidates.

    python -m retrieval_lab.rerank --dataset nfcorpus

The bi-encoder (Phase 0) embeds query and document *separately*, so a doc's
vector must serve every possible query — one summary for all readers. A
cross-encoder instead reads `[query] [SEP] [document]` as ONE sequence, letting
every query token attend to every document token. That is why it is more
accurate, and why it cannot do first-stage retrieval: there is no reusable
document vector to precompute, so scoring the full corpus would cost
323 x 3633 = ~1.2M forward passes instead of 323 x 100 = ~32K.

Read the result against the ORACLE CEILING (0.6263 on nfcorpus), not against
1.0 — see SESSIONS.md 2026-08-07. The reranker can only reorder what retrieval
already found; on 51 of 323 queries the top-100 contains nothing relevant at
all, and no reranker can score above 0 there.
"""
import argparse
import logging

from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import CrossEncoder

from .cache import cached_retrieval
from .data import load_beir
from .observability import op
from .oracle import oracle_rerank
from .retrieve import BI_ENCODER, doc_text, load_encoder, retrieve

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("retrieval_lab")

# NOT interchangeable with retrieve.BI_ENCODER despite the shared MiniLM-L6
# backbone — this one emits a score per (query, doc) pair, not an embedding.
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_cross_encoder(name: str = CROSS_ENCODER) -> CrossEncoder:
    return CrossEncoder(name)


@op  # traced in Weave when init_weave() ran; a no-op otherwise
def rerank(model: CrossEncoder, corpus: dict, queries: dict, results: dict) -> dict:
    """Re-score each query's retrieved candidates with the cross-encoder.

    Args:
        results: {query_id: {doc_id: bi_encoder_score}} — the Phase 0 candidates.
                 Taken as an argument (not recomputed) so this provably reranks
                 the same set `evaluate.py` and `oracle.py` scored.

    Return the SAME shape, {query_id: {doc_id: cross_encoder_score}}, with the
    same doc_ids per query — only the scores (and therefore the order) change.

    TODO(human): implement the reranking. Four steps:
      1. Loop over `results.items()` as (query_id, candidates). For each:
           candidate_ids = list(candidates)          # the doc_ids to re-score
      2. Build the pair list the cross-encoder wants — [(query_text, doc_text), ...]:
           pairs = [(queries[query_id], doc_text(corpus[d])) for d in candidate_ids]
         `doc_text` is imported above; it joins a BEIR doc's title + text.
      3. Score them all in one call (it batches internally):
           scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
         -> a numpy array, one float per pair, ALIGNED WITH candidate_ids by index.
         These are logits: unbounded, often negative. Fine for ranking — BEIR
         only reads the order — but don't compare them to Phase 0's cosines.
      4. Zip ids back to scores:
           reranked[query_id] = {d: float(s) for d, s in zip(candidate_ids, scores)}
         `float()` matters — pytrec_eval wants Python floats, not numpy scalars.

      DO NOT truncate to the top 10. Return all ~100 candidates reordered, and
      let the metric apply its own cutoff. Truncating collapses Recall@100 to
      Recall@10 and makes the ablation row incomparable to Phase 0's.

    Then delete the `raise NotImplementedError` below and run:
        pytest tests/test_rerank.py
    """
    raise NotImplementedError("TODO(human): see the docstring above")


def main(dataset: str, top_k: int, refresh: bool, limit: int | None) -> None:
    corpus, queries, qrels = load_beir(dataset)

    results = cached_retrieval(
        dataset, BI_ENCODER, top_k,
        compute=lambda: retrieve(load_encoder(BI_ENCODER), corpus, queries, top_k=top_k),
        refresh=refresh,
    )

    if limit:  # fast iteration while implementing — metrics won't match a full run
        results = dict(list(results.items())[:limit])
        log.warning("--limit %d: scoring a SUBSET; these numbers are not comparable "
                    "to the ablation table.", limit)

    log.info("Reranking %d queries x ~%d candidates with %s ...",
             len(results), top_k, CROSS_ENCODER)
    reranked = rerank(load_cross_encoder(), corpus, queries, results)

    # Same candidates, three orderings: the bi-encoder's, the cross-encoder's,
    # and the perfect one. The middle number is only meaningful between the two.
    base, _, _, _ = EvaluateRetrieval.evaluate(qrels, results, [10])
    rr, _, _, _ = EvaluateRetrieval.evaluate(qrels, reranked, [10])
    ceiling, _, _, _ = EvaluateRetrieval.evaluate(qrels, oracle_rerank(results, qrels), [10])

    b, r, c = base["NDCG@10"], rr["NDCG@10"], ceiling["NDCG@10"]

    log.info("\n=== Phase 1 (BEIR/%s, rerank top-%d) ===", dataset, top_k)
    log.info("Phase 0 · bi-encoder order  : %.4f", b)
    log.info("Phase 1 · cross-encoder     : %.4f  (%+.4f)", r, r - b)
    log.info("ceiling · perfect reorder   : %.4f", c)

    if c > b:
        log.info("\ncaptured %.0f%% of the available reranking headroom", 100 * (r - b) / (c - b))
    if r < b:
        log.warning("\nThe reranker made things WORSE. Check that pairs are "
                    "(query, doc) and not (doc, query), and that scores stayed "
                    "aligned with candidate_ids.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nfcorpus")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="rerank only the first N queries (fast smoke test while implementing)")
    args = p.parse_args()
    main(args.dataset, args.top_k, args.refresh, args.limit)
