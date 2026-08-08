"""The oracle rerank: Phase 1's ceiling, measured before Phase 1 is built.

    python oracle.py --dataset nfcorpus

A *cheating* reranker. It reads the qrels and orders each query's retrieved
candidates perfectly — but it may only REORDER what retrieval already found,
never add to it. So its NDCG@10 is the hard upper bound on anything a real
cross-encoder can achieve over the same top-k.

Reading the result:
  oracle ≈ baseline  -> retrieval's ordering is already near the best possible;
                        a reranker has nothing to win. Recall is the lever.
  oracle >> baseline -> the right docs are in the top-k, just badly ordered.
                        Exactly the job a cross-encoder does. Build it.

It never reaches 1.0, and the reason IS the finding: NDCG's denominator (IDCG)
is built from the *full* qrels, including relevant docs retrieval never
returned. The oracle can permute the numerator; it cannot recover those.
"""
import argparse
import logging
import math

from beir.retrieval.evaluation import EvaluateRetrieval

from .cache import cached_retrieval
from .data import load_beir
from .retrieve import BI_ENCODER, load_encoder, retrieve

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("retrieval_lab")


def oracle_rerank(results: dict, qrels: dict) -> dict:
    """Perfectly reorder each query's RETRIEVED docs by their true grade."""
    reranked: dict[str, dict[str, float]] = {}

    for query_id, docs in results.items():
        # Iterating `results`, never `qrels` — pulling docs from the ground
        # truth would measure a perfect *retriever*, which is not the question.
        grades = qrels.get(query_id, {})
        # NFCorpus grades are 1 and 2; a retrieved doc absent from qrels is
        # unjudged, so it grades 0 and sinks to the bottom.
        ordered = sorted(docs, key=lambda d: grades.get(d, 0), reverse=True)

        # Strictly descending synthetic scores. Using the grades themselves
        # would give identical NDCG (equally-graded docs contribute equal gain,
        # so tie order can't move the sum) but leaves "this is a total order"
        # implicit rather than stated.
        n = len(ordered)
        reranked[query_id] = {doc_id: 1.0 - i / n for i, doc_id in enumerate(ordered)}

    return reranked


def arithmetic_ceiling(results: dict, qrels: dict, k: int = 10) -> float:
    """Mean NDCG@k computed from first principles, bypassing oracle_rerank().

    Independent of the sort under test: for each query, take the grades of the
    retrieved-relevant docs, place the best k at ranks 1..k, and divide by
    IDCG@k over the *full* qrels. Any correct oracle must equal this exactly.

    This is the check `oracle >= baseline` structurally cannot make — that
    invariant holds for any improvement, including one that leaves gain on the
    table. Only this one says the oracle is *optimal*.
    """
    dcg = lambda grades: sum(g / math.log2(i + 2) for i, g in enumerate(grades))

    total, n = 0.0, 0
    for query_id, grades in qrels.items():
        if query_id not in results:
            continue
        ideal = dcg(sorted(grades.values(), reverse=True)[:k])
        if ideal == 0:
            continue
        # `d != query_id` mirrors BEIR's default ignore_identical_ids.
        retrieved = sorted((grades.get(d, 0) for d in results[query_id] if d != query_id),
                           reverse=True)[:k]
        total += dcg(retrieved) / ideal
        n += 1

    return total / n if n else 0.0


def per_query_ndcg(qrels: dict, run: dict, k: int) -> dict:
    """NDCG@k per query, for the sanity check below."""
    import pytrec_eval

    ev = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}"})
    return {q: m[f"ndcg_cut_{k}"] for q, m in ev.evaluate(run).items()}


def main(dataset: str, top_k: int, refresh: bool) -> None:
    corpus, queries, qrels = load_beir(dataset)

    # The same cached candidates evaluate.py scored — that identity is the
    # whole point of caching, and what makes the comparison below meaningful.
    results = cached_retrieval(
        dataset, BI_ENCODER, top_k,
        compute=lambda: retrieve(load_encoder(BI_ENCODER), corpus, queries, top_k=top_k),
        refresh=refresh,
    )

    oracle = oracle_rerank(results, qrels)

    base_ndcg, _, base_recall, _ = EvaluateRetrieval.evaluate(qrels, results, [10, top_k])
    orc_ndcg, _, _, _ = EvaluateRetrieval.evaluate(qrels, oracle, [10])

    # --- sanity check: an oracle can never rank worse than what it reordered.
    # There's no known-answer fixture for an oracle over real qrels, so this
    # per-query invariant is the check. A mean would hide individual failures.
    base_pq = per_query_ndcg(qrels, results, 10)
    orc_pq = per_query_ndcg(qrels, oracle, 10)
    regressions = [q for q in base_pq if orc_pq[q] < base_pq[q] - 1e-9]
    if regressions:
        raise AssertionError(
            f"{len(regressions)} queries scored WORSE after oracle reranking "
            f"(e.g. {regressions[:3]}) — the sort or the score assignment is wrong."
        )
    log.info("sanity check: oracle >= baseline on all %d queries", len(base_pq))

    # ... and the stronger one: is it OPTIMAL, not merely better?
    ceiling = arithmetic_ceiling(results, qrels, 10)
    if abs(orc_ndcg["NDCG@10"] - ceiling) > 1e-4:
        raise AssertionError(
            f"oracle NDCG@10 {orc_ndcg['NDCG@10']:.6f} != arithmetic ceiling "
            f"{ceiling:.6f} — the oracle is leaving gain on the table."
        )
    log.info("sanity check: oracle == arithmetic ceiling (%.6f) — provably optimal", ceiling)

    b, o = base_ndcg["NDCG@10"], orc_ndcg["NDCG@10"]
    headroom = o - b

    log.info("\n=== Phase 1 ceiling (BEIR/%s, top-%d candidates) ===", dataset, top_k)
    log.info("baseline NDCG@10 (bi-encoder order) : %.4f", b)
    log.info("oracle   NDCG@10 (perfect reorder)  : %.4f", o)
    log.info("available headroom for reranking    : %+.4f  (%.0f%% relative)",
             headroom, 100 * headroom / b)
    log.info("unreachable by reranking (recall)   : %.4f", 1.0 - o)
    log.info("Recall@%-3d (what capped the oracle) : %.4f", top_k, base_recall[f"Recall@{top_k}"])

    log.info("\n--- verdict (D3) ---")
    if headroom < 0.05:
        log.info("Headroom %+.3f is negligible — the bi-encoder already orders its "
                 "candidates near-optimally. A cross-encoder cannot pay off here; "
                 "recall is the binding constraint. Prioritize Phase 2.", headroom)
    else:
        log.info("Headroom %+.3f is material — the right docs ARE in the top-%d, just "
                 "badly ordered. That is exactly what a cross-encoder fixes. "
                 "Build Phase 1.", headroom, top_k)
        log.info("A real reranker capturing half this gap would land near NDCG@10 %.3f.",
                 b + headroom / 2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nfcorpus")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    main(args.dataset, args.top_k, args.refresh)
