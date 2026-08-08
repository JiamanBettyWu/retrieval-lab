"""Phase 0: baseline bi-encoder retrieval + BEIR evaluation.

    python evaluate.py --dataset nfcorpus

Produces the baseline numbers AND the headroom check that decides whether this
dataset is worth building a reranker on.
"""
import argparse
import logging

from beir.retrieval.evaluation import EvaluateRetrieval

from .cache import cached_retrieval
from .data import load_beir
from .observability import init_weave
from .retrieve import BI_ENCODER, load_encoder, retrieve

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("retrieval_lab")


def main(dataset: str, top_k: int, refresh: bool = False) -> None:
    init_weave()  # tracing on if weave + WANDB_API_KEY present; inert otherwise

    log.info("Loading BEIR/%s ...", dataset)
    corpus, queries, qrels = load_beir(dataset)
    log.info("corpus=%d docs | queries=%d | qrels=%d", len(corpus), len(queries), len(qrels))

    log.info("Retrieving top-%d with the bi-encoder ...", top_k)
    results = cached_retrieval(
        dataset, BI_ENCODER, top_k,
        # a thunk, so a cache hit never loads the encoder at all
        compute=lambda: retrieve(load_encoder(BI_ENCODER), corpus, queries, top_k=top_k),
        refresh=refresh,
    )

    ndcg, _map, recall, _precision = EvaluateRetrieval.evaluate(qrels, results, [10, top_k])
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, [10], metric="mrr")

    ndcg10 = ndcg["NDCG@10"]
    mrr10 = mrr["MRR@10"]
    recall_k = recall[f"Recall@{top_k}"]

    log.info("\n=== Phase 0 baseline (BEIR/%s, all-MiniLM-L6-v2) ===", dataset)
    log.info("NDCG@10       : %.4f", ndcg10)
    log.info("MRR@10        : %.4f", mrr10)
    log.info("Recall@%-7d: %.4f", top_k, recall_k)

    # --- Headroom check: is this dataset worth a reranker? ---
    log.info("\n--- headroom check ---")
    if ndcg10 > 0.9:
        log.info("NDCG@10 %.3f already near-perfect — little room for a reranker to show "
                 "lift. Consider a harder BEIR set.", ndcg10)
    else:
        log.info("NDCG@10 %.3f leaves room for reranking/fine-tuning to improve. Good.", ndcg10)
    if recall_k < 0.8:
        log.info("Recall@%d %.3f is low — the reranker is capped (it can't recover docs the "
                 "retriever missed). Raising retrieval recall is the lever.", top_k, recall_k)
    else:
        log.info("Recall@%d %.3f — most relevant docs reach the reranker. Good.", top_k, recall_k)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nfcorpus")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--refresh", action="store_true",
                   help="recompute retrieval, ignoring the cache (use after editing retrieve.py)")
    args = p.parse_args()
    main(args.dataset, args.top_k, args.refresh)
