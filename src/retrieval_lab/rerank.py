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
import random
import statistics

import pytrec_eval
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import CrossEncoder

from .cache import cached_retrieval
from .data import load_beir
from .observability import op
from .oracle import oracle_rerank
from .retrieve import BI_ENCODER, MODEL_HELP, doc_text, load_encoder, retrieve

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

    Returns the SAME shape, {query_id: {doc_id: cross_encoder_score}}, with the
    same doc_ids per query — only the scores (and therefore the order) change.

    Three things `tests/test_rerank.py` pins, each a silent failure otherwise:

    - **Pairs are `(query, doc)`, not `(doc, query)`.** The reversed order still
      produces scores and still ranks; it just ranks worse.
    - **`results` drives the loop, not `queries`.** `results` may be a strict
      subset — `--limit` makes it one — so iterating `queries` raises KeyError
      on the entrypoint while every shape assertion passes.
    - **All ~100 candidates come back reordered, never a truncated top-10.**
      Truncating collapses Recall@100 to Recall@10 and makes the ablation row
      incomparable to Phase 0's. `main()` re-checks this on every real run.

    The scores are logits — unbounded, often negative. Fine for ranking, since
    BEIR reads only the order, but not comparable to Phase 0's cosines.
    """
    reranked = {}

    for q, candidates in results.items():
        candidate_ids = list(candidates)
        pairs = [(queries[q], doc_text(corpus[d])) for d in candidate_ids]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        reranked[q] = {d: float(s) for d, s in zip(candidate_ids, scores)}

    return reranked


def per_query_ndcg(qrels: dict, run: dict) -> dict:
    """{query_id: NDCG@10}. BEIR's evaluate() only returns the corpus mean."""
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    return {q: m["ndcg_cut_10"] for q, m in ev.evaluate(run).items()}


def per_query_mrr(qrels: dict, run: dict) -> dict:
    """{query_id: reciprocal rank of the first relevant doc}."""
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    return {q: m["recip_rank"] for q, m in ev.evaluate(run).items()}


def n_relevant_in_top_k(qrels: dict, run: dict, q: str, k: int = 10) -> int:
    """How many of a query's top-k are relevant — the slot-filling count.

    NDCG@10 conflates 'is the first hit early' with 'are the ten slots full'.
    This separates the second out, which is what distinguishes a reranker that
    is merely weak from one that is actively evicting good documents.
    """
    top = sorted(run[q], key=run[q].get, reverse=True)[:k]
    return sum(1 for d in top if d in qrels.get(q, {}))


AVAIL_BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 10**9)]


def _bucket_label(lo: int, hi: int) -> str:
    if hi == 0:
        return "0 (unwinnable)"
    return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"


def _ranks(xs: list) -> list:
    """Ranks with ties averaged. Ranking is what makes the correlation below
    immune to the outliers that break a mean of ratios."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def _pearson(xs: list, ys: list) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def _partial(rx: list, ry: list, rz: list) -> float:
    """corr(x, y) with z held constant, on pre-ranked inputs.

    The point of the whole exercise: if two candidate explanations are
    correlated with each other, their raw correlations with the outcome say
    nothing about which one is doing the work.
    """
    rxy, rxz, ryz = _pearson(rx, ry), _pearson(rx, rz), _pearson(ry, rz)
    denom = ((1 - rxz**2) ** 0.5) * ((1 - ryz**2) ** 0.5)
    return (rxy - rxz * ryz) / denom if denom else 0.0


def spearman_permutation(xs: list, ys: list, trials: int = 20000) -> tuple:
    """Spearman rho plus a one-sided (rho < 0) permutation p-value.

    Permutation rather than a t-approximation because the capture ratios are
    heavy-tailed and nothing here is normal. Seeded, since the number gets
    reported and must not drift between runs.
    """
    rx, ry = _ranks(xs), _ranks(ys)
    rho = _pearson(rx, ry)
    rng = random.Random(0)
    shuffled, hits = rx[:], 0
    for _ in range(trials):
        rng.shuffle(shuffled)
        if _pearson(shuffled, ry) <= rho:
            hits += 1
    return rho, (hits + 1) / (trials + 1)


def breakdown(qrels: dict, results: dict, reranked: dict) -> None:
    """Why the mean moved — the two views that separate 'weak' from 'broken'.

    A corpus mean cannot distinguish a uniformly small gain from a mix of large
    wins and large losses, and the difference decides whether the next move is
    a better model or a bug hunt.
    """
    base = per_query_ndcg(qrels, results)
    rr = per_query_ndcg(qrels, reranked)
    orc = per_query_ndcg(qrels, oracle_rerank(results, qrels))
    delta = {q: rr[q] - base[q] for q in base}

    won = [d for d in delta.values() if d > 1e-9]
    lost = [d for d in delta.values() if d < -1e-9]
    tied = [d for d in delta.values() if abs(d) <= 1e-9]
    dead = [q for q in base if not set(results[q]) & set(qrels.get(q, {}))]

    log.info("\n--- per-query win/loss ---")
    log.info("improved   : %3d   mean %+.4f   best  %+.4f",
             len(won), statistics.mean(won) if won else 0.0, max(won, default=0.0))
    log.info("degraded   : %3d   mean %+.4f   worst %+.4f",
             len(lost), statistics.mean(lost) if lost else 0.0, min(lost, default=0.0))
    log.info("unchanged  : %3d", len(tied))
    log.info("unwinnable : %3d  (no relevant doc in the top-%d — 0.0 for any reranker)",
             len(dead), len(next(iter(results.values()), {})))

    # Does capture fall as more relevant docs become available? That is the
    # prediction of a qrel-density mismatch — the cross-encoder trained on
    # MS MARCO's ~1 relevant passage per query, NFCorpus has a median of 16.
    #
    # Only queries with headroom take part: the rest have a zero denominator
    # and could not demonstrate anything either way.
    # sorted(), not dict order: the permutation p-value below depends on the
    # order the queries are fed in, and a reported number must not shift with
    # an upstream dict's iteration order.
    live = sorted(q for q in base if orc[q] - base[q] > 1e-9)
    if not live:
        return
    avail = {q: len(set(results[q]) & set(qrels.get(q, {}))) for q in live}
    ratio = {q: (rr[q] - base[q]) / (orc[q] - base[q]) for q in live}

    def pooled(qs: list) -> float:
        """sum(gain) / sum(headroom) — weights each query by what was at stake.

        Preferred over the mean of per-query ratios, which is unstable: a query
        with 0.016 of headroom and a normal-sized loss yields a ratio near -10
        and single-handedly drags a 50-query bucket negative. Pooled never
        divides by an individual near-zero, and it is built the same way as the
        corpus-level headroom figure, so the two are directly comparable.
        """
        return sum(rr[q] - base[q] for q in qs) / sum(orc[q] - base[q] for q in qs)

    log.info("\n--- headroom captured, bucketed by relevant docs available ---")
    log.info("%-14s %5s %9s %9s %9s", "bucket", "n", "pooled", "median", "mean")
    for lo, hi in [(1, 2), (3, 5), (6, 10), (11, 10**9)]:
        qs = [q for q in live if lo <= avail[q] <= hi]
        if not qs:
            continue
        rs = [ratio[q] for q in qs]
        log.info("%-14s %5d %8.1f%% %8.1f%% %8.1f%%",
                 (f"{lo}-{hi}" if hi < 10**9 else f"{lo}+") + " relevant", len(qs),
                 100 * pooled(qs), 100 * statistics.median(rs), 100 * statistics.mean(rs))
    log.info("%-14s %5d %8.1f%% %8.1f%% %8.1f%%", "all", len(live), 100 * pooled(live),
             100 * statistics.median(ratio.values()), 100 * statistics.mean(ratio.values()))
    log.info("pooled is the trustworthy column; mean is shown only because it is the "
             "obvious estimator and it is misleading here (see pooled()'s docstring).")

    caps = [ratio[q] for q in live]
    av = [avail[q] for q in live]
    bs = [base[q] for q in live]
    hd = [orc[q] - base[q] for q in live]

    rho, p = spearman_permutation(av, caps)
    log.info("\n--- does capture fall as availability rises? ---")
    log.info("Spearman rho(availability, capture) = %+.4f   one-sided permutation "
             "p ~ %.4f  (n=%d)", rho, p, len(live))

    # rho alone would overclaim. Availability travels with baseline quality, so
    # "fewer relevant docs available" and "the bi-encoder was already worse" are
    # nearly the same set of queries. Partial correlations say whether either
    # variable survives controlling for the other; on nfcorpus neither does.
    ra, rc, rb, rh = _ranks(av), _ranks(caps), _ranks(bs), _ranks(hd)
    log.info("rho(baseline NDCG,  capture)        = %+.4f", _pearson(rb, rc))
    log.info("rho(headroom,       capture)        = %+.4f", _pearson(rh, rc))
    log.info("rho(availability,   baseline NDCG)  = %+.4f   <- the entanglement",
             _pearson(ra, rb))
    log.info("partial rho(availability, capture | baseline) = %+.4f",
             _partial(ra, rc, rb))
    log.info("partial rho(baseline, capture | availability) = %+.4f",
             _partial(rb, rc, ra))
    # Headroom is the other obvious confound, and it is exonerated: controlling
    # for it leaves availability's association intact rather than dissolving it.
    log.info("partial rho(availability, capture | headroom) = %+.4f",
             _partial(ra, rc, rh))
    log.info("\nA qrel-density mismatch predicts rho(availability, capture) < 0, and it "
             "is. But compare the two partials against BASELINE: if they are close to "
             "each other, availability and baseline quality are interchangeable here, "
             "and the density attribution is NOT established by this test alone — only "
             "the association is. (Partialling out HEADROOM is the control that comes "
             "back clean: availability survives it, so headroom is not doing the work.)")

    _gain_attribution(qrels, results, reranked, base, rr, orc)
    _metric_dissociation(qrels, results, reranked, base, rr)


def _gain_attribution(qrels, results, reranked, base, rr, orc) -> None:
    """Where the corpus-level gain actually comes from, in NDCG points.

    Capture is a percentage; the headline number is a sum of points. A bucket
    can have a mediocre capture rate and still dominate the total by having
    more room, so the two tables can rank buckets differently. This one answers
    "where did the score move", not "how efficiently was room used".
    """
    allq = sorted(base)
    avail = {q: len(set(results[q]) & set(qrels.get(q, {}))) for q in allq}
    total = sum(rr[q] - base[q] for q in allq)
    if abs(total) < 1e-12:
        return

    log.info("\n--- where the gain comes from (absolute NDCG points) ---")
    log.info("total summed gain %.4f over %d queries = %+.4f mean\n",
             total, len(allq), total / len(allq))
    log.info("%14s %5s %10s %11s %10s %9s",
             "bucket", "n", "sum gain", "% of total", "headroom", "capture")
    for lo, hi in AVAIL_BUCKETS:
        qs = [q for q in allq if lo <= avail[q] <= hi]
        if not qs:
            continue
        sg = sum(rr[q] - base[q] for q in qs)
        sh = sum(orc[q] - base[q] for q in qs)
        log.info("%14s %5d %+10.4f %10.1f%% %10.4f %9s", _bucket_label(lo, hi), len(qs),
                 sg, 100 * sg / total, sh,
                 f"{100 * sg / sh:+.1f}%" if sh > 1e-9 else "n/a")


def _metric_dissociation(qrels, results, reranked, base, rr) -> None:
    """The test that discriminates 'specialized' from 'simply weak'.

    A reranker that merely adds noise to a good ordering degrades EVERY metric.
    One trained to surface a single best answer should keep improving MRR --
    which only looks at the first hit -- while damaging NDCG@10 and the count
    of relevant docs in the top ten, both of which reward filling all ten slots.
    Opposite signs on the same queries is a qualitative prediction, much harder
    to explain away as 'less room to improve' than a difference in magnitudes.
    """
    allq = sorted(base)
    avail = {q: len(set(results[q]) & set(qrels.get(q, {}))) for q in allq}
    bm, rm = per_query_mrr(qrels, results), per_query_mrr(qrels, reranked)

    log.info("\n--- MRR vs NDCG@10 by bucket (does it still find rank 1?) ---")
    log.info("%14s %5s %10s %9s %13s %8s", "bucket", "n", "dNDCG@10", "dMRR",
             "rel@10 before", "after")
    for lo, hi in AVAIL_BUCKETS:
        qs = [q for q in allq if lo <= avail[q] <= hi]
        if not qs:
            continue
        dn = sum(rr[q] - base[q] for q in qs) / len(qs)
        dm = sum(rm[q] - bm[q] for q in qs) / len(qs)
        b10 = sum(n_relevant_in_top_k(qrels, results, q) for q in qs) / len(qs)
        a10 = sum(n_relevant_in_top_k(qrels, reranked, q) for q in qs) / len(qs)
        flag = "  <- MRR up, NDCG down" if dm > 1e-9 and dn < -1e-9 else ""
        log.info("%14s %5d %+10.4f %+9.4f %13.2f %8.2f%s",
                 _bucket_label(lo, hi), len(qs), dn, dm, b10, a10, flag)
    log.info("\nA bucket with dMRR > 0 and dNDCG < 0 is being actively mis-served: the "
             "reranker still promotes one good doc to rank 1 but evicts others from the "
             "top ten. Noise added to a good ordering would push BOTH down, so this "
             "pattern is evidence a 'find the single best answer' training objective is "
             "involved — the thing the correlation test above could not isolate.")


def main(dataset: str, top_k: int, refresh: bool, limit: int | None,
         show_breakdown: bool, model: str = BI_ENCODER) -> None:
    corpus, queries, qrels = load_beir(dataset)

    results = cached_retrieval(
        dataset, model, top_k,
        compute=lambda: retrieve(load_encoder(model), corpus, queries, top_k=top_k),
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
    def metrics(run: dict) -> dict:
        ndcg, _, recall, _ = EvaluateRetrieval.evaluate(qrels, run, [10, top_k])
        mrr = EvaluateRetrieval.evaluate_custom(qrels, run, [10], metric="mrr")
        return {**ndcg, **recall, **mrr}

    base, rr = metrics(results), metrics(reranked)
    ceiling = metrics(oracle_rerank(results, qrels))
    b, r, c = base["NDCG@10"], rr["NDCG@10"], ceiling["NDCG@10"]

    log.info("\n=== Phase 1 (BEIR/%s, rerank top-%d from %s) ===", dataset, top_k, model)
    # Columns are named for the STAGE, not the phase. They used to read
    # "Phase 0"/"Phase 1", which was true only while --model was hardcoded to
    # the baseline: run this over a Phase 2 adapter and the left column is
    # Phase 2 output, so the old header mislabelled every number under it.
    log.info("%-14s %10s %10s %10s", "", "retrieved", "reranked", "delta")
    # dict.fromkeys dedupes while keeping order — at --top-k 10 the last two
    # keys collapse into one and the row would otherwise print twice.
    for key in dict.fromkeys(("NDCG@10", "MRR@10", "Recall@10", f"Recall@{top_k}")):
        log.info("%-14s %10.4f %10.4f %+10.4f", key, base[key], rr[key], rr[key] - base[key])
    log.info("%-14s %10.4f   <- perfect reorder of the same candidates", "ceiling", c)

    # Reranking may only REORDER the candidate set, so Recall@top_k cannot move.
    # Any drift here means the reranker truncated its output, which would break
    # comparability with Phase 0 — a silent failure the corpus NDCG would hide.
    rk = f"Recall@{top_k}"
    if abs(rr[rk] - base[rk]) > 1e-9:
        log.error("\n%s CHANGED (%.5f -> %.5f). rerank() must return ALL candidates "
                  "reordered — never a truncated top-N.", rk, base[rk], rr[rk])

    if c > b:
        log.info("\ncaptured %.1f%% of the available reranking headroom "
                 "(%+.4f of a possible %+.4f)", 100 * (r - b) / (c - b), r - b, c - b)
        log.info("read against the %.4f ceiling, not 1.0", c)
    if r < b:
        log.warning("\nThe reranker made things WORSE. Check that pairs are "
                    "(query, doc) and not (doc, query), and that scores stayed "
                    "aligned with candidate_ids.")

    if show_breakdown:
        breakdown(qrels, results, reranked)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nfcorpus")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="rerank only the first N queries (fast smoke test while implementing)")
    p.add_argument("--breakdown", action="store_true",
                   help="per-query win/loss and headroom bucketed by relevant-doc count")
    p.add_argument("--model", default=BI_ENCODER, help=MODEL_HELP)
    args = p.parse_args()
    main(args.dataset, args.top_k, args.refresh, args.limit, args.breakdown, args.model)
