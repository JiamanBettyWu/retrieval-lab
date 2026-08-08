"""Fixture test for oracle_rerank(). Run:  python test_oracle.py

Pure function, no model, no download — runs instantly. The hand-built case has
a known-correct answer, which the real oracle run cannot have (its input is
real qrels), so this is where the logic gets pinned down.
"""
from beir.retrieval.evaluation import EvaluateRetrieval

from oracle import oracle_rerank

# The bi-encoder's (deliberately bad) ordering. D is retrieved but unjudged;
# E is relevant but was NEVER retrieved — it exists only in the qrels.
RESULTS = {
    "q1": {"A": 0.91, "B": 0.83, "C": 0.77, "D": 0.60},
    "q2": {"X": 0.88, "Y": 0.42},
}
QRELS = {
    "q1": {"C": 2, "A": 1, "E": 1},   # E is the un-retrieved relevant doc
    "q2": {"Y": 1},
}


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        raise AssertionError(f"{name}{' — ' + detail if detail else ''}")


def rank_order(scores):
    return sorted(scores, key=scores.get, reverse=True)


def main():
    oracle = oracle_rerank(RESULTS, QRELS)

    print("\n[1] reorders by true grade, descending")
    # C(grade 2) then A(grade 1) then the unjudged B/D in some order.
    check("q1 order is C, A, then unjudged",
          rank_order(oracle["q1"])[:2] == ["C", "A"],
          f"got {rank_order(oracle['q1'])}")
    check("q2 promotes the relevant doc Y over X",
          rank_order(oracle["q2"]) == ["Y", "X"],
          f"got {rank_order(oracle['q2'])}")

    print("\n[2] reorders only — never invents documents")
    # The load-bearing constraint. If E (relevant, un-retrieved) appears here,
    # the function is reading qrels for candidates and measuring a perfect
    # RETRIEVER, which would silently inflate the ceiling.
    for q in RESULTS:
        check(f"'{q}' candidate set unchanged",
              set(oracle[q]) == set(RESULTS[q]),
              f"added {set(oracle[q]) - set(RESULTS[q])}, dropped {set(RESULTS[q]) - set(oracle[q])}")

    print("\n[3] it is an upper bound, per query")
    import pytrec_eval
    ev = pytrec_eval.RelevanceEvaluator(QRELS, {"ndcg_cut.10"})
    base = {q: m["ndcg_cut_10"] for q, m in ev.evaluate(RESULTS).items()}
    orc = {q: m["ndcg_cut_10"] for q, m in ev.evaluate(oracle).items()}
    for q in base:
        check(f"'{q}' oracle {orc[q]:.4f} >= baseline {base[q]:.4f}",
              orc[q] >= base[q] - 1e-9)

    print("\n[4] the ceiling is capped by recall, not by ordering")
    # q1's best possible: C(2) at rank 1, A(1) at rank 2. But IDCG counts E too
    # — C(2), A(1), E(1) — so even a perfect reorder cannot reach 1.0.
    # q2 retrieved its only relevant doc, so a perfect reorder DOES reach 1.0.
    check("q1 oracle < 1.0 (E was never retrieved)", orc["q1"] < 1.0, f"got {orc['q1']:.4f}")
    check("q2 oracle == 1.0 (nothing was missed)", abs(orc["q2"] - 1.0) < 1e-9, f"got {orc['q2']:.4f}")

    ndcg, _, _, _ = EvaluateRetrieval.evaluate(QRELS, oracle, [10])
    print(f"\n       fixture oracle NDCG@10 = {ndcg['NDCG@10']:.4f}")
    print("\nAll checks passed.\n")


if __name__ == "__main__":
    main()
