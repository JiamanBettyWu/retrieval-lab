"""Fixture test for oracle_rerank(). Run:  pytest tests/test_oracle.py

Pure function — no model, no download, instant. The hand-built case has a
known-correct answer, which the real oracle run cannot have (its input is real
qrels), so this is where the logic gets pinned down.
"""
import pytest
import pytrec_eval

from retrieval_lab.oracle import arithmetic_ceiling, oracle_rerank

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

QUERY_IDS = list(RESULTS)


@pytest.fixture(scope="module")
def oracle():
    return oracle_rerank(RESULTS, QRELS)


def rank_order(scores):
    return sorted(scores, key=scores.get, reverse=True)


def ndcg_at_10(run):
    ev = pytrec_eval.RelevanceEvaluator(QRELS, {"ndcg_cut.10"})
    return {q: m["ndcg_cut_10"] for q, m in ev.evaluate(run).items()}


def test_orders_by_true_grade_descending(oracle):
    # C(grade 2), then A(grade 1), then the unjudged B/D in either order.
    assert rank_order(oracle["q1"])[:2] == ["C", "A"]


def test_promotes_the_relevant_doc(oracle):
    assert rank_order(oracle["q2"]) == ["Y", "X"]


@pytest.mark.parametrize("q", QUERY_IDS)
def test_candidate_set_is_unchanged(oracle, q):
    """The load-bearing constraint.

    If E (relevant, un-retrieved) ever appears here, the function is drawing
    candidates from qrels and therefore measuring a perfect RETRIEVER — which
    would silently inflate the ceiling Phase 1 gets judged against.
    """
    assert set(oracle[q]) == set(RESULTS[q])


@pytest.mark.parametrize("q", QUERY_IDS)
def test_is_an_upper_bound_per_query(oracle, q):
    base, orc = ndcg_at_10(RESULTS), ndcg_at_10(oracle)
    assert orc[q] >= base[q] - 1e-9


def test_ceiling_is_capped_by_recall_not_ordering(oracle):
    """q1 can't reach 1.0: IDCG counts E, which retrieval never returned.
    q2 can, because nothing relevant was missed. Same oracle, different cap —
    the gap below 1.0 is a recall statement, not an ordering one."""
    orc = ndcg_at_10(oracle)
    assert orc["q1"] < 1.0
    assert orc["q2"] == pytest.approx(1.0)


def test_oracle_is_optimal_not_merely_better(oracle):
    """`oracle >= baseline` passes for ANY improvement, including one leaving
    gain on the table. Only agreement with an independently-derived arithmetic
    ceiling establishes that the oracle is the maximum."""
    orc = ndcg_at_10(oracle)
    mean_ndcg = sum(orc.values()) / len(orc)
    assert mean_ndcg == pytest.approx(arithmetic_ceiling(RESULTS, QRELS, 10), abs=1e-9)
