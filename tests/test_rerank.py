"""Fixture tests for rerank(). Run:  pytest tests/test_rerank.py

Each test names one failure mode, so a red run tells you which step is wrong
rather than just "the number is low."

Uses the real cross-encoder on four docs (instant). A stub would hide the
thing most worth checking: that model.predict() is fed (query, doc) pairs in
that order, and that its output stays aligned with the candidate ids.

The second half pins the statistics behind `--breakdown`. Those numbers decide
a claim in the README, so they are held to the same standard as the retrieval
code — a correlation quietly depending on input order would be exactly the kind
of well-formed-but-wrong output this repo keeps getting bitten by.
"""
import pytest
from beir.retrieval.evaluation import EvaluateRetrieval

from retrieval_lab.rerank import (
    _partial,
    _ranks,
    load_cross_encoder,
    n_relevant_in_top_k,
    per_query_mrr,
    rerank,
    spearman_permutation,
)

CORPUS = {
    "MED-001": {"title": "Aspirin and heart disease",
                "text": "Low-dose aspirin reduces the risk of heart attack."},
    "MED-002": {"title": "Vitamin D deficiency",
                "text": "Vitamin D deficiency is linked to bone loss and rickets."},
    "MED-003": {"title": "Sourdough bread baking",
                "text": "A wild yeast starter gives sourdough its sour flavor."},
    "MED-004": {"title": "Volcanic activity in Iceland",
                "text": "Iceland sits on the Mid-Atlantic Ridge and has frequent eruptions."},
}

QUERIES = {
    "q1": "does aspirin prevent heart attacks?",
    "q2": "what happens when you lack vitamin D?",
    # Deliberately has NO entry in RESULTS below. `results` is the authoritative
    # set of what gets reranked and may be a strict subset of `queries` — the
    # --limit flag produces exactly that. Iterating `queries` instead raises
    # KeyError here; with identical key sets the bug is invisible.
    "q3": "how do volcanoes form?",
}

# A deliberately BAD first-stage ordering: the correct doc is ranked last both
# times. A working reranker must pull it to the front.
RESULTS = {
    "q1": {"MED-003": 0.90, "MED-004": 0.80, "MED-002": 0.70, "MED-001": 0.60},
    "q2": {"MED-004": 0.90, "MED-003": 0.80, "MED-001": 0.70, "MED-002": 0.60},
}
QRELS = {"q1": {"MED-001": 1}, "q2": {"MED-002": 1}}
EXPECTED_TOP = {"q1": "MED-001", "q2": "MED-002"}

# Keyed off RESULTS, not QUERIES — only retrieved queries get reranked.
QUERY_IDS = list(RESULTS)


@pytest.fixture(scope="module")
def reranked():
    return rerank(load_cross_encoder(), CORPUS, QUERIES, RESULTS)


def test_every_query_gets_an_entry(reranked):
    assert set(reranked) == set(RESULTS)


def test_iterates_results_not_queries(reranked):
    """`results` drives the loop, and it can be a strict subset of `queries`.

    `--limit N` truncates results while queries stays whole, so a rerank() that
    loops over `queries` and indexes `results[q]` raises KeyError on the real
    entrypoint while every shape test still passes. q3 exists only in QUERIES.
    """
    assert "q3" in QUERIES and "q3" not in RESULTS, "fixture no longer tests this"
    assert "q3" not in reranked


@pytest.mark.parametrize("q", QUERY_IDS)
def test_result_is_a_flat_dict(reranked, q):
    assert isinstance(reranked[q], dict)


@pytest.mark.parametrize("q", QUERY_IDS)
def test_candidate_set_is_unchanged(reranked, q):
    """Reranking REORDERS — it must not drop or add candidates.

    Truncating to the top 10 here would silently collapse Recall@100 to
    Recall@10 and make the Phase 1 ablation row incomparable to Phase 0's.
    """
    assert set(reranked[q]) == set(RESULTS[q])


@pytest.mark.parametrize("q", QUERY_IDS)
def test_scores_are_plain_floats(reranked, q):
    """model.predict returns numpy scalars; pytrec_eval wants Python floats."""
    assert all(isinstance(v, float) for v in reranked[q].values())


@pytest.mark.parametrize("q", QUERY_IDS)
def test_relevant_doc_is_promoted_to_rank_one(reranked, q):
    """The behavioural check.

    The correct doc starts LAST in RESULTS, so passing this means the model
    actually re-scored the pairs. Failing it while the shape tests pass points
    at pairs built as (doc, query) instead of (query, doc), or scores that
    drifted out of alignment with candidate_ids.
    """
    assert max(reranked[q], key=reranked[q].get) == EXPECTED_TOP[q]


def test_reranking_beats_the_input_ordering(reranked):
    """End-to-end through BEIR's evaluator, the way the real run is scored.

    The fixture's first-stage order is worst-possible and every relevant doc
    was retrieved, so a working reranker reaches exactly 1.0.
    """
    before, _, _, _ = EvaluateRetrieval.evaluate(QRELS, RESULTS, [10])
    after, _, _, _ = EvaluateRetrieval.evaluate(QRELS, reranked, [10])
    assert after["NDCG@10"] > before["NDCG@10"]
    assert after["NDCG@10"] == pytest.approx(1.0)


# --- the statistics behind --breakdown -------------------------------------
# These decide a claim in the README, so they get pinned like everything else.

def test_ranks_average_ties():
    """Tied values share the mean of the ranks they span, else the correlation
    silently depends on input order."""
    assert _ranks([10, 10, 20]) == [1.5, 1.5, 3.0]
    assert _ranks([5, 1, 3]) == [3.0, 1.0, 2.0]


def test_spearman_is_pm_one_on_monotone_input():
    rho, _ = spearman_permutation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], trials=10)
    assert rho == pytest.approx(1.0)
    rho, _ = spearman_permutation([1, 2, 3, 4, 5], [9, 7, 5, 3, 1], trials=10)
    assert rho == pytest.approx(-1.0)


def test_spearman_ignores_outlier_magnitude():
    """The reason the reported rho is rank-based: one wild value must not move
    it. Replacing the last y with -10000 changes no ranks, so rho is unchanged."""
    a, _ = spearman_permutation([1, 2, 3, 4], [4, 3, 2, 1], trials=10)
    b, _ = spearman_permutation([1, 2, 3, 4], [4, 3, 2, -10000], trials=10)
    assert a == pytest.approx(b)


def test_permutation_p_is_small_for_a_real_negative_trend():
    _, p = spearman_permutation(list(range(20)), list(range(20, 0, -1)), trials=2000)
    assert p < 0.01


def test_partial_correlation_zeroes_a_pure_confound():
    """y is driven only by z; x correlates with y solely through z. Holding z
    constant must collapse x's apparent effect — the check that dissolved the
    qrel-density attribution."""
    z = [1, 2, 3, 4, 5, 6, 7, 8]
    x = [v + 0.0 for v in z]        # x is a copy of z
    y = [2 * v for v in z]          # y is driven by z alone
    assert _partial(_ranks(x), _ranks(y), _ranks(z)) == pytest.approx(0.0, abs=1e-9)


def test_n_relevant_in_top_k_counts_only_the_cutoff():
    """Slot-filling count must respect the cutoff and read the score order,
    not dict insertion order."""
    run = {"q1": {"MED-001": 0.1, "MED-002": 0.9}}
    rel = {"q1": {"MED-001": 1}}
    assert n_relevant_in_top_k(rel, run, "q1", k=2) == 1
    assert n_relevant_in_top_k(rel, run, "q1", k=1) == 0   # MED-002 outranks it


def test_per_query_mrr_matches_hand_computed_rank():
    """Relevant doc sits third by score, so the reciprocal rank is 1/3."""
    run = {"q1": {"a": 0.9, "b": 0.8, "c": 0.7}}
    rel = {"q1": {"c": 1}}
    assert per_query_mrr(rel, run)["q1"] == pytest.approx(1 / 3)
