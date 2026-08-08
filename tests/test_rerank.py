"""Fixture test for rerank(). Run:  pytest tests/test_rerank.py

RED until rerank() is implemented — that's the point. Each test names one
failure mode, so a red run tells you which step is wrong rather than just
"the number is low."

Uses the real cross-encoder on four docs (instant). A stub would hide the
thing most worth checking: that model.predict() is fed (query, doc) pairs in
that order, and that its output stays aligned with the candidate ids.
"""
import pytest
from beir.retrieval.evaluation import EvaluateRetrieval

from retrieval_lab.rerank import load_cross_encoder, rerank

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
}

# A deliberately BAD first-stage ordering: the correct doc is ranked last both
# times. A working reranker must pull it to the front.
RESULTS = {
    "q1": {"MED-003": 0.90, "MED-004": 0.80, "MED-002": 0.70, "MED-001": 0.60},
    "q2": {"MED-004": 0.90, "MED-003": 0.80, "MED-001": 0.70, "MED-002": 0.60},
}
QRELS = {"q1": {"MED-001": 1}, "q2": {"MED-002": 1}}
EXPECTED_TOP = {"q1": "MED-001", "q2": "MED-002"}

QUERY_IDS = list(QUERIES)


@pytest.fixture(scope="module")
def reranked():
    return rerank(load_cross_encoder(), CORPUS, QUERIES, RESULTS)


def test_every_query_gets_an_entry(reranked):
    assert set(reranked) == set(RESULTS)


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
