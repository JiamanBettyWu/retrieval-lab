"""Fixture test for retrieve(). Run:  pytest tests/test_retrieve.py

A five-doc corpus small enough to reason about by hand. Each test targets ONE
failure mode, so a red run names its own cause — unlike the full BEIR run,
where every bug collapses into a single low NDCG number.

Deliberately uses the real MiniLM, not a stub: a stub returning numpy would
mask any mismatch between `convert_to_tensor=True` and what semantic_search
expects, which is one of the things worth verifying. Five docs encode instantly.
"""
import pytest
from beir.retrieval.evaluation import EvaluateRetrieval

from retrieval_lab.retrieve import load_encoder, retrieve

# Doc ids deliberately look like BEIR's (non-numeric, non-sequential) so that
# leaking a `corpus_id` row index through instead of a real id can't pass.
CORPUS = {
    "MED-001": {"title": "Aspirin and heart disease",
                "text": "Low-dose aspirin reduces the risk of heart attack."},
    "MED-002": {"title": "Vitamin D deficiency",
                "text": "Vitamin D deficiency is linked to bone loss and rickets."},
    "MED-003": {"title": "Sourdough bread baking",
                "text": "A wild yeast starter gives sourdough its sour flavor."},
    "MED-004": {"title": "Volcanic activity in Iceland",
                "text": "Iceland sits on the Mid-Atlantic Ridge and has frequent eruptions."},
    "MED-005": {"title": "The rules of cricket",
                "text": "A cricket match is played between two teams of eleven players."},
}

QUERIES = {
    "q1": "does aspirin prevent heart attacks?",
    "q2": "what happens when you lack vitamin D?",
}

# Ground truth: the one obviously-correct doc per query.
QRELS = {
    "q1": {"MED-001": 1},
    "q2": {"MED-002": 1},
}
EXPECTED_TOP = {"q1": "MED-001", "q2": "MED-002"}

QUERY_IDS = list(QUERIES)


@pytest.fixture(scope="module")
def model():
    return load_encoder()


@pytest.fixture(scope="module")
def results(model):
    """top_k=3 over a 5-doc corpus — exercises the truncating path."""
    return retrieve(model, CORPUS, QUERIES, top_k=3)


@pytest.fixture(scope="module")
def wide(model):
    """top_k=100 over a 5-doc corpus — exercises the clamping path."""
    return retrieve(model, CORPUS, QUERIES, top_k=100)


def test_every_query_gets_an_entry(results):
    assert set(results) == set(QUERIES)


@pytest.mark.parametrize("q", QUERY_IDS)
def test_result_is_a_flat_dict(results, q):
    """Catches returning a list of single-key {corpus_id: score} dicts."""
    assert isinstance(results[q], dict)


@pytest.mark.parametrize("q", QUERY_IDS)
def test_keys_are_real_corpus_doc_ids(results, q):
    """Catches `corpus_id` row indices leaking through unmapped.

    This is the silent one: unmapped ids don't crash, they just make
    pytrec_eval find zero overlap with the qrels and report NDCG@10 = 0.0,
    which reads as a weak model rather than a wrong id.
    """
    unknown = [k for k in results[q] if k not in CORPUS]
    assert not unknown, f"not corpus doc_ids: {unknown}"


@pytest.mark.parametrize("q", QUERY_IDS)
def test_scores_are_plain_floats(results, q):
    """pytrec_eval wants Python floats, not torch/numpy scalars."""
    assert all(isinstance(v, float) for v in results[q].values())


@pytest.mark.parametrize("q", QUERY_IDS)
def test_top_k_is_honored(results, q):
    assert len(results[q]) == 3


@pytest.mark.parametrize("q", QUERY_IDS)
def test_top_k_larger_than_corpus_clamps(wide, q):
    assert len(wide[q]) == len(CORPUS)


@pytest.mark.parametrize("q", QUERY_IDS)
def test_correct_doc_ranks_first(results, q):
    """Would catch query/doc args transposed in semantic_search."""
    assert max(results[q], key=results[q].get) == EXPECTED_TOP[q]


def test_beir_evaluator_accepts_the_shape(wide):
    """The integration check — shape-checking your own dict is not the same as
    the downstream consumer accepting it.

    One relevant doc per query, each correctly ranked first, so NDCG@10 must be
    exactly 1.0. Perfect-or-zero is maximally discriminating: 1.0 means the ids
    genuinely line up with the qrels; 0.0 means they don't, however right the
    dict looked.
    """
    ndcg, _map, _recall, _prec = EvaluateRetrieval.evaluate(QRELS, wide, [10])
    assert ndcg["NDCG@10"] == pytest.approx(1.0)
