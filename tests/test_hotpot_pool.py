"""Fixture tests for the hotpotqa-distractor-pool builder.

    pytest tests/test_hotpot_pool.py

Hand-built rows with known-correct answers, no download, instant — the same
reason `test_oracle.py` exists. The real build reads 7,405 real questions and
therefore has no answer anyone can check by eye; this is where the pooling
logic gets pinned.

What these pin, in order of how expensive the bug would be:
  - a paragraph shared by two questions collapses to ONE doc (that collapse is
    the entire reason the corpus is 66k and not 74k)
  - gold titles resolve to the SAME ids the corpus uses
  - doc ids are content-derived, never positional
"""
import pytest

from retrieval_lab.hotpot_pool import check, doc_id, pool_paragraphs

# Two questions that share the paragraph "Bardstown". Q1's golds are Fighting
# Cock + Bardstown; Q2's are Heaven Hill + Bardstown. "Nelson County" is a
# distractor for both and gold for neither.
ROWS = [
    {
        "id": "q1",
        "question": "Fighting Cock is produced in what Kentucky county?",
        "answer": "Nelson County",
        "supporting_facts": {"title": ["Fighting Cock", "Bardstown"]},
        "context": {
            "title": ["Fighting Cock", "Bardstown", "Nelson County"],
            "sentences": [
                ["Fighting Cock is a bourbon.", " It is made in Bardstown."],
                ["Bardstown is in Nelson County."],
                ["Nelson County is in Kentucky."],
            ],
        },
    },
    {
        "id": "q2",
        "question": "Heaven Hill is headquartered in which city?",
        "answer": "Bardstown",
        "supporting_facts": {"title": ["Heaven Hill", "Bardstown"]},
        "context": {
            "title": ["Heaven Hill", "Bardstown", "Nelson County"],
            # Bardstown appears again, here with a LONGER extraction.
            "sentences": [
                ["Heaven Hill is a distillery."],
                ["Bardstown is in Nelson County.", " It is the county seat."],
                ["Nelson County is in Kentucky."],
            ],
        },
    },
]


@pytest.fixture(scope="module")
def built():
    return pool_paragraphs(ROWS)


def test_shared_paragraphs_collapse(built):
    """The load-bearing one: 6 slots across 2 questions -> 4 unique docs.

    If this ever returns 6, dedup is off and the corpus is a per-question bag
    rather than a pool — every question would then compete only against its own
    distractors, which is not a retrieval task.
    """
    corpus, _, _, _ = built
    assert len(corpus) == 4
    assert {d["title"] for d in corpus.values()} == {
        "Fighting Cock", "Bardstown", "Nelson County", "Heaven Hill"
    }


def test_conflicting_text_keeps_the_longest(built):
    """Bardstown appears twice with different text; the fuller one survives."""
    corpus, _, _, _ = built
    bardstown = corpus[doc_id("Bardstown")]
    assert bardstown["text"] == "Bardstown is in Nelson County. It is the county seat."


def test_gold_ids_match_corpus_ids(built):
    """qrels and corpus must agree on ids, or NDCG is 0.0000 for no visible reason.

    This is the doc-id leak in its Phase 4 costume: both sides look well-formed
    and pytrec_eval simply finds no overlap.
    """
    corpus, _, qrels, _ = built
    assert qrels["q1"] == {doc_id("Fighting Cock"): 1, doc_id("Bardstown"): 1}
    assert set(qrels["q1"]) <= set(corpus)
    assert set(qrels["q2"]) <= set(corpus)


def test_distractors_are_not_labelled_relevant(built):
    """An oracle may only reorder what retrieval found — and only golds are gold.

    "Nelson County" is in both questions' candidate sets and neither's
    supporting_facts. Labelling it relevant would inflate every ceiling.
    """
    _, _, qrels, _ = built
    nelson = doc_id("Nelson County")
    assert nelson not in qrels["q1"]
    assert nelson not in qrels["q2"]


def test_answers_are_carried_through(built):
    _, _, _, answers = built
    assert answers == {"q1": "Nelson County", "q2": "Bardstown"}


def test_doc_ids_are_not_positional(built):
    corpus, _, _, _ = built
    assert not any(d.isdigit() for d in corpus)
    # ...and are a pure function of the title, so two builds agree.
    assert doc_id("Bardstown") == doc_id("Bardstown")
    assert doc_id("Bardstown") != doc_id("Nelson County")


def test_check_passes_on_a_valid_build(built):
    check(*built)


def test_check_catches_a_gold_doc_missing_from_the_corpus(built):
    corpus, queries, qrels, answers = built
    broken = {d: v for d, v in corpus.items() if d != doc_id("Bardstown")}
    with pytest.raises(AssertionError, match="absent from the corpus"):
        check(broken, queries, qrels, answers)


def test_check_catches_a_question_that_is_not_two_hop(built):
    corpus, queries, qrels, answers = built
    broken = {**qrels, "q1": {doc_id("Bardstown"): 1}}
    with pytest.raises(AssertionError, match="exactly 2 gold docs"):
        check(corpus, queries, broken, answers)


def test_check_catches_positional_doc_ids(built):
    _, queries, _, answers = built
    with pytest.raises(AssertionError, match="positional-index leak"):
        check({"0": {"title": "t", "text": "x"}}, queries, {"q1": {"0": 1}}, answers)
