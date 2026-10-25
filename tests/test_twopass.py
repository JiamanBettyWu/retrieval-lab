"""Tests for the two-pass refusal labeller (part 1 of issue #6).

    pytest tests/test_twopass.py

What is pinned: the order of the passes and the lock on pass A; what each pass
may and may not show; that each stored value matches the word on screen; that
annotators and guide versions never mix in one file; and that resume and skip
lose nothing. No model, no Ollama, no download.
"""
import json
from pathlib import Path

import pytest

from retrieval_lab import twopass
from retrieval_lab.generate import REFUSAL, Generation

CORPUS = {d: {"title": f"Title {d}", "text": f"PASSAGE-TEXT-{d}"} for d in ["dA", "dB", "dC"]}
QRELS = {"q1": {"dA": 1, "dC": 1}, "q2": {"dC": 1}, "q3": {"dB": 1}, "q4": {"dA": 1}}

REFUSED_RAW = ("<rationale>The passages do not name the architect.</rationale>\n"
               "<answer>INSUFFICIENT_CONTEXT</answer>")


def gen(query_id, refused=True, truncated=False, raw=REFUSED_RAW):
    return Generation(query_id=query_id, question=f"QUESTION-{query_id}?",
                      doc_ids=["dA", "dB", "dC"], raw=raw,
                      answer=REFUSAL if refused else "Oslo",
                      rationale="The passages do not name the architect.",
                      refused=refused, truncated=truncated)


GENS = [gen("q1"), gen("q2"), gen("q3", refused=False), gen("q4", truncated=None)]


def replies(*answers):
    it = iter(answers)
    return lambda _prompt="": next(it)


class Screen:
    """Collects everything the labeller prints."""
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        self.lines.append(s)

    @property
    def text(self):
        return "\n".join(self.lines)


def ticking(step=2.0):
    t = [0.0]
    def clock():
        t[0] += step
        return t[0]
    return clock


@pytest.fixture
def sheet(tmp_path):
    path = tmp_path / "twopass__x__betty.jsonl"
    twopass.write_twopass_sheet(GENS, CORPUS, QRELS, path, "betty", "0.1")
    return path


def label(path, *answers, **kw):
    screen = Screen()
    n = twopass.label_twopass(path, "betty", "0.1", ask=replies(*answers),
                              say=screen, clock=ticking(), **kw)
    return n, screen


def rows(path):
    return twopass.read_rows(path)


# ─────────────────────────── which rows ───────────────────────────

def test_only_refused_rows_that_were_not_cut_off_reach_the_sheet(sheet):
    assert [r["query_id"] for r in rows(sheet)] == ["q1", "q2", "q4"]


def test_cut_off_and_unrecorded_rows_are_counted():
    gens = [gen("q1"), gen("q2", truncated=True), gen("q4", truncated=None)]
    kept, counts = twopass.eligible_rows(gens)
    assert [g.query_id for g in kept] == ["q1", "q4"]
    assert counts == {"batch": 3, "refused": 3, "cut_off": 1, "not_recorded": 1}


def test_the_sheet_is_never_overwritten(sheet):
    with pytest.raises(FileExistsError):
        twopass.write_twopass_sheet(GENS, CORPUS, QRELS, sheet, "betty", "0.1")


def test_annotators_get_separate_files(tmp_path):
    gp = Path("cache/gen__ds__enc__top100__ctx10__qwen3-8b__v2__n30__seed2__exabc.json")
    a = twopass.twopass_path(tmp_path, gp, "betty")
    b = twopass.twopass_path(tmp_path, gp, "volunteer1")
    assert a != b and a.parent == b.parent == tmp_path
    assert "seed2__exabc" in a.name            # the batch key carries over whole
    for bad in ("../elsewhere", "betty\n", ""):
        with pytest.raises(ValueError):
            twopass.twopass_path(tmp_path, gp, bad)


# ─────────────────────────── what is shown ───────────────────────────

def test_pass_a_shows_no_passages_and_no_stratum(sheet):
    _, screen = label(sheet, "q")
    assert "QUESTION-q1?" in screen.text
    assert "The passages do not name the architect." in screen.text
    assert "PASSAGE-TEXT" not in screen.text
    assert "gold" not in screen.text.lower()


def test_newlines_in_a_passage_keep_its_indent(sheet):
    row = dict(rows(sheet)[0], passages=["[1] T. first line\nsecond line"])
    out = twopass.render(row, "B")
    assert "  [1] T. first line\n  second line" in out


def test_the_internal_refusal_marker_is_never_shown(sheet):
    _, screen = label(sheet, "q")
    assert REFUSAL not in screen.text
    assert "INSUFFICIENT_CONTEXT" in screen.text       # the model's own words


def test_pass_b_shows_passages_but_no_stratum_and_no_pass_a_label(sheet):
    label(sheet, *["a", "c"] * 3)                       # finish pass A
    _, screen = label(sheet, "q")
    assert "PASSAGE-TEXT-dA" in screen.text
    assert "[1] Title dA." in screen.text               # numbered as the prompt did
    assert "gold" not in screen.text.lower()
    assert "accept" not in screen.text.lower()
    assert "confident" not in screen.text.lower()


def test_no_stratum_after_a_label_either(sheet):
    """The old labeller revealed the stratum after each answer; this one never does."""
    _, screen = label(sheet, *["r", "u"] * 3)
    assert "gold" not in screen.text.lower()
    _, screen = label(sheet, "o", "n", "k", "k")
    assert "gold" not in screen.text.lower()


# ─────────────────────────── polarity ───────────────────────────

def test_each_stored_value_matches_the_word_on_screen(sheet):
    label(sheet, "a", "c", "r", "u", "a", "u")
    a = rows(sheet)
    assert [r["refusal_ok_response"] for r in a] == [True, False, True]   # accept, reject
    assert [r["confidence_response"] for r in a] == ["confident", "unsure", "unsure"]

    label(sheet, "o", "y", "k", "o", "n")
    b = rows(sheet)
    assert [r["refusal_ok_passages"] for r in b] == [False, True, False]  # over-refusal = False
    assert [r["needs_common_knowledge"] for r in b] == [True, None, False]


def test_the_on_screen_words_are_the_ones_in_the_mapping():
    assert twopass.PASS_A_CHOICES["a"] == ("accept", True)
    assert twopass.PASS_B_CHOICES["o"] == ("over-refusal", False)
    assert twopass.PASS_B_CHOICES["k"] == ("refusal OK", True)
    assert "a = accept" in twopass.PASS_A_QUESTION
    assert "o = over-refusal" in twopass.PASS_B_QUESTION
    assert "k = refusal OK" in twopass.PASS_B_QUESTION


def test_the_checkbox_is_asked_only_on_over_refusal(sheet):
    label(sheet, *["a", "c"] * 3)
    _, screen = label(sheet, "k", "k", "k")
    assert "average adult" not in screen.text
    assert [r["needs_common_knowledge"] for r in rows(sheet)] == [None, None, None]


# ─────────────────────────── order and lock ───────────────────────────

def test_pass_b_does_not_start_until_pass_a_is_complete(sheet):
    label(sheet, "a", "c", "s", "a", "c")              # q2 skipped
    _, screen = label(sheet, "q")
    assert "PASS A" in screen.text                      # still pass A
    assert "PASSAGE-TEXT" not in screen.text


def test_pass_b_begins_on_the_next_run_not_the_same_one(sheet):
    n, screen = label(sheet, *["a", "c"] * 3)
    assert n == 3
    assert "PASSAGE-TEXT" not in screen.text
    assert "Run --label again to begin pass B" in screen.text


def test_pass_a_is_read_only_once_pass_b_has_begun(sheet):
    label(sheet, *["a", "c"] * 3)
    label(sheet, "k", "q")
    data = rows(sheet)
    data[1]["refusal_ok_response"] = None               # a hand edit
    sheet.write_text("\n".join(json.dumps(r) for r in data) + "\n")
    with pytest.raises(AssertionError, match="changed after pass B began"):
        label(sheet, "a", "c")


def test_a_changed_pass_a_label_is_caught_once_pass_b_has_begun(sheet):
    """Not just a blanked label: an accept edited to reject is complete and
    in-vocabulary, so only the fingerprint taken as pass B began can see it."""
    label(sheet, *["a", "c"] * 3)
    label(sheet, "k", "q")
    data = rows(sheet)
    data[0]["refusal_ok_response"] = False              # accept -> reject, by hand
    sheet.write_text("\n".join(json.dumps(r) for r in data) + "\n")
    with pytest.raises(AssertionError, match="changed after pass B began"):
        label(sheet, "k")
    with pytest.raises(AssertionError, match="changed after pass B began"):
        twopass.validate_twopass(data, {"q1", "q2", "q4"})


def test_pass_a_can_change_freely_before_pass_b_begins(sheet):
    label(sheet, *["a", "c"] * 3)
    assert {r["pass_a_lock"] for r in rows(sheet)} == {None}


def test_a_different_guide_version_is_refused(sheet):
    with pytest.raises(AssertionError, match="guide version"):
        twopass.label_twopass(sheet, "betty", "0.2", ask=replies(), say=Screen())


def test_another_annotators_file_is_refused(sheet):
    with pytest.raises(AssertionError, match="labels by"):
        twopass.label_twopass(sheet, "volunteer1", "0.1", ask=replies(), say=Screen())


# ─────────────────────────── resume and timing ───────────────────────────

def test_quitting_keeps_every_label_given_and_resumes(sheet):
    label(sheet, "a", "c", "q")
    assert twopass.pass_a_done(rows(sheet)[0])
    assert not twopass.pass_a_done(rows(sheet)[1])
    _, screen = label(sheet, "q")
    assert "2 of 3 rows" in screen.text
    assert "QUESTION-q2?" in screen.text


def test_quitting_halfway_through_a_row_keeps_no_partial_answer(sheet):
    label(sheet, "r", "q")                              # label given, confidence not
    assert rows(sheet)[0]["refusal_ok_response"] is None


def test_seconds_run_from_display_to_the_label_not_the_follow_up(sheet):
    """The follow-up (confidence, checkbox) is not timed: the checkbox is asked
    only on over-refusals, so timing it would make pass B's seconds depend on
    the label itself."""
    label(sheet, "a", "c", "q")
    assert rows(sheet)[0]["seconds_response"] == 2.0    # ticks: start 2, label 4
    label(sheet, *["a", "c"] * 2)
    label(sheet, "o", "y", "k", "k")
    assert [r["seconds_passages"] for r in rows(sheet)] == [2.0, 2.0, 2.0]


def test_an_unrecognised_key_asks_again(sheet):
    label(sheet, "y", "accept", "a", "x", "c", "q")
    assert rows(sheet)[0]["refusal_ok_response"] is True


# ─────────────────────────── --check ───────────────────────────

def finished(sheet):
    label(sheet, *["a", "c"] * 3)
    label(sheet, "o", "y", "k", "k")
    return rows(sheet)


def test_a_finished_file_passes(sheet):
    twopass.validate_twopass(finished(sheet), {"q1", "q2", "q4"})


def test_a_duplicated_row_fails(sheet):
    data = finished(sheet)
    with pytest.raises(AssertionError, match="duplicated"):
        twopass.validate_twopass(data + [data[0]], {"q1", "q2", "q4"})


def test_a_missing_row_fails(sheet):
    with pytest.raises(AssertionError, match="missing"):
        twopass.validate_twopass(finished(sheet)[:2], {"q1", "q2", "q4"})


def test_a_checkbox_on_a_refusal_ok_row_fails(sheet):
    data = finished(sheet)
    data[1]["needs_common_knowledge"] = False
    with pytest.raises(AssertionError, match="needs_common_knowledge"):
        twopass.validate_twopass(data, {"q1", "q2", "q4"})


def test_an_unfinished_pass_fails(sheet):
    label(sheet, *["a", "c"] * 3)
    with pytest.raises(AssertionError, match="refusal_ok_passages"):
        twopass.validate_twopass(rows(sheet), {"q1", "q2", "q4"})
