"""Fixture tests for Phase 4c.1's worksheet plumbing.

    pytest tests/test_fixture.py

Hand labels are the only input in this repo that cannot be regenerated, so the
two things pinned here are both about *not losing them*: the `raw_sha` anchor
that catches labels pointing at regenerated text, and the overwrite guard.
Everything else in `fixture.py` is reporting.

`blank_labels()` is Betty's stub, so the tests that need a labelled row
monkeypatch it — the point is that the plumbing works for ANY axis set she
picks, which is also why `validate_labels` reads its field names from there
rather than hard-coding them.
No model, no Ollama, no download.
"""
import json

import pytest

from retrieval_lab import fixture
from retrieval_lab.generate import Generation

AXES = {"grounded": None, "refusal_ok": None}

CORPUS = {d: {"title": d, "text": f"text of {d}"} for d in ["dA", "dB", "dC"]}
QRELS = {"q1": {"dA": 1, "dC": 1}, "q2": {"dC": 1}}


def gen(query_id="q1", raw="<answer>yes</answer>", doc_ids=("dA", "dB")):
    return Generation(query_id=query_id, question="?", doc_ids=list(doc_ids),
                      raw=raw, answer="yes", rationale="because [1]", refused=False)


@pytest.fixture
def axes(monkeypatch):
    monkeypatch.setattr(fixture, "blank_labels", lambda: dict(AXES))


# ─────────────────────────── the stratum ───────────────────────────

def test_gold_in_context_counts_only_what_retrieval_delivered():
    """dC is gold but was not retrieved — the stratum is about the PROMPT."""
    assert fixture.gold_in_context(QRELS, "q1", ["dA", "dB"]) == 1
    assert fixture.gold_in_context(QRELS, "q1", ["dA", "dC"]) == 2
    assert fixture.gold_in_context(QRELS, "q2", ["dA", "dB"]) == 0


# ────────────────── the label ↔ generation binding ──────────────────

def test_labels_are_anchored_to_the_exact_answer_text(tmp_path, axes):
    """A regenerated batch must invalidate its labels LOUDLY.

    The load-bearing test in this file. Generation is nondeterministic even at
    temperature 0, so `--refresh` silently replaces the text a label was written
    about; nothing else in the pipeline would raise, and the fixture would go on
    reporting a κ computed against answers nobody read.
    """
    batch = [gen(raw="<answer>original</answer>")]
    path = fixture.write_sheet(batch, CORPUS, QRELS, tmp_path / "sheet.jsonl")

    assert fixture.load_labels(path, batch)                       # same text: fine

    regenerated = [gen(raw="<answer>reworded, same meaning</answer>")]
    with pytest.raises(AssertionError, match="has changed"):
        fixture.load_labels(path, regenerated)


def test_labels_from_a_different_draw_are_rejected(tmp_path, axes):
    """Wrong --seed or --n-queries: the ids do not exist in this batch."""
    path = fixture.write_sheet([gen("q1")], CORPUS, QRELS, tmp_path / "sheet.jsonl")
    with pytest.raises(AssertionError, match="not in this batch"):
        fixture.load_labels(path, [gen("q2")])


# ─────────────────────── not destroying labels ───────────────────────

def test_writing_over_an_existing_sheet_is_refused(tmp_path, axes):
    path = tmp_path / "sheet.jsonl"
    fixture.write_sheet([gen()], CORPUS, QRELS, path)
    with pytest.raises(FileExistsError):
        fixture.write_sheet([gen()], CORPUS, QRELS, path)


def test_a_row_that_raises_leaves_the_existing_sheet_intact(tmp_path, monkeypatch):
    """--force must not truncate the file before it knows it can write it."""
    monkeypatch.setattr(fixture, "blank_labels", lambda: dict(AXES))
    path = tmp_path / "sheet.jsonl"
    fixture.write_sheet([gen()], CORPUS, QRELS, path)
    before = path.read_text()

    monkeypatch.setattr(fixture, "blank_labels",
                        lambda: (_ for _ in ()).throw(NotImplementedError))
    with pytest.raises(NotImplementedError):
        fixture.write_sheet([gen()], CORPUS, QRELS, path, force=True)
    assert path.read_text() == before


# ──────────────────────────── the sheet ────────────────────────────

def test_the_sheet_never_shows_the_gold_answer(tmp_path, axes):
    """Groundedness is labelled blind — see the module docstring for why.

    Cheap to state, easy to break by adding one convenient field, and it would
    quietly turn every faithfulness label into a correctness label.
    """
    row = fixture.sheet_row(gen(), CORPUS, QRELS)
    assert "gold" not in json.dumps(row).lower().replace("gold_in_context", "")


def test_passages_are_numbered_the_way_the_prompt_numbered_them(axes):
    """A rationale citing [2] has to be checkable against passage [2]."""
    row = fixture.sheet_row(gen(doc_ids=("dA", "dB")), CORPUS, QRELS)
    assert row["passages"][0].startswith("[1] dA.")
    assert row["passages"][1].startswith("[2] dB.")


def test_the_applicable_axis_is_left_blank_in_a_fresh_row(axes):
    """The row arrives with exactly one decision to make — see the pre-fill tests."""
    row = fixture.sheet_row(gen(), CORPUS, QRELS)
    assert row["grounded"] is None


# ───────────────────────── the rubric check ─────────────────────────

VALUES = {"grounded": {True, False}, "refusal_ok": {True, False}}


@pytest.fixture
def rubric(monkeypatch, axes):
    monkeypatch.setattr(fixture, "LABEL_VALUES", {k: set(v) for k, v in VALUES.items()})


def row(**overrides):
    r = {"query_id": "q1", "grounded": True, "refusal_ok": fixture.NOT_APPLICABLE}
    r.update(overrides)
    return r


def test_a_finished_sheet_passes(rubric):
    fixture.validate_labels([row()], n_expected=1)


def test_a_blank_field_is_not_a_finished_fixture(rubric):
    with pytest.raises(AssertionError, match="blank label field"):
        fixture.validate_labels([row(grounded=None)], n_expected=1)


def test_a_value_outside_the_rubric_raises(rubric):
    """A typo'd label becomes a real category in the κ; nothing else catches it."""
    with pytest.raises(AssertionError, match="outside the rubric"):
        fixture.validate_labels([row(grounded="ture")], n_expected=1)


def test_not_applicable_is_allowed_on_every_axis(rubric):
    """Each row has exactly one axis that does not apply — it is a decision, not a blank."""
    fixture.validate_labels([row(grounded=fixture.NOT_APPLICABLE, refusal_ok=True)],
                            n_expected=1)


def test_an_axis_with_no_rubric_raises(monkeypatch, axes):
    """LABEL_VALUES and blank_labels() drifting apart must not pass silently."""
    monkeypatch.setattr(fixture, "LABEL_VALUES", {"grounded": {True, False}})
    with pytest.raises(AssertionError, match="every axis needs its allowed values"):
        fixture.validate_labels([row()], n_expected=1)


# ───────────────── the inapplicable axis, pre-filled ─────────────────

def test_a_refused_row_has_groundedness_prefilled_as_not_applicable(axes):
    """A refusal makes no claims — labelling it `grounded` would be a free point."""
    g = gen(); g.refused = True
    row = fixture.sheet_row(g, CORPUS, QRELS)
    assert row["grounded"] == fixture.NOT_APPLICABLE
    assert row["refusal_ok"] is None          # this is the axis to actually decide


def test_an_answered_row_has_refusal_ok_prefilled_as_not_applicable(axes):
    row = fixture.sheet_row(gen(), CORPUS, QRELS)
    assert row["refusal_ok"] == fixture.NOT_APPLICABLE
    assert row["grounded"] is None


def test_exactly_one_axis_is_prefilled_per_row(axes):
    """Pre-fill both and there is nothing left to label; pre-fill neither and the
    'n/a' typing it was meant to remove comes back."""
    for refused in (True, False):
        g = gen(); g.refused = refused
        row = fixture.sheet_row(g, CORPUS, QRELS)
        assert sum(row[a] == fixture.NOT_APPLICABLE for a in AXES) == 1


def test_a_renamed_axis_breaks_loudly(monkeypatch):
    """INAPPLICABLE_AXIS must follow blank_labels(), or a row silently loses its n/a."""
    monkeypatch.setattr(fixture, "blank_labels",
                        lambda: {"grounded": None, "refusal_calibrated": None})
    with pytest.raises(AssertionError, match="did not follow"):
        fixture.sheet_row(gen(), CORPUS, QRELS)      # answered row -> refusal axis


# ───────────────────────── the labelling loop ─────────────────────────

def sheet(tmp_path, *rows):
    p = tmp_path / "sheet.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def labelled_row(query_id="q1", refused=False, **overrides):
    """A sheet row as `write_sheet` produces it: one axis pre-filled, one blank."""
    r = {"query_id": query_id, "raw_sha": "abc123",
         "stratum": {"gold_in_context": 2, "refused": refused},
         "question": "?", "passages": ["[1] dA. text"],
         "answer": "yes", "rationale": "because [1]",
         "grounded": fixture.NOT_APPLICABLE if refused else None,
         "refusal_ok": None if refused else fixture.NOT_APPLICABLE}
    r.update(overrides)
    return r


def replies(*answers):
    """A fake terminal that reads from a script and ignores everything printed."""
    it = iter(answers)
    return lambda _prompt: next(it)


def test_it_asks_only_the_axis_that_applies(tmp_path, rubric):
    """A refused row is asked about its refusal; an answered row about grounding."""
    assert fixture.axis_to_label(labelled_row(refused=True)) == "refusal_ok"
    assert fixture.axis_to_label(labelled_row(refused=False)) == "grounded"
    assert fixture.axis_to_label(labelled_row(grounded=True)) is None


def test_answers_are_written_back(tmp_path, rubric):
    path = sheet(tmp_path, labelled_row("q1"), labelled_row("q2", refused=True))
    assert fixture.label_interactive(path, ask=replies("y", "n"), say=lambda *_: None) == 2

    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[0]["grounded"] is True and rows[0]["refusal_ok"] == fixture.NOT_APPLICABLE
    assert rows[1]["refusal_ok"] is False and rows[1]["grounded"] == fixture.NOT_APPLICABLE


def test_quitting_keeps_the_answers_already_given(tmp_path, rubric):
    """A closed laptop at row 20 must not cost rows 1-19."""
    path = sheet(tmp_path, labelled_row("q1"), labelled_row("q2"), labelled_row("q3"))
    assert fixture.label_interactive(path, ask=replies("y", "q"), say=lambda *_: None) == 1

    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[0]["grounded"] is True
    assert rows[1]["grounded"] is None and rows[2]["grounded"] is None


def test_it_resumes_where_it_stopped(tmp_path, rubric):
    """Rows already decided are not asked again — that is what makes it resumable."""
    path = sheet(tmp_path, labelled_row("q1", grounded=False), labelled_row("q2"))
    assert fixture.label_interactive(path, ask=replies("y"), say=lambda *_: None) == 1

    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[0]["grounded"] is False       # untouched
    assert rows[1]["grounded"] is True


def test_skip_leaves_the_row_blank(tmp_path, rubric):
    path = sheet(tmp_path, labelled_row("q1"))
    assert fixture.label_interactive(path, ask=replies("s"), say=lambda *_: None) == 0
    assert json.loads(path.read_text())["grounded"] is None


def test_an_unrecognised_key_reasks_rather_than_labelling(tmp_path, rubric):
    """A fat-fingered keystroke must not silently become a label."""
    path = sheet(tmp_path, labelled_row("q1"))
    fixture.label_interactive(path, ask=replies("maybe", "", "n"), say=lambda *_: None)
    assert json.loads(path.read_text())["grounded"] is False


def test_case_and_stray_whitespace_are_tolerated(tmp_path, rubric):
    """"Y " at 11pm is a yes, not a typo worth re-asking about."""
    path = sheet(tmp_path, labelled_row("q1"))
    fixture.label_interactive(path, ask=replies(" Y "), say=lambda *_: None)
    assert json.loads(path.read_text())["grounded"] is True


def test_the_stratum_is_withheld_while_deciding(rubric):
    """gold_in_context is anchoring bait for refusal_ok — see render_item's docstring."""
    row = labelled_row(refused=True)
    assert "gold docs in context" not in fixture.render_item(row)
    assert "gold docs in context" in fixture.render_item(row, reveal_stratum=True)


def test_a_non_binary_axis_refuses_to_guess(monkeypatch, axes):
    """y/n cannot map onto three categories; better to raise than to invent one."""
    monkeypatch.setattr(fixture, "LABEL_VALUES",
                        {"grounded": {"full", "partial", "none"}, "refusal_ok": {True, False}})
    with pytest.raises(NotImplementedError, match="binary axes only"):
        fixture.binary_axes()


def test_every_axis_has_its_polarity_on_screen(rubric):
    """The one corruption channel no checker can close.

    Inverted labels are in-vocabulary and internally consistent: --check passes,
    κ computes, the number is wrong. So an axis without a rendered question is a
    bug, not a cosmetic gap.
    """
    for axis in AXES:
        assert axis in fixture.AXIS_QUESTIONS
        assert "y =" in fixture.AXIS_QUESTIONS[axis]


def test_the_same_draw_under_a_new_prompt_version_is_not_an_overlap():
    """A prompt bump re-asks the same sampled questions — that is construction,
    not leakage. Warning about it trains the reader to scroll past warnings."""
    v1 = "cache/gen__ds__enc__top100__ctx10__qwen3-8b__v1__n30__seed1.json"
    v2 = "cache/gen__ds__enc__top100__ctx10__qwen3-8b__v2__n30__seed1.json"
    assert fixture.same_draw(v1, v2)
    assert not fixture.same_draw(v1, v1.replace("seed1", "seed2"))
    assert not fixture.same_draw(v1, v1.replace("n30", "n100"))
