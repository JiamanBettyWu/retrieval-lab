"""Tests for judging a two-pass sheet (issue #15).

    pytest tests/test_judge_twopass.py

What is pinned: which way "over-refusal" points on both sides of the pairing;
that an unfinished or edited sheet stops the run before any model is asked;
that the batch a sheet belongs to is read off its name and then checked; that
rulings are made once per batch however many sheets share it; and the 2×2 on a
hand-worked case. No model, no Ollama, no download — every sheet here is
written and labelled inside the test, from invented rows.
"""
import json
from dataclasses import asdict

import pytest

from retrieval_lab import cache, judge, twopass
from retrieval_lab.generate import REFUSAL, Generation
from retrieval_lab.stats import wilson_ci

DATASET = "toyset"
BATCH = f"{DATASET}__enc__top100__ctx10__gen__v2__n4__seed9"
CORPUS = {d: {"title": f"Title {d}", "text": f"PASSAGE-TEXT-{d}"} for d in ["dA", "dB", "dC"]}
QRELS = {"q1": {"dA": 1, "dC": 1}, "q2": {"dC": 1}, "q3": {"dB": 1}, "q4": {"dA": 1}}
RAW = ("<rationale>The passages do not name the architect.</rationale>\n"
       "<answer>INSUFFICIENT_CONTEXT</answer>")


def gen(query_id, refused=True, truncated=False):
    return Generation(query_id=query_id, question=f"QUESTION-{query_id}?",
                      doc_ids=["dA", "dB", "dC"], raw=RAW,
                      answer=REFUSAL if refused else "Oslo",
                      rationale="The passages do not name the architect.",
                      refused=refused, truncated=truncated)


# q3 is answered and q5 was cut off, so the batch's refusals are q1, q2, q4.
GENS = [gen("q1"), gen("q2"), gen("q3", refused=False), gen("q4"),
        gen("q5", truncated=True)]


def keys(*answers):
    it = iter(answers)
    return lambda _prompt="": next(it)


def ticking(step=2.0):
    t = [0.0]
    def clock():
        t[0] += step
        return t[0]
    return clock


def label(path, annotator, *answers, guide="0.1"):
    twopass.label_twopass(path, annotator, guide, ask=keys(*answers),
                          say=lambda s="": None, clock=ticking())


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A cache/ holding one generation batch, and a labels/ directory beside it."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(cache, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(judge, "CACHE_DIR", cache_dir)
    (cache_dir / f"gen__{BATCH}.json").write_text(json.dumps([asdict(g) for g in GENS]))
    (tmp_path / "labels").mkdir()
    return tmp_path


GUIDE = "0.1"
RUBRIC = judge.RUBRIC_FOR_GUIDE[GUIDE]      # the rubric a sheet under GUIDE is ruled under


def new_sheet(repo, annotator="betty", gold=None, guide=GUIDE):
    path = repo / "labels" / f"twopass__{BATCH}__{annotator}.jsonl"
    twopass.write_twopass_sheet(GENS, CORPUS, QRELS, path, annotator, guide, gold)
    return path


def finish(path, annotator="betty", guide="0.1"):
    """q1: accepted in pass A, over-refusal in pass B. q2 and q4: rejected, refusal OK."""
    label(path, annotator, "a", "c", "r", "c", "r", "u", guide=guide)
    label(path, annotator, "o", "n", "k", "k", guide=guide)
    return path


def ruling(query_id, verdict):
    return judge.Judgement(query_id=query_id, judge="toy-judge", digest="abc",
                           axis="refusal_ok", raw=f"<judge_answer>{verdict}</judge_answer>",
                           verdict=verdict, parsed=verdict is not None)


class FakeJudge:
    """Stands in for the model: answers from a table and counts its calls."""
    def __init__(self, monkeypatch, verdicts):
        self.calls = []
        self.rubrics = []
        self.verdicts = verdicts
        monkeypatch.setattr(judge, "judge_one", self)
        monkeypatch.setattr(judge, "ollama_digest", lambda model: "abc")
        monkeypatch.setattr(judge, "load_beir", lambda dataset: (CORPUS, {}, QRELS))

    def __call__(self, judge_name, digest, item, axis, timeout=600,
                 rubric=judge.RUBRIC_VERSION):
        self.calls.append((item["query_id"], axis))
        self.rubrics.append(rubric)
        return ruling(item["query_id"], self.verdicts[item["query_id"]])


# ─────────────────────────── polarity ───────────────────────────

def test_the_over_refusal_key_is_the_positive_class(repo):
    """Through `PASS_B_CHOICES`, not through a literal False: the stored value
    is only defined by the word that was on screen when it was entered."""
    assert twopass.PASS_B_CHOICES["o"] == ("over-refusal", False)
    assert twopass.PASS_A_CHOICES["a"] == ("accept", True)
    rows = twopass.read_rows(finish(new_sheet(repo)))
    js = [ruling("q1", False), ruling("q2", True), ruling("q4", True)]
    pairs = dict(zip(["q1", "q2", "q4"], judge.pair_with_pass_b(rows, js)))
    assert pairs["q1"] == {"over_refusal": True, "flagged": True, "accepted_in_pass_a": True}
    assert pairs["q2"] == {"over_refusal": False, "flagged": False,
                           "accepted_in_pass_a": False}


def test_a_judge_false_is_a_flag_and_the_rubric_says_so(repo):
    """`flagged` reads verdict False as over-refusal. That is only right while
    the prompt gives `false` that meaning, so the two are pinned together."""
    prompt = judge.build_judge_prompt("Q?", ["[1] p"], "", "r", "refusal_ok")
    assert "the answer is false (over-refusal)" in prompt
    rows = twopass.read_rows(finish(new_sheet(repo)))
    js = [ruling("q1", True), ruling("q2", False), ruling("q4", True)]
    flagged = [p["flagged"] for p in judge.pair_with_pass_b(rows, js)]
    assert flagged == [False, True, False]


def test_an_unparsed_ruling_is_passed_on_marked_not_dropped(repo):
    rows = twopass.read_rows(finish(new_sheet(repo)))
    js = [ruling("q1", None), ruling("q2", True), ruling("q4", True)]
    pairs = judge.pair_with_pass_b(rows, js)
    assert len(pairs) == 3
    assert pairs[0]["flagged"] is None


def test_a_sheet_row_with_no_ruling_raises(repo):
    rows = twopass.read_rows(finish(new_sheet(repo)))
    with pytest.raises(AssertionError, match="1 sheet rows have no ruling"):
        judge.pair_with_pass_b(rows, [ruling("q1", True), ruling("q2", True)])


# ─────────────────────── which batch a sheet is ───────────────────────

def test_the_batch_is_read_off_the_sheet_name(repo):
    path = repo / "labels" / f"twopass__{BATCH}__betty.jsonl"
    assert judge.sheet_generation_path(path, "betty") == \
        cache.CACHE_DIR / f"gen__{BATCH}.json"


def test_an_annotator_name_with_double_underscores_does_not_split_the_batch(repo):
    path = repo / "labels" / f"twopass__{BATCH}__v__1.jsonl"
    assert judge.sheet_generation_path(path, "v__1") == \
        cache.CACHE_DIR / f"gen__{BATCH}.json"


def test_a_file_not_named_like_a_sheet_is_refused(repo):
    with pytest.raises(ValueError, match="expected twopass__<batch>__betty.jsonl"):
        judge.sheet_generation_path(repo / "labels" / "copy-of-sheet.jsonl", "betty")


def test_a_sheet_from_another_dataset_is_refused(repo):
    path = finish(new_sheet(repo))
    with pytest.raises(ValueError, match="is not a otherset batch"):
        judge.open_sheet(path, "otherset", lambda: QRELS)


def test_a_sheet_renamed_onto_another_batch_fails_the_anchor(repo):
    """The name is only a pointer; the `raw_sha` on every row is the proof."""
    other = f"{DATASET}__enc__top100__ctx10__gen__v2__n4__seed10"
    changed = [gen("q1"), gen("q2"), gen("q4")]
    for g in changed:
        g.raw = g.raw.replace("architect", "engineer")
    (cache.CACHE_DIR / f"gen__{other}.json").write_text(
        json.dumps([asdict(g) for g in changed]))
    path = finish(new_sheet(repo))
    moved = path.rename(path.with_name(f"twopass__{other}__betty.jsonl"))
    with pytest.raises(AssertionError, match="generation text that has changed"):
        judge.open_sheet(moved, DATASET, lambda: QRELS)


# ─────────────────── an unfinished sheet is never judged ───────────────────

def refuses_before_any_model(repo, monkeypatch, path, match):
    fake = FakeJudge(monkeypatch, {})
    monkeypatch.setattr(judge, "require_models", lambda models: pytest.fail(
        "the judge was looked for before the sheet was checked"))
    with pytest.raises(AssertionError, match=match):
        judge.main_twopass([str(path)], "toy-judge", DATASET)
    assert fake.calls == []
    assert not list(cache.CACHE_DIR.glob("judge__*"))


def test_a_sheet_with_pass_b_unfinished_is_refused(repo, monkeypatch):
    path = new_sheet(repo)
    label(path, "betty", "a", "c", "r", "c", "r", "u")      # pass A only
    refuses_before_any_model(repo, monkeypatch, path, "no valid 'refusal_ok_passages'")


def test_a_sheet_with_no_labels_is_refused(repo, monkeypatch):
    refuses_before_any_model(repo, monkeypatch, new_sheet(repo),
                             "no valid 'refusal_ok_response'")


def test_a_pass_a_label_changed_after_the_lock_is_refused(repo, monkeypatch):
    path = finish(new_sheet(repo))
    rows = twopass.read_rows(path)
    rows[1]["refusal_ok_response"] = True       # was "reject"; still in vocabulary
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    refuses_before_any_model(repo, monkeypatch, path, "pass A was changed after pass B")


def test_one_unfinished_sheet_stops_the_others_too(repo, monkeypatch):
    done = finish(new_sheet(repo, "betty"), "betty")
    half = new_sheet(repo, "v1")
    label(half, "v1", "a", "c", "r", "c", "r", "u")
    fake = FakeJudge(monkeypatch, {})
    with pytest.raises(AssertionError):
        judge.main_twopass([str(done), str(half)], "toy-judge", DATASET)
    assert fake.calls == []


def test_the_same_sheet_twice_is_refused(repo, monkeypatch):
    path = finish(new_sheet(repo))
    FakeJudge(monkeypatch, {})
    with pytest.raises(SystemExit, match="passed twice"):
        judge.main_twopass([str(path), str(path)], "toy-judge", DATASET)


# ─────────────────────── the run and its cache ───────────────────────

VERDICTS = {"q1": False, "q2": True, "q4": True}


def test_every_refusal_is_judged_once_on_the_refusal_axis_with_passages(repo, monkeypatch):
    path = finish(new_sheet(repo))
    fake = FakeJudge(monkeypatch, VERDICTS)
    seen = []
    monkeypatch.setattr(judge, "pair_with_pass_b",
                        lambda rows, js: seen.append((rows, js)) or [])
    judge.main_twopass([str(path)], "toy-judge", DATASET)
    # the answered row (q3) and the cut-off row (q5) never reach the judge
    assert fake.calls == [("q1", "refusal_ok"), ("q2", "refusal_ok"), ("q4", "refusal_ok")]
    cached = cache.CACHE_DIR / (f"judge__gen__{BATCH}__refusals__toy-judge__"
                                f"{RUBRIC}.json")
    assert cached.exists()
    assert [j.query_id for j in seen[0][1]] == ["q1", "q2", "q4"]


def test_the_sheets_guide_version_picks_the_rubric_and_each_rubric_has_its_own_cache(
        repo, monkeypatch):
    """A ruling is scored against a label, so it must be made under the rule
    the label was made under. The sheet says which guide that was; no flag and
    no default may say otherwise. Rulings under one rubric must not land in,
    be read from, or disturb the file of another."""
    old = finish(new_sheet(repo, "betty", guide="0.1"), "betty")
    new = finish(new_sheet(repo, "v1", guide="0.2"), "v1", guide="0.2")
    fake = FakeJudge(monkeypatch, VERDICTS)
    stem = f"judge__gen__{BATCH}__refusals__toy-judge__"

    judge.main_twopass([str(old)], "toy-judge", DATASET)
    assert fake.rubrics == ["v3"] * 3
    assert (cache.CACHE_DIR / f"{stem}v3.json").exists()
    assert not (cache.CACHE_DIR / f"{stem}v4.json").exists()

    judge.main_twopass([str(new)], "toy-judge", DATASET)     # judged again, under v4
    assert fake.rubrics == ["v3"] * 3 + ["v4"] * 3
    v3_bytes = (cache.CACHE_DIR / f"{stem}v3.json").read_bytes()

    judge.main_twopass([str(old)], "toy-judge", DATASET)     # reads its own file back
    judge.main_twopass([str(new)], "toy-judge", DATASET)
    assert len(fake.rubrics) == 6
    assert (cache.CACHE_DIR / f"{stem}v3.json").read_bytes() == v3_bytes


def test_two_sheets_under_different_guides_in_one_run_each_get_their_own_rubric(
        repo, monkeypatch, caplog):
    old = finish(new_sheet(repo, "betty", guide="0.1"), "betty")
    new = finish(new_sheet(repo, "v1", guide="0.2"), "v1", guide="0.2")
    fake = FakeJudge(monkeypatch, VERDICTS)
    with caplog.at_level("INFO"):
        judge.main_twopass([str(old), str(new)], "toy-judge", DATASET)
    assert sorted(fake.rubrics) == ["v3"] * 3 + ["v4"] * 3
    assert "betty · guide 0.1 · rubric v3 ·" in caplog.text
    assert "v1 · guide 0.2 · rubric v4 ·" in caplog.text


def test_a_guide_version_with_no_rubric_stops_the_run_before_anything_is_judged(
        repo, monkeypatch):
    path = finish(new_sheet(repo, guide="0.3"), guide="0.3")
    fake = FakeJudge(monkeypatch, VERDICTS)
    with pytest.raises(SystemExit, match="guide 0.3, and no rubric"):
        judge.main_twopass([str(path)], "toy-judge", DATASET)
    assert fake.calls == []


def test_the_judge_reads_the_corpus_passages(repo):
    items = judge.judge_items(twopass.eligible_rows(GENS)[0], CORPUS)
    assert [it["query_id"] for it in items] == ["q1", "q2", "q4"]
    assert items[0]["passages"] == ["[1] Title dA. PASSAGE-TEXT-dA",
                                    "[2] Title dB. PASSAGE-TEXT-dB",
                                    "[3] Title dC. PASSAGE-TEXT-dC"]
    assert {it["axis"] for it in items} == {"refusal_ok"}


def test_two_sheets_on_one_batch_share_one_set_of_rulings(repo, monkeypatch):
    a = finish(new_sheet(repo, "betty"), "betty")
    b = finish(new_sheet(repo, "v1"), "v1")
    fake = FakeJudge(monkeypatch, VERDICTS)
    judge.main_twopass([str(a), str(b)], "toy-judge", DATASET)
    assert len(fake.calls) == 3


def test_a_filtered_sheet_judges_only_its_strata_under_its_own_key(repo, monkeypatch):
    """A `--gold` sheet leaves strata off, and nobody labelled those rows: the
    judge is not asked about them, and the smaller set gets its own cache file
    so it cannot be mistaken for the full batch's."""
    full = finish(new_sheet(repo, "betty"), "betty")
    part = new_sheet(repo, "v1", gold={0, 2})        # only q1 has 0 or 2 gold passages
    assert [r["query_id"] for r in twopass.read_rows(part)] == ["q1"]
    label(part, "v1", "a", "c")
    label(part, "v1", "o", "n")
    fake = FakeJudge(monkeypatch, VERDICTS)
    paired = []
    real = judge.pair_with_pass_b
    monkeypatch.setattr(judge, "pair_with_pass_b",
                        lambda rows, js: paired.append(real(rows, js)) or paired[-1])
    judge.main_twopass([str(full)], "toy-judge", DATASET)
    judge.main_twopass([str(part)], "toy-judge", DATASET)
    assert [q for q, _axis in fake.calls] == ["q1", "q2", "q4", "q1"]
    assert [len(p) for p in paired] == [3, 1]
    names = sorted(f.name.split("__refusals")[1] for f in cache.CACHE_DIR.glob("judge__*"))
    assert names == [f"-gold02__toy-judge__{RUBRIC}.json",
                     f"__toy-judge__{RUBRIC}.json"]


def test_a_spreadsheet_file_reads_from_the_whole_batch_rulings(repo):
    """A spreadsheet annotator's file holds part of the batch and no filter."""
    rows = [{"query_id": "q1", "timing": "session"}]
    got = judge.judged_set(rows, GENS, lambda: pytest.fail("no filter, no qrels"))
    assert [g.query_id for g in got["judged"]] == ["q1", "q2", "q4"]
    assert got["key_suffix"] == "__refusals"


def test_a_second_run_reads_the_cache(repo, monkeypatch):
    path = finish(new_sheet(repo))
    fake = FakeJudge(monkeypatch, VERDICTS)
    judge.main_twopass([str(path)], "toy-judge", DATASET)
    judge.main_twopass([str(path)], "toy-judge", DATASET)
    assert len(fake.calls) == 3


def test_cached_rulings_for_other_rows_of_the_same_count_are_refused(repo, monkeypatch):
    """`judge_batch` used to compare counts only."""
    FakeJudge(monkeypatch, VERDICTS)
    items = judge.judge_items(twopass.eligible_rows(GENS)[0], CORPUS)
    judge.judge_batch("toy-judge", items, "k")
    swapped = [dict(items[0], query_id="q9")] + items[1:]
    with pytest.raises(AssertionError, match="different draw"):
        judge.judge_batch("toy-judge", swapped, "k")


# ─────────────────────────── score_twopass ───────────────────────────
def pair(over_refusal, flagged, accepted=False):
    return {"over_refusal": over_refusal, "flagged": flagged,
            "accepted_in_pass_a": accepted}


# The docstring's worked example: 10 rows, 4 over-refusals in pass B, the judge
# flags 5 rows and 3 of them are among the 4. Two of the 4 were accepted in
# pass A, and the judge flagged one of those two.
WORKED = ([pair(True, True, accepted=True), pair(True, True), pair(True, True),
           pair(True, False, accepted=True)]
          + [pair(False, True), pair(False, True, accepted=True)]
          + [pair(False, False, accepted=True)] * 2 + [pair(False, False)] * 2)


def test_the_two_by_two_on_a_hand_worked_case():
    s = judge.score_twopass(WORKED)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (3, 2, 1, 4)
    assert s["unparsed"] == 0
    assert s["precision"] == (3, 5, wilson_ci(3, 5))
    assert s["recall"] == (3, 4, wilson_ci(3, 4))


def test_recall_on_the_rows_pass_a_accepted():
    """Only over-refusals that pass A accepted are in this denominator: the two
    accepted rows that pass B called refusal OK are not."""
    assert judge.score_twopass(WORKED)["recall_flips"] == (1, 2, wilson_ci(1, 2))


def test_row_order_does_not_change_the_score():
    assert judge.score_twopass(WORKED[::-1]) == judge.score_twopass(WORKED)


def test_no_over_refusals_gives_an_empty_recall_not_a_crash():
    s = judge.score_twopass([pair(False, False), pair(False, True)])
    assert s["recall"] == (0, 0, None)
    assert s["recall_flips"] == (0, 0, None)
    assert s["precision"] == (0, 1, wilson_ci(0, 1))


def test_an_unparsed_ruling_counts_as_not_flagged():
    """The rule, fixed before any real ruling was read: a reply that did not
    parse flagged nothing, so on an over-refusal it is a miss (fn) and stays in
    recall's denominator. It is also counted on its own, in `unparsed`."""
    s = judge.score_twopass(WORKED + [pair(True, None, accepted=True), pair(False, None)])
    assert s["unparsed"] == 2
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (3, 2, 2, 5)
    assert s["recall"] == (3, 5, wilson_ci(3, 5))
    assert s["recall_flips"] == (1, 3, wilson_ci(1, 3))
    assert s["precision"] == (3, 5, wilson_ci(3, 5))


# ─────────────────────────── the printed table ───────────────────────────
# Fed a hand-built score, because the stub raises before `report_twopass` is
# reached: without this, its first run would be on real rulings.

def test_the_table_prints_counts_and_intervals(caplog):
    score = {"tp": 3, "fp": 2, "fn": 1, "tn": 4, "unparsed": 1,
             "precision": (3, 5, wilson_ci(3, 5)), "recall": (3, 4, wilson_ci(3, 4)),
             "recall_flips": (0, 0, None)}
    with caplog.at_level("INFO", logger="retrieval_lab"):
        judge.report_twopass("toy sheet", score)
    out = caplog.text
    assert "toy sheet" in out
    assert "3/5 = 60.0%" in out and "95% Wilson" in out
    assert "3/4 = 75.0%" in out
    assert "0/0" in out and "0/0 =" not in out      # an empty rate has no percentage
    assert "unparsed rulings: 1" in out


# ─────────────────────────── the cap on the judge ───────────────────────────

def one_ruling(monkeypatch, reply):
    seen = {}
    def fake(prompt, model, timeout, max_tokens):
        seen["max_tokens"] = max_tokens
        return reply
    monkeypatch.setattr(judge, "ollama_generate", fake)
    item = {"query_id": "q1", "question": "Q?", "passages": ["[1] p"],
            "answer": "", "rationale": "r"}
    return judge.judge_one("toy-judge", "abc", item, "refusal_ok"), seen


def test_the_judge_call_is_capped(monkeypatch):
    j, seen = one_ruling(monkeypatch, {"response": "<judge_answer>false</judge_answer>",
                                       "done_reason": "stop"})
    assert seen["max_tokens"] == judge.JUDGE_MAX_TOKENS
    assert (j.verdict, j.parsed, j.truncated) == (False, True, False)


def test_a_reply_the_cap_ended_is_unparsed_even_with_a_verdict_tag_in_it(monkeypatch):
    looping = "<judge_answer>true</judge_answer> and again " * 50
    j, _ = one_ruling(monkeypatch, {"response": looping, "done_reason": "length"})
    assert (j.verdict, j.parsed, j.truncated) == (None, False, True)
    assert j.raw == looping                     # kept as emitted, for reading later


def test_rulings_cached_before_the_cap_still_load():
    old = {"query_id": "q1", "judge": "m", "digest": "abc", "axis": "refusal_ok",
           "raw": "<judge_answer>true</judge_answer>", "verdict": True,
           "parsed": True, "seconds": 1.0}
    assert judge.Judgement(**old).truncated is None


# ─────────────────────────── review of PR #16 ───────────────────────────

def sheet_dict(batch, ids, guide="0.1", flagged=True):
    return {"gen_path": cache.CACHE_DIR / f"gen__{batch}.json", "guide_version": guide,
            "rows": [{"query_id": q} for q in ids],
            "pairs": [pair(True, flagged) for _ in ids]}


def test_sheets_from_different_batches_pool_into_one_list(repo):
    pairs, why_not = judge.pooled_pairs([sheet_dict("a", ["q1", "q2"]),
                                         sheet_dict("b", ["q1"])])
    assert len(pairs) == 3 and why_not == ""      # q1 of batch a is not q1 of batch b


def test_a_row_on_two_sheets_is_not_pooled(repo):
    """Two annotators on one batch share ONE ruling per row. Pooled, each
    ruling would be counted twice and the interval computed over twice the rows."""
    pairs, why_not = judge.pooled_pairs([sheet_dict("a", ["q1", "q2"]),
                                         sheet_dict("a", ["q2", "q4"])])
    assert pairs is None and "1 rows are on more than one sheet" in why_not


def test_sheets_under_different_guide_versions_are_not_pooled(repo):
    pairs, why_not = judge.pooled_pairs([sheet_dict("a", ["q1"], guide="0.1"),
                                         sheet_dict("b", ["q1"], guide="0.2")])
    assert pairs is None and "guide versions 0.1, 0.2" in why_not


def test_two_annotators_on_one_batch_get_two_tables_and_no_pool(repo, monkeypatch, caplog):
    a = finish(new_sheet(repo, "betty"), "betty")
    b = finish(new_sheet(repo, "v1"), "v1")
    FakeJudge(monkeypatch, VERDICTS)
    with caplog.at_level("INFO", logger="retrieval_lab"):
        judge.main_twopass([str(a), str(b)], "toy-judge", DATASET)
    assert "no pooled table: 3 rows are on more than one sheet" in caplog.text
    assert "pooled (" not in caplog.text
    assert "betty · guide 0.1 ·" in caplog.text and "v1 · guide 0.1 ·" in caplog.text


def test_a_file_that_is_not_a_two_pass_sheet_gets_a_plain_error(repo):
    other = repo / "labels" / "fixture__toyset__n30__seed1.jsonl"
    other.write_text(json.dumps({"query_id": "q1", "grounded": True}) + "\n")
    with pytest.raises(ValueError, match="is not a two-pass sheet"):
        judge.locate_sheet(other, DATASET)
    empty = repo / "labels" / f"twopass__{BATCH}__betty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="is not a two-pass sheet: this file has no rows"):
        judge.locate_sheet(empty, DATASET)
    with pytest.raises(FileNotFoundError, match="no sheet at"):
        judge.locate_sheet(repo / "labels" / "missing.jsonl", DATASET)


def test_a_wrong_dataset_stops_before_the_corpus_loads(repo, monkeypatch):
    path = finish(new_sheet(repo))
    monkeypatch.setattr(judge, "load_beir", lambda dataset: pytest.fail(
        "the corpus was loaded before the sheet's name was checked"))
    with pytest.raises(ValueError, match="is not a otherset batch"):
        judge.main_twopass([str(path)], "toy-judge", "otherset")


def test_check_file_checks_an_annotator_given_on_its_own(repo):
    """Given only one of the two, it used to re-read both from the file and
    compare the file with itself."""
    path = finish(new_sheet(repo))
    gen_path = cache.CACHE_DIR / f"gen__{BATCH}.json"
    with pytest.raises(AssertionError, match="not 'v1'"):
        twopass.check_file(path, GENS, gen_path, lambda: QRELS, annotator="v1")
    with pytest.raises(AssertionError, match="not '9.9'"):
        twopass.check_file(path, GENS, gen_path, lambda: QRELS, guide_version="9.9")
    rows, minutes = twopass.check_file(path, GENS, gen_path, lambda: QRELS, annotator="betty")
    assert len(rows) == 3 and minutes is None


def test_sheet_batch_undoes_twopass_path(repo):
    gen_path = cache.CACHE_DIR / f"gen__{BATCH}.json"
    for annotator in ("betty", "v__1"):
        path = twopass.twopass_path(repo / "labels", gen_path, annotator)
        assert twopass.sheet_batch(path, annotator) == twopass.batch_name(gen_path)


def test_the_bakeoff_says_its_refusal_rows_were_labelled_under_another_rubric(caplog):
    scored = {"m": {"kappa": 0.5, "kappa_refusal": 0.1, "n": 16, "n_refusal": 14,
                    "miss_rate": 0.0, "sec_per_call": 1.0, "digest": "abc"}}
    with caplog.at_level("INFO", logger="retrieval_lab"):
        judge.report(scored)
    assert ("NOT LIKE FOR LIKE" in caplog.text) == (judge.RUBRIC_VERSION != "v2")
