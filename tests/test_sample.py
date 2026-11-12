"""Tests for the per-stratum sample of a batch's refused rows (`twopass --sample`).

    pytest tests/test_sample.py

What is pinned: the draw takes exactly its targets and only rows that would go
on a sheet; it is a pure function of its inputs, so `--check` and the judge can
recompute it; a short stratum is an error unless a fill stratum is named; and
every sheet written after the draw — terminal or spreadsheet — holds the
sampled rows only and says so on each row. No model, no Ollama, no download.
"""
import csv
import json
from pathlib import Path

import pytest

from retrieval_lab import judge, sheets, twopass
from retrieval_lab.generate import REFUSAL, Generation

DOCS = ["dA", "dB", "dC"]
CORPUS = {d: {"title": f"Title {d}", "text": f"PASSAGE-TEXT-{d}"} for d in DOCS}
REFUSED_RAW = ("<rationale>The passages do not name the architect.</rationale>\n"
               "<answer>INSUFFICIENT_CONTEXT</answer>")

# 24 refused rows: gold_in_context 2 for q00-q05, 1 for q06-q19, 0 for q20-q23.
QIDS = [f"q{i:02d}" for i in range(24)]
GOLD = {q: (2 if i < 6 else 1 if i < 20 else 0) for i, q in enumerate(QIDS)}
QRELS = {q: {d: 1 for d in DOCS[:g]} | ({"dX": 1, "dY": 1} if g == 0 else
                                          {"dX": 1} if g == 1 else {})
         for q, g in GOLD.items()}


def gen(query_id, refused=True, truncated=False):
    return Generation(query_id=query_id, question=f"QUESTION-{query_id}?",
                      doc_ids=DOCS, raw=REFUSED_RAW,
                      answer=REFUSAL if refused else "Oslo",
                      rationale="The passages do not name the architect.",
                      refused=refused, truncated=truncated)


# q98 answered and q99 was cut off: neither goes on a sheet, so neither may be drawn.
QRELS |= {"q98": {"dA": 1, "dB": 1}, "q99": {"dA": 1, "dB": 1}}
GENS = [gen(q) for q in QIDS] + [gen("q98", refused=False), gen("q99", truncated=True)]
STRATA = sheets.batch_strata(GENS, QRELS)


def spec(seed=0, fill_from=None, **targets):
    """spec(g2=3, g1=5) → the spec for --sample 2=3 1=5."""
    return twopass.sample_spec({int(k[1:]): n for k, n in targets.items()}, seed, fill_from)


def drawn(sp):
    return twopass.draw_sample(STRATA, sp)["query_ids"]


def by_stratum(ids):
    return {g: sorted(q for q in ids if GOLD[q] == g) for g in (0, 1, 2)}


# ─────────────────────────── the draw ───────────────────────────

def test_the_draw_takes_exactly_its_targets():
    got = twopass.draw_sample(STRATA, spec(g2=3, g1=5, g0=2))
    assert {g: len(q) for g, q in by_stratum(got["query_ids"]).items()} == {2: 3, 1: 5, 0: 2}
    assert got["taken"] == {"0": 2, "1": 5, "2": 3}
    assert got["available"] == {"0": 4, "1": 14, "2": 6}


def test_only_rows_that_go_on_a_sheet_can_be_drawn():
    assert "q98" not in STRATA and "q99" not in STRATA
    everything = drawn(spec(g2=6, g1=14, g0=4))
    assert everything == QIDS


def test_the_same_inputs_give_the_same_rows_and_another_seed_does_not():
    assert drawn(spec(g2=3, g1=5)) == drawn(spec(g2=3, g1=5))
    assert drawn(spec(g2=3, g1=5)) != drawn(spec(seed=1, g2=3, g1=5))


def test_the_draw_does_not_depend_on_the_order_of_the_rows():
    backwards = dict(reversed(list(STRATA.items())))
    assert twopass.draw_sample(backwards, spec(g2=3, g1=5)) == \
        twopass.draw_sample(STRATA, spec(g2=3, g1=5))


def test_one_stratums_rows_do_not_change_with_anothers_target():
    small, large = by_stratum(drawn(spec(g2=2, g1=5))), by_stratum(drawn(spec(g2=5, g1=5, g0=1)))
    assert small[1] == large[1]


def test_a_larger_target_keeps_the_smaller_targets_rows():
    small, large = by_stratum(drawn(spec(g2=2, g1=4))), by_stratum(drawn(spec(g2=2, g1=9)))
    assert set(small[1]) < set(large[1])


def test_a_short_stratum_is_an_error_without_a_fill():
    with pytest.raises(ValueError, match=r"gold=2 has 6, target 8.*--fill-from"):
        drawn(spec(g2=8, g1=5))


def test_a_short_stratum_is_made_up_from_the_fill_stratum():
    got = twopass.draw_sample(STRATA, spec(fill_from=1, g2=8, g1=5))
    assert got["taken"] == {"1": 7, "2": 6}
    assert len(got["query_ids"]) == 13
    # the fill continues gold=1's own order: the 5 it would have had are among the 7
    assert set(by_stratum(drawn(spec(g2=6, g1=5)))[1]) < set(by_stratum(got["query_ids"])[1])


def test_a_fill_stratum_need_not_have_a_target_of_its_own():
    got = twopass.draw_sample(STRATA, spec(fill_from=1, g2=8, g0=2))
    assert got["taken"] == {"0": 2, "1": 2, "2": 6}


def test_a_fill_stratum_that_runs_out_is_an_error():
    with pytest.raises(ValueError, match="gold=1 has 14, but its target plus the fill is 16"):
        drawn(spec(fill_from=1, g2=8, g1=14))
    with pytest.raises(ValueError, match="gold=1 has 14, target 15"):
        drawn(spec(fill_from=1, g2=3, g1=15))


def test_a_one_stratum_sample_is_refused():
    with pytest.raises(ValueError, match="at least two strata"):
        drawn(spec(g1=5))
    with pytest.raises(ValueError, match="at least two strata"):
        drawn(spec(g1=5, g2=0))


@pytest.mark.parametrize("pairs, message", [
    (["2:40"], "write GOLD=N"), (["3=4"], "write GOLD=N"), (["2=x"], "write GOLD=N"),
    (["2=4", "2=5"], "gold=2 twice"),
])
def test_targets_that_cannot_be_read_are_refused(pairs, message):
    with pytest.raises(ValueError, match=message):
        twopass.parse_targets(pairs)


def test_targets_are_read():
    assert twopass.parse_targets(["2=40", "1=60", "0=8"]) == {2: 40, 1: 60, 0: 8}


# ─────────────────────────── the record ───────────────────────────

def test_the_record_is_written_once_and_read_back_as_the_spec(tmp_path):
    path, sp = tmp_path / "sample__x.json", spec(seed=3, fill_from=1, g2=3, g1=5)
    assert twopass.read_sample(path) is None
    twopass.write_sample(path, sp, twopass.draw_sample(STRATA, sp))
    record = twopass.read_sample(path)
    assert twopass.record_spec(record) == sp and record["query_ids"] == drawn(sp)
    assert twopass.verify_sample(record, STRATA, path.name) == sp
    with pytest.raises(FileExistsError, match="fixed once drawn"):
        twopass.write_sample(path, sp, twopass.draw_sample(STRATA, sp))


# ─────────────────────────── a terminal sheet ───────────────────────────

def test_a_sheet_holds_the_sampled_rows_only_in_batch_order(tmp_path):
    path, sp = tmp_path / "twopass__x__betty.jsonl", spec(g2=3, g1=5, g0=2)
    counts = twopass.write_twopass_sheet(GENS, CORPUS, QRELS, path, "betty", "0.1", sample=sp)
    rows = twopass.read_rows(path)
    assert [r["query_id"] for r in rows] == drawn(sp)       # QIDS sort in batch order
    assert all(r["sample"] == sp for r in rows)
    assert counts["on_sheet"] == 10 and counts["other_strata"] == 14


def test_check_recomputes_the_sample_from_the_batch(tmp_path):
    path, sp = tmp_path / "twopass__x__betty.jsonl", spec(g2=3, g1=5)
    twopass.write_twopass_sheet(GENS, CORPUS, QRELS, path, "betty", "0.1", sample=sp)
    assert twopass.expected_ids(twopass.read_rows(path), GENS, lambda: QRELS) == set(drawn(sp))


def test_rows_under_different_samples_in_one_file_are_refused(tmp_path):
    path = tmp_path / "twopass__x__betty.jsonl"
    twopass.write_twopass_sheet(GENS, CORPUS, QRELS, path, "betty", "0.1", sample=spec(g2=3, g1=5))
    rows = twopass.read_rows(path)
    rows[0]["sample"] = spec(seed=9, g2=3, g1=5)
    with pytest.raises(AssertionError, match="different samples"):
        twopass.expected_ids(rows, GENS, lambda: QRELS)
    del rows[0]["sample"]
    with pytest.raises(AssertionError, match="different samples"):
        twopass.expected_ids(rows, GENS, lambda: QRELS)


def test_a_gold_filter_and_a_sample_together_are_refused(tmp_path):
    with pytest.raises(ValueError, match="--gold does not apply"):
        twopass.write_twopass_sheet(GENS, CORPUS, QRELS, tmp_path / "t.jsonl", "betty", "0.1",
                                    gold={1, 2}, sample=spec(g2=3, g1=5))


def test_a_sheet_without_a_sample_records_none(tmp_path):
    path = tmp_path / "twopass__x__betty.jsonl"
    twopass.write_twopass_sheet(GENS, CORPUS, QRELS, path, "betty", "0.1")
    assert all("sample" not in r for r in twopass.read_rows(path))


# ─────────────────────────── spreadsheet annotators ───────────────────────────

def sampled_assignment(sp):
    ids = set(drawn(sp))
    assignment = sheets.make_assignment({q: g for q, g in STRATA.items() if q in ids},
                                        ["v1", "v2"], 4, seed=0)
    assignment["sample"] = sp
    return assignment


def test_exported_rows_record_the_sample_and_the_sheet_does_not_show_it(tmp_path):
    sp = spec(g2=3, g1=5, g0=2)
    assignment = sampled_assignment(sp)
    jsonl = tmp_path / "twopass__x__v1.jsonl"
    csv_path, _template, _risky = sheets.export_sheet("A", jsonl, assignment, "v1", "0.1",
                                                      GENS, CORPUS, QRELS)
    rows = twopass.read_rows(jsonl)
    assert {r["query_id"] for r in rows} == sheets.assigned_ids(assignment, "v1") <= set(drawn(sp))
    assert all(r["sample"] == sp for r in rows)
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    assert not any(word in " ".join(header).lower() for word in ("sample", "gold", "stratum"))
    assert "sample" not in csv_path.read_text(encoding="utf-8-sig").lower()


def test_the_judge_reads_the_sampled_rows_only_under_their_own_name(tmp_path):
    sp = spec(g2=3, g1=5, g0=2)
    jsonl = tmp_path / "twopass__x__v1.jsonl"
    sheets.export_sheet("A", jsonl, sampled_assignment(sp), "v1", "0.1", GENS, CORPUS, QRELS)
    got = judge.judged_set(twopass.read_rows(jsonl), GENS, lambda: QRELS)
    assert [g.query_id for g in got["judged"]] == drawn(sp)

    other = tmp_path / "twopass__x__betty.jsonl"
    twopass.write_twopass_sheet(GENS, CORPUS, QRELS, other, "betty", "0.1", sample=sp)
    same = judge.judged_set(twopass.read_rows(other), GENS, lambda: QRELS)
    assert same["key_suffix"] == got["key_suffix"]           # one sample, one set of rulings

    plain = tmp_path / "twopass__y__betty.jsonl"
    twopass.write_twopass_sheet(GENS, CORPUS, QRELS, plain, "betty", "0.1")
    whole = judge.judged_set(twopass.read_rows(plain), GENS, lambda: QRELS)
    assert whole["key_suffix"] != got["key_suffix"] and len(whole["judged"]) == 24


# ─────────────────────────── the command line ───────────────────────────

@pytest.fixture
def cli(tmp_path, monkeypatch):
    """`twopass.main` on GENS, writing into tmp_path."""
    monkeypatch.setattr(twopass, "load_generations", lambda *a, **k: GENS)
    monkeypatch.setattr(twopass, "load_beir", lambda dataset: (CORPUS, {}, QRELS))
    monkeypatch.setattr(twopass, "generation_cache_path",
                        lambda *a, **k: Path("cache/gen__x.json"))

    def run(annotator=None, **kw):
        twopass.main("d", 30, 10, 100, 0, annotator, "0.1" if annotator else None,
                     tmp_path, kw.pop("label", False), kw.pop("check", False), **kw)
    return run, tmp_path


def test_the_sample_is_drawn_once_and_every_later_sheet_uses_it(cli):
    run, out = cli
    run(sample=["2=3", "1=5", "0=2"], sample_seed=4)
    sp = spec(seed=4, g2=3, g1=5, g0=2)
    assert twopass.record_spec(twopass.read_sample(out / "sample__x.json")) == sp

    run(annotator="betty")
    assert [r["query_id"] for r in twopass.read_rows(out / "twopass__x__betty.jsonl")] == drawn(sp)

    run(assign=["v1", "v2"], shared="4")
    assignment = sheets.read_assignment(out / "assign__x.json")
    assert assignment["sample"] == sp and assignment["n_rows"] == 10
    covered = set(assignment["shared"]) | {q for own in assignment["own"].values() for q in own}
    assert covered == set(drawn(sp))

    with pytest.raises(SystemExit, match="already has sheets or an assignment"):
        run(sample=["2=3", "1=5"])


def test_a_sample_after_a_sheet_is_refused(cli):
    run, out = cli
    run(annotator="betty")
    with pytest.raises(SystemExit, match="twopass__x__betty.jsonl"):
        run(sample=["2=3", "1=5"])
    assert not (out / "sample__x.json").exists()


def test_a_sample_that_cannot_be_drawn_writes_nothing(cli):
    run, out = cli
    with pytest.raises(SystemExit, match="gold=2 has 6, target 8"):
        run(sample=["2=8", "1=5"])
    assert not (out / "sample__x.json").exists()
    run(sample=["2=8", "1=5"], fill_from=1)
    assert json.loads((out / "sample__x.json").read_text())["taken"] == {"1": 7, "2": 6}


def test_gold_is_refused_once_a_sample_exists(cli):
    run, _out = cli
    run(sample=["2=3", "1=5"])
    with pytest.raises(SystemExit, match="--gold does not apply"):
        run(annotator="betty", gold=[1, 2])


def test_the_sample_flags_need_the_sample_action(cli):
    run, _out = cli
    with pytest.raises(SystemExit, match="go with --sample"):
        run(annotator="betty", fill_from=1)


# ─────────────────────────── a sampled sheet, all the way through ───────────────────────────
# The `sample` key rides on every row, so it has to survive labelling, the
# pass A lock, the spreadsheet import and the final check — the check
# `judge --sheet` also runs before it reads a file.

GEN_PATH = Path("cache/gen__x.json")


def keys(*answers):
    it = iter(answers)
    return lambda _prompt="": next(it)


def test_a_sampled_terminal_sheet_passes_the_check_once_labelled(tmp_path):
    path, sp = tmp_path / "twopass__x__betty.jsonl", spec(g2=3, g1=5, g0=2)
    twopass.write_twopass_sheet(GENS, CORPUS, QRELS, path, "betty", "0.1", sample=sp)
    clock = iter(range(0, 10_000, 2))
    for answers in (["a", "c"] * 10, ["k"] * 10):           # pass A, then pass B
        twopass.label_twopass(path, "betty", "0.1", ask=keys(*answers),
                              say=lambda s="": None, clock=lambda: float(next(clock)))
    rows, minutes = twopass.check_file(path, GENS, GEN_PATH, lambda: QRELS)
    assert minutes is None and [r["query_id"] for r in rows] == drawn(sp)
    assert all(r["sample"] == sp and r["pass_a_lock"] for r in rows)

    rows.pop()                                               # a sampled row gone missing
    twopass.save_rows(path, rows)
    with pytest.raises(AssertionError, match="file covers 9 rows, the batch has 10"):
        twopass.check_file(path, GENS, GEN_PATH, lambda: QRELS)


def fill(csv_path, values):
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        table = list(csv.DictReader(f))
    out = csv_path.with_name("filled_" + csv_path.name)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows([{**r, **values} for r in table])
    return out


def sittings(path, start, end):
    with path.open("w", newline="") as f:
        csv.writer(f).writerows([["sitting", sheets.COL_START, sheets.COL_END], [1, start, end]])
    return path


def test_a_sampled_spreadsheet_file_passes_the_check_once_imported(tmp_path):
    sp = spec(g2=3, g1=5, g0=2)
    assignment = sampled_assignment(sp)
    sheets.write_assignment(sheets.assignment_path(tmp_path, GEN_PATH), assignment)
    jsonl = tmp_path / "twopass__x__v1.jsonl"

    a_csv, _t, _r = sheets.export_sheet("A", jsonl, assignment, "v1", "0.1", GENS, CORPUS, QRELS)
    sheets.import_sheet("A", jsonl, fill(a_csv, {sheets.COL_A: "accept",
                                                 sheets.COL_CONFIDENCE: "confident"}),
                        sittings(tmp_path / "sa.csv", "14:00", "14:30"), assignment, "v1", "0.1")
    b_csv, _t, _r = sheets.export_sheet("B", jsonl, assignment, "v1", "0.1")
    assert "sample" not in b_csv.read_text(encoding="utf-8-sig").lower()
    sheets.import_sheet("B", jsonl, fill(b_csv, {sheets.COL_B: "refusal OK",
                                                 sheets.COL_COMMON: ""}),
                        sittings(tmp_path / "sb.csv", "20:00", "21:15"), assignment, "v1", "0.1")

    rows, minutes = twopass.check_file(jsonl, GENS, GEN_PATH, lambda: QRELS)
    assert minutes == {"A": 30.0, "B": 75.0}
    assert {r["query_id"] for r in rows} == sheets.assigned_ids(assignment, "v1")
    assert all(r["sample"] == sp for r in rows)
    # the judge still reads the whole sample, not just this annotator's part of it
    judged = judge.judged_set(rows, GENS, lambda: QRELS)["judged"]
    assert [g.query_id for g in judged] == drawn(sp)


def finished_spreadsheet_file(tmp_path, sp):
    assignment = sampled_assignment(sp)
    sheets.write_assignment(sheets.assignment_path(tmp_path, GEN_PATH), assignment)
    jsonl = tmp_path / "twopass__x__v1.jsonl"
    a_csv, _t, _r = sheets.export_sheet("A", jsonl, assignment, "v1", "0.1", GENS, CORPUS, QRELS)
    sheets.import_sheet("A", jsonl, fill(a_csv, {sheets.COL_A: "accept",
                                                 sheets.COL_CONFIDENCE: "confident"}),
                        sittings(tmp_path / "sa.csv", "14:00", "14:30"), assignment, "v1", "0.1")
    b_csv, _t, _r = sheets.export_sheet("B", jsonl, assignment, "v1", "0.1")
    sheets.import_sheet("B", jsonl, fill(b_csv, {sheets.COL_B: "refusal OK",
                                                 sheets.COL_COMMON: ""}),
                        sittings(tmp_path / "sb.csv", "20:00", "21:15"), assignment, "v1", "0.1")
    return jsonl, assignment


def test_a_spreadsheet_file_whose_rows_lost_the_sample_fails_the_check(tmp_path):
    jsonl, _assignment = finished_spreadsheet_file(tmp_path, spec(g2=3, g1=5, g0=2))
    rows = twopass.read_rows(jsonl)
    for r in rows:
        del r["sample"]
    twopass.save_rows(jsonl, rows)
    with pytest.raises(AssertionError, match="do not record the sample"):
        twopass.check_file(jsonl, GENS, GEN_PATH, lambda: QRELS)


def test_an_assignment_that_is_not_its_samples_rows_fails_the_check(tmp_path):
    jsonl, assignment = finished_spreadsheet_file(tmp_path, spec(g2=3, g1=5, g0=2))
    other = spec(seed=9, g2=3, g1=5, g0=2)          # same sizes, different rows
    assert set(drawn(other)) != set(drawn(assignment["sample"]))
    rows = twopass.read_rows(jsonl)
    for r in rows:
        r["sample"] = other
    twopass.save_rows(jsonl, rows)
    apath = sheets.assignment_path(tmp_path, GEN_PATH)
    apath.write_text(json.dumps({**assignment, "sample": other}))
    with pytest.raises(AssertionError, match="does not cover the rows its sample draws"):
        twopass.check_file(jsonl, GENS, GEN_PATH, lambda: QRELS)


# ─────────────────────────── a batch that changed under its sample ───────────────────────────

def test_a_record_that_no_longer_matches_the_batch_is_refused(cli):
    run, out = cli
    run(sample=["2=3", "1=5", "0=2"])
    path = out / "sample__x.json"
    record = json.loads(path.read_text())
    record["query_ids"] = sorted(set(QIDS) - set(record["query_ids"]))[:10]
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="lists different rows than its spec draws"):
        run(annotator="betty")
    with pytest.raises(AssertionError, match="lists different rows than its spec draws"):
        run(assign=["v1", "v2"])
    assert not list(out.glob("twopass__*")) and not list(out.glob("assign__*"))


def test_a_spec_the_batch_can_no_longer_fill_says_the_batch_changed():
    fewer = {q: g for q, g in STRATA.items() if not (g == 2 and q > "q01")}
    with pytest.raises(AssertionError, match="no longer gives its recorded sample"):
        twopass.sampled_ids(fewer, spec(g2=3, g1=5))


def test_a_batch_with_no_refusals_says_so_before_any_draw(tmp_path):
    answered = [gen("q00", refused=False)]
    with pytest.raises(ValueError, match="no refused rows"):
        twopass.write_twopass_sheet(answered, CORPUS, QRELS, tmp_path / "t.jsonl", "betty",
                                    "0.1", sample=spec(g2=3, g1=5))


# ─────────────────────────── what counts as already started ───────────────────────────

def test_an_empty_or_unreadable_label_file_still_counts_as_started(tmp_path):
    (tmp_path / "twopass__x__v1.jsonl").write_text("")
    (tmp_path / "twopass__x__v2.jsonl").write_text("{not json")
    assert twopass.batch_files(tmp_path, GEN_PATH) == ["twopass__x__v1.jsonl",
                                                       "twopass__x__v2.jsonl"]


def test_another_batchs_files_do_not_count(tmp_path):
    # batch "x__ex1" starts with "x"; its file is told apart by the row's annotator
    twopass.write_twopass_sheet(GENS, CORPUS, QRELS, tmp_path / "twopass__x__ex1__betty.jsonl",
                                "betty", "0.1")
    (tmp_path / "twopass__y__betty.jsonl").write_text("{not json")
    (tmp_path / "assign__y.json").write_text("{}")
    assert twopass.batch_files(tmp_path, GEN_PATH) == []
