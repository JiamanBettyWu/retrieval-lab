"""Tests for the spreadsheet export/import of the two-pass labeller (part 2 of issue #6).

    pytest tests/test_sheets.py

What is pinned: the assignment is reproducible, balanced by stratum and never
rewritten; the pass A sheet physically lacks passages and strata; pass B is
refused before pass A is imported complete, and locks it; a changed pass A is
refused after that; import rejects missing, extra, duplicated and
out-of-vocabulary rows; the stored values match the words in the headers; and
a finished sheet file passes the same validation as a terminal one, with
sittings in place of per-row seconds. No model, no Ollama, no download.
"""
import csv
import json

import pytest

from retrieval_lab import sheets, twopass
from retrieval_lab.generate import REFUSAL, Generation

DOCS = ["dA", "dB", "dC"]
CORPUS = {d: {"title": f"Title {d}", "text": f"PASSAGE-TEXT-{d}"} for d in DOCS}
REFUSED_RAW = ("<rationale>The passages do not name the architect.</rationale>\n"
               "<answer>INSUFFICIENT_CONTEXT</answer>")

# 12 refused rows: gold_in_context 2 for q00-q03, 1 for q04-q09, 0 for q10-q11.
QIDS = [f"q{i:02d}" for i in range(12)]
GOLD = {q: (2 if i < 4 else 1 if i < 10 else 0) for i, q in enumerate(QIDS)}
QRELS = {q: {d: 1 for d in DOCS[:g]} | ({"dX": 1, "dY": 1} if g == 0 else
                                          {"dX": 1} if g == 1 else {})
         for q, g in GOLD.items()}


def gen(query_id, refused=True, truncated=False):
    return Generation(query_id=query_id, question=f"QUESTION-{query_id}?",
                      doc_ids=DOCS, raw=REFUSED_RAW,
                      answer=REFUSAL if refused else "Oslo",
                      rationale="The passages do not name the architect.",
                      refused=refused, truncated=truncated)


GENS = [gen(q) for q in QIDS] + [gen("q98", refused=False), gen("q99", truncated=True)]


@pytest.fixture
def assignment():
    return sheets.make_assignment(sheets.batch_strata(GENS, QRELS), ["v1", "v2"], 4, seed=0)


@pytest.fixture
def jsonl(tmp_path):
    return tmp_path / "twopass__x__v1.jsonl"


def export(which, jsonl, assignment, annotator="v1", version="0.1"):
    """Export once; later calls reuse the file (re-exporting is refused)."""
    csv_path, template = sheets.sheet_paths(jsonl, annotator, which)
    if csv_path.exists():
        return csv_path, template, 0
    return sheets.export_sheet(which, jsonl, assignment, annotator, version,
                               GENS, CORPUS, QRELS)


def read_table(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fill(csv_path, values: dict, out=None):
    """Write `values` ({column: value or callable(query_id)}) into a copy of the sheet."""
    table = read_table(csv_path)
    for r in table:
        for col, v in values.items():
            r[col] = v(r["query_id"]) if callable(v) else v
    out = out or csv_path.with_name("filled_" + csv_path.name)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    return out


def sittings(tmp_path, *pairs, name="sittings.csv"):
    path = tmp_path / name
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sitting", sheets.COL_START, sheets.COL_END])
        w.writerows([[i, a, b] for i, (a, b) in enumerate(pairs, 1)])
    return path


def do_pass_a(tmp_path, jsonl, assignment, label="accept", confidence="confident"):
    csv_path, _, _ = export("A", jsonl, assignment)
    filled = fill(csv_path, {sheets.COL_A: label, sheets.COL_CONFIDENCE: confidence})
    return sheets.import_sheet("A", jsonl, filled, sittings(tmp_path, ("14:00", "14:30")),
                               assignment, "v1", "0.1")


def do_pass_b(tmp_path, jsonl, assignment, label="refusal OK", common=""):
    csv_path, _, _ = export("B", jsonl, assignment)
    filled = fill(csv_path, {sheets.COL_B: label, sheets.COL_COMMON: common})
    return sheets.import_sheet("B", jsonl, filled,
                               sittings(tmp_path, ("20:00", "21:15"), name="sb.csv"),
                               assignment, "v1", "0.1")


# ─────────────────────────── assignment ───────────────────────────

def test_assignment_is_reproducible_from_its_seed():
    strata = sheets.batch_strata(GENS, QRELS)
    a = sheets.make_assignment(strata, ["v1", "v2"], 4, seed=7)
    assert a == sheets.make_assignment(strata, ["v1", "v2"], 4, seed=7)
    assert a != sheets.make_assignment(strata, ["v1", "v2"], 4, seed=8)


def test_assignment_covers_every_row_once_apart_from_the_shared_ones(assignment):
    own = assignment["own"]
    assert set(own["v1"]).isdisjoint(own["v2"])
    assert set(assignment["shared"]).isdisjoint(own["v1"] + own["v2"])
    assert sorted(own["v1"] + own["v2"] + assignment["shared"]) == QIDS
    assert len(assignment["shared"]) == 4


def test_cut_off_and_answered_rows_are_never_assigned(assignment):
    every = assignment["shared"] + assignment["own"]["v1"] + assignment["own"]["v2"]
    assert "q98" not in every and "q99" not in every


def test_assignment_is_balanced_by_stratum(assignment):
    # 4 shared of 12, in proportion: 2 → 1.33, 1 → 2, 0 → 0.67 → {1, 2, 1}
    shared = [GOLD[q] for q in assignment["shared"]]
    assert sorted(shared) == [0, 1, 1, 2]
    for stratum in (0, 1, 2):
        counts = [sum(GOLD[q] == stratum for q in assignment["own"][a]) for a in ("v1", "v2")]
        assert max(counts) - min(counts) <= 1


def test_shared_all_gives_every_annotator_every_row():
    a = sheets.make_assignment(sheets.batch_strata(GENS, QRELS), ["v1", "v2"], "all", seed=0)
    assert sheets.assigned_ids(a, "v1") == sheets.assigned_ids(a, "v2") == set(QIDS)


def test_assignment_is_never_rewritten(tmp_path, assignment):
    path = tmp_path / "assign__x.json"
    sheets.write_assignment(path, assignment)
    with pytest.raises(FileExistsError):
        sheets.write_assignment(path, assignment)


@pytest.mark.parametrize("names, n", [(["v1", "v1"], 0), (["v 1"], 0), (["v1"], 13), ([], 0)])
def test_bad_assignments_are_refused(names, n):
    with pytest.raises(ValueError):
        sheets.make_assignment(sheets.batch_strata(GENS, QRELS), names, n, seed=0)


def test_an_annotator_outside_the_assignment_is_refused(assignment):
    with pytest.raises(KeyError):
        sheets.assigned_ids(assignment, "betty")


# ─────────────────────────── what each sheet shows ───────────────────────────

def test_pass_a_sheet_holds_no_passage_or_stratum_text(jsonl, assignment):
    csv_path, _, _ = export("A", jsonl, assignment)
    text = csv_path.read_text()
    assert "PASSAGE-TEXT" not in text and "Title d" not in text
    for word in ("gold_in_context", "stratum", "shared", "__REFUSED__", REFUSAL):
        assert word not in text
    assert sheets.COL_PASSAGES not in read_table(csv_path)[0]
    assert "QUESTION-q" in text and "do not name the architect" in text


def test_pass_a_sheet_holds_only_the_assigned_rows(jsonl, assignment):
    csv_path, _, _ = export("A", jsonl, assignment)
    assert {r["query_id"] for r in read_table(csv_path)} == sheets.assigned_ids(assignment, "v1")


def test_sheet_order_does_not_follow_the_ids(jsonl, assignment):
    csv_path, _, _ = export("A", jsonl, assignment)
    ids = [r["query_id"] for r in read_table(csv_path)]
    assert ids != sorted(ids)


def test_pass_b_sheet_holds_passages_but_no_pass_a_labels(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment, label="reject", confidence="unsure")
    csv_path, _, _ = export("B", jsonl, assignment)
    table = read_table(csv_path)
    assert sheets.COL_A not in table[0] and sheets.COL_CONFIDENCE not in table[0]
    assert all("[1] Title dA. PASSAGE-TEXT-dA" in r[sheets.COL_PASSAGES] for r in table)
    text = csv_path.read_text()
    assert "reject" not in text and "unsure" not in text and "stratum" not in text


def test_headers_spell_out_every_word_the_import_accepts():
    for which, columns in sheets.LABEL_COLUMNS.items():
        for col, (_field, words) in columns.items():
            assert all(w in col for w in words)


# ─────────────────────────── order and locking ───────────────────────────

def test_pass_b_export_is_refused_before_pass_a_is_imported(jsonl, assignment):
    export("A", jsonl, assignment)
    with pytest.raises(AssertionError, match="pass A is not imported complete"):
        export("B", jsonl, assignment)


def test_pass_b_export_locks_pass_a(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    assert {r["pass_a_lock"] for r in twopass.read_rows(jsonl)} == {None}
    export("B", jsonl, assignment)
    rows = twopass.read_rows(jsonl)
    assert {r["pass_a_lock"] for r in rows} == {twopass.pass_a_digest(rows)}


def test_pass_a_can_be_reimported_freely_before_pass_b(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment, label="accept")
    do_pass_a(tmp_path, jsonl, assignment, label="reject")
    assert {r["refusal_ok_response"] for r in twopass.read_rows(jsonl)} == {False}


def test_a_changed_pass_a_is_refused_after_pass_b_export(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment, label="accept")
    export("B", jsonl, assignment)
    with pytest.raises(AssertionError, match="differ from the ones locked"):
        do_pass_a(tmp_path, jsonl, assignment, label="reject")
    assert {r["refusal_ok_response"] for r in twopass.read_rows(jsonl)} == {True}


def test_an_identical_pass_a_reimport_after_pass_b_export_changes_nothing(tmp_path, jsonl,
                                                                          assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    export("B", jsonl, assignment)
    assert do_pass_a(tmp_path, jsonl, assignment) == "sittings"


def test_pass_b_import_is_refused_before_pass_b_export(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    fake = tmp_path / "b.csv"
    with fake.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([sheets.COL_ANNOTATOR, sheets.COL_ID, sheets.COL_B, sheets.COL_COMMON])
        w.writerows([["v1", q, "refusal OK", ""] for q in sheets.assigned_ids(assignment, "v1")])
    with pytest.raises(AssertionError, match="pass B was never exported"):
        sheets.import_sheet("B", jsonl, fake, sittings(tmp_path, ("9:00", "9:30")),
                            assignment, "v1", "0.1")


def test_a_sheet_file_cannot_be_labelled_in_the_terminal(tmp_path, jsonl, assignment):
    export("A", jsonl, assignment)
    with pytest.raises(AssertionError, match="labelled through spreadsheets"):
        twopass.label_twopass(jsonl, "v1", "0.1", ask=lambda _="": "q", say=lambda _="": None)


# ─────────────────────────── import validation ───────────────────────────

def test_polarity_follows_the_words(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment, label="reject", confidence="unsure")
    do_pass_b(tmp_path, jsonl, assignment, label="over-refusal", common="no")
    r = twopass.read_rows(jsonl)[0]
    assert r["refusal_ok_response"] is False and r["confidence_response"] == "unsure"
    assert r["refusal_ok_passages"] is False and r["needs_common_knowledge"] is False


def test_case_and_spacing_are_forgiven(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment, label="  Accept ", confidence="CONFIDENT")
    do_pass_b(tmp_path, jsonl, assignment, label="Refusal  ok")
    r = twopass.read_rows(jsonl)[0]
    assert r["refusal_ok_response"] is True and r["refusal_ok_passages"] is True


@pytest.mark.parametrize("value", ["yes", "true", "TRUE", "", "acept"])
def test_words_outside_the_vocabulary_are_refused(tmp_path, jsonl, assignment, value):
    with pytest.raises(AssertionError, match="must be one of"):
        do_pass_a(tmp_path, jsonl, assignment, label=value)
    assert all(r["refusal_ok_response"] is None for r in twopass.read_rows(jsonl))


def test_common_knowledge_is_required_on_an_over_refusal(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    with pytest.raises(AssertionError, match="for an over-refusal"):
        do_pass_b(tmp_path, jsonl, assignment, label="over-refusal", common="")


def test_common_knowledge_must_be_blank_on_refusal_ok(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    with pytest.raises(AssertionError, match="must be blank"):
        do_pass_b(tmp_path, jsonl, assignment, label="refusal OK", common="yes")


def _filled_a(jsonl, assignment):
    csv_path, _, _ = export("A", jsonl, assignment)
    return read_table(fill(csv_path, {sheets.COL_A: "accept", sheets.COL_CONFIDENCE: "unsure"}))


def _write(path, table):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    return path


@pytest.mark.parametrize("damage, message", [
    (lambda t: t[1:], "missing"),
    (lambda t: t + [t[0]], "appear twice"),
    (lambda t: t + [{**t[0], "query_id": "q98"}], "not assigned"),
])
def test_missing_extra_and_duplicated_rows_are_refused(tmp_path, jsonl, assignment,
                                                       damage, message):
    path = _write(tmp_path / "bad.csv", damage(_filled_a(jsonl, assignment)))
    with pytest.raises(AssertionError, match=message):
        sheets.import_sheet("A", jsonl, path, sittings(tmp_path, ("9:00", "9:30")),
                            assignment, "v1", "0.1")


def test_a_renamed_header_is_refused(tmp_path, jsonl, assignment):
    table = [{("label" if k == sheets.COL_A else k): v for k, v in r.items()}
             for r in _filled_a(jsonl, assignment)]
    path = _write(tmp_path / "bad.csv", table)
    with pytest.raises(AssertionError, match="lacks the column"):
        sheets.import_sheet("A", jsonl, path, sittings(tmp_path, ("9:00", "9:30")),
                            assignment, "v1", "0.1")


def test_blank_lines_and_a_byte_order_mark_are_tolerated(tmp_path, jsonl, assignment):
    path = _write(tmp_path / "ok.csv", _filled_a(jsonl, assignment))
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes() + b",,,,,,\n")
    assert sheets.import_sheet("A", jsonl, path, sittings(tmp_path, ("9:00", "9:30")),
                               assignment, "v1", "0.1") == "imported"


def test_the_pass_is_read_from_the_header(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    a_csv, _, _ = export("A", jsonl, assignment)
    b_csv, _, _ = export("B", jsonl, assignment)
    assert sheets.sheet_pass(a_csv) == "A" and sheets.sheet_pass(b_csv) == "B"


# ─────────────────────────── sittings ───────────────────────────

@pytest.mark.parametrize("start, end, minutes", [
    ("14:05", "15:10", 65), ("2:05 PM", "3:10 PM", 65), ("14:05:00", "15:10:30", 65.5),
    ("23:30", "0:15", 45),
])
def test_sittings_are_read_in_common_clock_formats(tmp_path, start, end, minutes):
    assert sheets.parse_sittings(sittings(tmp_path, (start, end)))[0]["minutes"] == minutes


@pytest.mark.parametrize("start, end", [("14:00", "14:00"), ("9:00", "18:30"), ("nine", "ten")])
def test_impossible_sittings_are_refused(tmp_path, start, end):
    with pytest.raises(AssertionError):
        sheets.parse_sittings(sittings(tmp_path, (start, end)))


def test_a_pass_with_no_sitting_is_refused(tmp_path):
    with pytest.raises(AssertionError, match="no sitting"):
        sheets.parse_sittings(sittings(tmp_path))


# ─────────────────────────── the finished file ───────────────────────────

def test_a_finished_sheet_file_passes_the_check(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    do_pass_b(tmp_path, jsonl, assignment, label="over-refusal", common="yes")
    rows = twopass.read_rows(jsonl)
    assert sheets.check_sheet_file(jsonl, rows, assignment, "v1") == {"A": 30.0, "B": 75.0}
    assert all(r["seconds_response"] is None for r in rows)


def test_the_check_refuses_a_file_without_sittings(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    do_pass_b(tmp_path, jsonl, assignment)
    sheets.sittings_path(jsonl).unlink()
    with pytest.raises(AssertionError, match="no sittings"):
        sheets.check_sheet_file(jsonl, twopass.read_rows(jsonl), assignment, "v1")


def test_the_check_refuses_per_row_seconds_typed_into_a_sheet_file(tmp_path, jsonl,
                                                                   assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    do_pass_b(tmp_path, jsonl, assignment)
    rows = twopass.read_rows(jsonl)
    rows[0]["seconds_passages"] = 12.0     # pass B: outside the pass A lock
    with pytest.raises(AssertionError, match="per-row seconds"):
        twopass.validate_twopass(rows, sheets.assigned_ids(assignment, "v1"))


def test_the_label_file_stores_the_same_row_shape_as_the_terminal(tmp_path, jsonl, assignment):
    export("A", jsonl, assignment)
    terminal = tmp_path / "twopass__x__betty.jsonl"
    twopass.write_twopass_sheet(GENS, CORPUS, QRELS, terminal, "betty", "0.1")
    sheet_keys = set(twopass.read_rows(jsonl)[0])
    assert sheet_keys == set(twopass.read_rows(terminal)[0]) | {"timing"}
    assert json.loads(jsonl.read_text().splitlines()[0])["stratum"]["gold_in_context"] in (0, 1, 2)


# ─────────────────────────── review fixes ───────────────────────────

def test_one_annotators_sheet_is_refused_under_another_name(tmp_path, assignment):
    """With every row shared, only the annotator column tells two sheets apart."""
    every = sheets.make_assignment(sheets.batch_strata(GENS, QRELS), ["v1", "v2"], "all", seed=0)
    v1_file, v2_file = tmp_path / "twopass__x__v1.jsonl", tmp_path / "twopass__x__v2.jsonl"
    v1_csv, _, _ = export("A", v1_file, every, annotator="v1")
    export("A", v2_file, every, annotator="v2")
    filled = fill(v1_csv, {sheets.COL_A: "accept", sheets.COL_CONFIDENCE: "unsure"})
    with pytest.raises(AssertionError, match="not 'v2'"):
        sheets.import_sheet("A", v2_file, filled, sittings(tmp_path, ("9:00", "9:30")),
                            every, "v2", "0.1")


def test_a_pass_a_sitting_can_be_corrected_after_the_lock(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    export("B", jsonl, assignment)
    csv_path, _, _ = export("A", jsonl, assignment)
    filled = fill(csv_path, {sheets.COL_A: "accept", sheets.COL_CONFIDENCE: "confident"})
    assert sheets.import_sheet("A", jsonl, filled, sittings(tmp_path, ("10:00", "10:45")),
                               assignment, "v1", "0.1") == "sittings"
    record = json.loads(sheets.sittings_path(jsonl).read_text())
    assert record["A"][0]["minutes"] == 45


def test_an_existing_sheet_is_never_overwritten(jsonl, assignment):
    sheets.export_sheet("A", jsonl, assignment, "v1", "0.1", GENS, CORPUS, QRELS)
    with pytest.raises(FileExistsError, match="returned, filled sheet"):
        sheets.export_sheet("A", jsonl, assignment, "v1", "0.1", GENS, CORPUS, QRELS)


def test_sheets_carry_a_byte_order_mark_for_excel(jsonl, assignment):
    csv_path, template, _ = export("A", jsonl, assignment)
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert template.read_bytes().startswith(b"\xef\xbb\xbf")


def test_a_sheet_saved_in_another_encoding_gets_a_clear_error(tmp_path, jsonl, assignment):
    csv_path, _, _ = export("A", jsonl, assignment)
    bad = tmp_path / "cp1252.csv"
    bad.write_bytes(csv_path.read_bytes() + "Zoë".encode("cp1252"))
    with pytest.raises(AssertionError, match="not UTF-8"):
        sheets.parse_labels(bad, "A", sheets.assigned_ids(assignment, "v1"), "v1")


def test_an_option_copied_with_its_explanation_is_accepted(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    do_pass_b(tmp_path, jsonl, assignment,
              label="refusal OK: a step is missing, the question is unclear, or the "
                    "passages disagree")
    assert {r["refusal_ok_passages"] for r in twopass.read_rows(jsonl)} == {True}


def test_pass_a_import_is_refused_beside_unlocked_pass_b_labels(tmp_path, jsonl, assignment):
    do_pass_a(tmp_path, jsonl, assignment)
    rows = twopass.read_rows(jsonl)
    rows[0]["refusal_ok_passages"] = True          # pass B label, no lock: a hand edit
    twopass.save_rows(jsonl, rows)
    with pytest.raises(AssertionError, match="never locked"):
        do_pass_a(tmp_path, jsonl, assignment, label="reject")


def test_a_file_mixing_terminal_and_sheet_rows_is_refused(jsonl, assignment):
    export("A", jsonl, assignment)
    rows = twopass.read_rows(jsonl)
    del rows[0]["timing"]
    with pytest.raises(AssertionError, match="mixes terminal rows and spreadsheet rows"):
        twopass.file_timing(rows)
