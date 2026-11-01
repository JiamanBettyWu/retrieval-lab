"""Spreadsheet export/import for the two-pass labeller (part 2 of issue #6).

For annotators who label in a spreadsheet instead of the terminal. The passes,
their order and the stored file are the same as `twopass.py`'s; only the screen
differs. Driven from `twopass`'s command line:

    python -m retrieval_lab.twopass <batch> --assign V1 V2 --shared 30 --assign-seed 0
    python -m retrieval_lab.twopass <batch> --annotator V1 --guide-version 0.1 --export-sheet pass-a
    python -m retrieval_lab.twopass <batch> --annotator V1 --guide-version 0.1 --import-sheet FILLED.csv --sittings SITTINGS.csv
    python -m retrieval_lab.twopass <batch> --annotator V1 --guide-version 0.1 --export-sheet pass-b
    python -m retrieval_lab.twopass <batch> --annotator V1 --guide-version 0.1 --import-sheet FILLED.csv --sittings SITTINGS.csv
    python -m retrieval_lab.twopass <batch> --annotator V1 --guide-version 0.1 --check

**Assignment is written once.** Each annotator gets their own rows plus an
optional shared set that every annotator labels, split at random and balanced
by `gold_in_context`. The record is never rewritten: changing who labels what
after labelling has started would change what the agreement numbers describe.

**The pass A sheet physically lacks the passages.** Not hidden columns — a
spreadsheet can unhide those. The pass B sheet does not carry the annotator's
pass A labels. Neither carries a stratum field.

**Exporting pass B is the moment pass A is locked**, the equivalent of the
terminal labeller stamping the lock as pass B begins. A pass A import after
that is refused if any label differs from the locked ones.

**Timing is per sitting, not per row.** A spreadsheet cannot time a row, so the
annotator writes the start and end of each sitting in a second, small sheet.
Per-pass totals give the pass B / pass A time ratio; per-row seconds stay empty
on these rows, which carry `timing: "session"` so validation knows why.

**The column headers are the stored values' only definition**, as the terminal
prompts are: they are built from `twopass`'s `*_QUESTION` and `*_CHOICES`, and
the import accepts exactly those words.
"""
import csv
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path

from .fixture import gold_in_context, save_rows
from .generate import Generation
from .twopass import (COMMON_KNOWLEDGE_CHOICES, COMMON_KNOWLEDGE_QUESTION,
                      CONFIDENCE_CHOICES, CONFIDENCE_QUESTION, PASS_A_CHOICES,
                      PASS_A_QUESTION, PASS_B_CHOICES, PASS_B_QUESTION, TIMING_SESSION,
                      batch_name, check_annotator, check_header, check_lock, eligible_rows,
                      file_timing, lock_pass_a, pass_a_done, pass_b_started, read_rows,
                      twopass_row, validate_twopass)

TIMING = TIMING_SESSION
MAX_SITTING_MINUTES = 8 * 60      # longer is a typo, not a sitting
SITTING_ROWS = 15                 # blank rows in the sittings template
CLOCK_FORMATS = ("%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p", "%I:%M%p")
FORMULA_START = ("=", "+", "-", "@")


# ─────────────────────────── assignment ───────────────────────────

def assignment_path(out_dir: Path, generation_path: Path) -> Path:
    return Path(out_dir) / f"assign__{batch_name(generation_path)}.json"


def _allocate(n: int, sizes: dict) -> dict:
    """Split n across groups in proportion to their sizes (largest remainder)."""
    total = sum(sizes.values())
    exact = {s: n * size / total for s, size in sizes.items()}
    quota = {s: int(e) for s, e in exact.items()}
    left = n - sum(quota.values())
    for s in sorted(sizes, key=lambda s: (quota[s] - exact[s], s))[:left]:
        quota[s] += 1
    return quota


def make_assignment(strata: dict[str, int], annotators: list[str], n_shared,
                    seed: int) -> dict:
    """Split the rows between annotators: `{"shared": [...], "own": {name: [...]}}`.

    `strata` maps query id → `gold_in_context`. `n_shared` is a count or "all".
    The shared rows are taken from each stratum in proportion to its size; the
    rest are dealt to the annotators in turn within each stratum, so each
    annotator's own rows match the batch's mix to within one row per stratum.
    """
    if not annotators:
        raise ValueError("name at least one annotator")
    if len(set(annotators)) != len(annotators):
        raise ValueError(f"annotator named twice: {annotators}")
    for a in annotators:
        check_annotator(a)
    ids = sorted(strata)
    n = len(ids) if n_shared == "all" else int(n_shared)
    if not 0 <= n <= len(ids):
        raise ValueError(f"--shared {n_shared}: the batch has {len(ids)} rows to label")

    rng = random.Random(seed)
    by_stratum = {}
    for s in sorted(set(strata.values())):
        group = [q for q in ids if strata[q] == s]
        rng.shuffle(group)
        by_stratum[s] = group
    quota = _allocate(n, {s: len(g) for s, g in by_stratum.items()})

    shared, own, turn = [], {a: [] for a in annotators}, 0
    for s, group in by_stratum.items():
        shared += group[:quota[s]]
        for q in group[quota[s]:]:
            own[annotators[turn % len(annotators)]].append(q)
            turn += 1
    return {"seed": seed, "n_rows": len(ids), "shared": sorted(shared),
            "own": {a: sorted(q) for a, q in own.items()}}


def write_assignment(path: Path, assignment: dict) -> None:
    if path.exists():
        raise FileExistsError(
            f"{path} already exists — the assignment is fixed once written, since "
            "changing who labels what changes what the agreement numbers describe."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(assignment, indent=1) + "\n")


def read_assignment(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"no assignment at {path} — write it first with --assign")
    return json.loads(path.read_text())


def assigned_ids(assignment: dict, annotator: str) -> set[str]:
    if annotator not in assignment["own"]:
        raise KeyError(f"{annotator!r} is not in the assignment "
                       f"(it names {sorted(assignment['own'])})")
    return set(assignment["own"][annotator]) | set(assignment["shared"])


def batch_strata(gens: list[Generation], qrels: dict) -> dict[str, int]:
    """`gold_in_context` for every row that goes on a sheet."""
    return {g.query_id: gold_in_context(qrels, g.query_id, g.doc_ids)
            for g in eligible_rows(gens)[0]}


# ─────────────────────────── the sheets ───────────────────────────

def header(question: str, suffix: str = "") -> str:
    """A column header from a terminal prompt: its question and every option."""
    first, *option_lines = question.split("\n")
    options = [re.sub(r"^\w = ", "", o.strip())
               for line in option_lines for o in line.split("|")]
    return f"{first.rstrip(':')} [{' | '.join(options)}]{suffix}"


COL_ROW, COL_ANNOTATOR, COL_ID, COL_QUESTION = "row", "annotator", "query_id", "question"
COL_ANSWER, COL_RATIONALE, COL_PASSAGES = "response: answer", "response: rationale", "passages"
COL_A = header(PASS_A_QUESTION)
COL_CONFIDENCE = header(CONFIDENCE_QUESTION)
COL_B = header(PASS_B_QUESTION)
COL_COMMON = header(COMMON_KNOWLEDGE_QUESTION, " — fill only for over-refusal")
COL_START, COL_END = "start (HH:MM)", "end (HH:MM)"

# column → (stored field, {word on screen: stored value})
LABEL_COLUMNS = {
    "A": {COL_A: ("refusal_ok_response", dict(PASS_A_CHOICES.values())),
          COL_CONFIDENCE: ("confidence_response", dict(CONFIDENCE_CHOICES.values()))},
    "B": {COL_B: ("refusal_ok_passages", dict(PASS_B_CHOICES.values())),
          COL_COMMON: ("needs_common_knowledge", dict(COMMON_KNOWLEDGE_CHOICES.values()))},
}
TEXT_COLUMNS = {"A": [COL_QUESTION, COL_ANSWER, COL_RATIONALE],
                "B": [COL_QUESTION, COL_PASSAGES, COL_ANSWER, COL_RATIONALE]}


def sheet_paths(jsonl_path: Path, annotator: str, which: str) -> tuple[Path, Path]:
    """The CSV to fill and its sittings template. Short names, since they are
    what an annotator sees; the hash ties them to the label file (batch and annotator)."""
    tag = hashlib.sha256(jsonl_path.name.encode()).hexdigest()[:8]
    base = jsonl_path.parent / "sheets" / f"{annotator}_pass-{which.lower()}_{tag}"
    return base.with_suffix(".csv"), base.with_name(base.name + "_sittings.csv")


def sittings_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_name(jsonl_path.stem + ".sittings.json")


def display_order(rows: list[dict], seed: int, annotator: str, which: str) -> list[dict]:
    """Shuffled, so neither position nor sort order shows which rows are shared."""
    out = sorted(rows, key=lambda r: r["query_id"])
    random.Random(f"{seed}:{annotator}:{which}").shuffle(out)
    return out


def sheet_cells(row: dict, which: str) -> dict:
    cells = {COL_ID: row["query_id"], COL_QUESTION: row["question"],
             COL_ANSWER: row["reply_answer"] or "(none given)",
             COL_RATIONALE: row["reply_rationale"]}
    if which == "B":
        cells[COL_PASSAGES] = "\n\n".join(row["passages"])
    return cells


def write_sheet(rows: list[dict], which: str, seed: int, annotator: str,
                csv_path: Path, template_path: Path) -> int:
    """Write the sheet to fill and a blank sittings template. Returns how many
    text cells start with a formula character (see `export_sheet`'s log)."""
    for p in (csv_path, template_path):
        if p.exists():
            raise FileExistsError(
                f"{p} already exists — it may be a returned, filled sheet. Move it "
                "elsewhere (or delete a blank one) to export again.")
    columns = [COL_ROW, COL_ANNOTATOR, COL_ID, *TEXT_COLUMNS[which], *LABEL_COLUMNS[which]]
    risky = 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel reads a BOM-less file as cp1252 and garbles accents.
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for n, row in enumerate(display_order(rows, seed, annotator, which), 1):
            cells = sheet_cells(row, which)
            risky += sum(cells[c].startswith(FORMULA_START) for c in TEXT_COLUMNS[which])
            w.writerow({COL_ROW: n, COL_ANNOTATOR: annotator, **cells})
    with template_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["sitting", COL_START, COL_END])
        w.writerows([[i, "", ""] for i in range(1, SITTING_ROWS + 1)])
    return risky


def new_sheet_rows(gens: list[Generation], corpus: dict, qrels: dict, ids: set[str],
                   annotator: str, guide_version: str) -> list[dict]:
    kept = [g for g in eligible_rows(gens)[0] if g.query_id in ids]
    if {g.query_id for g in kept} != ids:
        raise AssertionError("the assignment names rows this batch does not have to label")
    return [{**twopass_row(g, corpus, qrels, annotator, guide_version), "timing": TIMING}
            for g in kept]


def load_sheet_rows(jsonl_path: Path, annotator: str, guide_version: str,
                    ids: set[str]) -> list[dict]:
    rows = read_rows(jsonl_path)
    check_header(rows, annotator, guide_version)
    if file_timing(rows) != TIMING:
        raise AssertionError(f"{jsonl_path.name} is a terminal-labelled file, not a sheet file")
    if {r["query_id"] for r in rows} != ids:
        raise AssertionError(f"{jsonl_path.name} does not cover this annotator's assigned rows")
    return rows


def export_sheet(which: str, jsonl_path: Path, assignment: dict, annotator: str,
                 guide_version: str, gens=None, corpus=None, qrels=None) -> tuple[Path, Path, int]:
    """Write the CSV for pass `which` ("A" or "B"). Pass A creates the
    annotator's label file on first export; pass B requires pass A imported
    complete, and locks it."""
    ids = assigned_ids(assignment, annotator)
    if not jsonl_path.exists():
        if which == "B":
            raise FileNotFoundError(f"no label file at {jsonl_path} — export pass A first")
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        save_rows(jsonl_path, new_sheet_rows(gens, corpus, qrels, ids, annotator,
                                             guide_version))
    rows = load_sheet_rows(jsonl_path, annotator, guide_version, ids)

    if which == "B":
        missing = sum(not pass_a_done(r) for r in rows)
        if missing:
            raise AssertionError(
                f"pass A is not imported complete ({missing} of {len(rows)} rows have "
                "no label) — import the filled pass A sheet first")
        if rows[0]["pass_a_lock"] is None:
            lock_pass_a(rows)
            save_rows(jsonl_path, rows)
        check_lock(rows, required=True)

    csv_path, template_path = sheet_paths(jsonl_path, annotator, which)
    risky = write_sheet(rows, which, assignment["seed"], annotator, csv_path, template_path)
    return csv_path, template_path, risky


# ─────────────────────────── import ───────────────────────────

def _norm(s: str | None) -> str:
    return " ".join((s or "").split()).lower()


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    try:
        with Path(path).open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            table = list(reader)
            return [c.strip() for c in reader.fieldnames or []], table
    except UnicodeDecodeError:
        raise AssertionError(f"{Path(path).name} is not UTF-8 — save it again as "
                             "'CSV UTF-8' (Excel) or download it as CSV (Google Sheets)")


def parse_labels(path: Path, which: str, ids: set[str], annotator: str) -> dict[str, dict]:
    """The filled sheet as {query id: {stored field: value}}, or an error that
    names every problem row. Case and spacing are forgiven; words are not."""
    columns, table = read_csv(path)
    needed = [COL_ANNOTATOR, COL_ID, *LABEL_COLUMNS[which]]
    absent = [c for c in needed if c not in columns]
    if absent:
        raise AssertionError(f"{Path(path).name} lacks the column(s) {absent} — is it "
                             f"the pass {which} sheet, with its header row unchanged?")
    table = [{k.strip(): v for k, v in r.items() if k} for r in table]
    table = [r for r in table if any(_norm(v) for v in r.values())]   # blank lines
    owners = {_norm(r.get(COL_ANNOTATOR)) for r in table}
    if owners != {annotator.lower()}:
        raise AssertionError(f"{Path(path).name} is {sorted(owners)}'s sheet, not "
                             f"{annotator!r}'s — import it under its own annotator")

    seen = [(r.get(COL_ID) or "").strip() for r in table]
    dupes = sorted({q for q in seen if seen.count(q) > 1})
    if dupes:
        raise AssertionError(f"{len(dupes)} rows appear twice (e.g. {dupes[:3]})")
    if set(seen) != ids:
        missing, extra = sorted(ids - set(seen)), sorted(set(seen) - ids)
        raise AssertionError(
            f"the sheet's rows do not match the assignment: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(extra)} not assigned (e.g. {extra[:3]})")

    problems, out = [], {}
    for r in table:
        qid, labels = (r.get(COL_ID) or "").strip(), {}
        for col, (field, words) in LABEL_COLUMNS[which].items():
            vocab = {w.lower(): v for w, v in words.items()}
            # "refusal OK: a step is missing…" as the header shows it counts
            # as "refusal OK"; no option word contains a colon.
            given = _norm(r.get(col)).split(":")[0].strip()
            if field == "needs_common_knowledge":
                if labels.get("refusal_ok_passages") is False:
                    if given not in vocab:
                        problems.append(f"{qid}: '{col[:40]}…' must be one of "
                                        f"{list(words)} for an over-refusal, got {given!r}")
                    labels[field] = vocab.get(given)
                elif given:
                    problems.append(f"{qid}: '{col[:40]}…' must be blank unless "
                                    f"over-refusal, got {given!r}")
                else:
                    labels[field] = None
                continue
            if given not in vocab:
                problems.append(f"{qid}: '{col[:40]}…' must be one of {list(words)}, "
                                f"got {given!r}")
            labels[field] = vocab.get(given)
        out[qid] = labels
    if problems:
        raise AssertionError(f"{len(problems)} problem(s) in {Path(path).name}:\n  "
                             + "\n  ".join(problems[:10]))
    return out


def sheet_pass(path: Path) -> str:
    """Which pass a filled sheet belongs to, read from its header."""
    columns, _ = read_csv(path)
    found = [w for w in ("A", "B") if all(c in columns for c in LABEL_COLUMNS[w])]
    if len(found) != 1:
        raise AssertionError(f"{Path(path).name} has neither pass's label columns — "
                             "is its header row unchanged?")
    return found[0]


def _clock(s: str) -> datetime:
    s = " ".join(s.split()).upper()
    for fmt in CLOCK_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(s)


def parse_sittings(path: Path) -> list[dict]:
    """The sittings sheet as [{"start", "end", "minutes"}]. A sitting that
    passes midnight is counted as such."""
    columns, table = read_csv(path)
    if COL_START not in columns or COL_END not in columns:
        raise AssertionError(f"{Path(path).name} is not a sittings sheet "
                             f"(needs the columns {COL_START!r} and {COL_END!r})")
    out, problems = [], []
    for n, r in enumerate(table, 1):
        start, end = (r.get(COL_START) or "").strip(), (r.get(COL_END) or "").strip()
        if not start and not end:
            continue
        try:
            minutes = (_clock(end) - _clock(start)).total_seconds() / 60
        except ValueError as e:
            problems.append(f"line {n}: cannot read the time {str(e)!r} (use HH:MM)")
            continue
        if minutes < 0:
            minutes += 24 * 60
        if not 0 < minutes <= MAX_SITTING_MINUTES:
            problems.append(f"line {n}: {start}–{end} is {minutes:.0f} minutes")
            continue
        out.append({"start": start, "end": end, "minutes": round(minutes, 1)})
    if problems:
        raise AssertionError(f"{len(problems)} problem(s) in {Path(path).name}:\n  "
                             + "\n  ".join(problems))
    if not out:
        raise AssertionError(f"{Path(path).name} records no sitting — every pass needs "
                             "at least one start and end time")
    return out


def import_sheet(which: str, jsonl_path: Path, csv_path: Path, sittings_csv: Path,
                 assignment: dict, annotator: str, guide_version: str) -> str:
    """Validate a filled sheet and write its labels into the annotator's file.

    Returns "imported", or "sittings" for a pass A re-import identical to the
    locked labels (only its sittings are taken, so a mistyped time can still be
    corrected). Nothing is written unless every row is valid.
    """
    ids = assigned_ids(assignment, annotator)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"no label file at {jsonl_path} — export pass A first")
    rows = load_sheet_rows(jsonl_path, annotator, guide_version, ids)
    labels = parse_labels(csv_path, which, ids, annotator)
    sittings = parse_sittings(sittings_csv)
    locked = rows[0]["pass_a_lock"] is not None

    if which == "A" and locked:
        check_lock(rows, required=True)
        changed = [r["query_id"] for r in rows
                   if any(r[f] != labels[r["query_id"]][f] for f in labels[r["query_id"]])]
        if changed:
            raise AssertionError(
                f"{len(changed)} pass A labels differ from the ones locked when pass B "
                "was exported — pass A cannot change once the passages have been shown")
    elif which == "A" and pass_b_started(rows):
        raise AssertionError("this file has pass B labels but pass A was never locked")
    if which == "B":
        if not locked:
            raise AssertionError("pass B was never exported for this file — export it first")
        check_lock(rows, required=True)

    record_path = sittings_path(jsonl_path)
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    record[which] = sittings
    if which == "A" and locked:
        record_path.write_text(json.dumps(record, indent=1) + "\n")
        return "sittings"
    for r in rows:
        r.update(labels[r["query_id"]])
    save_rows(jsonl_path, rows)
    record_path.write_text(json.dumps(record, indent=1) + "\n")
    return "imported"


# ─────────────────────────── check ───────────────────────────

def check_sheet_file(jsonl_path: Path, rows: list[dict], assignment: dict,
                     annotator: str) -> dict:
    """Both passes complete for exactly the assigned rows, with sittings for
    each pass. Returns the minutes per pass."""
    validate_twopass(rows, assigned_ids(assignment, annotator))
    record_path = sittings_path(jsonl_path)
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    minutes = {}
    for which in ("A", "B"):
        if not record.get(which):
            raise AssertionError(f"no sittings recorded for pass {which} ({record_path.name})")
        minutes[which] = round(sum(s["minutes"] for s in record[which]), 1)
    return minutes


def is_sheet_file(path: Path) -> bool:
    return path.exists() and file_timing(read_rows(path)) == TIMING

