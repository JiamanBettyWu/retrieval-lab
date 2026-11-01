"""Two-pass labelling of refusals, in the terminal (part 1 of issue #6).

    python -m retrieval_lab.twopass --n-queries 30 --seed 2 --annotator NAME --guide-version 0.1          # write the sheet
    python -m retrieval_lab.twopass --n-queries 30 --seed 2 --annotator NAME --guide-version 0.1 --label  # label it, resumably
    python -m retrieval_lab.twopass --n-queries 30 --seed 2 --annotator NAME --guide-version 0.1 --check  # verify it

Reads one cached generation batch and labels its refused rows twice:

  - **pass A** shows the question and the model's reply (its own answer text
    and rationale) — no passages;
  - **pass B** shows the same rows with the retrieved passages.

Each pass runs over the whole batch before the next begins: pass B will not
start while any row lacks a pass A label, and once any pass B label exists the
pass A labels are read-only. Neither pass shows `gold_in_context` or any other
stratum field, and pass B does not show the row's pass A label.

**Its own file per (batch, annotator).** This is a separate sheet kind from
`fixture.py`'s one-pass sheet, which `judge.py` reads — neither is touched here.
Two annotators never share a file, so neither can overwrite the other's labels.

**`guide_version` on every row.** A rule change in the middle of a batch
changes what the labels after it mean, so the labeller refuses to continue a
file under a different version than the one it was written with.

**The words on screen are the stored values' only definition** — the same
lesson as `fixture.AXIS_QUESTIONS`: labels entered under a polarity remembered
backwards are all in-vocabulary and pass every check. So each prompt spells out
both options in the labelling guide's words, every row, and the keys are the
words' initials rather than y/n.
"""
import argparse
import hashlib
import json
import logging
import re
import shlex
import textwrap
import time
from pathlib import Path

from .cache import generation_cache_path
from .data import load_beir
from .fixture import (LABELS_DIR, batch_flags, gold_in_context, load_generations,
                      load_labels, raw_sha, save_rows)
from .generate import GENERATOR, PROMPT_VERSION, Generation, excluded_ids
from .retrieve import BI_ENCODER, MODEL_HELP

log = logging.getLogger("retrieval_lab")

# Two timings per pass: `seconds_*` runs from the row being shown to the main
# label, `seconds_*_total` to the row's last keypress (the follow-up question
# included). The first compares labels fairly — only over-refusals get the
# checkbox; the second is the full cost of a row.
PASS_A_FIELDS = ("refusal_ok_response", "confidence_response",
                 "seconds_response", "seconds_response_total")
PASS_B_FIELDS = ("refusal_ok_passages", "needs_common_knowledge",
                 "seconds_passages", "seconds_passages_total")

# key → (word on screen, stored value). These sentences must match the
# labelling guide (§3.1, §3.3, §4.1, §4.3); change one and change both.
PASS_A_CHOICES = {"a": ("accept", True), "r": ("reject", False)}
CONFIDENCE_CHOICES = {"c": ("confident", "confident"), "u": ("unsure", "unsure")}
PASS_B_CHOICES = {"o": ("over-refusal", False), "k": ("refusal OK", True)}
COMMON_KNOWLEDGE_CHOICES = {"y": ("yes", True), "n": ("no", False)}

PASS_A_QUESTION = ("Would you accept this refusal?\n"
                   "    a = accept   |   r = reject")
CONFIDENCE_QUESTION = ("Confidence in that label — the evidence inside the response, "
                       "not whether you know the answer:\n"
                       "    c = confident   |   u = unsure")
PASS_B_QUESTION = ("Do the passages support an answer to the question?\n"
                   "    o = over-refusal: they do, and it refused anyway\n"
                   "    k = refusal OK: a step is missing, the question is unclear, "
                   "or the passages disagree")
COMMON_KNOWLEDGE_QUESTION = ("Does reaching the answer need a fact that is not in the "
                             "passages but that an average adult would know?\n"
                             "    y = yes   |   n = no")

ANNOTATOR_RE = re.compile(r"[A-Za-z0-9_-]+")

# Rows labelled in a spreadsheet (`sheets.py`) carry `timing: "session"`: a
# sheet cannot time a row, so their `seconds_*` fields stay empty and the time
# is recorded per sitting instead.
TIMING_SESSION = "session"


def batch_name(generation_path: Path) -> str:
    """The batch's own cache name, the key every label file of it shares, so
    every key component — exclusions included — carries over unchanged."""
    return Path(generation_path).stem.removeprefix("gen__")


def check_annotator(annotator: str) -> None:
    if not ANNOTATOR_RE.fullmatch(annotator):
        raise ValueError(f"annotator {annotator!r}: use letters, digits, - or _ only")


def twopass_path(out_dir: Path, generation_path: Path, annotator: str) -> Path:
    """One file per (batch, annotator)."""
    check_annotator(annotator)
    return Path(out_dir) / f"twopass__{batch_name(generation_path)}__{annotator}.jsonl"


def file_timing(rows: list[dict]) -> str:
    """"session" for a file labelled in spreadsheets, "row" for a terminal one.
    A file mixing the two was edited by hand, and neither path may take it."""
    kinds = {r.get("timing") == TIMING_SESSION for r in rows}
    if len(kinds) > 1:
        raise AssertionError("this file mixes terminal rows and spreadsheet rows")
    return TIMING_SESSION if kinds == {True} else "row"


def eligible_rows(gens: list[Generation]) -> tuple[list[Generation], dict[str, int]]:
    """The refused rows that go on the sheet, plus counts for the summary.

    A row the output cap cut off (`truncated: true`) is left off: the model
    never finished the reply, so it is neither an answer nor a refusal. A row
    whose cap was not recorded (`truncated: null`) stays, and is counted.
    """
    refused = [g for g in gens if g.refused]
    kept = [g for g in refused if g.truncated is not True]
    return kept, {
        "batch": len(gens),
        "refused": len(refused),
        "cut_off": len(refused) - len(kept),
        "not_recorded": sum(g.truncated is None for g in kept),
    }


def model_reply(gen: Generation) -> tuple[str, str]:
    """The model's own answer text and rationale, as it wrote them.

    The answer is read from `raw` because `Generation.answer` holds the internal
    `__REFUSED__` marker on a refusal, which must never reach the screen. Same
    rule as `parse_answer` (first `<answer>` tag), without its error log: a
    tagless refusal is expected here, not a parse failure.
    """
    found = re.search(r"<answer>(.*?)</answer>", gen.raw, flags=re.DOTALL)
    return (found.group(1).strip() if found else ""), gen.rationale


def twopass_row(gen: Generation, corpus: dict, qrels: dict, annotator: str,
                guide_version: str) -> dict:
    """One sheet row. `stratum` is stored for analysis and never rendered."""
    answer, rationale = model_reply(gen)
    return {
        "query_id": gen.query_id,
        "raw_sha": raw_sha(gen.raw),
        "annotator": annotator,
        "guide_version": guide_version,
        "stratum": {
            "gold_in_context": gold_in_context(qrels, gen.query_id, gen.doc_ids),
            "refused": gen.refused,
        },
        "question": gen.question,
        "reply_answer": answer,
        "reply_rationale": rationale,
        "passages": [f"[{i}] {corpus[d]['title']}. {corpus[d]['text']}"
                     for i, d in enumerate(gen.doc_ids, 1)],
        **{f: None for f in PASS_A_FIELDS + PASS_B_FIELDS},
        "pass_a_lock": None,
    }


def keep_gold(gens: list[Generation], qrels: dict, gold) -> list[Generation]:
    """The rows whose `gold_in_context` is in `gold`; all of them when `gold` is None."""
    if gold is None:
        return list(gens)
    return [g for g in gens if gold_in_context(qrels, g.query_id, g.doc_ids) in gold]


def write_twopass_sheet(gens: list[Generation], corpus: dict, qrels: dict, path: Path,
                        annotator: str, guide_version: str, gold=None) -> dict[str, int]:
    """Write a fresh sheet. Never overwrites: the file holds hand labels.

    `gold` (e.g. {1, 2}) leaves the other strata off. The filter runs here, at
    write time, and is recorded on every row as `gold_kept`, so `--check` knows
    which rows to expect. Whoever writes the sheet knows which strata are on it;
    which of them a given row is stays hidden, which is why `gold` must name at
    least two strata — a one-stratum sheet would show every row's stratum.
    """
    if gold is not None and len(gold) < 2:
        raise ValueError(f"--gold {' '.join(map(str, sorted(gold)))}: name at least two "
                         "strata; a one-stratum sheet shows every row's stratum")
    if path.exists():
        raise FileExistsError(
            f"{path} already exists — it holds {annotator}'s labels. "
            "Continue it with --label; it is never rewritten."
        )
    eligible, counts = eligible_rows(gens)
    kept = keep_gold(eligible, qrels, gold)
    counts["other_strata"] = len(eligible) - len(kept)
    counts["on_sheet"] = len(kept)
    if not eligible:
        raise ValueError("this batch has no refused rows to label")
    if not kept:
        raise ValueError(f"--gold leaves none of the batch's {len(eligible)} refused rows")
    rows = [twopass_row(g, corpus, qrels, annotator, guide_version) for g in kept]
    if gold is not None:
        for r in rows:
            r["gold_kept"] = sorted(gold)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_rows(path, rows)
    return counts


def expected_ids(rows: list[dict], gens: list[Generation], load_qrels) -> set[str]:
    """The query ids a finished file must cover: the batch's eligible rows,
    narrowed by the `gold_kept` filter recorded at write time, if any.
    `load_qrels` is called only when a filter is recorded."""
    filters = {tuple(r["gold_kept"]) if r.get("gold_kept") is not None else None
               for r in rows}
    if len(filters) > 1:
        raise AssertionError("rows in this file were written under different --gold filters")
    eligible = eligible_rows(gens)[0]
    gold = filters.pop() if filters else None
    if gold is not None:
        eligible = keep_gold(eligible, load_qrels(), set(gold))
    return {g.query_id for g in eligible}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def required_fields(row: dict, fields: tuple) -> list[str]:
    """The fields a finished row must fill: no per-row seconds on a sheet row."""
    if row.get("timing") == TIMING_SESSION:
        return [f for f in fields if not f.startswith("seconds_")]
    return list(fields)


def pass_a_done(row: dict) -> bool:
    return all(row[f] is not None for f in required_fields(row, PASS_A_FIELDS))


def pass_b_done(row: dict) -> bool:
    return all(row[f] is not None for f in required_fields(row, PASS_B_FIELDS)
               if f != "needs_common_knowledge")


def pass_b_started(rows: list[dict]) -> bool:
    return any(row[f] is not None for row in rows for f in PASS_B_FIELDS)


def pass_a_digest(rows: list[dict]) -> str:
    """A fingerprint of every pass A label in the file."""
    labels = sorted((r["query_id"], *(r[f] for f in PASS_A_FIELDS)) for r in rows)
    return hashlib.sha256(json.dumps(labels).encode()).hexdigest()[:16]


def lock_pass_a(rows: list[dict]) -> None:
    """Stamp the pass A fingerprint on every row, once, as pass B begins."""
    digest = pass_a_digest(rows)
    for r in rows:
        r["pass_a_lock"] = digest


def check_lock(rows: list[dict], required: bool = False) -> None:
    """Pass A must be exactly as it was when pass B began.

    Catches a CHANGED pass A label, not just a blanked one: an accept edited to
    reject after the passages were seen is in-vocabulary and complete, so
    nothing else would notice it.
    """
    locks = {r.get("pass_a_lock") for r in rows}
    if locks == {None}:
        # The lock is stamped before the first pass B row is shown, so pass B
        # labels without one mean the file was edited — and re-stamping it now
        # would accept whatever pass A it holds.
        if required or pass_b_started(rows):
            raise AssertionError("this file has pass B labels but pass A was never locked")
        return
    if len(locks) != 1 or locks != {pass_a_digest(rows)}:
        raise AssertionError(
            "pass A labels differ from the ones recorded when pass B began — pass A "
            "was changed after pass B began. Restore the file from a copy."
        )


def current_pass(rows: list[dict]) -> str | None:
    """"A", "B", or None when both passes are complete.

    Raises if pass A changed after pass B began — a blanked label (pass B labels
    beside an incomplete pass A) or an edited one (`check_lock`).
    """
    if not all(pass_a_done(r) for r in rows):
        if pass_b_started(rows):
            raise AssertionError(
                "this file has pass B labels but an incomplete pass A — pass A "
                "was changed after pass B began. Restore the file from a copy."
            )
        return "A"
    check_lock(rows)
    if not all(pass_b_done(r) for r in rows):
        return "B"
    return None


def check_header(rows: list[dict], annotator: str, guide_version: str) -> None:
    """Every row must belong to this annotator and this guide version."""
    who = {r["annotator"] for r in rows}
    if who != {annotator}:
        raise AssertionError(f"file holds labels by {sorted(who)}, not {annotator!r}")
    versions = {r["guide_version"] for r in rows}
    if versions != {guide_version}:
        raise AssertionError(
            f"file was written under guide version {sorted(versions)}, not "
            f"{guide_version!r} — a rule change mid-batch changes what the labels "
            "after it mean. Finish under the old version, or start a new batch."
        )


def render(row: dict, which: str, width: int = 88) -> str:
    """One row as text. Pass B adds the passages. Never a stratum field, never
    the row's pass A label."""
    def wrap(s, indent=""):
        # Each line of the source wrapped on its own: textwrap mangles widths
        # and drops the indent after an embedded newline.
        return "\n".join(
            textwrap.fill(line, width=width, initial_indent=indent,
                          subsequent_indent=indent) if line.strip() else ""
            for line in s.split("\n"))
    parts = [wrap(f"QUESTION: {row['question']}"), ""]
    if which == "B":
        parts += [wrap(p, "  ") for p in row["passages"]] + [""]
    parts += [wrap(f"RESPONSE — answer   : {row['reply_answer'] or '(none given)'}"),
              wrap(f"RESPONSE — rationale: {row['reply_rationale']}")]
    return "\n".join(parts)


def choose(choices: dict, ask) -> str:
    """Ask until the reply is a choice key, `s` (skip) or `q` (save and quit)."""
    keys = "/".join(list(choices) + ["s", "q"])
    while True:
        reply = ask(f"  [{keys}] ").strip().lower()
        if reply in choices or reply in ("s", "q"):
            return reply


def label_twopass(path: Path, annotator: str, guide_version: str,
                  ask=input, say=print, clock=time.monotonic, width: int = 88) -> int:
    """Label the current pass, one row at a time. Returns the number labelled.

    Saves after every row, so quitting costs at most the row on screen. A
    skipped or unfinished row keeps no partial answer and comes back next time,
    timed from scratch. The pass ends when its last row is labelled; pass B
    begins on the NEXT run, never in the same one.
    """
    rows = read_rows(path)
    check_header(rows, annotator, guide_version)
    if file_timing(rows) == TIMING_SESSION:
        raise AssertionError(
            f"{path.name} is labelled through spreadsheets (--export-sheet / "
            "--import-sheet); its rows have no per-row timing to add to"
        )
    which = current_pass(rows)
    if which is None:
        say(f"{path.name}: both passes complete ({len(rows)} rows). Verify with --check.")
        return 0

    if which == "B" and rows[0].get("pass_a_lock") is None:
        lock_pass_a(rows)
        save_rows(path, rows)

    todo = [i for i, r in enumerate(rows)
            if not (pass_a_done(r) if which == "A" else pass_b_done(r))]
    say(f"{path.name}: PASS {which}{' (with passages)' if which == 'B' else ''} — "
        f"{len(todo)} of {len(rows)} rows to label.\n"
        "  s = skip this row   q = save and quit\n")

    done = 0
    for n, i in enumerate(todo, 1):
        row = rows[i]
        say("\n" + "─" * width)
        say(f"[pass {which}  {n}/{len(todo)}]  {row['query_id']}\n")
        say(render(row, which, width))
        say("")
        start = clock()

        if which == "A":
            say(f"  {PASS_A_QUESTION}")
            k1 = choose(PASS_A_CHOICES, ask)
            if k1 == "q":
                break
            if k1 == "s":
                continue
            seconds = round(clock() - start, 1)   # to the label, not the follow-up
            say(f"  → {PASS_A_CHOICES[k1][0]}\n")
            say(f"  {CONFIDENCE_QUESTION}")
            k2 = choose(CONFIDENCE_CHOICES, ask)
            if k2 == "q":
                break
            if k2 == "s":
                continue
            row["refusal_ok_response"] = PASS_A_CHOICES[k1][1]
            row["confidence_response"] = CONFIDENCE_CHOICES[k2][1]
            row["seconds_response"] = seconds
            row["seconds_response_total"] = round(clock() - start, 1)
            say(f"  → {PASS_A_CHOICES[k1][0]}, {CONFIDENCE_CHOICES[k2][0]}")
        else:
            say(f"  {PASS_B_QUESTION}")
            k1 = choose(PASS_B_CHOICES, ask)
            if k1 == "q":
                break
            if k1 == "s":
                continue
            seconds = round(clock() - start, 1)   # to the label, not the checkbox
            ck = None
            if PASS_B_CHOICES[k1][1] is False:           # over-refusal only
                say(f"  → {PASS_B_CHOICES[k1][0]}\n")
                say(f"  {COMMON_KNOWLEDGE_QUESTION}")
                k2 = choose(COMMON_KNOWLEDGE_CHOICES, ask)
                if k2 == "q":
                    break
                if k2 == "s":
                    continue
                ck = COMMON_KNOWLEDGE_CHOICES[k2][1]
            row["refusal_ok_passages"] = PASS_B_CHOICES[k1][1]
            row["needs_common_knowledge"] = ck
            row["seconds_passages"] = seconds
            row["seconds_passages_total"] = round(clock() - start, 1)
            say(f"  → {PASS_B_CHOICES[k1][0]}"
                + ("" if ck is None else f", common knowledge needed: {'yes' if ck else 'no'}"))

        save_rows(path, rows)          # after every row, not at the end
        done += 1

    left = sum(not (pass_a_done(r) if which == "A" else pass_b_done(r)) for r in rows)
    if left:
        say(f"\nsaved. {done} labelled this session; pass {which} has {left} rows to go.")
    elif which == "A":
        say(f"\npass A complete ({len(rows)} rows). Its labels are now fixed. "
            "Run --label again to begin pass B.")
    else:
        say(f"\npass B complete ({len(rows)} rows). Verify with --check.")
    return done


def validate_twopass(rows: list[dict], expected_ids: set[str]) -> None:
    """The file is a finished, in-vocabulary label set for exactly these rows."""
    ids = {r["query_id"] for r in rows}
    if len(ids) != len(rows):
        raise AssertionError(f"{len(rows) - len(ids)} duplicated rows — one row per query id")
    if ids != expected_ids:
        raise AssertionError(
            f"file covers {len(ids)} rows, the batch has {len(expected_ids)} to label "
            f"({len(expected_ids - ids)} missing, {len(ids - expected_ids)} extra)"
        )
    check_lock(rows)        # before completeness: a changed pass A outranks an unfinished pass B
    allowed = {
        "refusal_ok_response": {True, False},
        "confidence_response": {"confident", "unsure"},
        "refusal_ok_passages": {True, False},
    }
    for field, values in allowed.items():
        bad = [r["query_id"] for r in rows if r[field] not in values]
        if bad:
            raise AssertionError(f"{len(bad)} rows have no valid '{field}' (e.g. {bad[:3]})")
    timed = [r for r in rows if r.get("timing") != TIMING_SESSION]
    for field in ("seconds_response", "seconds_passages"):
        total = f"{field}_total"
        bad = [r["query_id"] for r in timed
               if not all(isinstance(r[f], (int, float)) for f in (field, total))
               or not 0 <= r[field] <= r[total]]
        if bad:
            raise AssertionError(
                f"{len(bad)} rows have no valid '{field}' / '{total}' (e.g. {bad[:3]})")
    # A sheet row's time is per sitting; a per-row number on one was typed in.
    bad = [r["query_id"] for r in rows if r.get("timing") == TIMING_SESSION
           and any(r[f] is not None for f in PASS_A_FIELDS + PASS_B_FIELDS
                   if f.startswith("seconds_"))]
    if bad:
        raise AssertionError(f"{len(bad)} sheet rows carry per-row seconds (e.g. {bad[:3]})")
    # The checkbox exists only on over-refusals: missing there, or present
    # anywhere else, is a file that was edited by hand.
    bad = [r["query_id"] for r in rows
           if (r["needs_common_knowledge"] not in (True, False)
               if r["refusal_ok_passages"] is False
               else r["needs_common_knowledge"] is not None)]
    if bad:
        raise AssertionError(
            f"{len(bad)} rows have `needs_common_knowledge` set where it does not "
            f"apply, or missing where it does (e.g. {bad[:3]})"
        )
    check_lock(rows, required=True)


def main(dataset: str, n_queries: int, n_context: int, top_k: int, seed: int,
         annotator: str | None, guide_version: str | None, out_dir: Path, label: bool,
         check: bool, model: str = BI_ENCODER, generator: str = GENERATOR,
         exclude: list[str] | None = None, assign: list[str] | None = None,
         shared: str = "0", assign_seed: int = 0, export_sheet: str | None = None,
         import_sheet: Path | None = None, sittings: Path | None = None,
         gold: list[int] | None = None) -> None:
    from . import sheets          # imports this module; deferred to avoid a cycle

    actions = [label, check, bool(assign), bool(export_sheet), bool(import_sheet)]
    if sum(actions) > 1:
        raise SystemExit("use one of --label, --check, --assign, --export-sheet, --import-sheet")
    # A flag without its parent would otherwise fall through to the default
    # action and write a new terminal sheet.
    if sittings and not import_sheet:
        raise SystemExit("--sittings goes with --import-sheet")
    if (shared != "0" or assign_seed != 0) and not assign:
        raise SystemExit("--shared and --assign-seed go with --assign")
    if gold and any(actions):
        raise SystemExit("--gold applies only when writing a terminal sheet: --label and "
                         "--check read it back from the file, and the spreadsheet actions "
                         "(--assign, --export-sheet, --import-sheet) do not take it")
    excluded = excluded_ids(exclude)
    gens = load_generations(dataset, model, top_k, n_context, generator, n_queries,
                            seed, excluded)
    gen_path = generation_cache_path(dataset, model, top_k, n_context, generator,
                                     PROMPT_VERSION, n_queries, seed, excluded)
    flags = batch_flags(dataset, n_queries, n_context, top_k, seed, model, generator, exclude)
    if Path(out_dir) != LABELS_DIR:
        flags += f" --out {shlex.quote(str(out_dir))}"

    if assign:
        _corpus, _queries, qrels = load_beir(dataset)
        assignment = sheets.make_assignment(sheets.batch_strata(gens, qrels), assign,
                                            shared, assign_seed)
        apath = sheets.assignment_path(out_dir, gen_path)
        sheets.write_assignment(apath, assignment)
        log.info("\n=== assignment (seed %d) ===", assign_seed)
        log.info("rows to label  : %d", assignment["n_rows"])
        log.info("shared         : %d  (every annotator labels these)", len(assignment["shared"]))
        for name, own in assignment["own"].items():
            log.info("%-15s: %d own + %d shared = %d", name, len(own),
                     len(assignment["shared"]), len(own) + len(assignment["shared"]))
        log.info("\nassignment -> %s", apath)
        return

    if not annotator or not guide_version:
        raise SystemExit("--annotator and --guide-version are required")
    path = twopass_path(out_dir, gen_path, annotator)
    flags += f" --annotator {annotator} --guide-version {shlex.quote(guide_version)}"

    if export_sheet or import_sheet:
        assignment = sheets.read_assignment(sheets.assignment_path(out_dir, gen_path))
        if path.exists():
            load_labels(path, gens)     # anchor check: the batch was not regenerated
        if export_sheet:
            which = {"pass-a": "A", "pass-b": "B"}[export_sheet]
            corpus = qrels = None
            if which == "A" and not path.exists():
                corpus, _queries, qrels = load_beir(dataset)
            csv_path, template, risky = sheets.export_sheet(
                which, path, assignment, annotator, guide_version, gens, corpus, qrels)
            log.info("\n=== pass %s sheet for %s ===", which, annotator)
            log.info("rows           : %d", len(sheets.assigned_ids(assignment, annotator)))
            log.info("sheet          : %s", csv_path)
            log.info("sittings sheet : %s", template)
            log.info("\nIn Google Sheets, import with 'Convert text to numbers, dates, and "
                     "formulas' unticked%s.",
                     f" — {risky} text cells start with = + - or @" if risky else "")
            log.info("Import it filled: python -m retrieval_lab.twopass %s --import-sheet "
                     "<filled.csv> --sittings <sittings.csv>", flags)
            return
        if not sittings:
            raise SystemExit("--import-sheet needs --sittings <the filled sittings sheet>")
        which = sheets.sheet_pass(import_sheet)
        result = sheets.import_sheet(which, path, import_sheet, sittings, assignment,
                                     annotator, guide_version)
        if result == "sittings":
            log.info("pass A labels identical to the locked ones; sittings updated")
        else:
            log.info("pass %s imported: %d rows -> %s", which,
                     len(sheets.assigned_ids(assignment, annotator)), path)
        if which == "A":
            log.info("Next: python -m retrieval_lab.twopass %s --export-sheet pass-b", flags)
        else:
            log.info("Next: python -m retrieval_lab.twopass %s --check", flags)
        return

    if label or check:
        if not path.exists():
            raise FileNotFoundError(
                f"no sheet at {path} — write it first with:\n"
                f"  python -m retrieval_lab.twopass {flags}"
            )
        rows = load_labels(path, gens)  # anchor check: the batch was not regenerated
        if label:
            label_twopass(path, annotator, guide_version)
            return
        check_header(rows, annotator, guide_version)
        if sheets.is_sheet_file(path):
            assignment = sheets.read_assignment(sheets.assignment_path(out_dir, gen_path))
            minutes = sheets.check_sheet_file(path, rows, assignment, annotator)
            log.info("labels OK: %d rows, both passes complete, anchored to the batch",
                     len(rows))
            log.info("time: pass A %.1f min (%.2f per row), pass B %.1f min (%.2f per row)",
                     minutes["A"], minutes["A"] / len(rows),
                     minutes["B"], minutes["B"] / len(rows))
            return
        validate_twopass(rows, expected_ids(rows, gens, lambda: load_beir(dataset)[2]))
        log.info("labels OK: %d rows, both passes complete, anchored to the batch", len(rows))
        return

    corpus, _queries, qrels = load_beir(dataset)
    counts = write_twopass_sheet(gens, corpus, qrels, path, annotator, guide_version,
                                 set(gold) if gold else None)
    log.info("\n=== two-pass sheet (%s) ===", annotator)
    log.info("batch rows     : %d", counts["batch"])
    log.info("refused        : %d", counts["refused"])
    log.info("cut off        : %d  (left off the sheet: the cap ended the reply)",
             counts["cut_off"])
    log.info("not recorded   : %d  (on the sheet; the cap was not recorded)",
             counts["not_recorded"])
    if gold:
        log.info("other strata   : %d  (left off the sheet: --gold %s)",
                 counts["other_strata"], " ".join(map(str, sorted(gold))))
    log.info("on the sheet   : %d", counts["on_sheet"])
    log.info("\nsheet -> %s", path)
    log.info("Label it: python -m retrieval_lab.twopass %s --label", flags)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hotpotqa-distractor-pool")
    p.add_argument("--n-queries", type=int, required=True, help="which batch")
    p.add_argument("--seed", type=int, required=True, help="which batch")
    p.add_argument("--exclude", action="append", default=[], metavar="PATH",
                   help="the --exclude files the batch was generated with, or its "
                        "exclude__…json record; repeatable")
    p.add_argument("--n-context", type=int, default=10)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--annotator", help="one file per annotator (required except with --assign)")
    p.add_argument("--guide-version",
                   help="the labelling guide's version; recorded on every row "
                        "(required except with --assign)")
    p.add_argument("--out", type=Path, default=LABELS_DIR,
                   help=f"directory for the sheet (default {LABELS_DIR})")
    p.add_argument("--label", action="store_true",
                   help="label the current pass, one row at a time, resumably")
    p.add_argument("--check", action="store_true",
                   help="verify both passes are complete and anchored to the batch")
    p.add_argument("--assign", nargs="+", metavar="NAME",
                   help="split the batch between these spreadsheet annotators; written once")
    p.add_argument("--shared", default="0",
                   help="with --assign: rows every annotator labels, a count or 'all'")
    p.add_argument("--assign-seed", type=int, default=0, help="with --assign: the split's seed")
    p.add_argument("--export-sheet", choices=["pass-a", "pass-b"],
                   help="write this annotator's CSV for one pass")
    p.add_argument("--import-sheet", type=Path, metavar="CSV",
                   help="read a filled CSV back (its pass is read from the header)")
    p.add_argument("--sittings", type=Path, metavar="CSV",
                   help="with --import-sheet: the filled sittings sheet for that pass")
    p.add_argument("--model", default=BI_ENCODER, help=MODEL_HELP)
    p.add_argument("--generator", default=GENERATOR)
    p.add_argument("--gold", type=int, nargs="+", choices=[0, 1, 2], metavar="N",
                   help="when writing a terminal sheet: keep only refusals with this many "
                        "gold passages in context (e.g. --gold 1 2); recorded in the file")
    args = p.parse_args()
    main(args.dataset, args.n_queries, args.n_context, args.top_k, args.seed,
         args.annotator, args.guide_version, args.out, args.label, args.check,
         args.model, args.generator, args.exclude, args.assign, args.shared,
         args.assign_seed, args.export_sheet, args.import_sheet, args.sittings, args.gold)
