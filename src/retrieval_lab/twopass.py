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


def twopass_path(out_dir: Path, generation_path: Path, annotator: str) -> Path:
    """One file per (batch, annotator). The batch's own name is the key, so
    every key component — exclusions included — carries over unchanged."""
    if not ANNOTATOR_RE.fullmatch(annotator):
        raise ValueError(f"annotator {annotator!r}: use letters, digits, - or _ only")
    batch = generation_path.stem.removeprefix("gen__")
    return Path(out_dir) / f"twopass__{batch}__{annotator}.jsonl"


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


def write_twopass_sheet(gens: list[Generation], corpus: dict, qrels: dict, path: Path,
                        annotator: str, guide_version: str) -> dict[str, int]:
    """Write a fresh sheet. Never overwrites: the file holds hand labels."""
    if path.exists():
        raise FileExistsError(
            f"{path} already exists — it holds {annotator}'s labels. "
            "Continue it with --label; it is never rewritten."
        )
    kept, counts = eligible_rows(gens)
    if not kept:
        raise ValueError("this batch has no refused rows to label")
    rows = [twopass_row(g, corpus, qrels, annotator, guide_version) for g in kept]
    path.parent.mkdir(parents=True, exist_ok=True)
    save_rows(path, rows)
    return counts


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pass_a_done(row: dict) -> bool:
    return all(row[f] is not None for f in PASS_A_FIELDS)


def pass_b_done(row: dict) -> bool:
    return all(row[f] is not None for f in PASS_B_FIELDS if f != "needs_common_knowledge")


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
    for field in ("seconds_response", "seconds_passages"):
        total = f"{field}_total"
        bad = [r["query_id"] for r in rows
               if not all(isinstance(r[f], (int, float)) for f in (field, total))
               or not 0 <= r[field] <= r[total]]
        if bad:
            raise AssertionError(
                f"{len(bad)} rows have no valid '{field}' / '{total}' (e.g. {bad[:3]})")
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
         annotator: str, guide_version: str, out_dir: Path, label: bool, check: bool,
         model: str = BI_ENCODER, generator: str = GENERATOR,
         exclude: list[str] | None = None) -> None:
    excluded = excluded_ids(exclude)
    gens = load_generations(dataset, model, top_k, n_context, generator, n_queries,
                            seed, excluded)
    gen_path = generation_cache_path(dataset, model, top_k, n_context, generator,
                                     PROMPT_VERSION, n_queries, seed, excluded)
    path = twopass_path(out_dir, gen_path, annotator)
    flags = (batch_flags(dataset, n_queries, n_context, top_k, seed, model,
                         generator, exclude)
             + f" --annotator {annotator} --guide-version {shlex.quote(guide_version)}"
             + (f" --out {shlex.quote(str(out_dir))}" if Path(out_dir) != LABELS_DIR else ""))

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
        validate_twopass(rows, {g.query_id for g in eligible_rows(gens)[0]})
        log.info("labels OK: %d rows, both passes complete, anchored to the batch", len(rows))
        return

    corpus, _queries, qrels = load_beir(dataset)
    counts = write_twopass_sheet(gens, corpus, qrels, path, annotator, guide_version)
    log.info("\n=== two-pass sheet (%s) ===", annotator)
    log.info("batch rows     : %d", counts["batch"])
    log.info("refused        : %d", counts["refused"])
    log.info("cut off        : %d  (left off the sheet: the cap ended the reply)",
             counts["cut_off"])
    log.info("not recorded   : %d  (on the sheet; the cap was not recorded)",
             counts["not_recorded"])
    log.info("on the sheet   : %d", counts["refused"] - counts["cut_off"])
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
    p.add_argument("--annotator", required=True, help="one file per annotator")
    p.add_argument("--guide-version", required=True,
                   help="the labelling guide's version; recorded on every row")
    p.add_argument("--out", type=Path, default=LABELS_DIR,
                   help=f"directory for the sheet (default {LABELS_DIR})")
    p.add_argument("--label", action="store_true",
                   help="label the current pass, one row at a time, resumably")
    p.add_argument("--check", action="store_true",
                   help="verify both passes are complete and anchored to the batch")
    p.add_argument("--model", default=BI_ENCODER, help=MODEL_HELP)
    p.add_argument("--generator", default=GENERATOR)
    args = p.parse_args()
    main(args.dataset, args.n_queries, args.n_context, args.top_k, args.seed,
         args.annotator, args.guide_version, args.out, args.label, args.check,
         args.model, args.generator, args.exclude)
