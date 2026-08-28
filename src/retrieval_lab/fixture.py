"""Phase 4c.1: turn a generation batch into a hand-labelling worksheet.

    python -m retrieval_lab.fixture --n-queries 30 --seed 1          # write the sheet
    python -m retrieval_lab.fixture --n-queries 30 --seed 1 --label  # label it, resumably
    python -m retrieval_lab.fixture --n-queries 30 --seed 1 --check  # verify the labels

This module does not judge anything and does not score anything. It reads a
cached generation batch and emits one JSONL row per item with blank label
fields for Betty to fill in; `--check` reads them back and validates them
against the batch they were written from.

**Why the fixture is drawn held-out (`--seed 1`).** The labels choose a judge
(D13's bake-off) and then that judge scores the `seed 0` batches. Selecting the
judge on the same items it will be graded on is the same leak as tuning a
threshold on the test set: the winning candidate is partly winning because it
matched 30 specific labels. Different seed, different questions, no leak — but
`sample_queries` draws fresh from the shared pool rather than excluding the
earlier draw, so overlap is *possible* and this module counts it out loud
rather than assuming it away (see `overlap_with`).

**Why every row carries `raw_sha`.** Generation is nondeterministic even at
temperature 0 — `cache.generation_cache_path` explains why at length. So a
label is only meaningful against the exact answer text it was written for: run
`--refresh` on the batch, and 30 hand labels silently start pointing at text
that no longer exists, with nothing raising. Hashing the generation's `raw`
into the label row makes that failure loud, and it is the same class of bug as
a leaked `corpus_id`: well-formed output, wrong meaning, no exception.

**The sheet is blind to the gold answer, on purpose.** Groundedness asks "is
every claim supported by these passages", which the passages alone can settle;
correctness asks "is this the right answer", which needs the gold span. They
are orthogonal — the probe produced an answer that was *right* while citing a
passage that never made the claim — and showing the gold answer while labelling
groundedness leaks one into the other. `answers.jsonl` is therefore never read
by the writer. Correctness is scored automatically by token-F1 later; it does
not need a hand label.

`stratum` DOES read qrels, which is analysis-only — the licence `oracle.py`
has, and it never touches the generation path. It exists so the sheet shows
whether this 30-draw actually covers the interesting strata (an answer
committed on partial gold is where faithfulness is genuinely at stake) instead
of finding out after the labelling evening.
"""
import argparse
import hashlib
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path

from .cache import cached_retrieval, generation_cache_path
from .data import load_beir
from .generate import (GENERATOR, PROMPT_VERSION, Generation, cached_generations,
                       context_for, sample_queries)
from .retrieve import BI_ENCODER, MODEL_HELP

log = logging.getLogger("retrieval_lab")

LABELS_DIR = Path("data/labels")

# The value that means "this axis does not apply to this row" — a refused answer
# has no claims to ground, an answered one has no refusal to judge. Distinct from
# None, which means "not labelled yet": one is a finished decision, the other is
# an unfinished evening, and `--check` has to tell them apart. When κ is computed
# these rows leave that axis's DENOMINATOR rather than scoring as agreement —
# the same rule D14 settled for refusals and correctness.
NOT_APPLICABLE = "n/a"


# ─────────────────────────── Betty's stub ───────────────────────────

def blank_labels() -> dict:
    """The label axes — one dict of empty fields, merged into every sheet row.

    CONCEPT — this dict IS the fixture's definition, and three things downstream
    are shaped by it:

    1. **What κ is computed over.** Cohen's κ needs two raters assigning the
       same *categorical* label to the same items. Every axis you put here
       becomes a κ column (judge vs you); every axis you leave out cannot be
       reported no matter how good the judge is. A free-text field is a note,
       not an axis — it never becomes a κ.

    2. **How many categories, and whether they are ordered.** Binary
       (grounded / not) makes κ easy to read and easy to agree on, but collapses
       "one unsupported clause in an otherwise-grounded answer" into the same
       bucket as a fabrication. A 3-point scale keeps that distinction and costs
       you agreement — raters disagree far more on middle categories, so κ drops
       and you cannot tell whether the judge got worse or the scale got harder.
       Pick on purpose; the README row has to say which.

    3. **Refused items need their own axis or they distort the main one.** ~1 in
       3 of these rows will be `__REFUSED__` (the n=100 batch refused 34%).
       A refusal has no claims, so it is trivially "grounded" — score it on the
       groundedness axis and every refusal becomes a free point that inflates
       agreement without measuring anything. The question worth labelling for a
       refusal is different: *should* it have refused, given these passages?
       That is the over-refusal / calibrated-refusal split the n=100 batch
       estimated at 5 vs 29, and it is what D15's stratified claim rests on.

    Return the axes as `{field_name: None}` (or a sentinel like ""), one entry
    per thing you will actually decide per item. `validate_labels` reads the
    keys from here, so the sheet, the checker and the loader stay in sync with
    whatever you choose — you should not have to edit them.

    Note the sheet deliberately does NOT show the gold answer, so do not add a
    correctness axis here; token-F1 scores that from `answers.jsonl` without a
    human. See this module's docstring for why the blinding matters.
    """
    return {"grounded": None, "refusal_ok": None}


# The allowed values per axis — the rubric, in code rather than in your head.
#
# CONCEPT — this exists because a mislabelled row is not a visible error. Type
# "ture", or "partial " with a trailing space, or "yes" three weeks after you
# wrote true, and nothing raises: κ just quietly gains a category that is really
# a typo, and no amount of staring at the number reveals it. Same class as a
# leaked `corpus_id` — well-formed, wrong meaning, no exception. `validate_labels`
# turns that into a raise naming the offending row.
#
# THE RUBRIC OF RECORD — settled 2026-08-23, binary on both axes.
#
# Written down before row 1 was labelled, because a boundary invented at row 40
# means rows 1–39 were labelled under a different rule and the κ silently mixes
# two scales. That is rater drift, and it is unrecoverable without relabelling.
# The README row reporting κ must state the scale: a κ is not comparable across
# scales, and "we hand-labelled it" does not tell a reader whether a borderline
# answer counted as grounded.
#
#   grounded (answered rows only)
#     true  — EVERY clause of the rationale is supported by the passages it
#             cites. Strict by choice: one unsupported clause makes the row
#             false, even if the rest checks out and even if the short answer is
#             correct. The test is "could a reader verify this answer from the
#             cited passages", not "is this answer right".
#     false — any clause unsupported, miscited, or reasoned past what the
#             passage establishes.
#     Both probe cases in LEARNINGS.md land on false under this rule, and they
#     are the two shapes to expect:
#       · 7th Sea — cited [1] for a claim [1] never makes. Decorative citation.
#       · Vienna  — reasoned from a real in-context passage (Gluck's biography)
#                   toward a conclusion that passage does not establish. The
#                   subject is right, the inference is not carried by the text.
#                   This is the harder call and it is deliberately `false`: the
#                   passage gets you to "a Gluck opera", not to the answer.
#
#   refusal_ok (refused rows only)
#     true  — refusing was the right call on these passages.
#     false — an OVER-REFUSAL, judged by a deliberately narrow test: the
#             rationale itself names the answer correctly, and the model refused
#             anyway. Seed-1 row 1 is the type case — it identifies Edmund
#             Mortimer, then emits the sentinel on a technicality.
#     "CORRECTLY" MEANS: what these ten passages ESTABLISH — not what HotpotQA's
#     gold answer says. The two come apart exactly where it matters. On a row
#     with `gold_in_context: 0` the gold answer is not reachable from the context
#     the model saw, so a refusal there is calibrated regardless of the fact that
#     an answer exists in the world; grading against gold would mark the model
#     down for information it never had. And the bar is ESTABLISH, not SUGGEST —
#     the Vienna case's passages point toward an answer without carrying it, and
#     reading "points toward" as "was derivable" would turn calibrated refusals
#     into over-refusals. This is the same strictness `grounded` uses, and it is
#     why the sheet stays blind to the gold answer: shown one, a labeller
#     inevitably starts checking "did the rationale name THE GOLD ANSWER", which
#     is a different, easier, and wrong question.
#
#     NOTE the narrowness, and that it is a real choice: a refusal whose
#     rationale does NOT name the answer scores `true` under this rule even if
#     the answer was in fact derivable from the passages. That makes the axis
#     CONSERVATIVE — it counts only over-refusals the model's own text convicts
#     it of — which buys reproducibility (two raters agree on "does the
#     rationale name it") at the cost of undercounting the true over-refusal
#     rate. Any README claim from this axis is therefore a LOWER BOUND, and
#     should say so.
LABEL_VALUES: dict[str, set] = {
    "grounded": {True, False},
    "refusal_ok": {True, False},
}

# THE BOUNDARIES — written down 2026-08-26, after seed 1 was labelled and read
# back off the sheet. They are recorded here rather than in anyone's head because
# a boundary that is not written is a boundary that drifts, and drift is
# unrecoverable without relabelling. `build_judge_prompt` in `judge.py` must
# mirror these two rules; a judge graded against labels it was given a different
# rubric for measures rubric disagreement, not judge quality.
#
#   grounded — `true` requires EVERY clause of the rationale to be supported by
#     the passages it cites. Strict, and strict in three specific ways, each of
#     which appears in seed 1 as a `false`:
#       - UNCITED ASSERTION. A factual clause carrying no citation at all is
#         unsupported, even when a passage happens to support it. ("The Mark-8
#         used the Intel 8008, the Comx-35 the RCA 1802" — both true, neither
#         cited.) The test is whether a reader could VERIFY the answer from the
#         citations given, not whether the claim is true.
#       - INFERENCE PAST THE PASSAGE. A clause the passage makes plausible but
#         does not establish. ("holy servant of Christ" and an association with
#         Exeter Cathedral do not establish "saints".)
#       - UNSUPPORTED NEGATIVE. A denial the passage does not license, including
#         one drawn from a passage's silence. ("...works in film, but not
#         photography" — the passage lists photography.)
#     `grounded` is INDEPENDENT OF CORRECTNESS: rows whose short answer is right
#     are labelled `false` when the rationale is not verifiable from what it
#     cites ("yes"; "Edmund Mortimer"). This is the axis the sheet is blinded to
#     the gold answer to protect.
#
#   refusal_ok — `true` = "refusing was the right call given these passages",
#     which is a question about the PASSAGES, not about the model's reasoning.
#       - `true` when the answer genuinely is not derivable from the context.
#         (Asked BJ's Wholesale Club's location count, the passages give U.S.
#         Vision's ~650 and never BJ's own. Not derivable; refusing is correct.)
#       - `false` when the passages DID support an answer and the model failed to
#         see it — including refusals turning on the question's exact phrasing.
#         (Passage [6] establishes Hund's first rule and its importance in
#         chemistry; refusing because no passage says the words "first rule of
#         chemistry" is misreading present context, not calibration.)
#     Note the asymmetry this creates, and it is the reason the axis yields only
#     a LOWER BOUND on over-refusal: a refusal is scored against what the context
#     supports, so a `true` here means "not derivable", never "the model was
#     right to be unsure".


# ─────────────────────── plumbing (wired, working) ───────────────────────

def sheet_path(dataset: str, n_queries: int, seed: int) -> Path:
    """Where one draw's worksheet lives. One function, because `judge.py` reads
    these sheets too and a second copy of the format is a second thing to drift."""
    return LABELS_DIR / f"fixture__{dataset}__n{n_queries}__seed{seed}.jsonl"


def raw_sha(raw: str) -> str:
    """Short content hash of a generation's raw text — the label's anchor."""
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def gold_in_context(qrels: dict, query_id: str, doc_ids: list[str]) -> int:
    """How many of this query's gold docs retrieval actually put in the prompt.

    Analysis-only (reads qrels), never on the generation path. HotpotQA is
    2-gold-per-question, so this is 0, 1 or 2 and names the stratum.
    """
    return len(set(doc_ids) & set(qrels.get(query_id, {})))


def load_generations(dataset: str, model: str, top_k: int, n_context: int,
                     generator: str, n_queries: int, seed: int) -> list[Generation]:
    """Read one cached generation batch. Never generates — raises if absent.

    Writing a worksheet must not be able to trigger a model run: a batch that
    appeared as a side effect of asking for a sheet is a batch nobody logged.
    """
    path = generation_cache_path(dataset, model, top_k, n_context, generator,
                                 PROMPT_VERSION, n_queries, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"no generation batch at {path} — draw it first with:\n"
            f"  python -m retrieval_lab.generate --n-queries {n_queries} --seed {seed}"
        )
    return cached_generations(path, compute=lambda: (_ for _ in ()).throw(
        RuntimeError("unreachable: the cache file exists")))


def same_draw(a, b) -> bool:
    """Are these two cache files the same sample, differing only in prompt version?

    `sample_queries` keys on (n_queries, seed) and nothing else, so two files
    whose names match once the `__v<N>__` token is removed hold the same question
    ids by construction.
    """
    strip = lambda p: re.sub(r"__v\d+__", "__", Path(p).name)
    return strip(a) == strip(b)


def overlap_with(qids: list[str], own_path) -> dict[str, list[str]]:
    """Which other batches on disk share query ids with this draw.

    `sample_queries` re-draws from the whole eligible pool per (n, seed); it does
    not exclude an earlier draw. Held-out is therefore an *intent*, and this
    counts the exceptions instead of trusting the seed.
    """
    from .cache import CACHE_DIR

    here = set(qids)
    shared = {}
    for path in sorted(CACHE_DIR.glob("gen__*.json")):
        # Exclude this batch by PATH, not by "it shares every id" — a larger
        # batch that happens to contain all 30 (a full-pool run) is a real
        # overlap, and the count test would have called it clean.
        if path.resolve() == Path(own_path).resolve():
            continue
        # ... and exclude the SAME draw under a different prompt version. A
        # prompt bump re-asks the same sampled questions by construction, so its
        # old file shares every id without that being a leak. Warning about it
        # is worse than useless: a warning that fires on a correct state is one
        # you learn to scroll past, and the next real overlap scrolls past too.
        if same_draw(path, own_path):
            continue
        rows = json.loads(path.read_text())
        common = sorted(here & {r["query_id"] for r in rows})
        if common:
            shared[path.name] = common
    return shared


def sheet_row(gen: Generation, corpus: dict, qrels: dict) -> dict:
    """One worksheet line: what to read, what to fill in, and what pins it.

    Field order is chosen for hand-editing — the blank label fields sit last so
    they are where the cursor lands. `passages` is rendered with the same [1]…[n]
    labels `build_prompt` used, so a citation in the rationale can be checked
    against the passage it names without cross-referencing doc ids.
    """
    return {
        "query_id": gen.query_id,
        "raw_sha": raw_sha(gen.raw),
        "stratum": {
            "gold_in_context": gold_in_context(qrels, gen.query_id, gen.doc_ids),
            "refused": gen.refused,
        },
        "question": gen.question,
        "passages": [f"[{i}] {corpus[d]['title']}. {corpus[d]['text']}"
                     for i, d in enumerate(gen.doc_ids, 1)],
        "answer": gen.answer,
        "rationale": gen.rationale,
        **labels_for(gen),
    }


# Which axis does NOT apply to which kind of row, keyed by `Generation.refused`.
# A refusal makes no claims, so it cannot be grounded; an answer cannot be an
# over-refusal. Both are decisions the `refused` flag already knows, so the sheet
# fills them in rather than asking for 60 hand-typed "n/a"s — the fewer fields a
# human types, the fewer a human typos.
#
# Rename an axis in `blank_labels()` and this is the one other place to update;
# `labels_for` raises rather than silently skipping an axis it cannot find.
INAPPLICABLE_AXIS = {True: "grounded", False: "refusal_ok"}


def labels_for(gen: Generation) -> dict:
    """Blank label fields for one row, with the inapplicable axis pre-filled."""
    labels = blank_labels()
    axis = INAPPLICABLE_AXIS[gen.refused]
    if axis not in labels:
        raise AssertionError(
            f"INAPPLICABLE_AXIS names '{axis}' but the axes are {sorted(labels)} — "
            "an axis was renamed in blank_labels() and this mapping did not follow."
        )
    labels[axis] = NOT_APPLICABLE
    return labels


def write_sheet(gens: list[Generation], corpus: dict, qrels: dict, path: Path,
                force: bool = False) -> Path:
    """Write the worksheet, refusing to clobber labels already filled in.

    The overwrite guard is the whole reason this is a function: these rows are
    the only artifact in the repo that cannot be regenerated, and a second run
    of the writer is exactly what would silently blank an evening of them.
    """
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists — refusing to overwrite hand labels. "
            "Pass --force if you are certain it is unlabelled."
        )
    # Every row is built BEFORE the file is opened: opening for write truncates,
    # so a row that raises halfway through would leave --force having destroyed
    # labels for a sheet it then failed to write.
    lines = [json.dumps(sheet_row(g, corpus, qrels), ensure_ascii=False) for g in gens]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def load_labels(path: Path, gens: list[Generation]) -> list[dict]:
    """Read the sheet back and prove it still describes THIS batch.

    Three ways a label set goes quietly wrong, all raised here:
      - a row for a query id the batch does not contain (wrong seed / wrong n)
      - a `raw_sha` mismatch (the batch was --refresh'd under the labels)
      - a row with a label field still blank — raised by `validate_labels`,
        which `--check` runs immediately after this
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {g.query_id: g for g in gens}

    missing = [r["query_id"] for r in rows if r["query_id"] not in by_id]
    if missing:
        raise AssertionError(
            f"{len(missing)} labelled rows are not in this batch (e.g. {missing[:3]}) — "
            "the labels were written against a different draw."
        )

    stale = [r["query_id"] for r in rows if r["raw_sha"] != raw_sha(by_id[r["query_id"]].raw)]
    if stale:
        raise AssertionError(
            f"{len(stale)} labels point at generation text that has changed "
            f"(e.g. {stale[:3]}) — the batch was regenerated under them. The labels "
            "are not recoverable by re-running; restore the batch or relabel."
        )
    return rows


def validate_labels(rows: list[dict], n_expected: int) -> None:
    """Blank-field and coverage check. Axis names come from `blank_labels()`.

    A blank field RAISES rather than warns: `--check` is the thing that says
    "this fixture is finished", and a half-labelled sheet passing it is the
    failure the whole module is built to prevent. A short sheet only warns —
    dropping rows (an overlapping query id, say) is a legitimate choice.

    Filled fields are checked against `LABEL_VALUES` for the same reason: a
    typo'd label is indistinguishable from a real category once it reaches a κ.
    """
    axes = list(blank_labels())

    if set(LABEL_VALUES) != set(axes):
        raise AssertionError(
            f"LABEL_VALUES covers {sorted(LABEL_VALUES)} but the axes are "
            f"{sorted(axes)} — every axis needs its allowed values, or the "
            "rubric and the sheet have drifted apart."
        )

    blank = [r["query_id"] for r in rows
             if any(r.get(a) in (None, "") for a in axes)]
    if blank:
        raise AssertionError(
            f"{len(blank)}/{len(rows)} rows still have a blank label field "
            f"(e.g. {blank[:5]}) — this sheet is not a finished fixture."
        )

    # Reported per axis rather than as one flat count: a whole axis typed in the
    # wrong vocabulary and one slip on one row are different mistakes.
    for axis in axes:
        allowed = set(LABEL_VALUES[axis]) | {NOT_APPLICABLE}
        bad = [(r["query_id"], r[axis]) for r in rows if r[axis] not in allowed]
        if bad:
            raise AssertionError(
                f"{len(bad)} rows have a value for '{axis}' outside the rubric "
                f"{sorted(map(str, allowed))} — e.g. {bad[:3]}. A typo'd label "
                "becomes a real category in the κ and nothing else would catch it."
            )

    if len(rows) != n_expected:
        log.warning("sheet holds %d rows, batch holds %d", len(rows), n_expected)

    for axis in axes:
        scored = [r[axis] for r in rows if r[axis] != NOT_APPLICABLE]
        counts = {v: sum(x == v for x in scored) for v in sorted(set(scored), key=str)}
        log.info("%-12s n=%-3d (%d n/a)  %s", axis, len(scored),
                 len(rows) - len(scored), counts)


def report_strata(gens: list[Generation], qrels: dict) -> None:
    """Print the (gold_in_context × refused) grid this draw actually landed on.

    The strata are the reason to look at a sheet before labelling it: the cell
    that matters most for faithfulness — answered while holding only half the
    gold chain — is also the one a 30-draw can miss entirely.
    """
    grid: dict[tuple[int, bool], int] = {}
    for g in gens:
        key = (gold_in_context(qrels, g.query_id, g.doc_ids), g.refused)
        grid[key] = grid.get(key, 0) + 1

    log.info("\n=== strata in this draw (n=%d) ===", len(gens))
    log.info("%-16s %-10s %s", "gold in context", "answered", "refused")
    for gold in sorted({k[0] for k in grid}, reverse=True):
        log.info("%-16d %-10d %d", gold, grid.get((gold, False), 0), grid.get((gold, True), 0))


# ───────────────────────── the labelling loop ─────────────────────────
#
# Reading JSONL is bearable; hand-EDITING it is not — typing `true` at the end of
# a 6,000-character line, sixty times, is how a row gets corrupted at 11pm. This
# renders one row, asks the one question that row needs, and writes the answer
# back. Ergonomics only: it decides nothing, and every value it writes is one
# `validate_labels` would already have accepted.


# What `y` MEANS, on screen, every single row.
#
# This is the one corruption channel no checker can close. Every other mistake
# the sheet can absorb is catchable — a typo fails the rubric check, a blank
# fails --check, a regenerated batch fails the raw_sha anchor. But labels typed
# under a remembered-backwards polarity are all in-vocabulary and internally
# consistent: --check passes, κ computes, and the number is simply wrong. So the
# question is rendered in full above the prompt rather than trusting the slug to
# carry it at 11pm — and it doubles as the rubric being on-screen while labelling.
#
# These sentences MUST match the boundaries recorded above LABEL_VALUES. Sharpen
# one and you sharpen both — they are the same rubric, and this is the copy that
# is actually on screen at 11pm.
#
# AMENDED 2026-08-26: `refusal_ok` previously read "n = OVER-REFUSAL: the
# rationale itself names the answer correctly, and it refused anyway", which
# tests the RATIONALE. The boundary of record tests the PASSAGES, and the two
# disagree on one case — passages support an answer, rationale never names it.
# The rationale-based wording scored that `y`; the rule below scores it `n`,
# because a refusal is over-refusal whether or not the model's own text exposes
# it. Seed 1 was labelled under the OLD wording (see SESSIONS.md 2026-08-26).
AXIS_QUESTIONS = {
    "grounded":   "y = EVERY clause is supported by the passages it cites   |   "
                  "n = any clause unsupported, miscited, or reasoned past what "
                  "the passage establishes",
    "refusal_ok": "y = these passages genuinely do NOT support an answer   |   "
                  "n = OVER-REFUSAL: they DO support one and it refused anyway "
                  "(check the passages yourself; the rationale may be wrong)",
}


def axis_to_label(row: dict) -> str | None:
    """Which axis this row still needs a decision on, or None if it is done.

    The other axis is already `NOT_APPLICABLE` — `sheet_row` filled it from the
    `refused` flag — so there is exactly one question per row, and a row whose
    remaining axis is filled is a row to skip on resume.
    """
    pending = [a for a in blank_labels() if row.get(a) in (None, "")]
    if len(pending) > 1:
        raise AssertionError(
            f"{row['query_id']} has {len(pending)} unanswered axes {pending} — "
            "the sheet was written without the inapplicable axis pre-filled."
        )
    return pending[0] if pending else None


def render_item(row: dict, width: int = 88, reveal_stratum: bool = False) -> str:
    """One row as readable text. **`stratum` is withheld by default.**

    `gold_in_context` is anchoring bait while labelling `refusal_ok`: see a 0 and
    it is very tempting to write "the refusal was right" without reading the
    passages. That label would then be a function of the qrels rather than of the
    passages — and the whole point of hand labels is that a human read the thing.
    So the render shows it only AFTER the answer is in, as feedback.
    """
    import textwrap

    wrap = lambda s, indent="": textwrap.fill(
        s, width=width, initial_indent=indent, subsequent_indent=indent)

    parts = [f"Q: {row['question']}", ""]
    parts += [wrap(p, "  ") for p in row["passages"]]
    parts += ["", wrap(f"ANSWER   : {row['answer']}"),
              wrap(f"RATIONALE: {row['rationale']}")]
    if reveal_stratum:
        s = row["stratum"]
        parts += ["", f"[gold docs in context: {s['gold_in_context']}  "
                      f"refused: {s['refused']}]"]
    return "\n".join(parts)


def save_rows(path: Path, rows: list[dict]) -> None:
    """Rewrite the sheet atomically, after every single answer.

    Atomic because the alternative is a truncated JSONL if the terminal dies
    mid-write, and this file is the one thing here that cannot be regenerated.
    After every answer rather than at the end, so quitting — or a crash, or a
    closed laptop — costs at most the row in progress.
    """
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    tmp.replace(path)


def binary_axes() -> dict[str, tuple]:
    """The (yes_value, no_value) pair per axis, from `LABEL_VALUES`.

    The prompt maps y/n onto real label values rather than hard-coding True and
    False, so it stays correct if the rubric changes. A non-binary axis raises
    instead of guessing which of three categories `y` means.
    """
    pairs = {}
    for axis, allowed in LABEL_VALUES.items():
        values = set(allowed) - {NOT_APPLICABLE}
        if values != {True, False}:
            raise NotImplementedError(
                f"--label handles binary axes only; '{axis}' allows {sorted(map(str, values))}. "
                "Label it in the file, or extend the prompt to offer those values."
            )
        pairs[axis] = (True, False)
    return pairs


def label_interactive(path: Path, ask=input, say=print, width: int = 88) -> int:
    """Render → ask → write, one row at a time. Returns the number labelled.

    `ask`/`say` are injected so the loop is testable without a terminal — the
    same reason `cached_generations` takes `compute` as a thunk.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    pairs = binary_axes()
    todo = [i for i, r in enumerate(rows) if axis_to_label(r)]
    if not todo:
        say(f"{path.name}: all {len(rows)} rows already labelled — nothing to do.")
        return 0

    say(f"{path.name}: {len(todo)} of {len(rows)} rows still need a decision.\n"
        "  y = yes   n = no   s = skip   q = save and quit\n")

    done = 0
    for n, i in enumerate(todo, 1):
        row = rows[i]
        axis = axis_to_label(row)
        yes, no = pairs[axis]

        say("\n" + "─" * width)
        say(f"[{n}/{len(todo)}]  {row['query_id']}\n")
        say(render_item(row, width))
        say("")

        say(f"  {AXIS_QUESTIONS[axis]}")
        while True:
            reply = ask(f"  {axis}?  [y/n/s/q] ").strip().lower()
            if reply in ("y", "n", "s", "q"):
                break
            say("  (y, n, s or q)")

        if reply == "q":
            say(f"\nsaved. {done} labelled this session, "
                f"{len(todo) - done} still to go.")
            return done
        if reply == "s":
            continue

        rows[i][axis] = yes if reply == "y" else no
        save_rows(path, rows)          # after every answer, not at the end
        done += 1
        say(f"  → {axis}={rows[i][axis]}   "
            f"(gold in context: {row['stratum']['gold_in_context']})")

    say(f"\nfinished. {done} labelled. "
        f"Verify with: python -m retrieval_lab.fixture --check ...")
    return done


def main(dataset: str, n_queries: int, n_context: int, top_k: int, seed: int,
         check: bool, force: bool, label: bool = False, model: str = BI_ENCODER,
         generator: str = GENERATOR) -> None:
    corpus, queries, qrels = load_beir(dataset)
    gens = load_generations(dataset, model, top_k, n_context, generator, n_queries, seed)
    path = sheet_path(dataset, n_queries, seed)

    if label:
        load_labels(path, gens)     # anchor check BEFORE editing: a sheet whose
        label_interactive(path)     # batch was regenerated must not be labelled
        return

    if check:
        rows = load_labels(path, gens)
        validate_labels(rows, len(gens))
        log.info("labels OK: %d rows, all anchored to the batch on disk", len(rows))
        return

    report_strata(gens, qrels)

    shared = overlap_with([g.query_id for g in gens],
                          generation_cache_path(dataset, model, top_k, n_context,
                                                generator, PROMPT_VERSION,
                                                n_queries, seed))
    if shared:
        log.warning("\n⚠️  this draw overlaps earlier batches — held-out is not clean:")
        for name, common in shared.items():
            log.warning("   %-70s %d shared (%s)", name, len(common), ", ".join(common[:3]))
        log.warning("   Drop them from the fixture, or label them and say so.")
    else:
        log.info("\nheld-out: no query id here appears in any other cached batch")

    write_sheet(gens, corpus, qrels, path, force)
    log.info("\nworksheet -> %s (%d rows)", path, len(gens))
    log.info("Label it, then: python -m retrieval_lab.fixture "
             "--n-queries %d --seed %d --check", n_queries, seed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hotpotqa-distractor-pool")
    p.add_argument("--n-queries", type=int, default=30, help="which batch to sheet")
    p.add_argument("--n-context", type=int, default=10)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--seed", type=int, default=1, help="1 = the held-out fixture draw")
    p.add_argument("--check", action="store_true",
                   help="read the labels back and verify them against the batch")
    p.add_argument("--label", action="store_true",
                   help="interactive labelling: one row at a time, resumable")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing sheet (destroys hand labels)")
    p.add_argument("--model", default=BI_ENCODER, help=MODEL_HELP)
    p.add_argument("--generator", default=GENERATOR)
    args = p.parse_args()
    main(args.dataset, args.n_queries, args.n_context, args.top_k, args.seed,
         args.check, args.force, args.label, args.model, args.generator)
