"""Phase 4c.2: bake off judge candidates against the hand-labelled fixture.

    python -m retrieval_lab.judge --bakeoff                    # every candidate
    python -m retrieval_lab.judge --bakeoff --judge gemma3:4b  # just one

This module scores JUDGES, not generations. It reads a labelled fixture sheet,
asks each candidate model to rule on the same rows Betty ruled on, and reports
how well each agrees with her — plus how fast it is and how often it fails to
emit parseable output. Scoring the real Phase 4 batches comes later and reuses
`judge_batch`; that run is only trustworthy because this one happened first.

**This is the fixture-first rule, one level up.** `pytest` proves the retrieval
pipeline on five docs with a known `NDCG@10 = 1.0` before anyone trusts a number
from `evaluate.py`. Same shape here: a judge that cannot reproduce 30 known
labels cannot be trusted on 500 unknown ones, and this finds that out in minutes
rather than after the spend.

**Why the candidates differ by SIZE rather than by vendor** (issue #2 / D13).
The recorded pick was "27B, because bigger is safer" — never measured. The
principle worth testing: judging here is *classification against a rubric*, and
everything it rules on is already in the prompt, so parameter count may buy very
little. A bake-off of four similarly-sized models would establish which vendor we
like; one spanning 4B → 24B establishes whether size matters. `mistral-small`
(~24B at 4-bit) is the ceiling probe — if it cannot separate grounded from
ungrounded, no smaller model will — and `gemma3:4b` is the floor. A tie closes
D13 with a finding *and* a judge roughly 6x faster, which is what should set `n`
for the full config sweep.

**No qwen3 candidate, ever.** `qwen3:8b` is the generator (D10); a model grading
its own output flatters itself, and dodging that self-bias is the whole reason
the judge is cross-model.

**Reading the numbers, decided before they exist.** The `grounded` axis has 16
labelled rows. At that size a κ difference under ~0.2 is noise, so the decision
rule is "big gap or no gap, then throughput" — not "0.71 beats 0.68". And
`refusal_ok` (14 rows, 12/2) is reported as DESCRIPTIVE ONLY: two minority cases
cannot certify anything, which is the measured reason seed 2 was not labelled
(2026-08-26). Certification reads the `grounded` axis.

**Scope: Ollama transport only.** D10 names Sonnet 5 as the primary judge, and
certifying it against this same fixture needs an API path rather than
`call_ollama`. That is deliberate future work, not an oversight — the fixture,
the rubric and the scoring below are all transport-agnostic; only `judge_one`
would gain a branch.

────────────────────────────────────────────────────────────────────────────
The four functions that decide what this module MEANS are Betty's:
    build_judge_prompt()  the rubric, rendered for a model — must MIRROR the
                          boundaries recorded above LABEL_VALUES in fixture.py
    parse_judgement()     the seam between a chat model and a κ
    cohen_kappa()         the agreement statistic the README will publish
    rank_candidates()     what "separates grounded from ungrounded" means as a
                          number, which is the actual bake-off decision
Everything else — transport, timing, caching, digests, validation, the table —
is wired below.
────────────────────────────────────────────────────────────────────────────
"""
import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

from .cache import CACHE_DIR, generation_cache_path, judgement_cache_path
from .data import load_beir
from .fixture import (NOT_APPLICABLE, load_generations, load_labels,
                      sheet_path)
from .generate import GENERATOR, PROMPT_VERSION, Generation, call_ollama
from .observability import op
from .retrieve import BI_ENCODER, MODEL_HELP

log = logging.getLogger("retrieval_lab")

OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# The bake-off slate — deliberately size-spanning, deliberately non-qwen3.
# `mistral-small` is the ~24B ceiling probe; `gemma3:4b` is the floor. Add the
# 8B/14B middle only if these two show a gap worth locating.
CANDIDATES = ["mistral-small", "gemma3:4b"]

# Bump on EVERY edit to build_judge_prompt, exactly as PROMPT_VERSION works for
# the generator — it is the only thing standing between a reworded rubric and a
# cache serving judgements made under the previous wording. It is in
# `judgement_cache_path`, so a bump invalidates judgements without touching the
# generations they scored.
RUBRIC_VERSION = "v1"

# Which axis applies to which row, mirroring fixture.INAPPLICABLE_AXIS. A refusal
# makes no claims to ground; an answer cannot be an over-refusal. The judge is
# asked only the applicable question, so its output lines up with the sheet
# without either side having to guess.
AXIS_FOR = {True: "refusal_ok", False: "grounded"}


@dataclass
class Judgement:
    """One candidate's ruling on one row — the unit κ is computed over."""
    query_id: str
    judge: str
    digest: str             # WHICH weights ruled; see `ollama_digest`
    axis: str               # "grounded" or "refusal_ok"
    raw: str                # exactly what the judge emitted, never cleaned
    verdict: object = None  # True / False, or None when parsing missed
    parsed: bool = True     # False = a format failure, not a judgement
    seconds: float = 0.0    # wall-clock for THIS call; see `judge_batch`


# ─────────────────────────── Betty's stubs ───────────────────────────

def build_judge_prompt(question: str, passages: list[str], answer: str,
                       rationale: str, axis: str) -> str:
    """Render one row into a prompt asking the judge for a binary ruling.

    CONCEPT — this function is where a bake-off can quietly become meaningless,
    and it fails in a way no test catches: **if the rubric you give the judge is
    not the rubric you labelled under, κ measures rubric disagreement rather than
    judge quality**, and a low number reads as "these models are bad at judging"
    when it actually means "we asked them a different question". Same class as
    `NDCG@10 = 0.0000` meaning leaked `corpus_id` — well-formed output, wrong
    meaning, nothing raises.

    So the rubric of record is the comment block above `LABEL_VALUES` in
    `fixture.py`, and this prompt has to mirror it. Concretely, that block names
    three specific ways `grounded` is false — uncited assertion, inference past
    the passage, unsupported negative — and one asymmetry in `refusal_ok` (it
    rules on what the PASSAGES support, not on whether the model's reasoning
    looked careful). A judge not told about the uncited-assertion rule will score
    the Mark-8 row `grounded=true` and be marked wrong for following a rubric it
    was never given.

    Four design decisions live here, and they are yours:

    1. **How much of the rubric to include.** Naming the three failure modes
       makes the judge stricter and closer to your labels; it also risks teaching
       it to hunt for those three and miss a fourth. The alternative — state the
       principle only ("verifiable from the citations given") — tests whether the
       judge derives the boundary itself. Both are defensible; they measure
       different things, and the README has to say which one produced the κ.

    2. **Whether to show few-shot examples.** Your seed-1 rows contain textbook
       cases of each failure mode. Using them as examples would likely raise κ —
       and would be a leak: the fixture is held-out precisely so the judge is not
       selected on the items it is graded on. If you want few-shot, the examples
       must come from a batch that is NOT this fixture (`seed 2` is sitting there
       unlabelled and unused).

    3. **One axis per call, or both.** `axis` is passed in so you can ask exactly
       one question per row, which is what `AXIS_FOR` sets up. Asking both wastes
       tokens on the inapplicable one and invites the judge to rule on an axis
       the sheet marked `n/a` — but one call for both axes is half the latency,
       and latency is half the definition of done here.

    4. **The output format**, which `parse_judgement` must match exactly. The
       generator's `<rationale>…</rationale><answer>…</answer>` contract works and
       is already proven against these models; reusing its shape is a reasonable
       default. Asking for a rationale BEFORE the verdict is not cosmetic — it
       gives the model somewhere to do the work, and it gives you something to
       read when a judgement looks wrong.

    `passages` arrives pre-rendered as "[1] Title. Text" strings, in the same
    numbering the generator saw, so citations in `rationale` refer to the same
    passages the judge is looking at.

    Bump RUBRIC_VERSION whenever you touch this.
    """
    raise NotImplementedError("Betty's — see the CONCEPT block above")


def parse_judgement(raw: str) -> object:
    """Pull the binary verdict out of raw judge output. None when it is not there.

    CONCEPT — the same seam as `parse_answer`, with one extra trap. There, a
    parse miss produced an empty answer and a token-F1 of 0: visibly broken. Here
    a sloppy parser is worse, because the natural fallback is *guessing*. Search
    the text for "true" and you will match "it is not true that every clause is
    supported" and score it as agreement. A judge that emitted nothing usable
    must be COUNTED, never guessed at — `validate` raises on a batch with too
    many misses, and the bake-off table reports the miss rate as its own column
    precisely because small models fail structured output more often, and a
    candidate that cannot emit the format is disqualified regardless of its κ.

    Return `True`, `False`, or `None` for "no parseable verdict". Do not return a
    default; `None` is what keeps a format failure distinguishable from a ruling.

    Note the axis vocabulary is binary on both axes (`LABEL_VALUES`), so this
    returns a bool rather than a string — but the model will not emit a bool. The
    mapping from whatever word you asked for to True/False is this function's
    real work, and it must be strict: an unrecognised word is `None`, not False.
    """
    raise NotImplementedError("Betty's — see the CONCEPT block above")


def cohen_kappa(rater_a: list, rater_b: list) -> float:
    """Cohen's κ between two raters over the same items, in order.

    CONCEPT — κ exists because **raw accuracy lies under class imbalance**, and
    this fixture is imbalanced on both axes. A judge that answers "grounded" to
    everything scores 10/16 = 62% raw accuracy on your `grounded` split while
    knowing nothing at all; on `refusal_ok` (12/2) a constant "ok" scores 86%.
    Publishing either as "our judge agrees with humans 86% of the time" would be
    true and worthless. κ subtracts the agreement two raters would reach by
    chance given their own marginal rates, so the do-nothing judge scores ≈ 0.

        κ = (p_o − p_e) / (1 − p_e)

    `p_o` is the observed agreement rate — the easy half. `p_e` is the expected
    agreement *by chance*, computed from each rater's own marginals: for each
    category, multiply the two raters' rates of using it, and sum. That marginal
    detail is the whole point — κ is defined relative to how often each rater
    actually says "true", not relative to 50/50.

    Two properties worth knowing before you read a number out of this:
      - κ = 1 is perfect, 0 is chance, and NEGATIVE is possible (worse than
        chance). A negative κ from a judge usually means an inverted polarity in
        `parse_judgement`, not a hostile model — check the parser first.
      - κ is unstable on small n, which is why the decision rule for this
        bake-off (module docstring) is "big gap or no gap", and why `refusal_ok`
        with two minority rows is reported as descriptive only.

    Both lists must already exclude rows either rater left `n/a` or unparsed —
    dropping them is the D14 convention (an inapplicable row leaves the
    DENOMINATOR rather than scoring as agreement). `align()` below does that
    pairing for you and is deliberately not part of this stub.

    Rounding out the definition is worth doing by hand once: build the 2x2
    confusion matrix on paper for a case you can check, then make this reproduce
    it. `tests/test_judge.py` should pin at least perfect agreement (κ = 1),
    total disagreement, and the constant-rater case (κ = 0) — that last one is
    the property the whole metric exists for.
    """
    raise NotImplementedError("Betty's — see the CONCEPT block above")


def rank_candidates(scored: dict) -> list[tuple]:
    """Order the candidates. THIS is the bake-off's decision, in code.

    CONCEPT — issue #2's definition of done names two things, "separation *and*
    throughput", and deliberately does not say how to trade them off. That
    trade-off is the decision, and writing it as a function rather than eyeballing
    a table is what makes it reviewable later — the same reason `rerank.py
    --breakdown` prints every figure the README quotes instead of a notebook
    producing them once.

    `scored` arrives as {judge: {"kappa": float, "kappa_refusal": float,
    "n": int, "miss_rate": float, "sec_per_call": float, "digest": str}}.

    What has to be decided:

    1. **Disqualification before ranking.** A candidate whose `miss_rate` is high
       is not a worse judge, it is a broken one — it never emitted the format, so
       its κ is computed over whichever subset it happened to manage, which is
       not a random subset (models fail formatting on the hard rows). Pick a
       threshold and apply it as a gate, not as a penalty term.

    2. **When κ counts as a tie.** The module docstring commits to "a difference
       under ~0.2 on 16 rows is noise" BEFORE the numbers exist, which is the
       honest order to decide it in. Encode that: candidates within the band are
       tied, and ties break on throughput. If you instead let κ order them
       strictly, a 0.03 gap picks a judge that is 6x slower, and the sweep size
       for all of Phase 4 gets set by noise.

    3. **What `kappa_refusal` is allowed to do.** Per the 2026-08-26 decision it
       is descriptive: two minority rows cannot certify. If it enters the ranking
       at all, say so in the README row — but the argument for keeping it out is
       that a judge should not win on an axis we agreed we cannot measure.

    Return `[(judge, reason), ...]` best-first, where `reason` is the one-line
    justification the table prints. Making each candidate carry its own reason
    means the table shows WHY the winner won, which is what a reader of the
    README needs and what "measured, not asserted" means here.
    """
    raise NotImplementedError("Betty's — see the CONCEPT block above")


# ─────────────────────── plumbing (wired, working) ───────────────────────

def ollama_digest(model: str) -> str:
    """The digest of the weights currently behind a model tag. Short, or "?".

    **A model tag is not a pin.** `mistral-small:latest` today and six months
    from now can be different weights with the same name, and nothing would
    raise — the same silent-staleness shape as a `retrieve.py` edit under a warm
    cache. This repo commits `uv.lock` because its claims are version-dependent;
    a judge model is that same kind of dependency, and the bake-off's losers get
    deleted from disk once it is over.

    Recording the digest in every judgement row is what keeps a deleted model's
    number re-derivable: `ollama pull` rebuilds the weights, and a digest that
    does not match says so out loud instead of quietly re-answering the question.
    """
    try:
        with urllib.request.urlopen(OLLAMA_TAGS_URL, timeout=10) as r:
            tags = json.loads(r.read())["models"]
    except (urllib.error.URLError, KeyError, TimeoutError):
        return "?"
    wanted = model if ":" in model else f"{model}:latest"
    for t in tags:
        if t.get("name") == wanted:
            return (t.get("digest") or "?")[:12]
    return "?"


def require_models(models: list[str]) -> None:
    """Fail before judging, not 200 calls in, if a candidate is not pulled."""
    missing = [m for m in models if ollama_digest(m) == "?"]
    if missing:
        raise RuntimeError(
            f"not pulled (or Ollama is down): {', '.join(missing)}. "
            f"Run `ollama pull {missing[0]}`, and check `ollama serve` is up."
        )


@op
def judge_one(judge: str, digest: str, item: dict, axis: str,
              timeout: int = 600) -> Judgement:
    """One judgement, timed. Traced by Weave when it is live.

    The clock wraps the transport call only — prompt building and parsing are
    ours and are the same for every candidate, so including them would blur the
    throughput comparison that issue #2 says should choose the sweep size.
    """
    prompt = build_judge_prompt(item["question"], item["passages"],
                                item["answer"], item["rationale"], axis)
    t0 = time.perf_counter()
    raw = call_ollama(prompt, model=judge, timeout=timeout)
    seconds = time.perf_counter() - t0

    verdict = parse_judgement(raw)
    return Judgement(query_id=item["query_id"], judge=judge, digest=digest,
                     axis=axis, raw=raw, verdict=verdict,
                     parsed=verdict is not None, seconds=seconds)


def validate(js: list[Judgement], judge: str) -> None:
    """Raise on a batch that must not be cached. Mirrors `generate.validate`.

    Called inside `compute()`, before the cache write, for the reason spelled out
    there: checking afterwards means the operator reads the error, fixes the
    parser, and the next run is a silent cache HIT serving the poisoned batch.
    Judging is nondeterministic like generation, so no rerun-and-diff catches it.

    Note this raises on TRANSPORT failure but only warns on a high miss rate —
    a candidate that cannot emit the format is a *finding* about that candidate,
    and the bake-off has to be able to report it rather than crash on it.
    `rank_candidates` is where a miss rate turns into a disqualification.
    """
    empty = [j.query_id for j in js if not j.raw.strip()]
    if empty:
        raise AssertionError(
            f"{judge}: {len(empty)} judgements are empty (e.g. {empty[:3]}) — "
            "that is a transport failure, not a model with no opinion. "
            "Nothing cached."
        )

    missed = [j.query_id for j in js if not j.parsed]
    if missed:
        log.warning("  %s: %d/%d judgements did not parse — reported as "
                    "miss_rate, not silently dropped", judge, len(missed), len(js))


def cached_judgements(path, compute, refresh: bool = False) -> list[Judgement]:
    """Same thunk contract as `cached_retrieval` / `cached_generations`.

    **The per-call `seconds` is persisted rather than measured at the call site**
    on purpose: a cache HIT would otherwise report a throughput of "instant" and
    silently rank a slow judge first. Timing belongs to the judgement, not to the
    run that reads it.
    """
    if path.exists() and not refresh:
        rows = json.loads(path.read_text())
        log.info("judgement cache HIT  %s (%d rulings)", path, len(rows))
        return [Judgement(**r) for r in rows]

    log.info("judgement cache MISS %s — judging ...", path)
    js = compute()

    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps([asdict(j) for j in js], indent=1))
    log.info("judgements cached -> %s", path)
    return js


def judge_items(gens: list[Generation], corpus: dict) -> list[dict]:
    """What the judge is shown, built from the CORPUS — never from the sheet.

    The worksheet already carries rendered `passages`, and reusing them would be
    one line shorter. It would also make a human-edited file the judge's source
    of truth: the sheet is hand-touched sixty times in an evening, and nothing
    checks its passage strings against the corpus (`load_labels` anchors `raw_sha`,
    which covers the ANSWER text only). Rendering from `doc_ids` here means the
    judge reads exactly what the generator read, by construction.

    The judge is also never shown the human label or the stratum — it is being
    graded against them.
    """
    return [{
        "query_id": g.query_id,
        "question": g.question,
        "passages": [f"[{i}] {corpus[d]['title']}. {corpus[d]['text']}"
                     for i, d in enumerate(g.doc_ids, 1)],
        "answer": g.answer,
        "rationale": g.rationale,
        "axis": AXIS_FOR[g.refused],
    } for g in gens]


def judge_batch(judge: str, items: list[dict], generation_key: str,
                refresh: bool = False) -> list[Judgement]:
    """Every row, judged by one candidate, cached under (generations, judge, rubric).

    `generation_key` is the generation cache file's stem, so a judgement file
    names the exact batch it scored — the split `judgement_cache_path` documents:
    swapping a judge invalidates judgements only, never the expensive generations.
    """
    digest = ollama_digest(judge)
    path = judgement_cache_path(generation_key, judge, RUBRIC_VERSION)

    def compute():
        js = []
        for i, item in enumerate(items, 1):
            js.append(judge_one(judge, digest, item, item["axis"]))
            if i % 10 == 0 or i == len(items):
                log.info("  %s: judged %d/%d", judge, i, len(items))
        validate(js, judge)     # before the cache write, never after
        return js

    js = cached_judgements(path, compute, refresh)

    # Post-hoc, like generate's: catches a HIT written when the SHEET was
    # different (rows dropped, a relabelled draw), which validate cannot see
    # because that batch was valid when it was written.
    if len(js) != len(items):
        raise AssertionError(
            f"{path.name} holds {len(js)} rulings but the batch has {len(items)} "
            "rows — it was written against a different draw. Pass --refresh."
        )
    return js


def align(rows: list[dict], js: list[Judgement], axis: str) -> tuple[list, list]:
    """Pair human labels with judgements on one axis, dropping what cannot pair.

    Three kinds of row leave the DENOMINATOR here rather than scoring as
    agreement — the D14 convention the fixture already applies to `n/a`:
      - the axis does not apply to this row (`n/a` on the sheet)
      - the judge was asked a different axis (it is asked only the applicable one)
      - the judge's output did not parse (a format failure is not a ruling)

    Returns (human, judge) as equal-length lists in matching order, ready for
    `cohen_kappa`. Kept out of the stub deliberately: which rows are eligible is
    a convention already settled, while what to DO with them is the statistic.
    """
    by_id = {j.query_id: j for j in js if j.axis == axis and j.parsed}
    human, judged = [], []
    for row in rows:
        j = by_id.get(row["query_id"])
        if j is None or row.get(axis) == NOT_APPLICABLE:
            continue
        human.append(row[axis])
        judged.append(j.verdict)
    return human, judged


def score_candidate(rows: list[dict], js: list[Judgement], judge: str) -> dict:
    """Everything the bake-off table needs about one candidate, as plain numbers."""
    h_g, j_g = align(rows, js, "grounded")
    h_r, j_r = align(rows, js, "refusal_ok")
    applicable = [j for j in js if any(r["query_id"] == j.query_id for r in rows)]

    return {
        "kappa": cohen_kappa(h_g, j_g) if h_g else float("nan"),
        "kappa_refusal": cohen_kappa(h_r, j_r) if h_r else float("nan"),
        "n": len(h_g),
        "n_refusal": len(h_r),
        "miss_rate": sum(not j.parsed for j in applicable) / max(len(applicable), 1),
        "sec_per_call": sum(j.seconds for j in applicable) / max(len(applicable), 1),
        "digest": next((j.digest for j in js), "?"),
    }


def report(scored: dict) -> None:
    """The bake-off table. Every column is a thing issue #2 asks for."""
    log.info("\n=== Phase 4c.2 judge bake-off (rubric %s) ===", RUBRIC_VERSION)
    log.info("%-18s %8s %6s %8s %10s  %-12s", "judge", "kappa", "n",
             "miss%", "sec/call", "digest")
    for judge, s in scored.items():
        log.info("%-18s %8.3f %6d %7.1f%% %9.2fs  %-12s", judge, s["kappa"],
                 s["n"], 100 * s["miss_rate"], s["sec_per_call"], s["digest"])

    log.info("\ndescriptive only — 2 minority rows cannot certify (2026-08-26):")
    for judge, s in scored.items():
        log.info("  %-18s refusal_ok kappa %6.3f over n=%d",
                 judge, s["kappa_refusal"], s["n_refusal"])

    log.info("\nκ on n=16 is noisy: a gap under ~0.2 is a tie, broken on "
             "throughput. Decision rule fixed before the numbers existed.")
    for i, (judge, reason) in enumerate(rank_candidates(scored), 1):
        log.info("  %d. %-18s %s", i, judge, reason)


def main(dataset: str, n_queries: int, n_context: int, top_k: int, seed: int,
         judges: list[str], refresh: bool, model: str = BI_ENCODER,
         generator: str = GENERATOR) -> None:
    corpus, _queries, _qrels = load_beir(dataset)
    gens = load_generations(dataset, model, top_k, n_context, generator,
                            n_queries, seed)

    # load_labels re-checks every raw_sha against the batch, so a fixture whose
    # generations were regenerated under it cannot reach a κ.
    path = sheet_path(dataset, n_queries, seed)
    rows = load_labels(path, gens)
    log.info("fixture: %s (%d rows)", path, len(rows))

    require_models(judges)
    generation_key = generation_cache_path(dataset, model, top_k, n_context,
                                           generator, PROMPT_VERSION, n_queries,
                                           seed).stem

    items = judge_items(gens, corpus)
    scored = {}
    for judge in judges:
        js = judge_batch(judge, items, generation_key, refresh)
        scored[judge] = score_candidate(rows, js, judge)

    report(scored)
    log.info("\nNext: record the winner and its digest in issue #2, delete the "
             "losers (`ollama rm`), then score the seed 0 batches.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hotpotqa-distractor-pool")
    p.add_argument("--n-queries", type=int, default=30,
                   help="which fixture draw to read (part of its cache key)")
    p.add_argument("--seed", type=int, default=1,
                   help="the HELD-OUT draw; seed 0 is what the winner will score")
    p.add_argument("--n-context", type=int, default=10)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--judge", action="append", dest="judges",
                   help="candidate model (repeatable); defaults to CANDIDATES")
    p.add_argument("--bakeoff", action="store_true",
                   help="score every candidate and print the comparison table")
    p.add_argument("--refresh", action="store_true", help="re-judge, ignoring the cache")
    p.add_argument("--model", default=BI_ENCODER, help=MODEL_HELP)
    p.add_argument("--generator", default=GENERATOR)
    args = p.parse_args()
    main(args.dataset, args.n_queries, args.n_context, args.top_k, args.seed,
         args.judges or CANDIDATES, args.refresh, args.model, args.generator)
