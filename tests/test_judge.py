"""Fixture tests for the Phase 4c.2 bake-off. Run:  pytest tests/test_judge.py

No model is called and no fixture sheet is read — every input here is a literal,
so the suite runs in milliseconds and says nothing about whether a judge is any
good. What it pins is the four functions that decide what a κ MEANS, for the
same reason `test_rerank.py` pins `spearman_permutation`: these numbers go in
the README, and every failure mode below is well-formed output with the wrong
meaning, which is the shape this repo keeps getting bitten by.

Three of them were live bugs during implementation, caught by hand and pinned
here so they cannot come back silently:

  · `passages` interpolated as a list rather than joined — the judge reads a
    Python repr, quotes and commas included.
  · the FORMAT block containing a literal `<judge_answer>true</judge_answer>`,
    which a judge that restates the format gets parsed AS a verdict of true.
  · `rank_candidates` anchoring its tie band on every candidate rather than on
    the survivors, letting a disqualified model's inflated κ set the bar.

The rubric WORDING is deliberately not asserted — it is meant to be revised, and
a test that breaks on rewording would train you to edit the test. What is
asserted is structure: which axis's rules appear, that the rationale is present,
and that nothing verdict-shaped is echoable out of the prompt.
"""
import math

import pytest

from retrieval_lab.judge import (
    KAPPA_TIE_BAND,
    MAX_MISS_RATE,
    build_judge_prompt,
    cohen_kappa,
    parse_judgement,
    rank_candidates,
)

PASSAGES = ["[1] Edmund Mortimer. An English nobleman of the Welsh marches.",
            "[2] Roger Mortimer. A rival claimant to the same title."]
QUESTION = "Who was the father of Edmund Mortimer?"
RATIONALE = "Passage [1] names him but does not give a father."


# ──────────────────────────── build_judge_prompt ────────────────────────────

@pytest.mark.parametrize("axis", ["grounded", "refusal_ok"])
def test_passages_are_joined_not_repr(axis):
    """A list interpolated into an f-string renders as a Python repr.

    The judge then reads `['[1] Edmund…', '[2] Roger…']` — quotes, commas, and
    any newline in the passage body escaped as a literal \\n. Nothing raises and
    the prompt still looks plausible in a log, which is why this is pinned.
    """
    prompt = build_judge_prompt(QUESTION, PASSAGES, "Roger", RATIONALE, axis)
    assert "['[1]" not in prompt and "', '" not in prompt
    for p in PASSAGES:
        assert p in prompt


@pytest.mark.parametrize("axis", ["grounded", "refusal_ok"])
def test_rationale_reaches_the_prompt(axis):
    """Both axes rule on the RATIONALE, so omitting it makes the bake-off void.

    `grounded` asks whether every clause of the rationale is supported;
    `refusal_ok`'s narrow test is whether the rationale itself names the answer.
    On a refused row `answer` is only the sentinel, so a prompt without the
    rationale gives the judge nothing to rule on — and its κ then measures
    nothing, while looking like a normal number.
    """
    prompt = build_judge_prompt(QUESTION, PASSAGES, "Roger", RATIONALE, axis)
    assert RATIONALE in prompt


def test_each_axis_gets_only_its_own_rules():
    """One question per call, mirroring AXIS_FOR.

    A prompt carrying both rubrics invites a ruling on the axis the sheet marked
    n/a, which `align` would then pair against a label that does not exist.
    """
    grounded = build_judge_prompt(QUESTION, PASSAGES, "Roger", RATIONALE, "grounded")
    refusal = build_judge_prompt(QUESTION, PASSAGES, "I cannot answer",
                                 RATIONALE, "refusal_ok")
    assert "GROUNDED" in grounded and "OVER REFUSAL" not in grounded
    assert "OVER REFUSAL" in refusal and "GROUNDED in the" not in refusal


def test_refusal_rubric_keeps_the_narrowness_clause():
    """The conservative boundary, without which κ measures rubric disagreement.

    `fixture.py` scores a refusal `true` when the rationale does not name the
    answer EVEN IF the answer was derivable. Drop that from the prompt and the
    judge convicts on derivability — stricter than the labels it is graded
    against — and the gap reads as judge quality rather than as our own edit.
    """
    refusal = build_judge_prompt(QUESTION, PASSAGES, "I cannot answer",
                                 RATIONALE, "refusal_ok")
    assert "derivable" in refusal
    assert "ESTABLISH" in refusal and "SUGGEST" in refusal


def test_prompt_contains_no_echoable_verdict():
    """The FORMAT block must not contain a filled-in `<judge_answer>` tag.

    A judge that restates the format before ruling then emits a verdict-shaped
    string it never meant. Because `parse_judgement` reads the LAST tag this is
    survivable, but a prompt with a literal verdict in it is one refactor away
    from a systematic bias toward whichever value was written there.
    """
    for axis in ("grounded", "refusal_ok"):
        prompt = build_judge_prompt(QUESTION, PASSAGES, "Roger", RATIONALE, axis)
        assert "<judge_answer>true</judge_answer>" not in prompt
        assert "<judge_answer>false</judge_answer>" not in prompt


def test_judge_tags_do_not_collide_with_the_generator_s():
    """`<judge_answer>` is a separate namespace from the generator's `<answer>`.

    Generated text flows into this prompt. Sharing a tag name would let content
    from the thing being judged be read as the judgement of it.
    """
    prompt = build_judge_prompt(QUESTION, PASSAGES, "Roger", RATIONALE, "grounded")
    assert "<judge_answer>" in prompt and "<judge_rationale>" in prompt


# ───────────────────────────── parse_judgement ─────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("<judge_rationale>ok</judge_rationale><judge_answer>true</judge_answer>", True),
    ("<judge_answer>false</judge_answer>", False),
    ("<judge_answer>TRUE</judge_answer>", True),
    ("<judge_answer>\n  false\n</judge_answer>", False),
])
def test_verdicts_that_parse(raw, expected):
    assert parse_judgement(raw) is expected


def test_the_substring_trap():
    """The reason this is not `"true" in raw`.

    A rationale arguing the answer is NOT grounded contains the word "true".
    Searching for it scores a false verdict as agreement — invisible, and biased
    toward whichever value the rubric text happens to mention.
    """
    assert parse_judgement(
        "It is not true that every clause is supported.") is None


@pytest.mark.parametrize("raw", [
    "",
    "I think the answer is grounded.",
    "<judge_answer>yes</judge_answer>",
    "<judge_answer>grounded</judge_answer>",
    "<judge_answer>true.</judge_answer>",
    "<judge_answer>**true**</judge_answer>",
    "<judge_answer></judge_answer>",
])
def test_unrecognised_output_is_a_miss_never_a_guess(raw):
    """`None` is the third outcome, and it must stay distinguishable.

    A default of False would turn every format failure into a ruling, inflating
    n and quietly biasing κ. The miss rate is a REPORTED column and a
    disqualification gate; it only works if misses are counted, not absorbed.
    """
    assert parse_judgement(raw) is None


def test_generator_tags_alone_do_not_parse():
    """A reply in the generator's `<answer>` namespace is a format failure.

    This is the point of the separate tag: text carrying generator tags can
    reach the judge, and must never be readable as the judge's own verdict.
    """
    assert parse_judgement("<rationale>x</rationale><answer>true</answer>") is None


@pytest.mark.parametrize("raw,expected", [
    ("FORMAT: <judge_answer>...</judge_answer>\n"
     "<judge_rationale>Unsupported.</judge_rationale>"
     "<judge_answer>false</judge_answer>", False),
    ("<judge_answer>true</judge_answer> — on reflection, no: "
     "<judge_answer>false</judge_answer>", False),
])
def test_the_last_tag_wins(raw, expected):
    """The verdict follows the rationale, so the final tag is the settled one.

    First-match reads an echoed format block or an abandoned verdict. This
    deliberately diverges from `generate.parse_answer`, whose format block has
    no filled-in tag to echo.
    """
    assert parse_judgement(raw) is expected


def test_round_trips_the_format_the_prompt_asks_for():
    """The contract between the two functions, asserted in one place.

    `build_judge_prompt` and `parse_judgement` are a matched pair; editing the
    format in one and not the other produces a 100% miss rate that reads as a
    model failure.
    """
    prompt = build_judge_prompt(QUESTION, PASSAGES, "Roger", RATIONALE, "grounded")
    assert "<judge_rationale>...</judge_rationale>" in prompt
    reply = "<judge_rationale>Cited [1] supports it.</judge_rationale>\n" \
            "<judge_answer>true</judge_answer>"
    assert parse_judgement(reply) is True


# ─────────────────────────────── cohen_kappa ───────────────────────────────

HUMAN = [True] * 10 + [False] * 6          # the seed-1 `grounded` balance


def test_perfect_agreement_is_one():
    assert cohen_kappa(HUMAN, HUMAN) == pytest.approx(1.0)


def test_the_constant_rater_scores_zero():
    """The property the whole metric exists for.

    A judge answering `true` to all 16 rows gets 10/16 = 62.5% raw accuracy
    while knowing nothing. Publishing that as "agrees with humans 62% of the
    time" would be true and worthless; κ subtracts it to 0.
    """
    assert cohen_kappa(HUMAN, [True] * 16) == pytest.approx(0.0)


def test_reproduces_a_hand_worked_confusion_matrix():
    """TT=9, TF=1, FT=2, FF=4 over n=16.

    p_o = 13/16 = 0.8125; the judge says true 11/16, the human 10/16, so
    p_e = .625(.6875) + .375(.3125) = 0.5469 and κ = .2656/.4531 = 0.5862.
    Derived on paper independently of this code — the same discipline as
    `arithmetic_ceiling()` checking the oracle's sort.
    """
    judge = [True] * 9 + [False] * 1 + [True] * 2 + [False] * 4
    assert cohen_kappa(HUMAN, judge) == pytest.approx(0.5862, abs=1e-4)


def test_worse_than_chance_is_negative():
    """κ < 0 is legal and diagnostic.

    A negative κ from a real judge almost always means inverted polarity in
    `parse_judgement`, not a hostile model — so it must not be clamped to 0,
    which would hide the bug it points at.
    """
    assert cohen_kappa(HUMAN, [not h for h in HUMAN]) < 0


def test_marginals_not_a_coin_flip():
    """p_e is built from each rater's OWN rate, not assumed 50/50.

    Under a 50/50 assumption p_e is always 0.5 and the constant rater above
    would score 0.25 instead of 0 — the failure κ exists to prevent.
    """
    balanced = [True] * 8 + [False] * 8
    skewed_agreement = cohen_kappa(HUMAN, [True] * 10 + [False] * 6)
    balanced_agreement = cohen_kappa(balanced, balanced)
    assert skewed_agreement == pytest.approx(1.0)
    assert balanced_agreement == pytest.approx(1.0)
    # ... but identical p_o, different p_e, on a partial-agreement case:
    a = cohen_kappa(HUMAN, [True] * 11 + [False] * 5)
    b = cohen_kappa(balanced, [True] * 9 + [False] * 7)
    assert a != pytest.approx(b)


def test_degenerate_agreement_is_nan_not_one():
    """Both raters constant on the same value: p_e == 1, and κ is 0/0.

    Returning 1.0 would report "perfect agreement" for a comparison that
    distinguished nothing. Reachable on `refusal_ok`, where 12 of 14 rows are
    `true` and dropped misses can leave an all-true remainder.
    """
    assert math.isnan(cohen_kappa([True] * 12, [True] * 12))


def test_no_pairable_rows_is_nan_not_a_crash():
    """`align` drops every row when a candidate never emits the format.

    That is an outcome the bake-off REPORTS — the miss-rate column exists for
    it — so it must not take the run down with a ZeroDivisionError that names
    neither the model nor the axis.
    """
    assert math.isnan(cohen_kappa([], []))


def test_length_mismatch_raises():
    """Zip would silently score the shorter prefix and call it a κ."""
    with pytest.raises(AssertionError):
        cohen_kappa([True, False], [True])


# ────────────────────────────── rank_candidates ──────────────────────────────

def candidate(kappa, miss_rate, sec_per_call, n=16):
    return {"kappa": kappa, "kappa_refusal": 0.5, "n": n, "n_refusal": 14,
            "miss_rate": miss_rate, "sec_per_call": sec_per_call,
            "digest": "abc123"}


def test_within_the_band_throughput_decides():
    """The decision rule, fixed before the numbers existed.

    A 0.06 κ gap on 16 rows is noise. Letting it order the candidates strictly
    would pick a judge ~6x slower and set the sweep size for all of Phase 4 on
    a difference the fixture cannot resolve.
    """
    ranked = rank_candidates({
        "mistral-small": candidate(0.64, 0.00, 24.0),
        "gemma3:4b": candidate(0.58, 0.03, 4.1),
    })
    assert [j for j, _ in ranked] == ["gemma3:4b", "mistral-small"]
    assert "throughput" in ranked[0][1]


def test_outside_the_band_kappa_decides():
    """Speed does not rescue a candidate that is genuinely worse."""
    ranked = rank_candidates({
        "mistral-small": candidate(0.71, 0.00, 24.0),
        "gemma3:4b": candidate(0.34, 0.03, 4.1),
    })
    assert [j for j, _ in ranked] == ["mistral-small", "gemma3:4b"]


def test_anchor_reads_survivors_only():
    """The bug this pins: a gated-out model must not set the tie band.

    A disqualified candidate's κ is INFLATED — it ruled only on the rows it
    managed to format, and models fail formatting on the hard ones. Anchored on
    all candidates, the broken 0.90 below puts both real candidates outside the
    band, `tied` empties, and the tie rule silently stops existing.
    """
    ranked = rank_candidates({
        "broken-8b": candidate(0.90, 0.40, 2.0),
        "mistral-small": candidate(0.64, 0.00, 24.0),
        "gemma3:4b": candidate(0.58, 0.03, 4.1),
    })
    assert [j for j, _ in ranked[:2]] == ["gemma3:4b", "mistral-small"]
    assert "0.64" in ranked[0][1], "the anchor should be the best SURVIVOR"


def test_ties_are_anchored_not_pairwise():
    """"Within 0.2 of each other" is not transitive.

    0.70 ~ 0.55 and 0.55 ~ 0.40, but 0.70 is not ~ 0.40. A pairwise comparator
    makes `sorted` order-dependent without raising. Anchored on max(κ), `c` is
    excluded despite being the fastest — which is the point.
    """
    ranked = rank_candidates({
        "a": candidate(0.70, 0.0, 20.0),
        "b": candidate(0.55, 0.0, 8.0),
        "c": candidate(0.40, 0.0, 2.0),
    })
    assert [j for j, _ in ranked] == ["b", "a", "c"]


def test_disqualified_sort_last_regardless_of_kappa():
    """A broken model with a flattering κ must never head the table.

    The README table is read top-down; a gate that removed a candidate from
    contention but left it sorted by score would undo itself in presentation.
    """
    ranked = rank_candidates({
        "broken-8b": candidate(0.95, 0.50, 2.0),
        "gemma3:4b": candidate(0.30, 0.00, 4.1),
    })
    assert [j for j, _ in ranked] == ["gemma3:4b", "broken-8b"]
    assert ranked[-1][1].startswith("disqualified")


def test_nan_kappa_is_disqualified_not_ranked_low():
    """`score_candidate` emits nan when no row pairs.

    Every comparison with nan is False, so a nan κ silently never ties and can
    poison `max`. It is the same category as a format failure: meaningless, not
    low.
    """
    ranked = rank_candidates({
        "mistral-small": candidate(0.64, 0.00, 24.0),
        "gemma3:4b": candidate(float("nan"), 0.90, 4.1, n=0),
    })
    assert [j for j, _ in ranked] == ["mistral-small", "gemma3:4b"]
    assert "κ undefined" in ranked[-1][1]


def test_the_gate_boundary_is_exclusive():
    """3 misses of 30 is exactly MAX_MISS_RATE, and it PASSES.

    The reasoning recorded on the constant is "4+ misses of 30", built on the
    argument that a gate firing at small counts disqualifies usable judges on
    sampling noise. `>=` would silently move the gate one miss stricter than the
    comment it is justified by.
    """
    assert 3 / 30 == pytest.approx(MAX_MISS_RATE)
    ranked = rank_candidates({"gemma3:4b": candidate(0.58, 3 / 30, 4.1)})
    assert not ranked[0][1].startswith("disqualified")

    ranked = rank_candidates({"gemma3:4b": candidate(0.58, 4 / 30, 4.1)})
    assert ranked[0][1].startswith("disqualified")


def test_all_disqualified_returns_the_block():
    """No survivors is a reportable outcome, not an exception."""
    ranked = rank_candidates({
        "a": candidate(float("nan"), 1.0, 1.0, n=0),
        "b": candidate(0.9, 0.8, 2.0),
    })
    assert len(ranked) == 2
    assert all(r.startswith("disqualified") for _, r in ranked)


def test_every_candidate_carries_a_reason():
    """The table has to show WHY the winner won — "measured, not asserted"."""
    ranked = rank_candidates({
        "mistral-small": candidate(0.64, 0.00, 24.0),
        "gemma3:4b": candidate(0.58, 0.03, 4.1),
        "broken-8b": candidate(0.95, 0.50, 2.0),
    })
    assert len(ranked) == 3
    for judge, reason in ranked:
        assert isinstance(reason, str) and reason.strip()


def test_the_band_is_read_from_the_constant():
    """The rule the docstring commits to must be the rule the code applies."""
    assert KAPPA_TIE_BAND == 0.2
    just_inside = rank_candidates({
        "fast": candidate(0.70 - KAPPA_TIE_BAND, 0.0, 2.0),
        "slow": candidate(0.70, 0.0, 20.0),
    })
    assert just_inside[0][0] == "fast"
    just_outside = rank_candidates({
        "fast": candidate(0.70 - KAPPA_TIE_BAND - 0.01, 0.0, 2.0),
        "slow": candidate(0.70, 0.0, 20.0),
    })
    assert just_outside[0][0] == "slow"
