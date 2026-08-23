"""Fixture tests for Phase 4a's plumbing.

    pytest tests/test_generate.py

Pins the wiring, plus the contracts of `parse_answer` and `should_refuse`.
`build_prompt` is not called from here — the prompt is exercised by probes
against a live model, not by fixtures.
No model, no Ollama, no download.

The three load-bearing ones are `test_context_comes_from_results_not_qrels` (the
same constraint `test_oracle.py` pins one level down), the cache-key tests (a
stale generation cache cannot be caught by re-running, because generation is
nondeterministic even at temperature 0), and
`test_a_partial_parse_still_returns_raw` — see the parsing section below for why
that one is not a formality.
"""
import json

import pytest

from retrieval_lab.cache import generation_cache_path, judgement_cache_path
from retrieval_lab.generate import (
    Generation,
    cached_generations,
    SENTINEL,
    context_for,
    parse_answer,
    sample_queries,
    should_refuse,
    validate,
)

# q3 is retrieved but UNLABELLED — eligible for nothing.
RESULTS = {
    "q1": {"dA": 0.9, "dB": 0.5, "dC": 0.7},
    "q2": {"dB": 0.8, "dD": 0.2},
    "q3": {"dA": 0.4},
}
QRELS = {"q1": {"dC": 1, "dZ": 1}, "q2": {"dD": 1}, "q3": {}}
CORPUS = {d: {"title": d, "text": f"text of {d}"} for d in ["dA", "dB", "dC", "dD", "dZ"]}

KEY = dict(dataset="hotpotqa-distractor-pool", retriever="sentence-transformers/all-MiniLM-L6-v2",
           top_k=100, n_context=10, generator="qwen3:8b", prompt_version="v0",
           n_queries=50, seed=0)


def test_sampling_is_deterministic():
    """A rerun must score the SAME questions, or every number moves silently."""
    a = sample_queries(RESULTS, QRELS, n=1, seed=0)
    b = sample_queries(RESULTS, QRELS, n=1, seed=0)
    assert a == b


def test_sampling_excludes_unlabelled_queries():
    """q3 has no qrels: no oracle context and no correctness score can exist."""
    assert "q3" not in sample_queries(RESULTS, QRELS, n=99)


def test_sampling_returns_everything_when_n_exceeds_the_pool():
    assert sample_queries(RESULTS, QRELS, n=99) == ["q1", "q2"]


def test_context_is_ordered_best_first_and_truncated():
    doc_ids, docs = context_for(RESULTS, CORPUS, "q1", n_context=2)
    assert doc_ids == ["dA", "dC"]                      # 0.9, 0.7 — not dB at 0.5
    assert [d["title"] for d in docs] == ["dA", "dC"]


def test_context_comes_from_results_not_qrels():
    """The load-bearing one, inherited from `oracle.py`'s discipline.

    dZ is relevant and was never retrieved. If it ever appears in a prompt, this
    module is generating from the labels — which is milestone 4b's CEILING run,
    not a real config, and every comparison against that ceiling collapses.
    """
    doc_ids, _ = context_for(RESULTS, CORPUS, "q1", n_context=10)
    assert "dZ" not in doc_ids
    assert set(doc_ids) == set(RESULTS["q1"])


@pytest.mark.parametrize("field,value", [
    ("prompt_version", "v1"),
    ("generator", "qwen3:14b"),
    ("n_context", 5),
    ("top_k", 50),
    ("dataset", "fiqa"),
    ("retriever", "some/finetuned-encoder"),
    ("n_queries", 15),
    ("seed", 1),
])
def test_every_key_component_changes_the_cache_path(field, value):
    """Anything that changes what the model was asked — or WHICH questions were
    asked — must change the filename.

    Generation is nondeterministic even at temperature 0, so a stale hit cannot
    be caught by re-running and diffing the way a stale retrieval cache can.
    `n_queries` and `seed` are here because a batch of 15 is not a prefix of a
    batch of 100; without them one config owns one filename, and drawing a
    held-out fixture batch overwrites the batch it was meant to validate against.
    """
    assert generation_cache_path(**KEY) != generation_cache_path(**{**KEY, field: value})


def test_the_same_config_reuses_one_path():
    assert generation_cache_path(**KEY) == generation_cache_path(**KEY)


def test_swapping_a_judge_does_not_invalidate_generations():
    """D10's intent, with the blast radius it actually wanted.

    Both judges are tentative until they clear the 4c.1 fixture, so a swap is
    expected. It must invalidate judgements without discarding the expensive
    half.
    """
    gen = generation_cache_path(**KEY)
    a = judgement_cache_path(gen.stem, "claude-sonnet-5", "r1")
    b = judgement_cache_path(gen.stem, "Qwen3.8-27B", "r1")
    c = judgement_cache_path(gen.stem, "claude-sonnet-5", "r2")
    assert len({a, b, c}) == 3
    assert generation_cache_path(**KEY) == gen      # untouched by any of it


def test_cache_hit_round_trips_and_skips_compute(tmp_path):
    """A hit must not need Ollama at all — that is what makes re-judging cheap."""
    path = tmp_path / "gen.json"
    row = Generation(query_id="q1", question="why?", doc_ids=["dA"], raw="raw text",
                     answer="because", rationale="dA says so", refused=False)
    path.write_text(json.dumps([row.__dict__]))

    def boom():
        raise AssertionError("compute() ran on a cache hit")

    assert cached_generations(path, boom) == [row]


def test_cache_miss_writes_what_it_computed(tmp_path):
    path = tmp_path / "gen.json"
    row = Generation(query_id="q1", question="why?", doc_ids=["dA"], raw="r",
                     answer="a", rationale="s", refused=True)
    assert cached_generations(path, lambda: [row]) == [row]
    assert cached_generations(path, lambda: []) == [row]   # now a hit


def _gen(qid, raw="raw", answer="a", refused=False):
    return Generation(query_id=qid, question="q?", doc_ids=["dA"], raw=raw,
                      answer=answer, rationale="s", refused=refused)


def test_a_poisoned_batch_is_never_written_to_disk(tmp_path):
    """The bug this guard exists for.

    If validation ran AFTER the cache write, the run would raise, the operator
    would fix the parser — and the next run would be a silent cache HIT serving
    the poisoned batch. Generation is nondeterministic, so no rerun-and-diff
    catches that. The file must simply not exist.
    """
    path = tmp_path / "gen.json"
    unparsed = [_gen(f"q{i}", answer="") for i in range(5)]
    with pytest.raises(AssertionError, match="Nothing cached"):
        cached_generations(path, lambda: validate(unparsed) or unparsed)
    assert not path.exists()


def test_empty_generations_raise_before_caching(tmp_path):
    path = tmp_path / "gen.json"
    rows = [_gen("q1", raw="   ")]
    with pytest.raises(AssertionError, match="transport or prompt failure"):
        cached_generations(path, lambda: validate(rows) or rows)
    assert not path.exists()


def test_refusals_are_not_counted_as_parse_misses():
    """A refusal legitimately has no short answer — it is a measurement, not a miss."""
    assert validate([_gen(f"q{i}", answer="", refused=True) for i in range(5)]) == []


def test_a_few_parse_misses_are_tolerated():
    rows = [_gen("q0", answer="")] + [_gen(f"q{i}") for i in range(1, 10)]
    assert validate(rows) == ["q0"]


# ───────────────────────── parse_answer's contract ─────────────────────────
#
# These pin the seam between a chat model's output and a scoring metric. The
# shape they enforce comes from measurement, not taste: `build_prompt` asks for
# `<rationale>…</rationale>` then `<answer>…</answer>`, with the refusal
# sentinel INSUFFICIENT_CONTEXT *inside* the answer tags — a natural-language
# sentinel because `__REFUSED__` read to the model as a directive about the
# whole reply and went tagless 3/8 (LEARNINGS, 2026-08-19).

WELL_FORMED = "<rationale>Nelson County is where it sits [1]</rationale>\n<answer>Nelson County</answer>"


def test_a_well_formed_reply_splits_into_answer_and_rationale():
    assert parse_answer(WELL_FORMED) == ("Nelson County", "Nelson County is where it sits [1]")


def test_the_sentinel_is_returned_verbatim_rather_than_interpreted():
    """The parser EXTRACTS; `should_refuse` recognises; the pipeline normalises.

    Collapsing those would merge "the model declined" with "my rule classified
    this as a decline" into one string, which is the model-side/pipeline-side
    distinction `should_refuse`'s docstring calls not interchangeable — and the
    reason `REFUSAL` is a different word from the one the prompt asks for.
    """
    raw = "<rationale>the passages do not say [1][2]</rationale><answer>INSUFFICIENT_CONTEXT</answer>"
    answer, _ = parse_answer(raw)
    assert answer == "INSUFFICIENT_CONTEXT"


def test_a_missing_answer_tag_returns_raw_in_the_rationale_slot():
    """`("", raw)` is the miss contract, and `raw` is not decoration.

    It is the only channel by which `should_refuse` can see text the parser did
    not recognise — such as a sentinel the model emitted without its tags.
    """
    assert parse_answer("INSUFFICIENT_CONTEXT") == ("", "INSUFFICIENT_CONTEXT")


def test_a_partial_parse_still_returns_raw():
    """LOAD-BEARING. A rationale that parses must not mask an answer that did not.

    The failure this blocks, in full: the model writes a rationale and then emits
    the sentinel *without* answer tags. If the rationale slot carried the parsed
    rationale instead of `raw`, the word INSUFFICIENT_CONTEXT would vanish before
    `should_refuse` ever saw it — so a correct refusal gets booked as parser
    drift by `validate()`, and refusal rate (the metric D11 kept) quietly loses
    those queries to the parse-miss column. Same shape as `NDCG@10 = 0.0000`
    meaning wrong doc ids: well-formed output, wrong ledger.

    Re-breakable by the natural refactor: splitting the two lookups into
    independent `try` blocks that each fall through to a default instead of
    returning early. The empty answer then travels with the *parsed* rationale
    and `raw` is dropped. Nothing raises when that happens.
    """
    raw = "<rationale>the passages do not state this [1][2]</rationale>\nINSUFFICIENT_CONTEXT"
    assert parse_answer(raw) == ("", raw)


def test_empty_answer_tags_count_as_a_miss():
    """The invariant is "answer empty", not "answer tag absent".

    `<answer></answer>` matches the regex, so the IndexError guard never fires;
    without an explicit emptiness check the miss would skip the `raw` path.
    """
    raw = "<rationale>r [1]</rationale><answer></answer>"
    assert parse_answer(raw) == ("", raw)


def test_a_missing_rationale_does_not_discard_a_good_answer():
    """Token-F1 needs the answer alone; a missing rationale only degrades 4c.

    Failing the whole parse here would book a perfectly scoreable answer against
    `validate()`'s 20% parse-miss gate.
    """
    raw = "<answer>Nelson County</answer>"
    assert parse_answer(raw) == ("Nelson County", raw)


def test_parsing_does_not_normalise_the_answer():
    """Whitespace only. Normalisation belongs to the correctness scorer.

    HotpotQA gold answers carry literal punctuation — `'"Alceste"'` includes its
    quote marks — and standard normalisation moved gold-context accuracy from
    9/15 to ~13/15. Doing it in two places would double-normalise somewhere no
    test is watching.
    """
    answer, _ = parse_answer('<rationale>r [2]</rationale><answer>  "Alceste"  </answer>')
    assert answer == '"Alceste"'


def test_a_multiline_rationale_survives():
    """Rationales span lines, so the pattern needs DOTALL; without it this
    returns a miss and the faithfulness judge reads nothing."""
    raw = "<rationale>line one [1]\nline two [2]</rationale>\n<answer>yes</answer>"
    assert parse_answer(raw) == ("yes", "line one [1]\nline two [2]")


def test_the_first_answer_block_wins():
    """Documents a decision, not a law: qwen3 sometimes restates its answer, and
    first-vs-last was chosen without evidence. Revisit against a real ten-passage
    batch; until then this test is what makes the choice visible when it changes."""
    raw = "<rationale>r [1]</rationale><answer>AOL</answer> ... <answer>Sesame Street</answer>"
    answer, _ = parse_answer(raw)
    assert answer == "AOL"


# ─────────────────────── should_refuse's contract ───────────────────────
#
# The gate is MODEL-side: the prompt asks for SENTINEL when the passages do not
# support an answer, and this function detects it. It therefore measures the
# model's own calibration, not a threshold anyone chose — a distinction the
# README row has to state, because a score-threshold gate would produce a
# differently-meaning number of the same name.
#
# As of D14 (2026-08-22) this function drives TWO published numbers, not one:
# refusal rate, and — because refusals leave the correctness denominator — which
# queries correctness is computed over at all. A false positive here deletes a
# scoreable answer from the metric silently.


def test_the_sentinel_in_the_answer_is_a_refusal():
    assert should_refuse((SENTINEL, "the passages do not say [1][2]")) is True


def test_a_normal_answer_is_not_a_refusal():
    assert should_refuse(("Nelson County", "it sits there [1]")) is False


def test_a_sentinel_that_lost_its_tags_is_still_a_refusal():
    """The payoff for `parse_answer`'s `("", raw)` contract.

    When the model emits the sentinel without answer tags the parser cannot
    return it as an answer — so it hands back `raw`, and this is the only place
    that looks. Without this branch a genuine refusal is booked as parser drift
    by `validate()` and vanishes from refusal rate.
    """
    raw = f"<rationale>the passages do not say [1]</rationale>\n{SENTINEL}"
    assert should_refuse(("", raw)) is True


def test_an_unparseable_answer_is_not_laundered_into_a_refusal():
    """The mirror-image error, and the worse one because it is silent.

    Returning True for every empty answer would move parse misses into the
    refusal column, where `validate()`'s 20% gate cannot see them (it exempts
    `refused` rows) — so a broken parser would read as a cautious model and
    nothing would raise. An empty answer with no sentinel anywhere in `raw` is a
    parse miss, and it must stay one.
    """
    assert should_refuse(("", "I'm not sure how to answer that.")) is False


def test_a_rationale_that_merely_mentions_the_sentinel_is_not_a_refusal():
    """LOAD-BEARING. The rationale is consulted ONLY when the answer is empty.

    The prompt hands the model the sentinel string, and models narrate the rules
    they were given — so a perfectly good answer can arrive alongside a rationale
    that names SENTINEL in passing. Ungated, that returns True: `generate_one`
    overwrites a parsed answer with REFUSAL, `validate()` exempts the row,
    refusal rate inflates, and under D14 the query drops out of the correctness
    denominator. Two headline numbers move and neither moves informatively.

    Regresses the moment anyone removes the `not answer` gate. Nothing raises.
    """
    rationale = (f"If the passages did not cover this I would say {SENTINEL}, "
                 "but [2] gives it")
    assert should_refuse(("Nelson County", rationale)) is False


def test_sentinel_detection_survives_case_and_trailing_text():
    """The model may lower-case it or append its reasoning to it. Normalising
    here is safe precisely because this is classification, not scoring — the
    place normalisation must NOT happen is `parse_answer`."""
    assert should_refuse((SENTINEL.lower(), "r [1]")) is True
    assert should_refuse((f"{SENTINEL} — [2] lacks the link", "r [1]")) is True


def test_the_gate_and_the_prompt_use_the_same_sentinel():
    """SENTINEL is one constant because it appears in two places that must agree.

    Reword the prompt's sentinel without updating the detector and refusal rate
    drops to 0.0%, which reads as a *result* rather than as a broken gate. This
    test fails instead.
    """
    from retrieval_lab.generate import build_prompt
    rendered = build_prompt("Q?", [{"title": "T", "text": "x"}])
    assert SENTINEL in rendered
