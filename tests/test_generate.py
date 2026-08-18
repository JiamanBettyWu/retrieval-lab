"""Fixture tests for Phase 4a's plumbing.

    pytest tests/test_generate.py

Pins the wiring only — `build_prompt`, `parse_answer` and `should_refuse` are
stubs, and these tests deliberately never call them. No model, no Ollama, no
download.

The two load-bearing ones are `test_context_comes_from_results_not_qrels` (the
same constraint `test_oracle.py` pins one level down) and the cache-key tests
(a stale generation cache cannot be caught by re-running, because generation is
nondeterministic even at temperature 0).
"""
import json

import pytest

from retrieval_lab.cache import generation_cache_path, judgement_cache_path
from retrieval_lab.generate import (
    Generation,
    cached_generations,
    context_for,
    sample_queries,
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
           top_k=100, n_context=10, generator="qwen3:8b", prompt_version="v0")


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
])
def test_every_key_component_changes_the_cache_path(field, value):
    """Anything that changes what the model was asked must change the filename.

    Generation is nondeterministic even at temperature 0, so a stale hit cannot
    be caught by re-running and diffing the way a stale retrieval cache can.
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
