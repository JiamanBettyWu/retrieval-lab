"""Fixture test for retrieve(). Run:  python test_retrieve.py

A five-doc corpus small enough to reason about by hand. Each assertion targets
ONE failure mode, so a red test names its own cause — unlike the full BEIR run,
where every bug collapses into a single low NDCG number.

No pytest: bare asserts keep this runnable with the Phase 0 deps as-is.
"""
from beir.retrieval.evaluation import EvaluateRetrieval

from retrieve import load_encoder, retrieve

# Doc ids deliberately look like BEIR's (non-numeric, non-sequential) so that
# leaking a row index through instead of a real id can't accidentally pass.
CORPUS = {
    "MED-001": {"title": "Aspirin and heart disease",
                "text": "Low-dose aspirin reduces the risk of heart attack."},
    "MED-002": {"title": "Vitamin D deficiency",
                "text": "Vitamin D deficiency is linked to bone loss and rickets."},
    "MED-003": {"title": "Sourdough bread baking",
                "text": "A wild yeast starter gives sourdough its sour flavor."},
    "MED-004": {"title": "Volcanic activity in Iceland",
                "text": "Iceland sits on the Mid-Atlantic Ridge and has frequent eruptions."},
    "MED-005": {"title": "The rules of cricket",
                "text": "A cricket match is played between two teams of eleven players."},
}

QUERIES = {
    "q1": "does aspirin prevent heart attacks?",
    "q2": "what happens when you lack vitamin D?",
}

# Ground truth: the one obviously-correct doc per query.
QRELS = {
    "q1": {"MED-001": 1},
    "q2": {"MED-002": 1},
}
EXPECTED_TOP = {"q1": "MED-001", "q2": "MED-002"}


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        raise AssertionError(f"{name}{' — ' + detail if detail else ''}")


def main():
    model = load_encoder()

    print("\n[1] shape of the returned dict")
    results = retrieve(model, CORPUS, QUERIES, top_k=3)

    check("every query has an entry",
          set(results) == set(QUERIES),
          f"got {set(results)}, want {set(QUERIES)}")

    for q, hits in results.items():
        # Catches error #1: a list of single-key dicts instead of a flat dict.
        check(f"results['{q}'] is a dict",
              isinstance(hits, dict),
              f"got {type(hits).__name__}")

        # Catches error #2: corpus_id row indices leaking through unmapped.
        check(f"results['{q}'] keys are real corpus doc_ids",
              all(k in CORPUS for k in hits),
              f"unknown ids: {[k for k in hits if k not in CORPUS]}")

        # pytrec_eval wants plain Python floats, not torch/numpy scalars.
        check(f"results['{q}'] values are floats",
              all(isinstance(v, float) for v in hits.values()),
              f"got types {sorted({type(v).__name__ for v in hits.values()})}")

        check(f"results['{q}'] honors top_k=3", len(hits) == 3, f"got {len(hits)}")

    print("\n[2] ranking is sane")
    for q, want in EXPECTED_TOP.items():
        top = max(results[q], key=results[q].get)
        # A wrong doc here means the query/doc args to semantic_search are swapped.
        check(f"'{q}' ranks {want} first", top == want, f"got {top}")

    print("\n[3] top_k larger than the corpus clamps, not crashes")
    wide = retrieve(model, CORPUS, QUERIES, top_k=100)
    check("top_k=100 over 5 docs returns 5",
          all(len(h) == len(CORPUS) for h in wide.values()),
          f"got {[len(h) for h in wide.values()]}")

    print("\n[4] BEIR's evaluator accepts the shape (the real integration check)")
    # Shape-checking your own dict is not the same as pytrec_eval accepting it.
    # With one relevant doc per query, all of it correctly ranked first,
    # NDCG@10 must be exactly 1.0. A 0.0 here means the ids don't line up.
    ndcg, _map, recall, _prec = EvaluateRetrieval.evaluate(QRELS, wide, [10])
    print(f"       NDCG@10 = {ndcg['NDCG@10']:.4f}   Recall@10 = {recall['Recall@10']:.4f}")
    check("NDCG@10 == 1.0 on the fixture",
          abs(ndcg["NDCG@10"] - 1.0) < 1e-9,
          f"got {ndcg['NDCG@10']:.4f} (0.0 => doc_ids don't match qrels)")

    print("\nAll checks passed.\n")


if __name__ == "__main__":
    main()
