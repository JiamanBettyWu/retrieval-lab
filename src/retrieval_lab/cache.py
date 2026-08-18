"""On-disk cache for retrieval runs.

Every later phase re-scores the SAME first-stage candidates: the Phase 1
cross-encoder reranks them, and the Phase 2 oracle bounds what that reranking
could achieve. Recomputing embeddings per run isn't just slow (~1 min for
NFCorpus) — it leaves "did both stages see identical candidates?" resting on
determinism rather than on evidence. Caching makes it a fact.

The key covers dataset + model + top_k, but NOT the contents of retrieve().
Change that function and the cache goes stale silently — pass refresh=True
(or `--refresh`) after touching retrieval.
"""
import json
import logging
from pathlib import Path

CACHE_DIR = Path("cache")

log = logging.getLogger("retrieval_lab")


def cache_path(dataset: str, model_name: str, top_k: int) -> Path:
    """One file per (dataset, model, top_k) — the things that change the run."""
    slug = model_name.replace("/", "__")
    return CACHE_DIR / f"{dataset}__{slug}__top{top_k}.json"


def cached_retrieval(dataset: str, model_name: str, top_k: int, compute, refresh: bool = False) -> dict:
    """Return {query_id: {doc_id: score}}, from disk if present.

    `compute` is a zero-arg callable producing the results. It stays a thunk so
    a cache hit skips loading the encoder entirely, not just the encoding.
    """
    path = cache_path(dataset, model_name, top_k)

    if path.exists() and not refresh:
        results = json.loads(path.read_text())
        log.info("retrieval cache HIT  %s (%d queries)", path, len(results))
        return results

    log.info("retrieval cache MISS %s — computing ...", path)
    results = compute()

    CACHE_DIR.mkdir(exist_ok=True)
    # JSON round-trips this shape exactly: BEIR ids are already strings and the
    # scores are Python floats (retrieve() casts them), so nothing is coerced.
    path.write_text(json.dumps(results))
    log.info("retrieval cached -> %s", path)
    return results


def generation_cache_path(dataset: str, retriever: str, top_k: int, n_context: int,
                          generator: str, prompt_version: str) -> Path:
    """One file per thing that changes what the generator was asked.

    Phase 4 needs a stricter key than retrieval does, for a reason retrieval
    doesn't have: **generation is nondeterministic even at temperature 0**, so a
    stale hit cannot be caught by "just re-run it and compare".

    `prompt_version` is in the key because editing a prompt invalidates results
    exactly as silently as editing `retrieve.py` does — and unlike `retrieve.py`
    there is no `--refresh` habit built around it yet. Bump it on every prompt
    edit; that is the entire contract.

    The judge model IDs are deliberately NOT here — see `judgement_cache_path`.
    """
    slug = lambda s: s.replace("/", "__").replace(":", "-")
    return (CACHE_DIR / f"gen__{dataset}__{slug(retriever)}__top{top_k}"
                        f"__ctx{n_context}__{slug(generator)}__{prompt_version}.json")


def judgement_cache_path(generation_key: str, judge: str, rubric_version: str) -> Path:
    """Judgements key off the generations they scored, PLUS judge and rubric.

    **This splits what `TODO.md` D10 recorded as one key** ("all three model IDs
    go in the generation cache key alongside `prompt_version`"). D10's intent is
    honoured — swapping a judge must never silently reuse scores produced by a
    different judge — but folding judge IDs into the *generation* key would also
    discard every generation whenever only a judge changed, and D10 explicitly
    expects the judge to change (both are marked tentative until they clear the
    4c.1 fixture). Generations are the expensive half; re-scoring them is cheap.

    So: one generation cache, one judgement cache per judge, the second keyed on
    the first. A swapped judge invalidates judgements only, which is exactly the
    blast radius the decision was protecting against.
    """
    slug = lambda s: s.replace("/", "__").replace(":", "-")
    return CACHE_DIR / f"judge__{generation_key}__{slug(judge)}__{rubric_version}.json"
