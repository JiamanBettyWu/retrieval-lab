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
