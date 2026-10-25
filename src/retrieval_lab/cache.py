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
import hashlib
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


def exclusion_tag(excluded) -> str:
    """A short, stable name for a set of excluded query ids; "" for no exclusion.

    Hashes the ids themselves, not the files they came from: excluding the v1
    and the v2 file of one draw is the same exclusion and must name the same
    batch. `hashlib`, not `hash()` — Python salts `hash()` per process, and the
    point of the key is that another machine finds the same file.
    """
    if not excluded:
        return ""
    return hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest()[:10]


def exclusion_suffix(excluded) -> str:
    """The filename token for an exclusion: `__ex<tag>`, or "" for none.

    One function because a batch and its labelling sheet must carry the same
    token — a second copy of the format is a link that can silently break.
    """
    tag = exclusion_tag(excluded)
    return f"__ex{tag}" if tag else ""


def generation_cache_path(dataset: str, retriever: str, top_k: int, n_context: int,
                          generator: str, prompt_version: str, n_queries: int,
                          seed: int, excluded=frozenset()) -> Path:
    """One file per thing that changes what the generator was asked.

    Phase 4 needs a stricter key than retrieval does, for a reason retrieval
    doesn't have: **generation is nondeterministic even at temperature 0**, so a
    stale hit cannot be caught by "just re-run it and compare".

    `prompt_version` is in the key because editing a prompt invalidates results
    exactly as silently as editing `retrieve.py` does — and unlike `retrieve.py`
    there is no `--refresh` habit built around it yet. Bump it on every prompt
    edit; that is the entire contract.

    **`n_queries` and `seed` are in the key too**, which the retrieval cache has
    no equivalent of. They do not change what the model was *asked* — they change
    *which questions* were asked, and a file holding 15 answers is not a prefix of
    the file holding 100: `sample_queries` draws a fresh sample per n, so the two
    sets overlap only by chance. Leaving them out meant one filename per config
    and therefore one batch on disk at a time, so drawing a held-out fixture batch
    overwrote the batch being scored. Now every distinct draw is its own file and
    a fixture set can sit beside the set it validates a judge for.

    **`excluded` joins the key for the same reason.** A draw that leaves out
    earlier batches holds different questions from the plain draw with the same
    `(n, seed)`, so without it the plain batch would be served as a cache HIT.
    The suffix appears only when something is excluded, so every batch written
    before exclusion existed keeps its name — and the labels and judgements
    keyed off those names stay valid. The hash is opaque on purpose (a list of
    ids does not fit in a filename); `exclusion_record_path` says what it hides.

    The judge model IDs are deliberately NOT here — see `judgement_cache_path`.
    """
    slug = lambda s: s.replace("/", "__").replace(":", "-")
    return (CACHE_DIR / f"gen__{dataset}__{slug(retriever)}__top{top_k}"
                        f"__ctx{n_context}__{slug(generator)}__{prompt_version}"
                        f"__n{n_queries}__seed{seed}{exclusion_suffix(excluded)}.json")


def exclusion_record_path(generation_path: Path) -> Path:
    """Where a batch drawn with exclusions records WHAT it excluded.

    Named `exclude__…`, not `gen__…`, so nothing that globs for generation
    batches (`fixture.overlap_with`) mistakes the record for one.
    """
    return generation_path.with_name(f"exclude__{generation_path.name}")


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
