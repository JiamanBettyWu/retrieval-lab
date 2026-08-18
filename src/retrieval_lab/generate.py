"""Phase 4a: generate answers from retrieved context, cached.

    python -m retrieval_lab.generate --dataset hotpotqa-distractor-pool --n-queries 50

Takes `results` as an argument rather than re-retrieving, for the same reason
`oracle_rerank` does: every Phase 4 number must be about the SAME first-stage
candidates the retrieval phases were scored on. Re-retrieving here would make
"the generator and the reranker saw identical context" an assumption about
determinism instead of a fact on disk.

**Why generation gets a stricter cache than retrieval.** Retrieval is
deterministic — a stale cache is recoverable by re-running and diffing. A
generator is not, even at temperature 0 (batching, kv-cache layout and
quantisation all leak in), so two runs of "the same" config differ slightly and
there is no diff that proves staleness. The key therefore covers everything that
changes what the model was asked: dataset, retriever, top_k, context size,
generator, and `prompt_version`. See `cache.generation_cache_path`.

Scope: this module generates. It does not score. Faithfulness/relevance judging
is 4c and lands in `judge.py`; correctness (token-F1 against HotpotQA's gold
short answers) is the amended scope decision in `docs/plan.md`.

────────────────────────────────────────────────────────────────────────────
STUBS — three functions below raise NotImplementedError and are Betty's:
    build_prompt()   the RAG prompt, and what PROMPT_VERSION means
    parse_answer()   splitting a short answer out of a paragraph
    should_refuse()  the refusal gate, which is a measurement not a feature
Everything else (cache, sampling, transport, sanity checks) is wired.
────────────────────────────────────────────────────────────────────────────
"""
import argparse
import json
import logging
import random
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

from .cache import CACHE_DIR, cached_retrieval, generation_cache_path
from .data import load_beir
from .observability import op
from .retrieve import BI_ENCODER, MODEL_HELP, load_encoder, retrieve

log = logging.getLogger("retrieval_lab")

GENERATOR = "qwen3:8b"          # D10: local via Ollama, deliberately not the judge
OLLAMA_URL = "http://localhost:11434/api/generate"

# Bump on EVERY prompt edit. It is the only thing standing between a reworded
# prompt and a cache that silently serves answers to the previous wording.
PROMPT_VERSION = "v0"

REFUSAL = "__REFUSED__"


@dataclass
class Generation:
    """One (question, context, answer) triple — the unit the judge will score."""
    query_id: str
    question: str
    doc_ids: list[str]      # the context, by id, so the judge can re-read it
    raw: str                # exactly what the model emitted, never cleaned
    answer: str             # the short answer, for token-F1 against gold
    rationale: str          # the model's supporting text, for faithfulness
    refused: bool


# ─────────────────────────── Betty's stubs ───────────────────────────

def build_prompt(question: str, docs: list[dict]) -> str:
    """Render the retrieval context and question into one prompt string.

    CONCEPT — the three jobs this prompt has, and they pull against each other:

    1. **Ground the answer.** Instruct the model to use only the passages. This
       is what makes faithfulness measurable at all: without it, an answer drawn
       from parametric memory is indistinguishable from a grounded one, and your
       own probe showed `qwen3:8b` will happily invent a university donor.

    2. **Emit a citable structure.** The judge has to check claims against
       passages, so passages need stable labels ([1], [2], ...) that the model
       reuses. Ungrounded *citations* are a real failure mode — the probe caught
       an answer citing [1] for a claim [1] never made — and you can only catch
       it if citations are parseable.

    3. **Emit a SHORT answer separately from the rationale.** HotpotQA's gold is
       a span ("Nelson County"); the model emits paragraphs. Token-F1 over a
       paragraph scores near 0 no matter how right it is. So ask for both — a
       short answer line and its supporting rationale — and let `parse_answer`
       split them. Deciding that output format IS this function's real design
       work; everything downstream depends on the shape you pick here.

    `docs` arrives ordered best-first: [{"title": str, "text": str}, ...].

    Bump PROMPT_VERSION whenever you touch this.
    """
    raise NotImplementedError("build_prompt is yours — see the concept notes above")


def parse_answer(raw: str) -> tuple[str, str]:
    """Split raw model output into (short_answer, rationale).

    CONCEPT — this is the seam where a scoring metric meets a chat model, and it
    fails in a specific way: if parsing misses, you get an empty short answer,
    token-F1 scores 0, and the run looks like a *model* failure rather than a
    *parser* failure. Same shape as `NDCG@10 = 0.0000` meaning wrong doc ids.

    So parse strictly and make misses countable rather than silent — `main()`
    already logs a parse-failure rate and refuses to write a cache file when it
    is implausibly high. Return ("", raw) when you cannot find a short answer;
    do not guess, and do not fall back to the whole paragraph.

    Whatever contract you choose here must match what `build_prompt` asks for.
    """
    raise NotImplementedError("parse_answer is yours — see the concept notes above")


def should_refuse(question: str, docs: list[dict], parsed: tuple[str, str]) -> bool:
    """Decide whether this counts as a refusal.

    CONCEPT — the refusal gate is a MEASUREMENT, not a feature (D11 killed the
    framework, not the branch). The number it produces is **refusal rate per
    retrieval configuration**, and the question it answers is: does worse
    retrieval make the model refuse more, or does it make the model hallucinate
    more? Those are very different failure profiles for a RAG system, and only
    one of them is safe.

    The design choice is *where* refusal is detected:
      - **model-side** — the prompt tells it to say so when unsupported, and you
        detect that string. Measures the model's own calibration.
      - **pipeline-side** — a conditional on retrieval scores refuses before
        generating. Measures a threshold you chose.
    They measure different things and are not interchangeable. Pick one on
    purpose, and say which in the README row.

    `parsed` is `parse_answer`'s output, so a model-side gate can key off it.
    """
    raise NotImplementedError("should_refuse is yours — see the concept notes above")


# ─────────────────────── plumbing (wired, working) ───────────────────────

def call_ollama(prompt: str, model: str = GENERATOR, timeout: int = 600) -> str:
    """One generation. Thinking off, temperature 0.

    `think: False` matters more than it looks: qwen3 is a reasoning model by
    default and otherwise wraps every answer in a <think> block, which both
    triples latency and puts unsupported speculation into text the judge will
    read as part of the answer.
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["response"]
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama unreachable at {OLLAMA_URL} ({e}). Start it with `ollama serve`, "
            f"and check `ollama list` includes {model}."
        ) from e


def sample_queries(results: dict, qrels: dict, n: int, seed: int = 0) -> list[str]:
    """Pick n query ids to generate for — deterministically.

    The full pool is 7,405 questions; Phase 4 scores a sample because judging is
    the expensive half. Sorting before sampling makes the draw reproducible
    across machines (dict order is insertion order, which depends on how the
    corpus was built), so a rerun scores the SAME questions rather than a fresh
    sample that would quietly change every number.

    Only queries with labels are eligible: an unlabelled one can be generated
    for but neither its oracle context nor its correctness score exists.
    """
    eligible = sorted(q for q in results if qrels.get(q))
    if n >= len(eligible):
        return eligible
    return sorted(random.Random(seed).sample(eligible, n))


def context_for(results: dict, corpus: dict, query_id: str, n_context: int) -> tuple[list[str], list[dict]]:
    """The top-n retrieved docs for one query, best-first.

    Reads `results`, never `qrels` — the same constraint `oracle.py` is pinned
    on. Drawing context from the labels here would make every config generate
    from perfect context, which is milestone 4b's *ceiling* run, not a real one.
    """
    ranked = sorted(results[query_id], key=results[query_id].get, reverse=True)[:n_context]
    return ranked, [corpus[d] for d in ranked]


@op
def generate_one(question: str, doc_ids: list[str], docs: list[dict],
                 query_id: str, model: str) -> Generation:
    """Prompt → model → parsed Generation. Traced by Weave when it is live."""
    raw = call_ollama(build_prompt(question, docs), model=model)
    answer, rationale = parse_answer(raw)
    refused = should_refuse(question, docs, (answer, rationale))
    return Generation(
        query_id=query_id, question=question, doc_ids=doc_ids, raw=raw,
        answer=REFUSAL if refused else answer, rationale=rationale, refused=refused,
    )


def cached_generations(path, compute, refresh: bool = False) -> list[Generation]:
    """Same contract as `cached_retrieval`: `compute` stays a thunk.

    A cache hit must not require Ollama to be running at all — that is what
    makes re-scoring a finished generation set cheap, which is the whole reason
    judgements are cached separately.
    """
    if path.exists() and not refresh:
        rows = json.loads(path.read_text())
        log.info("generation cache HIT  %s (%d answers)", path, len(rows))
        return [Generation(**r) for r in rows]

    log.info("generation cache MISS %s — generating ...", path)
    gens = compute()

    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps([asdict(g) for g in gens], indent=1))
    log.info("generations cached -> %s", path)
    return gens


def validate(gens: list[Generation]) -> list[str]:
    """Raise on a batch that must not be cached. Returns the parse-miss ids.

    **Called inside `compute()`, before the cache write, on purpose.** Checking
    after the write is worse than not checking: the run raises, the operator
    reads the error, fixes the parser — and the next run is a silent cache HIT
    serving the poisoned batch. Generation is nondeterministic, so no rerun-and-
    diff catches it. A bad batch must never reach disk.
    """
    empty = [g.query_id for g in gens if not g.raw.strip()]
    if empty:
        raise AssertionError(
            f"{len(empty)} generations are empty (e.g. {empty[:3]}) — that is a "
            "transport or prompt failure, not a model that had nothing to say. "
            "Nothing cached."
        )

    # A parser that silently misses looks exactly like a model that answers
    # badly, and only one of those is worth debugging. Same shape as
    # NDCG@10 = 0.0000 meaning wrong doc ids rather than a weak model.
    unparsed = [g.query_id for g in gens if not g.refused and not g.answer.strip()]
    if len(unparsed) > 0.2 * len(gens):
        raise AssertionError(
            f"{len(unparsed)}/{len(gens)} answers produced no short answer — "
            "parse_answer() and build_prompt() have drifted apart. Nothing cached."
        )
    return unparsed


def main(dataset: str, n_queries: int, n_context: int, top_k: int, seed: int,
         refresh: bool, model: str = BI_ENCODER, generator: str = GENERATOR) -> None:
    corpus, queries, qrels = load_beir(dataset)

    results = cached_retrieval(
        dataset, model, top_k,
        compute=lambda: retrieve(load_encoder(model), corpus, queries, top_k=top_k),
        refresh=False,  # never rebuild retrieval as a side effect of generating
    )

    qids = sample_queries(results, qrels, n_queries, seed)
    path = generation_cache_path(dataset, model, top_k, n_context, generator, PROMPT_VERSION)

    def compute():
        gens = []
        for i, qid in enumerate(qids, 1):
            doc_ids, docs = context_for(results, corpus, qid, n_context)
            gens.append(generate_one(queries[qid], doc_ids, docs, qid, generator))
            if i % 10 == 0 or i == len(qids):
                log.info("  generated %d/%d", i, len(qids))
        validate(gens)          # before the cache write, never after
        return gens

    gens = cached_generations(path, compute, refresh)

    # This one stays post-hoc deliberately: it also fires on a cache HIT whose
    # file was written under a different --n-queries or --seed, which validate()
    # cannot see because that batch was valid when it was written.
    if len(gens) != len(qids):
        raise AssertionError(
            f"cache holds {len(gens)} answers but this config asks for {len(qids)} "
            f"queries — {path.name} was written under different sampling"
        )

    unparsed = [g.query_id for g in gens if not g.refused and not g.answer.strip()]
    refusals = sum(g.refused for g in gens)
    log.info("\n=== Phase 4a (%s, %d queries, top-%d context, %s, prompt %s) ===",
             dataset, len(gens), n_context, generator, PROMPT_VERSION)
    log.info("generated      : %d", len(gens))
    log.info("refused        : %d  (%.1f%% — the per-config metric D11 kept)",
             refusals, 100 * refusals / len(gens))
    log.info("no short answer: %d  (parse misses, not model failures)", len(unparsed))
    log.info("cache          : %s", path)
    log.info("\nNext: 4b generates from qrel-perfect context for the ceiling, "
             "then 4c validates a judge against the fixture before it scores any "
             "of this.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="hotpotqa-distractor-pool")
    p.add_argument("--n-queries", type=int, default=50,
                   help="how many questions to generate for (the sample size is a "
                        "judging-cost decision, not a retrieval one)")
    p.add_argument("--n-context", type=int, default=10,
                   help="how many retrieved docs go in the prompt")
    p.add_argument("--top-k", type=int, default=100, help="retrieval depth (cached)")
    p.add_argument("--seed", type=int, default=0, help="fixes WHICH questions are sampled")
    p.add_argument("--refresh", action="store_true", help="regenerate, ignoring the cache")
    p.add_argument("--model", default=BI_ENCODER, help=MODEL_HELP)
    p.add_argument("--generator", default=GENERATOR)
    args = p.parse_args()
    main(args.dataset, args.n_queries, args.n_context, args.top_k, args.seed,
         args.refresh, args.model, args.generator)
