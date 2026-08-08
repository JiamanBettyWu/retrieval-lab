# Session log

Reverse-chronological journal — one dated entry per working session: what got
done, what was decided and why. History only; for what to do next see
[TODO.md](TODO.md).

---

## 2026-08-08 (Phase 1 lands at 0.3412; four wrong answers about why)

Second session of the day, continuing from the entry below. Betty implemented
`rerank()` herself and asked for a review. **Phase 1 is done and the ablation
table has its second row.** Most of the session went into explaining a result
that turned out to be much harder to attribute than to measure.

**The review.** The implementation was correct on every trap the scaffold
warned about — `(query, doc)` pair order, no truncation, `float()` cast — with
one real bug: it looped over `queries` and indexed `results[q]`. Harmless while
the two key sets match, which is why all 32 tests passed, but `--limit N`
truncates `results` and not `queries`, so the documented smoke-test command died
with `KeyError`. Betty fixed it. The deeper problem was that `test_rerank.py`
*couldn't* catch it: `QUERIES` and `RESULTS` had identical keys, so iterating
either was indistinguishable. Added `q3` to `QUERIES` with no `RESULTS` entry
plus an explicit `test_iterates_results_not_queries`, then verified the fix by
reintroducing the bug (11 errors) and restoring.

**The number.** `NDCG@10 0.3159 → 0.3412`, `+0.0253` — **8.1% of the 0.3104
headroom** the oracle measured on 2026-08-07. `Recall@100` came back identical
to five decimals (`0.31145`), which is worth more than it looks: reranking may
only reorder the candidate set, so that column is a free assertion that nothing
got truncated, holding over all 323 real queries where the fixture covers four
docs. `main()` now checks it explicitly and errors if it ever moves.

**Explaining the 8% — four states, three of them wrong, all of them reached the
README before being replaced.** Recording the sequence because each failure is a
distinct and generalizable trap:

1. *Asserted.* MRR@10 rose `+0.0630` while Recall@10 stayed flat (`+0.0021`).
   The obvious story: a train/test mismatch in **qrel density** — MS MARCO
   labels ~1 relevant passage per query, NFCorpus has a median of 16, so the
   model optimizes "find the single best answer" while NDCG@10 pays for filling
   ten slots. Tidy, consistent with two real measurements, and untested. It was
   written into the README as fact before anyone noticed it was a hypothesis.
2. *"Refuted."* The prediction — capture should fall as more relevant docs
   become available — was tested by bucketing the 245 queries with headroom,
   each normalized by its own `oracle − baseline`, and averaging: `+16.4% ·
   −22.0% · +7.1% · −12.9%`. No trend. README updated to say so.
3. *Supported.* **The estimator was broken.** A per-query ratio divides by that
   query's headroom; two queries with headroom near `0.016` and ordinary losses
   produced ratios of `−9.3` and `−10.7`, enough to drag a 54-query bucket from
   positive to `−22.0%`. Pooling instead (`sum(gain)/sum(headroom)`, which never
   divides by an individual near-zero and is built like the corpus-level figure)
   gives `+14.8% · +19.0% · +9.2% · −1.9%` — the 3–5 bucket flips sign. Spearman
   on ranks: `rho = −0.19`, one-sided permutation `p ≈ 0.001`, n=245.
4. *Unattributable.* Baseline NDCG@10 predicts capture equally well (`−0.1924`
   vs availability's `−0.1912`) and the two correlate with each other at
   `+0.5716`. Partial either out and both collapse symmetrically
   (`−0.1009` / `−0.1032`). Headroom is exonerated as a confound — controlling
   for it, availability *strengthens* to `−0.2129` — but availability and
   "the bi-encoder was already good here" are the same queries wearing two
   labels, and the second needs no story about MS MARCO at all.

**Betty broke the deadlock, and the move is the session's real lesson.** Her
question: if the model is trained to surface one best answer, shouldn't MRR
improve on dense queries even as NDCG@10 falls? That is a *qualitative*
prediction, and it is exactly what a correlation cannot deliver — "less room to
improve" explains any difference in **magnitude**, but it does not predict two
metrics moving in **opposite directions** on the same queries. It holds, and
only on the dense bucket:

```
availability     n   dNDCG@10      dMRR   rel@10 before -> after
       1-2      84    +0.0422   +0.1004      0.79 -> 0.83
       3-5      55    +0.0635   +0.0968      1.82 -> 2.00
      6-10      47    +0.0466   +0.0766      2.62 -> 2.91
       11+      86    -0.0123   +0.0289      5.78 -> 5.44
```

With 11+ relevant docs available the reranker still improves rank 1 while
*evicting* 0.34 relevant docs from the top ten. Noise added to a good ordering
would push all three down; instead the first position improves and coverage
degrades. Still short of proof — the dense bucket is also the high-baseline
bucket — but it is evidence a confound doesn't trivially absorb. The
generalization, recorded in `LEARNINGS.md`: **when two explanations are
entangled, don't refine the measurement, find the prediction on which they
disagree.** Three passes of better statistics could not do what one change of
question did.

**Attributing the gain in points rather than percentages** (capture is a ratio;
the headline is a sum) turned up the most actionable thing in the session:
**86% of the `+0.0253` comes from queries with 1–5 relevant docs available**,
while the 11+ bucket *subtracts* `1.06` NDCG points — 13% of what the rest
earned — despite holding the **largest headroom in the dataset** (33.3 points).
Suppressing the reranker on dense queries alone would give `0.3445` instead of
`0.3412`. Better still, given the MRR result: blend cross-encoder and bi-encoder
scores rather than replacing, keeping the rank-1 win without the eviction. That
is a concrete Phase 2 lever that costs no training.

**Everything the README quotes is now printed by `rerank.py --breakdown`.** The
analysis was originally scratchpad scripts, which is precisely the gap this repo
exists to close — a portfolio front door citing numbers no entrypoint can
regenerate. `main()` gained the full metric set (was NDCG@10 only), and
`breakdown()` gained per-query win/loss, the pooled/median/mean bucket table,
the correlation and partial-correlation block, gain attribution, and the
MRR-vs-NDCG dissociation. The statistics helpers are pinned by seven new tests:
tie-averaged ranks, Spearman `±1` on monotone input, outlier-immunity, a small
permutation p on a real trend, and a partial correlation that zeroes a pure
confound. Query order in `breakdown()` is `sorted()` and the permutation RNG is
seeded, deliberately — a reported statistic must not drift with dict ordering.
`p` is still a Monte Carlo estimate (SE ≈ `0.0002` at 20k shuffles) and the
README says so rather than quoting four significant figures.

**Notes for later.** The cross-encoder's `max_length` is 512, checked early and
ruled out as a cause — NFCorpus abstracts aren't being truncated to stubs.
Reading the availability buckets, note that "availability" means *relevant docs
retrieved into the top-100* (median 3), not relevant docs in the qrels (median
16); both numbers appear in the README and they are not interchangeable. The
capture table covers 245 queries (those with headroom) while the gain table
covers all 323 — the 51 zero-availability queries have zero headroom by
definition and vanish from any ratio.

## 2026-08-08 (Phase 1 scaffolded; why reranking precedes the recall work)

Continuation of the 2026-08-07 session below. No metrics moved; this was
interpretation plus a scaffold. Commit `847fb53`.

**Reading the oracle result correctly — the numbers that explain it.** Betty
asked whether Recall@100 0.3115 and oracle NDCG@10 0.6263 could be reconciled as
"recall is poor but the top-10 is found well." Half right. Pulled the actual
distribution of *retrieved-relevant* docs per query:

```
relevant docs per query       : median 16   mean 38.2
of those, retrieved in top-100: median  3   mean  8.0
queries with >=10 retrieved-relevant : 88/323
queries with  0 retrieved-relevant   : 51/323
```

So retrieval is **not** finding the top-10 well — the median query has only 3 of
its relevant docs anywhere in the 100-doc pool. The reconciliation is that the
two metrics have **different denominators**. Recall@100 is judged against every
relevant doc (median 16, mean 38); NDCG@10's IDCG counts only the best 10, so
everything beyond ten slots is free. The log discount compounds it: summing
`1/log2(rank+1)` over 10 slots gives 4.544, of which ranks 1–3 alone contribute
2.131 — **the top three positions carry 47% of all NDCG@10 weight.** Three
relevant docs, perfectly placed, capture roughly half the achievable score.
That is why a thin candidate pool still yields a 0.63 ceiling.

**51 of 323 queries (16%) have zero relevant docs in the top-100.** The oracle
scores 0.0 on those and no reranker can ever help them — a hard, quotable cap on
Phase 1. It also sharpens Phase 2's framing: recall work is *not* uniformly
valuable, and its single biggest win is dragging queries out of that zero
bucket, since each one moves from a guaranteed 0.0 to something positive. Worth
tracking that count as a Phase 2 metric alongside NDCG.

**Decision: do NOT reorder the phases.** Betty's follow-up was whether retrieval
quality should be improved before reranking. Pushed back, and the oracle is the
evidence: +0.3104 of NDCG@10 is available from reordering *the candidates as
they are today*, so improving retrieval first doesn't unlock that gain — it only
delays collecting it. The two levers compose rather than sequence (better
retrieval raises the ceiling a reranker works under; it doesn't replace the
reranker). Three supporting reasons: cost asymmetry (a pretrained cross-encoder
is an afternoon, LoRA fine-tuning is days), the ablation needs a rerank-alone row
to stay interpretable, and Phase 1 de-risks Phase 2 — if the cross-encoder
underdelivers against its ceiling, the two-stage architecture itself is in
question and you'd want to know that before spending days training. The kernel
Betty was right about: recall is the bigger *long-run* constraint (0.3737
unreachable vs 0.3104 reachable), which is exactly what Phase 2 exists for.

**Scaffolded Phase 1** (`src/retrieval_lab/rerank.py`, `tests/test_rerank.py`).
`rerank()` is left as a `TODO(human)` with a four-step docstring — Betty is
implementing it herself, same handover pattern as `retrieve()` on 2026-07-15.
Everything around it is wired: `main()` loads the cached candidates, reranks,
and prints Phase 0 / Phase 1 / ceiling side by side plus **the percentage of
available headroom captured**, so the result gets read against 0.6263 rather
than 1.0. `--limit N` reranks a subset for fast iteration and warns that those
numbers aren't ablation-comparable. The test suite is deliberately red until the
function lands (10 errors, all `NotImplementedError` — verified nothing else
breaks). Its fixture stacks the deck: the correct doc starts *last* in the input
ordering and every relevant doc was retrieved, so a correct reranker scores
exactly 1.0. Tests target the two traps expected here — pairs built as
`(doc, query)` instead of `(query, doc)`, which passes every shape check while
silently degrading ranking; and truncating to the top 10, which collapses
Recall@100 and breaks comparability with the Phase 0 row.

**Non-obvious:** the cross-encoder shares the MiniLM-L6 backbone with the Phase 0
bi-encoder but is a different model *type* — it emits one score per (query, doc)
pair and produces no reusable embedding, which is why it cannot do first-stage
retrieval (scoring the full corpus would be ~1.2M forward passes against ~32K
over the cached top-100). `cross-encoder/ms-marco-MiniLM-L-6-v2` downloads on
first use; the test run already pulled it. Its outputs are **logits** —
unbounded, often negative, not comparable to Phase 0's cosine scores. Fine for
ranking since BEIR reads only the order.

---

## 2026-08-07, later (Phase 1 ceiling measured, D3 resolved, src/ restructure)

Same working day as the entry below, which closed out Phase 0. This half
answered "is Phase 1 worth building?" *before* building it, then reorganized the
repo. Commits `31349fc`, `343d7d1`, `d8a9783`, `1cbab34`.

**Caching came first, and not for speed** (`src/retrieval_lab/cache.py`,
`31349fc`). Every later phase re-scores the *same* first-stage candidates — the
Phase 1 cross-encoder reranks them, the oracle bounds what that reranking could
achieve. Recomputing embeddings per run left "did both stages see identical
candidates?" resting on determinism instead of evidence; a JSON cache keyed on
(dataset, model, top_k) makes it a fact. Confirmed lossless — the cache-hit run
reproduced 0.3159 / 0.5046 / 0.3115 exactly. The `compute` argument is a thunk
so a cache hit skips loading the encoder entirely, not just the encoding. **The
key does not cover `retrieve.py`'s contents** — edit that file and the cache
goes stale silently, hence `--refresh`.

**The oracle rerank** (`src/retrieval_lab/oracle.py`). A cheating reranker: it
reads the qrels and orders each query's retrieved candidates perfectly, but may
only *reorder* what retrieval found, never add to it. Result on NFCorpus
top-100: **baseline 0.3159 → oracle 0.6263, headroom +0.3104 (98% relative)**.
Recall@10 tells the same story more plainly — reordering alone lifts it
0.155 → 0.276 against a Recall@100 of 0.3115, meaning nearly every relevant doc
the retriever found is already in the top-100 but buried below rank 10. The
residual `1 − 0.6263 = 0.3737` is unreachable by any reranker; that is recall,
and it belongs to Phase 2.

**D3 resolved: build Phase 1.** The top-100 contains roughly the right documents
and merely orders them badly, which is precisely what a cross-encoder repairs.
The oracle row is now a permanent third column in the README ablation table — a
ceiling to read Phase 1's result against rather than the meaningless 1.0. That
framing is itself the portfolio point: most RAG projects report the lift and
never the ceiling.

**A verification mistake worth remembering, because it was caught rather than
avoided.** The first version (`31349fc`) asserted only a per-query invariant:
oracle ≥ baseline on every query. That felt rigorous and isn't — it passes for
*any* improvement, including one leaving gain on the table, so it establishes
"better," not "best." For a number the README publishes as a hard upper bound,
best is the claim. `343d7d1` adds `arithmetic_ceiling()`, which derives the mean
NDCG@10 from first principles independently of the sort under test (place the
best k retrieved-relevant grades at ranks 1..k over IDCG@10 from full qrels).
It agrees to floating point: **0.626292 vs 0.626300**. Both invariants now
assert on every run. The general lesson: a monotonicity check and an optimality
check are different claims, and it is easy to ship the first while believing
you shipped the second.

**Metric facts pinned down along the way**, now recorded in `CLAUDE.md` so they
don't get re-derived: qrels values are relevance *grades* (NFCorpus uses 1 and
2, higher = better), not rank positions — six docs can share grade 2, which is
what proves it. `pytrec_eval` computes NDCG with **linear gain** (`gain = grade`),
not the exponential `2^grade − 1` common in papers; verified empirically rather
than assumed, with a case that discriminates the two (0.859719 linear vs
0.796708 exponential — pytrec_eval returned the former). And the ideal ranking
is *constructed* by sorting the graded docs, which is why IDCG counts relevant
docs retrieval never returned and why the oracle cannot reach 1.0.

**The restructure (the real decision this half).** Betty asked to move the code
into `src/`. That **reverses the flat-layout decision recorded in the entry
below** ("small lab; also sidesteps the package-import traps mise hit"), so it
was surfaced explicitly rather than quietly applied — she overruled it
deliberately after seeing the trade-off. At seven modules, with `rerank.py` and
`finetune.py` still to come, `pyproject.toml` plus an editable install makes
imports unambiguous rather than fragile, which was the actual concern behind the
original call. Two lighter options were offered and declined: `tests/` only
(zero import changes), and a `src/` directory with a `PYTHONPATH` shim — the
latter rejected in the framing because a path shim *is* the fragility the
original note warned about. Amendment recorded in `docs/plan.md` rather than
overwriting the old layout section. `d8a9783`.

Two things the move forced. The test files were `main()`-driven scripts with
bare asserts; under `pytest tests/` they would have collected **zero tests and
exited green** — worse than having no suite. Converted to real parametrized
`test_*` functions (22 tests, ~6s). And `requirements.txt` was deleted rather
than kept alongside `pyproject.toml`, since two dependency lists drift. Weave
moved to a `[tracing]` extra, matching its already-optional design. Verified
mechanical: both entrypoints reproduce their numbers exactly post-move, and
`git mv` throughout so `git log --follow` still reaches the original commits.

**`CLAUDE.md` initialized** (`1cbab34`) — setup/run commands for the new layout,
the pipeline architecture, the measurement discipline as the repo's actual
thesis, and the failure modes that cost time to rediscover (NDCG 0.0 = wrong
ids; cwd-relative `data/`/`cache/`; the cache key not covering `retrieve.py`;
NFCorpus's Recall@10 cap of 0.615 and the expected `NDCG@100 < NDCG@10`
inversion). Every command in it was run before committing.

**Non-obvious / quirks:**
- `data/` and `cache/` resolve against the **current working directory**, not
  the package root — entrypoints must be run from the repo root. Chosen over
  anchoring to a computed repo root, which gets fragile once a package is
  installed. Documented in the package docstring and README.
- A **Codex config exists at `~/.codex/config.toml`** (user-level, untouched and
  unread). `/import` would list what's portable into Claude Code.
- Betty added `scratch.ipynb` for experiments; gitignored at her request, along
  with `build/`, `dist/`, `*.egg-info/`, `.pytest_cache/` — `uv` doesn't create
  those but a plain `pip install -e .` would.
- Convention set this session: **no need to mark which `LEARNINGS.md` entries
  Claude wrote.** `CLAUDE.md` updated accordingly.

---

## 2026-08-07 (Phase 0 green — retrieve() fixed, baseline measured, D1 resolved)

Picked up the `retrieve()` WIP handed over on 2026-07-15 and closed out Phase 0.
Betty had finished the implementation but hadn't run it — the session started
from "I know there are probably bugs, but I'm not sure how to test it," so the
work was as much about *building the test* as fixing the code. Everything below
shipped in commit `5af1e0d`.

**The bug (and why it needed a test to find).** `retrieve.py:59` returned
`[{corpus_id: score}, ...]` — a list of single-key dicts, keyed by
`semantic_search`'s `corpus_id`, which is a positional index into the encode
order rather than a BEIR `doc_id`. Two errors in one line: wrong container, and
the step-4 `doc_ids[...]` lookup skipped entirely. The container error would have
surfaced quickly; the id error would not have. Nothing raises — `pytrec_eval`
simply finds zero overlap with the qrels and prints `NDCG@10: 0.0000`, which
reads as "MiniLM is bad at medical retrieval," not "my ids are wrong." A bug that
is invisible *because* its output is well-formed. Fixed to
`{doc_ids[hit["corpus_id"]]: float(hit["score"]) for hit in hits[i]}` — the
`float()` matters, pytrec_eval wants Python floats, not torch scalars. The stale
`TODO(human)` block in the docstring was replaced with a note on why
normalization makes cosine == dot product and why the id remap is needed.

**The test, and the reasoning behind its shape** (`test_retrieve.py`, new). A
five-document fixture — two medical docs matching the two queries, three
obvious distractors (sourdough, Iceland, cricket) — with hand-written qrels. It
runs in ~5s with no download and no pytest dependency (bare asserts; pytest
isn't in `requirements.txt` and wasn't worth adding to get a test running). Doc
ids are deliberately non-sequential strings (`MED-001`…) so a leaked row index
cannot accidentally pass. Each assertion targets one failure mode so a red test
names its own cause: dict-not-list, keys-exist-in-corpus, values-are-float,
top_k honored, top_k > corpus clamps, and rank-1 correctness (which would catch
transposed args to `semantic_search`). The load-bearing one is the last: it runs
BEIR's real `EvaluateRetrieval` over the fixture, which must score **exactly
`NDCG@10 = 1.0`**. Asserting on your own dict's shape is not the same as the
downstream consumer accepting it, and perfect-or-zero is maximally diagnostic —
a mid-range number would have said nothing. Deliberately *not* a stub model: a
stub returning numpy would have masked any mismatch between `convert_to_tensor=True`
and what `semantic_search` expects, and MiniLM on five docs is instant anyway.
Ran it against the broken code first to confirm it actually fails (`got list`),
then fixed. Sequencing is the point — fixture first, full BEIR run second; run
the real eval first and a low number has four possible causes, run it second and
it has one interpretation.

**Baseline (BEIR/nfcorpus, `all-MiniLM-L6-v2`, top-100):** NDCG@10 **0.3159**,
MRR@10 **0.5046**, Recall@100 **0.3115**. Weave tracing was live for the run
(`wandb.ai/bettyjiamanwu/retrieval-lab`, call `019fdf0a-4e1e-7d92-b4ed-f4bda58f3010`).
Recorded in the README ablation table.

**The recall-ceiling check (a near-miss worth remembering).** `evaluate.py`'s
headroom logic printed "Recall@100 0.311 is low — raising retrieval recall is the
lever," and that verdict was very nearly taken at face value. But NFCorpus was
picked *for* dense qrels, and dense qrels can cap recall arithmetically — a query
with 200 relevant docs caps Recall@100 at 0.50 no matter how good the retriever
is. Two hints pointed that way: MRR@10 (0.505) sitting far above NDCG@10 (0.316),
the signature of "finds one good doc fast, misses the tail," and NDCG@100 (0.295)
sitting *below* NDCG@10. So the ceiling was computed directly from the qrels
(`sum(min(n,100)/n)/|Q|`, counting only rel > 0): **0.9647**. Median 16 relevant
docs/query, mean 38, max 475, but only 22 of 323 queries exceed 100 — the top-100
cutoff barely binds. **The verdict stands as written**: 0.311 against a 0.965
ceiling is genuinely low retrieval, ~32% efficiency, and raising recall is a real
Phase 2 lever. The hypothesis was wrong, but the check was cheap and the byproduct
is useful — **Recall@10 caps at 0.6146**, not 1.0, which matters before that
column ever lands in an ablation table. The NDCG@100 < NDCG@10 inversion is
likewise expected with dense qrels (ideal-DCG denominator grows faster than
retrieved gain), not a bug.

**D1 resolved — NFCorpus, on empirical grounds.** The scaffold defaulted to it
and the headroom check confirmed it: NDCG@10 0.316 is far from 1.0, and the
top-100 holds roughly 2× the relevant docs the top-10 surfaces (Recall@10 0.155 →
Recall@100 0.311), so there is real material for Phase 1's reranker to reorder.
No reason to switch to SciFact.

**Next move decided, not yet done: measure Phase 1's ceiling before building it**
(now D3). This applies the Phase 0 discipline already recorded in `LEARNINGS.md`
one level down — an *oracle rerank*: take the top-100 the bi-encoder already
returns, reorder it perfectly using qrels as ground truth, compute NDCG@10. Pure
sort, no model inference, and it hard-bounds everything a real cross-encoder can
deliver. It also earns a third ablation column (off-the-shelf / +rerank / oracle),
which is a stronger portfolio artifact than the lift alone — most RAG projects
report the improvement and never the ceiling.

**Also updated:** `README.md` — Phase 0 ablation row filled in, a
headroom-verdict section added with the ceiling numbers and the two
reading-notes caveats, and `test_retrieve.py` documented in the run instructions
with *why* the fixture must precede the real eval. `LEARNINGS.md` — two new
devlog entries (the silent-0.0 bug and its test design; the check-the-ceiling
lesson). Both were written by Claude in the file's existing voice; Betty may want
to rewrite them in her own.

**Non-obvious:** `evaluate.py` recomputes retrieval from scratch on every run
(~1 min of encoding on this machine). Phase 1 must rerank the *same* top-100
Phase 0 produced, so caching retrieval results to disk is likely the first
refactor — otherwise every ablation row repays the encode cost and there's no
guarantee the reranker saw an identical candidate set. Also: with
`WANDB_API_KEY` set, `@op` on `retrieve` serializes the full corpus arg and all
result entries — it completed fine and wasn't slow, but it's noisy to debug
through.

---

## 2026-07-15 (Phase 0 scaffold + repo created)

Kicked off **retrieval-lab**, a build-to-learn project chosen off the `llm-wiki`
gap-map (concepts studied but never built: RAG, reranking, learning-to-rank,
embeddings, LoRA, fine-tuning). Full plan in `docs/plan.md` (mirrors the vault
note `notes/retrieval-lab`).

**Shipped:**
- Phase 0 scaffold — commit `3b4afee`. Flat layout at repo root: `data.py`
  (BEIR/NFCorpus load with qrels), `evaluate.py` (main: load → retrieve → BEIR
  metrics NDCG@10/MRR@10/Recall@100 → headroom check), `observability.py`
  (Weave init + `@op` import-shim lifted from the mise pattern), `retrieve.py`
  (Phase 0 bi-encoder — **left as a `TODO(human)` for Betty to implement**).
- `docs/plan.md` — commit `41a2b17`. Repo-adapted copy of the vault plan
  (wiki-links flattened, repo-layout section corrected to the actual flat tree).
- GitHub repo created **private**: https://github.com/JiamanBettyWu/retrieval-lab
  (`origin`, `main` tracks `origin/main`).

**WIP handed to Betty:** `retrieve()` in `retrieve.py`. She did step 1 (built
`doc_ids`/`doc_texts`/`query_ids`/`query_texts`) + a bare `model.encode()` stub;
the `raise NotImplementedError` marker is still in place. Remaining: encode with
`normalize_embeddings=True` → `st_util.semantic_search(top_k)` → map each hit's
`corpus_id` (a row index) back to the real BEIR `doc_id`, returning
`{query_id: {doc_id: score}}`. She'll finish it later today.

**Decisions with reasoning (baked into the scaffold):**
- **Evaluate on a labeled BEIR benchmark, not the wiki.** The wiki is too small
  and clean to *score* — near-perfect base retrieval = no headroom to show a
  reranker/fine-tune helping, and no qrels to compute NDCG. BEIR ships qrels and
  is hard enough to leave headroom. The wiki is reserved for the Phase 3 *demo*
  (qualitative, no labels). ("Evaluate ≠ demo.")
- **Baseline encoder = general `all-MiniLM-L6-v2`, deliberately NOT the
  MS-MARCO-tuned bi-encoder.** Phase 2 LoRA-fine-tunes on MS MARCO; starting from
  an already-MS-MARCO-specialized model would flatten the ablation. General
  baseline = headroom for the fine-tune to visibly help.
- **bi-encoder ≠ cross-encoder** (the name-collision trap): `all-MiniLM-L6-v2`
  (retriever, emits embeddings) and `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (Phase 1 reranker, emits a score) share the MiniLM-L6 backbone but are
  different model *types* — the reranker cannot do first-stage retrieval.
- **Zero-shot generalization is the headline claim** (Phase 2): fine-tune on
  MS MARCO, evaluate on an *unseen* BEIR domain — a stronger portfolio line than
  improving one's own test split.
- **Flat repo layout** (files at root, not `src/`) — small lab; also sidesteps
  the package-import traps mise hit. Weave is optional via the shim, so Phase 0
  runs with no weave and no `WANDB_API_KEY`.
- **Private repo for now**, flip public when Phase 0 is presentable (mise's path).

**Non-obvious / quirks:**
- **Two `gh` hosts are both "active"** (`github.com` = JiamanBettyWu,
  `github.gatech.edu` = jwu809). Repo was created with `GH_HOST=github.com`
  forced, to avoid landing on the GT Enterprise host. Use that env var for any
  future `gh` op on this repo.
- **The plan lives in two places** — `docs/plan.md` here and `notes/retrieval-lab`
  in the vault. Per the vault's code-project convention ("the repo is the
  source"), once this graduates to a `wiki/projects/` page the **repo's copy
  becomes canonical** and the vault harvests from it. Watch drift direction.
- `data/` (BEIR downloads) and `.venv/` are gitignored.
