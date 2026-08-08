# retrieval-lab

A retrieval-quality lab: build a two-stage RAG retriever, LoRA-fine-tune its
encoder, and **prove** each improvement on a labeled benchmark (BEIR) with
metrics traced in Weave. Then demo the validated pipeline as an "ask my second
brain" front door over a personal wiki.

> Full plan: [`docs/plan.md`](docs/plan.md). Current state and open decisions:
> [`TODO.md`](TODO.md).

## Phase 0 — baseline retriever + eval harness

An off-the-shelf **bi-encoder** (`all-MiniLM-L6-v2`) over a small BEIR set
(NFCorpus), scored against BEIR's `qrels`. Produces two things:

1. a **baseline** (NDCG@10, MRR@10) to improve on in later phases, and
2. the **headroom check** — is NDCG@10 well below 1.0 (room for a reranker to
   help), and is Recall@100 high enough that the reranker even has the right
   docs to reorder?

### Setup

```bash
uv venv && uv pip install -e '.[dev]'    # add ,tracing for Weave: -e '.[dev,tracing]'
export WANDB_API_KEY=...                 # optional — tracing is inert without it
```

An editable install, so `retrieval_lab` is importable from anywhere while the
source stays live under `src/`.

**To reproduce the numbers in the ablation table, use the lockfile instead:**

```bash
uv sync --locked --all-extras            # exact versions from uv.lock
```

`pyproject.toml` gives ranges (`sentence-transformers>=3.0`); `uv.lock` pins the
resolved version of every dependency and all ~180 transitive ones. Metrics move
with model and tokenizer versions, so a fresh resolve can produce a different
NDCG@10 and leave you unable to tell a real regression from a newer `torch`.
The command above is what makes "run this and you get 0.3412" checkable rather
than asserted — `uv pip install` ignores the lock and resolves fresh.

Two things to know before running it. **`uv sync` prunes**: it makes the venv
*exactly* match the lock, uninstalling anything else, where `uv pip install` only
ever adds. And it treats extras as exact too — `--extra dev` alone would remove
Weave, hence `--all-extras`. `--locked` fails rather than silently re-resolving
if `uv.lock` has drifted from `pyproject.toml`.

### Run

Run from the repo root — `data/` and `cache/` resolve against the working
directory.

```bash
pytest                                            # both fixture suites, ~6s, no download
python -m retrieval_lab.evaluate --dataset nfcorpus   # the baseline
python -m retrieval_lab.oracle   --dataset nfcorpus   # the ceiling a reranker could reach
python -m retrieval_lab.rerank   --dataset nfcorpus   # Phase 1 — cross-encoder rerank
python -m retrieval_lab.rerank   --dataset nfcorpus --breakdown   # + the analysis below
```

Every number in the Phase 1 section below comes from that last command — the
per-query win/loss counts and the bucket table included, so the analysis is
reproducible rather than a claim.

Retrieval results are cached to `cache/` per (dataset, model, top_k), so only
the first run pays the ~1 min encode — and every later stage provably reranks
the *same* candidate set rather than a re-encoded one. The cache key does not
cover `retrieve.py`'s contents: pass `--refresh` after editing it.

**Run `pytest` before trusting any number from `evaluate.py`.** The suites are
five-document fixtures with known-correct answers, and each test targets one
failure mode — so a red run names its own cause, where the full BEIR run
collapses every bug into a single low NDCG. The load-bearing test feeds the
fixture through BEIR's own evaluator, which must score exactly `NDCG@10 = 1.0`:
if `corpus_id` row indices ever leak through unmapped, `pytrec_eval` finds no
overlap with the qrels and reports `0.0` — which reads as "weak model," not
"wrong ids."

## Ablation (fills in as phases land)

| Stage | Model | NDCG@10 | MRR@10 | Recall@100 |
|---|---|---|---|---|
| Phase 0 · bi-encoder | `all-MiniLM-L6-v2` | **0.3159** | **0.5046** | **0.3115** |
| Phase 1 · + rerank | `+ cross-encoder/ms-marco-MiniLM-L-6-v2` | **0.3412** | **0.5675** | **0.3115** |
| Phase 2 · LoRA encoder | fine-tuned on MS MARCO | | | |
| _ceiling_ · **oracle rerank** | _perfect reorder of the same top-100_ | _0.6263_ | — | _0.3115_ |

The last row is not a system — it's the **measuring stick**. `oracle.py` reads
the qrels and orders each query's retrieved candidates perfectly, but may only
*reorder* what retrieval found, never add to it. So `0.6263` is the hard upper
bound on anything a real cross-encoder can reach over these candidates, and the
Phase 1 row should be read against it rather than against 1.0.

### Phase 1 result — +0.0253 NDCG@10, or **8.1% of the available headroom**

The cross-encoder moved NDCG@10 from `0.3159` to `0.3412`. Two ratios worth
keeping apart, since they coincide near 8% by accident:

- `0.0253 / 0.3159` = **+8.0% relative to baseline** — the usual way to report a gain.
- `0.0253 / 0.3104` = **8.1% of the measured headroom** — the number this repo cares
  about, because the denominator is the oracle ceiling rather than an implied 1.0.

Modest either way. The interesting part is what the split does and doesn't explain.

| | Phase 0 | Phase 1 | Δ |
|---|---|---|---|
| NDCG@10 | 0.3159 | 0.3412 | **+0.0253** (+8.0% rel) |
| MRR@10 | 0.5046 | 0.5675 | **+0.0630** (+12.5% rel) |
| Recall@10 | 0.1550 | 0.1570 | +0.0021 (flat) |
| Recall@100 | 0.3115 | 0.3115 | 0 — *by construction* |

**MRR rises sharply while Recall@10 stays flat.** The reranker is pulling *one*
relevant document to rank 1, but it is not bringing *more* relevant documents
into the top 10. That much is measured.

Per-query, the effect is real but noisy — **110 queries improved** (mean
`+0.1686`), **95 degraded** (mean `−0.1092`), 118 unchanged. Wins outweigh losses
in magnitude rather than in count, and the worst single regression is `−0.4477`,
with no catastrophic collapse of the kind that would indicate a bug rather than
a limitation. **51 queries are unwinnable** — their top-100 contains nothing
relevant, so they score 0.0 for any reranker.

#### A candidate explanation, and what testing it cost

The obvious story: a train/test mismatch in **qrel density**. MS MARCO labels
roughly one relevant passage per query, so the cross-encoder learns "find the
single best answer"; NFCorpus has a median of 16 and NDCG@10 pays for filling
all ten slots. It fits the MRR-up/Recall-flat shape exactly.

It also makes a falsifiable prediction: **capture should fall as more relevant
docs become available.** Testing that took three passes — a broken estimator, a
fix, and a confound check that undid the conclusion — and the wrong turns are
kept here because each one is a trap that generalizes past this repo.

Take the 245 queries with any headroom and normalize each by its own
`oracle − baseline`, so buckets with different ceilings stay comparable. Then
the question is how to aggregate within a bucket:

| relevant docs available | queries | **pooled** | median | mean |
|---|---|---|---|---|
| 1–2 | 69 | **+14.8%** | +0.0% | +16.4% |
| 3–5 | 54 | **+19.0%** | +0.0% | **−22.0%** |
| 6–10 | 47 | **+9.2%** | +4.7% | +7.1% |
| 11+ | 75 | **−1.9%** | −2.9% | −12.9% |
| all | 245 | **+8.7%** | +0.0% | −2.8% |

**The mean column is untrustworthy, and it is the one the obvious
implementation gives you.** A per-query ratio divides by that query's headroom;
when the headroom is `0.016` and the reranker loses a normal amount, the ratio
is `−10`. Two such queries drag the 3–5 bucket from `+19.0%` to `−22.0%` and
flip its sign. **Pooled** — `sum(gain) / sum(headroom)` across the bucket —
never divides by an individual near-zero, weights each query by what was
actually at stake, and is built exactly like the corpus-level 8.1% figure, so
the two are comparable. Its `all` row (`+8.7%`) sanity-checks against it.

Read the pooled column and capture does fall off at the top end — the reranker
**actively hurts** the densest bucket (`−1.9%`). But the sequence is a hump, not
a slope: `+14.8% → +19.0% → +9.2% → −1.9%`. The 1–2 bucket is exactly where the
density story predicts capture should be *highest*, and it sits below 3–5. The
sharpest form of the prediction fails even with the estimator fixed.

A rank-based test over all 245 queries, immune to the outliers that broke the
mean, does find a monotone component:

```
Spearman rho(availability, capture) = −0.1912
one-sided permutation p ≈ 0.001    (n = 245, 20k shuffles)
```

(The entrypoint prints `0.0010` reproducibly — the RNG is seeded and the query
order sorted, deliberately, so a reported statistic can't drift. But `p` is
still a Monte Carlo *estimate*: 20k shuffles put its standard error near
`0.0002`. Read it as "about one in a thousand," not four significant figures.)

**Then the confound check dissolved the attribution.** Availability is not the
only thing that predicts capture, and it is badly entangled with the others:

| | rho with capture |
|---|---|
| relevant docs available | **−0.1912** |
| baseline NDCG@10 | **−0.1924** |
| headroom (`oracle − baseline`) | +0.0929 |

Headroom isn't the culprit — partial out its effect and availability's
correlation survives at `−0.2129`. But **baseline NDCG explains the data exactly
as well**, and availability correlates with it at `+0.5716`. Partial out either:

```
partial rho(availability, capture | baseline)  = −0.1009
partial rho(baseline,     capture | availability) = −0.1032
```

**Symmetric.** Neither variable retains explanatory power once the other is
controlled, so this data cannot say which is responsible. "The reranker captures
less where more relevant docs are available" and "the reranker captures less
where the bi-encoder was already good" are, here, the same finding wearing two
labels — and the second needs no story about MS MARCO at all.

So the correlation establishes an association and **cannot attribute it.**
Separating two predictors correlated at 0.57 needs either an intervention or a
prediction that distinguishes them. The next section is the second one.

#### The test that does discriminate: MRR against NDCG

The correlation compared *magnitudes* — capture is smaller here than there —
and "the bi-encoder was already good" explains a magnitude difference perfectly
well. A **qualitative** prediction is harder to explain away.

If the cross-encoder is specifically a *find-the-single-best-answer* model, then
on dense queries it should keep improving **MRR** — which only looks at the
first relevant hit — while damaging **NDCG@10** and the count of relevant docs
in the top ten, both of which reward filling all ten slots. Opposite signs, same
queries. A reranker that were merely *weak*, adding noise to an already-good
ordering, would push all three down together.

| availability | n | ΔNDCG@10 | ΔMRR | rel@10 before | after |
|---|---|---|---|---|---|
| 1–2 | 84 | +0.0422 | +0.1004 | 0.79 | 0.83 |
| 3–5 | 55 | +0.0635 | +0.0968 | 1.82 | 2.00 |
| 6–10 | 47 | +0.0466 | +0.0766 | 2.62 | 2.91 |
| **11+** | 86 | **−0.0123** | **+0.0289** | **5.78** | **5.44** |

**The prediction holds on the dense bucket, and only there.** With 11+ relevant
docs available the reranker still improves rank 1 (`ΔMRR +0.0289`) while
*evicting* 0.34 relevant documents from the top ten and lowering NDCG@10. Every
other bucket improves on all three.

That dissociation is real evidence for the training-objective story in a way the
correlation was not: noise does not selectively improve the first position. The
model is confidently promoting one document and demoting others that had earned
their slots.

Two honest limits. The dense bucket is still the high-baseline bucket, so an
interaction can't be ruled out — this is stronger evidence, not proof. And the
effect is small in absolute terms (`−0.34` docs, `−0.0123` NDCG). The
intervention that would settle it remains a reranker trained on dense qrels.

#### Where the gain actually comes from

Capture is a percentage; the headline `+0.0253` is a sum of points. Those can
rank buckets differently, so it is worth attributing the score movement
directly:

| availability | n | sum gain | % of total | headroom | capture |
|---|---|---|---|---|---|
| 0 (unwinnable) | 51 | +0.0000 | 0.0% | 0.0000 | n/a |
| 1–2 | 84 | +3.5427 | 43.4% | 24.7608 | +14.3% |
| 3–5 | 55 | +3.4952 | 42.8% | 18.3583 | +19.0% |
| 6–10 | 47 | +2.1896 | 26.8% | 23.7778 | +9.2% |
| **11+** | 86 | **−1.0605** | **−13.0%** | **33.3467** | **−3.2%** |

**86% of the gain comes from queries with 1–5 relevant docs available.** The
dense bucket doesn't merely underperform — it *subtracts* 1.06 points, cancelling
13% of what the rest earned, despite having the **largest headroom in the
dataset** (33.3 points). The reranker was handed the biggest opportunity here
and moved backwards.

**The concrete Phase 2 lever:** if the reranker simply did *nothing* on dense
queries, NDCG@10 would be `0.3445` instead of `0.3412` — a 13% larger gain, from
a routing rule rather than a better model. Better still, given the MRR result:
*blend* cross-encoder and bi-encoder scores rather than replacing, keeping the
rank-1 improvement without the eviction. (The usual caveat applies — a rule keyed
on "dense" and one keyed on "baseline was good" would select nearly the same
queries, so this would work without confirming why.)

One further caveat the table surfaces: the median query captures **+0.0%**. The
corpus-level `+0.0253` is carried by a minority of queries; the typical one is
untouched. That is invisible in the headline number.

**Recall@100 is identical to five decimal places, and that is a test, not a
coincidence.** Reranking may only reorder the candidate set, never change it, so
any movement in that column would mean the reranker had truncated its output and
silently broken comparability with Phase 0. The equality is stronger evidence
than the fixture suite, because it holds over the real 323-query run.

The remaining `0.6263 − 0.3412 = 0.2851` is still reachable by a *better*
reranker over these same candidates. The `1 − 0.6263` above that is not — it is
recall, and it belongs to Phase 2.

#### The pre-build argument, and where it held up

Recorded before Phase 1 ran, kept here rather than rewritten: available headroom
was **+0.3104 — a 98% relative gain over baseline** — meaning the bi-encoder's
top-100 already *contains* roughly the right documents and is simply ordering
them badly, "precisely the failure a cross-encoder repairs."

The premise held; **the conclusion only partly did.** The misordering was real
and the ceiling was correctly measured, but an off-the-shelf MS MARCO
cross-encoder repairs only the *first-position* part of it (MRR `+0.0630`) and
barely touches the *fill-the-top-10* part (Recall@10 `+0.0021`). "A cross-encoder
repairs misordering" turned out to be too coarse a premise: misordering has
kinds, and this model reliably fixes only one of them. The bucket table above
gives weak support to qrel density as the reason, without settling it.

Recall@10 says it more plainly: reordering alone lifts it **0.155 → 0.276**,
nearly to Recall@100's 0.3115. Almost every relevant document the retriever
found is *already in the top-100 but buried below rank 10*; perfect ordering
pulls essentially all of them into the top 10.

`oracle.py` asserts two invariants on every run: that the oracle beats the
baseline on each query individually, and — the stronger claim — that its score
equals an **arithmetic ceiling derived independently of the sort under test**
(`0.626292`, matching to floating point). The first alone would pass for any
improvement, including one leaving gain on the table; only the second
establishes that `0.6263` is genuinely the maximum.

### Phase 0 headroom verdict — proceed

- **NDCG@10 0.316** — far below 1.0, so a reranker has room to show lift.
- **Recall@100 0.311** against a **0.965 achievable ceiling** (NFCorpus qrels are
  dense: median 16 relevant docs/query, but only 22 of 323 queries exceed 100, so
  the top-100 cutoff barely binds). Retrieval recall is genuinely low, not a
  denominator artifact — which makes raising recall a real Phase 2 lever.
- **Top-100 holds ~2× the relevant docs the top-10 surfaces** (Recall@10 0.155 →
  Recall@100 0.311), so there is material for Phase 1's reranker to reorder.

Two reading notes so these aren't mistaken for bugs: `NDCG@100` (0.295) sits
*below* `NDCG@10` (0.316) because dense qrels grow the ideal-DCG denominator
faster than retrieved gain; and **Recall@10 caps at 0.615**, not 1.0, since the
median query has 16 relevant docs and only 10 slots.

## Layout

```
src/retrieval_lab/
├── data.py           # BEIR download + load (ships the qrels labels)
├── retrieve.py       # the bi-encoder retrieval (Phase 0 core)
├── cache.py          # on-disk retrieval cache — pins the candidate set across phases
├── evaluate.py       # entrypoint: load → retrieve → BEIR metrics → headroom check
├── oracle.py         # entrypoint: the perfect-rerank ceiling (Phase 1's upper bound)
├── rerank.py         # entrypoint: cross-encoder rerank of the cached candidates (Phase 1)
└── observability.py  # Weave init + @op shim (works with or without weave)
tests/                # fixture suites, no download required
docs/plan.md          # the full multi-phase design doc
```

Gitignored and rebuildable: `data/` (BEIR downloads) and `cache/` (retrieval
runs — delete to force a recompute, or pass `--refresh`).
