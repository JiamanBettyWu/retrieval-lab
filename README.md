# retrieval-lab

A retrieval-quality lab: build a two-stage RAG retriever, LoRA-fine-tune its
encoder, and **measure** every change against a labeled benchmark (BEIR) with
metrics traced in Weave — including the changes that turn out to make things
worse, which is what Phase 2 did. Then put the validated retriever behind a
generator and measure *that* the same way — grounded answers scored by an
LLM-as-judge calibrated against hand labels.

> **Amended 2026-08-14:** the wiki demo UI (originally Phase 3) is **retired**,
> not deferred. Its deliverable was a front door onto the retriever Phase 2 had
> just measured to be worse, and a demo is not a measurement. Generation was
> promoted to Phase 4 and is the next real phase. See
> [`docs/plan.md`](docs/plan.md).

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

Three optional extras, all opt-in: **`dev`** (pytest), **`tracing`** (Weave),
and **`train`** (`peft` + `accelerate` + `datasets` — Phase 2 fine-tuning only,
so reproducing a *retrieval* number never pulls a training stack).

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

# Phase 2 — fine-tune, then point the same three entrypoints at the adapter
python -m retrieval_lab.finetune --tag r16-a32-lr1e3-100k --triples 100000 --k 10000 --lr 1e-3
python -m retrieval_lab.evaluate --dataset nfcorpus --model models/lora-r16-a32-lr1e3-100k
python -m retrieval_lab.oracle   --dataset nfcorpus --model models/lora-r16-a32-lr1e3-100k
python -m retrieval_lab.rerank   --dataset nfcorpus --model models/lora-r16-a32-lr1e3-100k
```

`--model` takes a hub id or a local adapter path and **keys the retrieval
cache**, so the baseline and every fine-tune keep separate candidate sets. The
`oracle` line is not optional bookkeeping: a ceiling belongs to the candidate set
it was measured over, so a new retriever needs a new ceiling.

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
| Phase 2 · LoRA encoder | fine-tuned on MS MARCO (r=16, α=32, lr 1e-3) | **0.2725** | **0.4482** | **0.2950** |
| Phase 2 · + rerank | LoRA encoder `+` the same cross-encoder | **0.3407** | **0.5705** | **0.2950** |
| _reference_ · off-the-shelf | `msmarco-MiniLM-L6-cos-v5` (fully MS MARCO tuned) | _0.2584_ | _0.4554_ | _0.2342_ |
| _ceiling_ · **oracle rerank** | _perfect reorder of the Phase 0 top-100_ | _0.6263_ | — | _0.3115_ |
| _ceiling_ · **oracle rerank** | _perfect reorder of the Phase 2 top-100_ | _0.6096_ | — | _0.2950_ |

The ceiling rows are not systems — they're the **measuring stick**. `oracle.py`
reads the qrels and orders each query's retrieved candidates perfectly, but may
only *reorder* what retrieval found, never add to it. So `0.6263` is the hard
upper bound on anything a real cross-encoder can reach over the Phase 0
candidates, and the Phase 1 row should be read against it rather than against
1.0.

**There are two ceilings because a ceiling belongs to a candidate set, not to a
dataset.** Phase 2 changes the retriever, so it changes what reaches the
reranker, so the bound moves with it — `0.6263 -> 0.6096`. Reusing the Phase 0
number would have silently scored Phase 2 against a bound it never had.

**Phase 4 is not in this table, on purpose.** Its columns are retrieval metrics,
and Phase 4's result is judge–human agreement on hand labels — a different
measurement with a different denominator, and stacking a κ into an NDCG table
would imply a comparability that does not exist. Its numbers live in
[Phase 4](#phase-4--generation-and-a-judge-that-had-to-be-measured-too) below.

**Phase 2 lost, and the row stays.** The project's rule is that every NFCorpus
number obtained gets reported; a table that only fills in when a phase wins is a
table you cannot trust when it says a phase won. What the loss turned out to
mean is below.

**This table is NFCorpus only. The Phase 2 loss is not.** It replicated on
**SciFact** and **FiQA** with identical ordering — base > LoRA > fully-MS-MARCO,
no exception on any of the three — which is what moved the explanation from
"biomedical domain gap" to training-distribution breadth. The three-dataset
table is in the Phase 2 section.

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

### Phase 2 result — the fine-tune lost **−0.0434 NDCG@10**, and the reranker absorbed **99%** of it

LoRA (r=16, α=32, `query`+`value`, 147,456 trainable params = 0.65% of the model)
on 100k MS MARCO triples with `MultipleNegativesRankingLoss`, evaluated zero-shot
on the untouched BEIR set. **The fine-tune worked on its own objective**: MS MARCO
dev loss fell from the base encoder's **0.6597** to **0.3129** for the best
config, and every config beat the base by a wide margin (within the
learning-rate search the spread is much smaller, `0.3960 -> 0.3129`). NFCorpus
NDCG@10 went the other way: `0.3159 -> 0.2725`, **−13.7% relative**.

That contrast is the phase's finding in one line — real in-domain gain, paid for
out of domain. Reproduce the base number with
`python -m retrieval_lab.finetune --dev-loss`.

**Before interpreting a bad number, rule out a broken one.** `NDCG@10 = 0.0000`
has a known cause in this repo; `0.2725` doesn't, so it needed its own checks.
The saved adapter carries the same three modules as the baseline
(`Transformer -> Pooling -> Normalize`, `max_seq_length=256`), and `oracle.py`
passed both invariants on the *new* candidate set — `oracle >= baseline` on all
323 queries, and `oracle == arithmetic_ceiling` at `0.609627` to six decimals.
The second derives the ceiling independently of the sort under test, so its
agreement over candidates it had never seen says the pipeline is intact and only
the embeddings moved.

Recall@100 fell with it (`0.3115 -> 0.2950`), and so did the ceiling
(`0.6263 -> 0.6096`). **That locates the failure in retrieval, not in ranking.**
Reordering cannot change which documents were found, so an unchanged candidate
set implies an unchanged ceiling; this one moved. Nothing downstream recovers a
document that never entered the top 100.

#### The damage is flat across a 50× learning-rate range

| lr | MS MARCO dev loss | cosine to base | NFCorpus NDCG@10 |
|---|---|---|---|
| — (baseline) | 0.6597 | 1.0000 | **0.3159** |
| 2e-5 | 0.3960 | 0.9757 | 0.2724 |
| 1e-4 | 0.3443 | 0.9438 | 0.2737 |
| 5e-4 | 0.3180 | 0.9473 | 0.2765 |
| 1e-3 | 0.3129 | 0.9294 | 0.2725 |

The baseline's dev loss is measured on the same slice at the same eval batch
size (the last 10,000 rows of the seed-42 shuffle, batch 8); MNRL loss is a
property of a batch, so a number taken at any other batch size is on a different
scale and does not belong in this column.

Across the four fine-tuned configs, dev loss spans 21%. NFCorpus spans 1.5%. The middle column is what makes that
strange rather than merely negative: **embedding drift from the base model does
grow with learning rate** — the four adapters are genuinely, progressively
different — while the damage doesn't move. A quantity that stops responding to
its driver has hit a floor. Had the damage been proportional to distance moved,
1e-3 (which drifted three times as far as 2e-5) would be three times as damaged.

`checkpoint-200` of the 2e-5 run bounds the other end. At step 200 — 6.4% of an
epoch, still inside the `warmup_ratio=0.1` ramp — the encoder sits at cosine
**0.9998** to base and scores `NDCG@10 0.3143`, `Recall@100 0.3132`,
`MRR@10 0.5056`: baseline within noise. So the damage is neither instantaneous
nor proportional. **It accrues between step 200 and one epoch, then saturates.**
The shape in between is unmeasured — the claim is "saturates before one epoch",
not "saturates at step N".

Which gives the finding: **zero-shot transfer degradation saturates before
in-domain fit does.** By one epoch, 2e-5 and 1e-3 have paid an identical NFCorpus
penalty despite 2e-5 being visibly undertrained on MS MARCO — its own eval curve
was still descending when it ran out of steps. So the intuitive mitigation, *use
a gentler learning rate to preserve general ability*, is **strictly dominated**:
the same penalty for a worse specialist. The levers that could work are different
in kind — fewer total steps, mixed-domain replay, or less adapter capacity.

#### The control, and why it changes the claim

`msmarco-MiniLM-L6-cos-v5` — same backbone size, MS MARCO tuned by the
sentence-transformers team — scores **0.2584** on NFCorpus. *Below* the
fine-tune, and 0.0575 below the general-purpose baseline.

```
general purpose ──────────── this fine-tune ─────── fully MS MARCO
    0.3159                      0.2725                  0.2584
       └────── −0.0434 ───────────┘                        │
       └───────────────── −0.0575 ─────────────────────────┘
```

The three models order cleanly along one axis of specialisation, and on NFCorpus
the fine-tune paid **75%** of the full penalty. **Nothing was broken.** The
encoder was moved toward MS MARCO, and this is what that move costs. Without the
control, the same numbers equally support *"our recipe is wrong"* — a different
claim, and the wrong one. It cost one evaluation.

It also relocates the floor: the four adapters converge at 0.272 but the axis
runs to 0.258, so the saturation point is likely **the capacity ceiling of
147,456 parameters on `query`+`value`**, exhausted identically by every learning
rate — which predicts a larger `rank`, or adding `key`, would push closer to
0.258. Untested. And the control differs from the fine-tune in several ways at
once (full fine-tune, different loss, different data recipe), so it is a
reference point, not a one-variable comparison; it cannot attribute the 0.0141
gap to LoRA.

#### The mechanism, and the prediction that refuted the first one

**The explanation above was originally "domain gap", and it was wrong.** Kept
here rather than rewritten, because how it was falsified is the point.

The recorded reading of the control was that MS MARCO specialisation costs you
on *biomedical* text — NFCorpus's vocabulary is far from MS MARCO's web queries.
That story makes a testable prediction, so it was **committed to git before the
run** (`66e798c`): on **FiQA**, whose natural-language financial questions look
much more like MS MARCO's, `msmarco-MiniLM-L6-cos-v5` should *beat*
`all-MiniLM-L6-v2`.

It lost — by the **largest margin of the three datasets**, on the set chosen
specifically because it should have reversed.

| dataset | `all-MiniLM-L6-v2` | LoRA r16 lr1e-3 | `msmarco-MiniLM-L6-cos-v5` |
|---|---|---|---|
| NFCorpus | **0.3159** | 0.2725 (−13.7%) | 0.2584 (−18.2%) |
| SciFact | **0.6451** | 0.5736 (−11.1%) | 0.4870 (−24.5%) |
| FiQA | **0.3687** | 0.2784 (−24.5%) | 0.2317 (−37.2%) |

Identical ordering on all three, no exception, no reversal — and the adapter
lands between the two every time (75%, 45% and 66% of the way along the axis).
So the *finding* replicates and the *explanation* does not survive.

**What replaces it: training-distribution breadth, not domain match.**
`all-MiniLM-L6-v2`'s model card records training on **1,170,060,424 sentence
pairs** across ~30 datasets; `msmarco-MiniLM-L6-cos-v5` trained on MS MARCO
alone. The fine-tune took the 1.17B-pair model and pulled it toward a
distribution defined by ~500k triples. **Narrowing the training distribution
costs retrieval quality everywhere** — which is what three datasets with no
exception show, and what the domain framing failed to predict. It also fits the
saturation across a 50× learning-rate range better than the domain story did: a
147,456-parameter adapter can only pull so far however hard it is pushed, so it
*strengthens* the capacity-ceiling reading above rather than competing with it.

The methodological point is worth more than the mechanism. A pre-registered
prediction on a dataset picked to break the hypothesis turns "our explanation
sounds right" into a result either way — and this one cost three evaluations.

#### Where the damage lands, and what the reranker does about it

Per-query against the Phase 0 run: **69 improved** (mean `+0.0895`), **136
degraded** (mean `−0.1487`), 118 unchanged, worst single query `−0.9197`.

| relevant docs/query | n | ΔNDCG@10 | ΔRecall@100 |
|---|---|---|---|
| 1–3 | 62 | −0.0529 | −0.0215 |
| 4–10 | 67 | −0.0225 | −0.0036 |
| 11+ | 194 | −0.0477 | −0.0193 |

Every bucket is down, recall included — **diffuse, not a collapsed subset**, and
specifically *not* the dense-query story the aggregate number alone could not
have ruled out.

Then the part that makes the architecture look good rather than the encoder look
bad. Reranking the degraded candidates gives `0.2725 -> 0.3407`, against Phase 1's
`0.3412` from baseline candidates:

**A first stage 13.7% worse ends up 0.15% worse end to end.** The cross-encoder
doesn't care what order candidates arrive in, only that they arrive — and
Recall@100 fell only 0.0165, so nearly all of them still did. The two-stage
design absorbed almost a full first-stage regression. One dataset, n=1, but it is
the kind of claim this table exists to support.

One figure needs care: headroom capture reads **20.2%** here against Phase 1's
8.1%. That is the denominator moving, not the reranker improving — a worse
initial ordering leaves more headroom. The absolute gain is the honest
comparison: `+0.0253 -> +0.0682`, more work done because there was more mess.

#### What this cost in method, stated plainly

The learning-rate search found its winner **at the edge of the grid twice
running**. It stopped at 1e-3 on a stated rule — returns fell to `−0.005` per
doubling and the dev curve went non-monotone (`0.3325 -> 0.3346 -> 0.3398`
mid-run before recovering, the stability edge showing up as variance rather than
divergence) — which is different from stopping silently.

**No configuration was ever run twice**, so every number here is n=1 with no
estimate of run-to-run variance. The early gaps (0.052, 0.026) are far too large
for that to matter. The `5e-4 -> 1e-3` gap of `0.0051`, which decided the winner,
is exactly the size where it might.

Selection used MS MARCO dev loss only — no hyperparameter was chosen by looking
at NFCorpus. That constraint is why the fine-tune was *able* to lose here: the
selector was a proxy, and this phase measured how far the proxy and the target
can diverge.

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

## Phase 4 — generation, and a judge that had to be measured too

The retriever now feeds `qwen3:8b`, which answers from the top 10 passages or
declines with a sentinel. Scoring those answers needs a judge, and **a judge is
itself a measuring instrument, so it gets the same treatment every other claim
in this repo gets**: validated against hand labels before any number it produces
is believed.

Dataset is `hotpotqa-distractor-pool` rather than NFCorpus — HotpotQA questions
are 2-hop with exactly two gold passages, so the count that survives retrieval
into context (`gold_in_context` ∈ {0,1,2}) is a free difficulty stratum.

### Judge validation — the 4B candidate parsed everything and agreed with nothing

30 generations were hand-labelled on two disjoint axes (a refusal makes no
claims, so scoring it for groundedness would award it a free point): **16
answered rows** labelled `grounded`, **14 refused rows** labelled `refusal_ok`.
Two local candidates were then scored against those labels.

| judge | κ `grounded` | n | 95% CI | raw agree | parse-miss | s/call | digest |
|---|---|---|---|---|---|---|---|
| **`mistral-small`** (23.6B) | **+0.586** | 16 | [+0.09, +1.00] | 13/16 | **0.0%** | 42.5 | `8039dd90c113` |
| `gemma3:4b` (4.3B) | **−0.257** | 16 | [−0.67, +0.14] | 5/16 | **0.0%** | 6.8 | `a2af6cc3eb7f` |

CIs are percentile bootstrap (10,000 resamples). The `grounded` rubric is the
strict one — every clause of the rationale must be supported by the passage it
cites, so a *factually correct* answer still scores `false` when the citation
does not carry it. That wording was sharpened in `b4e6b45`; **the κ above was
produced under the sharpened version**, and reverting it would change what the
number means.

**The finding is not "bigger is better."** `gemma3:4b` emitted well-formed,
parseable rulings on **30/30** rows and still landed below chance, ruling `false`
on 8 of the 11 rows a human labelled `true` — it has the *shape* of a strict
groundedness judgement without the discrimination. Both candidates had passed the
format smoke test 4/4 an hour earlier. **Parse rate is the cheap monitor you
would actually automate, and it cannot tell these two models apart.** Only labels
can. That is the same class of failure as `NDCG@10 = 0.0000` from leaked
`corpus_id`s: well-formed output, wrong meaning, nothing raises.

**This is a selection result, not a size result.** The two candidates differ in
vendor, architecture family and training corpus as well as parameter count — only
quantization is matched (Q4_K_M). Two points differing on four axes cannot
attribute the gap to size; that would need two sizes inside one family
(`gemma3:4b` vs `gemma3:27b`), which was not run. The ranking is secure at n=16;
the value of κ is not — `[+0.09, +1.00]` clears zero and little else.

Decision and reasoning:
[#2](https://github.com/JiamanBettyWu/retrieval-lab/issues/2).

### Generator behaviour — refusal tracks the evidence it was given

100 questions, prompt v2, temperature 0, split by how much gold retrieval
actually delivered:

| gold passages in context | queries | refusal rate | 95% CI |
|---|---|---|---|
| 2 of 2 | 51 | **9.8%** | [4.3%, 21.0%] |
| 1 of 2 | 41 | **63.4%** | [48.1%, 76.4%] |
| 0 of 2 | 8 | **87.5%** | [52.9%, 97.8%] |
| all | 100 | 38.0% | [29.1%, 47.8%] |

Monotonic, and the full-gold and partial-gold intervals do not overlap: **as
retrieval degrades this generator declines rather than invents**, which is the
safe half of the failure space.

**Descriptive, not causal.** The three strata contain *different questions*, so
"worse retrieval → more refusal" is confounded with "harder question → more
refusal". Separating them needs the paired design in
[`docs/plan.md`](docs/plan.md), where the query sample is held fixed across
retrieval configs.

**Two caveats that belong next to any number from this axis.** First, the
over-refusal count is a **lower bound**: `refusal_ok` scores a refusal against
what the passages establish, and the axis has only 2 minority rows, so its κ
(`mistral-small` 0.632, `gemma3:4b` −0.105, n=14) is reported as **descriptive
only** — 36% and 43% of bootstrap resamples respectively contain no minority row
at all. Second, refusals have at least two mechanisms — missing context, and
*misreading present context* — and this curve cannot separate them. A row where
both gold passages were present, the rationale names the answer, and the model
refuses on the question's exact phrasing is filed here as conservatism. It is a
comprehension failure wearing a refusal's clothes.

## Layout

```
src/retrieval_lab/
├── data.py           # BEIR download + load (ships the qrels labels)
├── retrieve.py       # the bi-encoder retrieval (Phase 0 core)
├── cache.py          # on-disk retrieval cache — pins the candidate set across phases
├── evaluate.py       # entrypoint: load → retrieve → BEIR metrics → headroom check
├── oracle.py         # entrypoint: the perfect-rerank ceiling (Phase 1's upper bound)
├── rerank.py         # entrypoint: cross-encoder rerank of the cached candidates (Phase 1)
├── finetune.py       # entrypoint: LoRA fine-tune on MS MARCO (Phase 2); --dev-loss scores
│                     #   any existing model on the dev slice without training
├── hotpot_pool.py    # entrypoint: build the hotpotqa-distractor-pool corpus (Phase 4)
├── generate.py       # entrypoint: answer from the top-n context, or refuse (Phase 4a)
├── fixture.py        # entrypoint: draw, label and verify the hand-labelling sheet (4c.1)
├── judge.py          # entrypoint: --smoke the format, --bakeoff the candidates (4c.2)
└── observability.py  # Weave init + @op shim (works with or without weave)
tests/                # fixture suites, no download required
docs/plan.md          # the full multi-phase design doc
```

Gitignored and rebuildable: `data/` (BEIR downloads) and `cache/` (retrieval
runs, generation batches and judgements — delete to force a recompute, or pass
`--refresh`). Hand-written labels under `data/labels/` are **not** rebuildable
and are the one artifact here that cost human time.
