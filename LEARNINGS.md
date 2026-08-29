# LEARNINGS — retrieval-lab

Append-only devlog, written while it's fresh — findings and the reasoning
behind them, recorded before they turn into hindsight.

## 2026-07 — Phase 0: the headroom check comes before any reranker
Phase 0's real job isn't "build a retriever" — it's to confirm the dataset can
even *show* a reranker helping. Two gates: NDCG@10 must be well below 1.0 (room
to improve), and Recall@100 must be high enough that the reranker has the right
docs to reorder — it can never recover a relevant doc the retriever left out of
the top-100. Cheap experiment that could kill the plan runs first.

## 2026-08-07 — a retrieval bug that scores 0.0 instead of crashing
`retrieve()` returned `{row_index: score}` instead of `{doc_id: score}` —
`semantic_search` hands back `corpus_id`, a positional index into the encode
order, not a BEIR id. Nothing raises. `pytrec_eval` just finds zero overlap with
the qrels and prints `NDCG@10: 0.0000`, which reads as "the model is bad on this
dataset." The bug is invisible *because* the output is well-formed.

The lesson generalizes past this repo: **asserting on your own output's shape is
not the same as the downstream consumer accepting it.** The check that actually
discriminates is a five-doc fixture with hand-written qrels run through the real
`EvaluateRetrieval` — with one relevant doc per query ranked first, it must score
exactly 1.0. Perfect-or-zero is maximally diagnostic; a mid-range number would
have told me nothing. Sequencing matters as much as the assertions: fixture
first, full BEIR run second. Run the real eval first and a low number has four
possible causes; run it second and it has one interpretation.

## 2026-08-07 — check the metric's ceiling before reading it as a failure
`Recall@100 = 0.311` looked like weak retrieval, but on a dataset picked for
*dense* qrels the cap could just as easily have been arithmetic — if a query has
200 relevant docs, top-100 caps recall at 0.50 no matter how good the retriever
is. Computed the actual ceiling from the qrels (`sum(min(n,100)/n)/|Q|`): **0.965**.
So the low recall is real, and "raise recall" is a genuine lever. But the same
calculation shows **Recall@10 caps at 0.615** — worth knowing before reporting it
in an ablation table against an implied 1.0. Any metric with k in its name has a
denominator worth checking before you interpret it.

## 2026-08-08 — two metrics disagreeing usually means two denominators
Recall@100 0.3115 and oracle NDCG@10 0.6263 look contradictory — poor retrieval,
decent ranking — until you notice they measure against different targets.
Recall@100 is judged against *every* relevant doc (median 16 per query, mean 38).
NDCG@10's IDCG counts only the best 10, so everything past ten slots is free. The
log discount widens the gap further: over 10 slots `1/log2(rank+1)` sums to 4.544,
and ranks 1–3 alone contribute 2.131 — **the top three positions carry 47% of all
NDCG@10 weight.** So the median query, with only 3 of its relevant docs anywhere
in the top-100, can still reach ~0.63 once those three are placed first.

The corollary is the useful part: a low metric is not automatically the thing to
fix. 51 of 323 queries have *zero* relevant docs in the top-100 and score 0.0 no
matter how good the reranker gets, while 88 already hold enough to fill a perfect
top-10. Recall work isn't uniformly valuable — its concentrated win is emptying
that zero bucket. Averages hide which queries the work would actually help.

## 2026-08-08 — build the cheap lever first when levers compose
Tempting conclusion from the above: retrieval is the real problem, so fix it
before reranking. Wrong ordering. The +0.3104 of headroom is available from
reordering *today's* candidates, so improving retrieval first doesn't unlock that
gain — it delays collecting it. The two levers compose (better retrieval raises
the ceiling a reranker operates under; it never replaces the reranker), and when
levers compose, order them by cost-to-value: a pretrained cross-encoder is an
afternoon, LoRA fine-tuning is days. Cheap-first also de-risks — if the reranker
underdelivers against a ceiling you already measured, the architecture itself is
in question, and that's worth knowing before spending days training.

## 2026-08-08 — the reranker underdelivered, and the split says why
Phase 1 landed +0.0253 NDCG@10 (0.3159 → 0.3412): **8.1% of the 0.3104 headroom**
the oracle said was there. The previous entry ended by calling that exact
scenario the reason to build cheap-first — underdeliver against a measured
ceiling and the architecture is in question. So: which part is in question?

The averaged NDCG can't say. The metric split can. MRR@10 rose `0.5046 → 0.5675`
(+12.5% relative), while Recall@10 moved `0.1550 → 0.1570` — flat. The reranker
reliably pulls *one* relevant doc to rank 1 and brings essentially no additional
relevant docs into the top 10.

**Then I nearly shipped an explanation I hadn't tested.** The story that fits
that shape is a train/test mismatch in qrel density: MS MARCO labels ~1 relevant
passage per query, NFCorpus has a median of 16, so the model optimizes "find the
single best answer" and NDCG@10 pays for filling ten slots. Tidy, plausible, and
I had it written into the README before noticing it was an assertion sitting in
a repo whose entire thesis is *don't assert, measure*.

It makes a prediction, and every input was already on disk: if density is the
cause, captured headroom should fall as more relevant docs become available.
Bucketing the 245 queries with any headroom, each normalized by its own
`oracle − baseline`, then averaging within buckets: **1–2: +16.4% · 3–5: −22.0%
· 6–10: +7.1% · 11+: −12.9%.** No trend. I wrote "the prediction isn't borne
out" into the README and considered it closed.

**That was the second mistake, and worse than the first.** The estimator was
broken. A per-query ratio divides by that query's headroom; two queries with
headroom near `0.016` and ordinary-sized losses produced ratios of `−9.3` and
`−10.7` — enough to drag a 54-query bucket from positive to `−22.0%`. Pooling
instead (`sum(gain)/sum(headroom)` per bucket, which never divides by an
individual near-zero) gives **+14.8% · +19.0% · +9.2% · −1.9%**. The 3–5 bucket
flips sign and a decline appears. Spearman on ranks, which outliers cannot move,
puts it at **rho = −0.19, one-sided permutation p ≈ 0.001 (n=245)**. A real
association, in the predicted direction.

**Then the confound check took it away again.** Baseline NDCG@10 correlates with
capture at **−0.1924** — indistinguishable from availability's −0.1912 — and the
two predictors correlate with each other at **+0.5716**. Partial either out and
both collapse to about the same place: `availability | baseline = −0.1009`,
`baseline | availability = −0.1032`. Symmetric. Neither survives controlling for
the other, so "captures less where more relevant docs are available" and
"captures less where the bi-encoder was already good" are the same 245 queries
wearing two labels — and the second explanation needs no story about MS MARCO
at all. (Headroom, the other obvious confound, is innocent: partial it out and
availability *strengthens* to −0.2129.)

At that point the correlation had gone as far as it could: real association,
unattributable cause.

**Betty broke the deadlock by asking for a different kind of prediction.** Her
observation: if the model is trained to surface *one* best answer, then on dense
queries MRR should still improve even as NDCG@10 falls — because MRR only looks
at the first hit and NDCG@10 pays for filling ten slots. That is a *qualitative*
prediction. The whole reason the correlation failed to discriminate is that
"less room to improve" explains any difference in **magnitude**; it does not
predict two metrics moving in **opposite directions** on the same queries.

It holds, and only on the dense bucket:

```
availability     n   dNDCG@10      dMRR   rel@10 before -> after
       1-2      84    +0.0422   +0.1004      0.79 -> 0.83
       3-5      55    +0.0635   +0.0968      1.82 -> 2.00
      6-10      47    +0.0466   +0.0766      2.62 -> 2.91
       11+      86    -0.0123   +0.0289      5.78 -> 5.44
```

With 11+ relevant docs available the reranker still improves rank 1 while
*evicting* 0.34 relevant docs from the top ten. Noise added to a good ordering
would push all three numbers down; instead the first position improves and
coverage degrades. That is a mechanism, not just a correlation.

Final state: association real, and the training-objective explanation now has
evidence a confound doesn't trivially absorb — still short of proof, since the
dense bucket is also the high-baseline bucket.

**The methodological lesson is the one I want to keep.** I spent three passes
trying to make a *correlation* discriminate between two predictors that sit at
0.57 with each other. It cannot, no matter how careful the estimator or how many
partials I stack on it. What broke the tie was changing the question — from "how
much does capture fall?" (magnitude, which both stories predict) to "do two
metrics move in opposite directions?" (a pattern only one story predicts). **When
two explanations are entangled, don't refine the measurement; find the
prediction on which they disagree.**

Two things I'd rather not relearn. **A mechanism that "explains" your numbers is
a hypothesis wearing a conclusion's clothes** — consistency is cheap, because
most wrong explanations are consistent with the evidence that suggested them.
And **a ratio with a small denominator is not a measurement.** Normalizing per
query felt like the careful move; it was the step that injected the error. When
the fix for one bias (buckets with different ceilings) is a division, check what
the denominator does near zero before trusting the mean of the result. Pooling
answers the same question without ever forming the unstable quantity.

A third thing, which is really the same thing one level up: **two correlated
predictors will each "confirm" a story that only one of them, or neither, is
driving.** Availability and baseline quality sit at 0.57 here. Any single
correlation between one of them and an outcome was always going to look like
support for whichever mechanism I had in mind first. The partial correlation
costs four lines and is the difference between a finding and a preference.

The sequence is the embarrassing part and the reason it's written down:
asserted a mechanism untested → "refuted" it with a broken estimator → found it
supported → found the support unattributable → found a prediction that
discriminates. Four of those five states reached the README before the last one
replaced them.

Two findings that survive independent of any of this. **The median query
captures +0.0%** while the corpus gain is `+0.0253` — the aggregate rides on a
minority, and the typical query is untouched. And attributing the gain in
absolute points rather than percentages: **86% of it comes from queries with 1–5
relevant docs available**, while the 11+ bucket *subtracts* 1.06 points, 13% of
what the rest earned, despite holding the largest headroom in the dataset (33.3
points). Suppressing the reranker on dense queries alone would put NDCG@10 at
`0.3445` rather than `0.3412`. Given the MRR result, blending the two scores
instead of replacing looks better still — keep the rank-1 win, drop the eviction.
That is the concrete thing Phase 2 inherits from Phase 1.

The diagnostic that separated "real limitation" from "bug," meanwhile, was
**per-query win/loss, not the mean**: 110 improved (mean +0.169), 95 degraded
(mean −0.109), 118 unchanged, worst regression −0.45. Balanced counts with wins
larger than losses is what a genuinely weak-but-positive signal looks like.
Mostly-improve with a few catastrophic collapses would have meant truncation or
misaligned scores — a different afternoon entirely.

Third thing, cheaper than both: **Recall@100 came out identical to five decimals
across the two runs (0.31145).** Reranking may only reorder the candidate set,
so that column is a free assertion that nothing got truncated to top-10 — and it
holds over all 323 real queries, where the fixture suite covers four docs. Look
for the invariant your pipeline can't help but satisfy, then check it anyway.

## 2026-08-12 — the lockfile earned its keep before it was tested

Phase 2 needs `peft`, `accelerate` and `datasets`, and adding them is the first
change to this repo's dependency set since the numbers in the README were
published. The obvious way to do it is to add the deps and start training. The
useful way is to add the deps, land them in a commit that touches *nothing*
else, and find out what moved before any new code exists to blame.

Nothing moved, and the way that was established is the part worth keeping.
Three checks, increasing in strength:

1. **The lock diff.** `uv lock` added exactly `peft`, `accelerate` and
   `psutil`; no existing package changed version. The remaining ~100 lines of
   diff are dependency *edges* being regrouped, not versions — worth reading
   the diff for `version =` lines specifically rather than eyeballing the
   line count and assuming.
2. **A full `--refresh` recompute.** All seven published figures came back
   identical to four decimals: baseline `0.3159`, rerank `0.3412`, MRR `0.5046`
   → `0.5675`, ceiling `0.6263`, 8.1% of headroom. A cached run would have
   proved nothing here — the cache is the thing under suspicion.
3. **A byte-for-byte diff of the candidate set.** Copied the cached top-100 out
   before refreshing, `cmp`'d it after. **Identical, floats included.** Matching
   metrics make identical candidates near-certain; this makes them a fact,
   which is the same distinction `cache.py` exists to enforce one level down.
   Two encode passes over the same corpus on the same hardware produced
   bit-identical cosine scores — MPS non-determinism is a thing I'd have
   hedged about, and now don't have to.

The reason to bother: this repo's whole claim is that its numbers mean
something. If the lock bump had arrived tangled with `finetune.py`, a metric
shift would have had two candidate causes and no cheap way to separate them —
the same "four possible causes" trap Phase 0 lost an afternoon to with
`NDCG@10 = 0.0000`. Isolating the commit costs one extra `git commit` and buys
an unambiguous bisect point. Do it before the interesting change, not after.

Two smaller things. `uv sync --all-extras` uninstalled an ipykernel tree that
had been `uv pip install`ed ad hoc — exactly the documented behaviour (sync
makes the venv *match* the lock, it doesn't merely add), but the first time it
has actually taken something away here. And `cache/reranked_nfcorpus.json`
turned out to be an orphan: `rerank.py` re-scores unconditionally and never
reads or writes it. Deleted. Retrieval is the *only* cached stage, which is
why getting the Phase 2 checkpoint into the cache key is a sufficient fix and
not merely a partial one — there is no second stale-value channel downstream.

## 2026-08-14 — the leak the smoke test was structurally incapable of finding

Phase 2's training path went in today, and the bug worth writing down was not
in the model. `load_triples` shuffled with seed 42, took the first `n` rows as
training, and then took the dev slice off the **unshuffled** dataset. Both
lines look right in isolation. The overlap:

```
--smoke   n=2,000   k=100    ->     1 of 100 dev rows in training  ( 1.0%)
real run  n=100,000 k=10,000 -> 2,001 of 10,000 dev rows in training (20.0%)
```

The leak scales with `n/502,931`, so it is ~1% at smoke size and 20% at the
size that matters. **The smoke run passed, and could not have done otherwise.**
It measures throughput and proves the path executes — that is its whole job,
and it is worth being explicit that "the smoke test is green" carries no
information about whether the logic is correct. The failure signature is also
nastier than Phase 0's `NDCG@10 = 0.0000`: a leaking dev set does not produce a
suspicious number, it produces a plausible one, and it would have read
*differently* plausible per config depending on how much each memorised —
corrupting the exact signal R1 says hyperparameters must be selected on.

The fix is one word (`dataset` → `dataset_shuffled`), but the durable part is
the invariant. Dev = the **last k rows of the same shuffle**, never
`range(n, n+k)`: the latter is disjoint only for the `n` you happened to use,
so a later 210k run would quietly swallow a dev slice parked at rows
100k–110k. Disjointness should be a property of the definition, not a
coincidence between two row counts. A guard on `n + k` keeps it that way.

Then a free result from measuring rather than assuming: **every row in
`msmarco-bm25/triplet` carries a unique query** — 100,000 distinct queries in
100,000 rows, and zero shared between the train and dev slices. So the split is
query-disjoint, not merely row-disjoint, and dev loss measures generalisation
to unseen questions. It also retires the MNRL false-negative worry from
2026-08-12: two rows for one query cannot land in a batch here, because no
query appears twice anywhere. That does *not* transfer to `triplet-hard`.

**Three library facts that would each have failed silently.** `bias="lora_only"`
does not mean "only the adapter's biases" — LoRA's `A`/`B` have no biases at
all, and `tuners_utils.py:497` unfreezes the *base layer's* bias for every
adapted module. So it trains pretrained weights that `save_pretrained` then
does not save: a checkpoint that cannot reproduce its own numbers. `bias="none"`
is the setting that matches the claim "the backbone is frozen".
`metric_for_best_model="eval_loss"` resolves `greater_is_better=False` on its
own (`training_args.py:1534` keys off the name ending in `"loss"`) — had it
defaulted True, `load_best_model_at_end` would have kept the *worst*
checkpoint, silently. And `fp16` is no longer CUDA-only: `accelerator.py:565`
lists `mps` as supported for torch >= 2.5.0. It stays off here anyway, because
`GradScaler` skips overflow steps and would muddy the steps/sec the smoke run
exists to measure — but the reason is now a measured trade-off rather than an
inherited belief, which is the difference that matters.

Smaller, and pleasant: LoRA initialises `B` to zeros, so an untrained adapter
is not merely close to the base model but **bit-identical** — max abs embedding
difference 0.0. Practically, that means a Phase 2 run that lands exactly on
0.3159 has not trained rather than failed to help. And `save_pretrained` writes
593,072 bytes for r=16 on query+value (147,456 params x 4B + header), which is
what lets `models/` stay gitignored as genuinely rebuildable.

## 2026-08-14 — the fine-tune lost, and the control is what made that a finding

Phase 2 finished and NFCorpus NDCG@10 went **down**: `0.3159 -> 0.2725`, a 13.7%
relative drop, with Recall@100 falling `0.3115 -> 0.2950` alongside it. R4 said
the fine-tune was allowed to lose. It did.

The first thing worth recording is that the loss was believable *before* it was
interpreted. `NDCG@10 = 0.0000` has a known cause here; `0.2725` does not, so it
needed its own checks: the saved adapter carries the same three modules as the
baseline (`Transformer -> Pooling -> Normalize`, `max_seq_length=256`), and
`oracle.py` passed both invariants on the *new* candidate set — `oracle >=
baseline` on all 323 queries, and `oracle == arithmetic_ceiling` to six decimals
at `0.609627`. That second check derives the ceiling independently of the sort
under test, so its agreement over a candidate set it had never seen says the
pipeline is intact and only the embeddings moved. R3 also settled itself: the
ceiling did not carry over, `0.6263 -> 0.6096`.

**A ceiling that moves is the more damaging result.** Reordering cannot change
which documents were retrieved, so an unchanged candidate set implies an
unchanged ceiling. This one dropped, which locates the failure in *retrieval*
rather than in *ranking* — and nothing downstream recovers a document that never
entered the top 100.

The expected shape of the damage was a slope: more MS MARCO specialisation,
worse biomedical transfer, monotone in learning rate. All four adapters say
otherwise.

| lr | dev loss | cos to base | NFCorpus NDCG@10 |
|---|---|---|---|
| — | — | 1.0000 | 0.3159 |
| 2e-5 | 0.3960 | 0.9757 | 0.2724 |
| 1e-4 | 0.3443 | 0.9438 | 0.2737 |
| 5e-4 | 0.3180 | 0.9473 | 0.2765 |
| 1e-3 | 0.3129 | 0.9294 | 0.2725 |

Dev loss spans 21%; NFCorpus spans 1.5%. The middle column is what makes the
flatness strange rather than merely negative — embedding drift from the base
model **does** grow with learning rate, so the models are genuinely and
progressively different. The drift scales and the damage does not, which is the
signature of a quantity that has hit a floor rather than one still responding to
its driver. Had the damage been proportional to distance moved, 1e-3 (which
drifted three times as far as 2e-5) would have been three times as damaged.

`checkpoint-200` of the 2e-5 run bounds the other end. At step 200 — 6.4% of an
epoch, still inside the `warmup_ratio=0.1` ramp — the encoder sits at cosine
**0.9998** to base and scores `NDCG@10 0.3143`, `Recall@100 0.3132`,
`MRR@10 0.5056`: baseline within noise, recall and MRR fractionally above it. So
the damage is neither instantaneous nor proportional. It accrues somewhere
between step 200 and one epoch and then stops. The shape in between was not
measured and three more checkpoints would be needed to trace it; the honest
claim is "saturates before one epoch", not "saturates at step N".

Which yields the finding worth keeping: **zero-shot transfer degradation
saturates before in-domain fit does.** By one epoch, 2e-5 and 1e-3 have paid an
identical NFCorpus penalty despite 2e-5 being visibly undertrained on MS MARCO —
its own eval curve was still descending when it ran out of steps. The intuitive
mitigation, "use a gentler learning rate to preserve general ability", is
therefore strictly dominated: the same penalty for a worse specialist. The
levers that could work are different in kind — fewer total steps, mixed-domain
replay, or less adapter capacity.

**The control is what turned all of this from a post-mortem into a result.**
`msmarco-MiniLM-L6-cos-v5` — same backbone size, MS MARCO tuned by people who do
it full-time — scores `NDCG@10 0.2584`, `Recall@100 0.2342` on NFCorpus. That is
*below* the fine-tune, and 0.0575 below the general-purpose baseline. So the
three models order cleanly along one axis of specialisation, `0.3159 -> 0.2725
-> 0.2584`, and the fine-tune paid 75% of the full penalty. Nothing was broken;
the encoder was moved toward MS MARCO, and this is what that move costs on
biomedical text. Without the control the same numbers support "our recipe is
wrong", which is a different claim and would have been the wrong one. It cost
one look.

It also relocates the floor. The four adapters converge at 0.272, but the axis
runs to 0.258, so the saturation point is probably not a law of training
dynamics — it is more likely the **capacity ceiling of 147,456 trainable
parameters on `query`+`value`**, exhausted identically by every learning rate.
That predicts a larger `rank`, or adding `key`, would push closer to 0.258.
Untested. The control differs from the fine-tune in several ways at once (full
fine-tune, different loss, different data recipe), so it is a reference point
and not a one-variable comparison; it cannot attribute the 0.0141 gap to LoRA.

Where the damage lands, from the cached runs at no extra cost: 69 queries
improved (mean `+0.0895`), 136 degraded (mean `-0.1487`), 118 unchanged, worst
single query `-0.9197`. Bucketed by qrel density the drop is `-0.0529` (1–3
relevant, n=62), `-0.0225` (4–10, n=67), `-0.0477` (11+, n=194), with recall down
in every bucket. **Diffuse, not a collapsed subset** — and specifically not the
D5 dense-query story, which the aggregate number alone could not have ruled out.

Then the part that makes the architecture look good rather than the encoder look
bad. Reranking the degraded candidates gives `0.2725 -> 0.3407`, against Phase
1's `0.3412` from baseline candidates. **A first stage 13.7% worse ends up 0.15%
worse end to end.** The cross-encoder does not care what order candidates arrive
in, only that they arrive, and Recall@100 only fell 0.0165 — so the two-stage
design absorbed nearly all of a substantial first-stage regression. One dataset,
n=1, but it is the kind of claim the ablation table exists to support. The
headroom-capture figure needs care here: it reads 20.2% against Phase 1's 8.1%,
which is the denominator moving rather than the reranker improving. The absolute
gain is the honest comparison, `+0.0253 -> +0.0682` — more work done because
there was more mess.

Two methodological notes. The lr search found its winner at the edge of the grid
twice running, and each extension was cheap relative to how badly a boundary
winner reads; the search stopped at 1e-3 on a stated rule (returns fell to
`-0.005` per doubling and the dev curve went non-monotone, bouncing `0.3325 ->
0.3346 -> 0.3398` mid-run before recovering — the stability edge showing up as
variance rather than as divergence), which is different from stopping silently.
And **no configuration was ever run twice**, so every number here is n=1 with no
estimate of run-to-run variance. The early gaps (0.052, 0.026) are far too large
for that to matter. The 5e-4/1e-3 gap of 0.0051, which decided the winner, is
exactly the size where it might.

Housekeeping worth not re-deriving: `@op` on `train()` serialised both Datasets
as call inputs and the trace died at 413 on an 80,466,260-byte payload — twice,
since the 1e-4 run had already launched before the fix. `postprocess_inputs`
substituting row count and column order fixed it (5e-4 uploaded clean), and the
column *order* is the more useful record anyway, because MNRL reads columns
positionally. Throughput measured 0.848 / 1.086 / 0.744 / 1.092 steps/sec across
four runs doing identical work, a ±30% swing from thermals and sleep, so a
single run's steps/sec is not a number to quote precisely. And `perf_counter`
disagreed with HF's `train_runtime` by 964s on the 2e-5 run, almost exactly the
one anomalous 981.7s eval — consistent with the laptop sleeping, since
`perf_counter` does not tick through sleep on macOS while `time.time` does.

## 2026-08-14 (pre-registration) — does the Phase 2 loss generalise, or is it NFCorpus?

Written and committed **before the run**, so the prediction cannot be adjusted
to fit the result.

The Phase 2 finding is n=1 dataset. On NFCorpus the specialisation axis ran
`all-MiniLM-L6-v2 0.3159 > LoRA 0.2725 > msmarco-MiniLM-L6-cos-v5 0.2584`, and
the conclusion drawn was that moving toward MS MARCO costs biomedical
retrieval. That conclusion has a competitor it cannot currently rule out:
**NFCorpus is a known weak spot for MS MARCO-trained models**, so the axis may
be a property of this one dataset rather than of specialisation.

The discriminating test needs a BEIR set whose *query distribution* resembles MS
MARCO's — natural questions rather than biomedical terms. FiQA (57,638 docs, 648
queries, financial forum QA) fits; SciFact (5,183 docs, 300 queries) is a cheap
second point, though its claim-style queries are expected to behave more like
NFCorpus.

**The model to watch is the control, not the adapter.** If domain match is the
mechanism, the axis should *reverse*:

> **Prediction.** On FiQA, `msmarco-MiniLM-L6-cos-v5` beats `all-MiniLM-L6-v2`
> on NDCG@10, and the LoRA adapter lands between them or above baseline.
>
> **Falsifier.** If the control also loses on FiQA, the domain-match explanation
> is wrong and the honest reading becomes "MS MARCO tuning costs retrieval
> quality broadly" — a different and more negative claim than the one currently
> in the README.

No retraining is involved: the adapters already exist, so this costs encoding
time only. R2 is extended past NFCorpus for this — every number obtained on any
dataset gets reported, win or lose. Without that rule, evaluating on new
datasets until one flatters the fine-tune is exactly the search this repo exists
to argue against.

## 2026-08-14 (result) — the prediction was wrong, and the mechanism is breadth, not domain

The FiQA prediction committed in `66e798c` was **refuted**. It said
`msmarco-MiniLM-L6-cos-v5` would beat `all-MiniLM-L6-v2` on FiQA, because FiQA's
natural-question queries resemble MS MARCO's where NFCorpus's biomedical terms do
not. The control lost on FiQA by the **largest margin of the three datasets** —
on the set chosen specifically because it should have reversed.

| dataset | `all-MiniLM-L6-v2` | LoRA r16 lr1e-3 | `msmarco-MiniLM-L6-cos-v5` |
|---|---|---|---|
| NFCorpus | **0.3159** | 0.2725 (−13.7%) | 0.2584 (−18.2%) |
| SciFact | **0.6451** | 0.5736 (−11.1%) | 0.4870 (−24.5%) |
| FiQA | **0.3687** | 0.2784 (−24.5%) | 0.2317 (−37.2%) |

Identical ordering on all three, no exceptions, no reversal — and the adapter
sits between the two every time (75%, 45% and 66% of the way along). So the
finding replicates, but the **domain-match explanation in the README is wrong**
and needs replacing.

What survives: `all-MiniLM-L6-v2`'s model card records training on
**1,170,060,424 sentence pairs** across ~30 datasets; `msmarco-MiniLM-L6-cos-v5`
trained on MS MARCO alone. The fine-tune took the 1.17B-pair model and pulled it
toward a distribution defined by ~500k triples. **Narrowing the training
distribution costs retrieval quality everywhere**, which is what three datasets
with no exception actually show, and which the domain framing did not predict.
It also explains the saturation across a 50x learning-rate range better: a
147,456-parameter adapter can only pull so far however hard it is pushed, which
strengthens the capacity-ceiling hypothesis (D8) rather than competing with it.

**The half that is still unmeasured, and it decides the claim.** The four Phase 2
dev losses are comparable only to each other; the base model's dev loss on the
same slice was never computed. Until it is, "traded out-of-domain quality for
in-domain gain" is unverified — if `all-MiniLM-L6-v2` already scores below
0.3129 on that slice, there was no gain to trade and Phase 2 is loss in both
directions. `dev_loss()` in `finetune.py` is scaffolded for exactly this.

**Evals ran at batch size 8, not 32.** `per_device_eval_batch_size` does not
inherit from `per_device_train_batch_size` — it defaults to 8, silently.
Confirmed against `checkpoint-3125/training_args.bin`: every Phase 2 run trained
at 32 and evaluated at 8. Three consequences. The base-model comparison must use
8 or it lands on a different scale. The random-guess floor for those dev losses
is `ln(8) = 2.079`, not `ln(32) = 3.466` — the latter applies to the train loss,
which did run at 32. And **the selector was lower-resolution than intended**:
ranking one document above 7 distractors is easier than above 31, so the loss
scale is compressed and configs sit closer together than they otherwise would.
That is a second independent reason to distrust the `0.0051` margin that picked
1e-3 over 5e-4, alongside the absent noise floor.

Also checked while in the args: warmup did run, despite
`SentenceTransformerTrainingArguments` storing it as `warmup_ratio=None,
warmup_steps=0.1`. `get_warmup_steps(3125)` returns 313 and the logged lr peaks
at epoch 0.1024 (step 320) before decaying, so the claim that `checkpoint-200`
sits inside the warmup ramp holds. Worth knowing that the stored attribute does
not read back the way it was set.

## 2026-08-16 — the base model's dev loss, and the trade survives

`0.6597`. `all-MiniLM-L6-v2`, untouched, on the last 10,000 rows of the seed-42
shuffle at eval batch 8 — the same slice and the same scale the four Phase 2
configs were scored on. Against `0.3960 / 0.3443 / 0.3180 / 0.3129`, every
fine-tune sits far below the model it started from, and the worst config still
improves on the base by more than the spread across all four configs combined.

So the claim that was unverified since 2026-08-14 is **verified**: Phase 2 really
did buy in-domain gain, and "traded out-of-domain quality for in-domain gain" is
the right shape of sentence. The alternative the measurement existed to rule out
— base already below 0.3129, meaning loss in both directions — is dead by a
margin too large to be a batching artifact.

The number also **corroborates against a figure recorded before it existed**.
The 2e-5 run's eval curve opened at `0.6263` at step 200 and fell monotonically
to `0.3960`. Step 200 is 200 steps of training past the base model, and it sits
just below `0.6597` — the ordering base > first-eval > final is exactly what a
model improving from this starting point produces. A base loss that had come
back at, say, 0.45 would have been well-formed, plausible, and inconsistent with
a curve already on disk. Worth remembering as a pattern: the cheapest check on a
new number is an old number nobody computed it from.

The two ways to get this wrong are both about comparability rather than code,
so they are now guards rather than prose. `k` and the eval batch size each get
a loud warning when they deviate from `10,000` and `8`, because both produce a
number that looks like a table row and is measured on a different task —
a different `k` is a different slice, not a cheaper sample of the same one.
`--dev-loss` takes `n` nowhere near the dev slice for the same reason: dev is
defined as the last `k` of the shuffle, so it is a function of `k` and the seed
alone, and `load_triples(0, k)` says so in code.

## 2026-08-17 — read the data before picking the dataset, and check your own sampler

Phase 4 was about to run on NFCorpus for no better reason than Phases 0–2 did.
Twenty questions — five each from nfcorpus, fiqa, scifact, hotpotqa, generated
by `qwen3:8b` from gold context and again from nothing — cost about ten minutes
and moved the decision twice.

**The disqualifying finding was one nobody had predicted.** NFCorpus queries are
not questions. `turnips`. `folic acid`. "To Snack or Not to Snack?" They are
NutritionFacts.org video titles, and "answer relevance" barely parses when the
query is one noun. The pre-probe worry had been that biomedical abstracts are
tedious to hand-annotate, which is true and beside the point.

**The argument that felt strongest died on contact.** The case against HotpotQA
was contamination: 2018 Wikipedia multi-hop, in every pretraining corpus, so the
generator would answer from memory and flatten every retrieval configuration.
It got **1 of 5 right with no context and 4 of 5 with it** — inventing a county,
a genre, and, memorably, both Temple University as Jack Guttentag's employer and
a "Benjamin Franklin Templeton" who donated it. An 8B model does not reliably
memorise long-tail entity facts. The prediction was well-reasoned, cheap to test,
and wrong; testing it cost less than defending it would have. It is also
generator-specific and n=5, so it is recorded as re-runnable rather than settled.

**The probe caught its own author, too.** For `folic acid` the gold context came
back as three papers on depression and antioxidants, and the model correctly
refused. That read as broken labelling. It was mostly a broken *sampler*: all 12
gold docs sit at grade 1, so `sorted(key=-grade)[:3]` was an arbitrary draw from
a 12-way tie, and it happened to skip the four papers that do discuss folate.
The dataset is defensible — a NutritionFacts page on folate and mood cites the
surrounding depression literature, and document-level relevance is what these
qrels claim.

What survives is sharper than the accusation. **95% of NFCorpus test labels are
grade 1 (11,758 against 576), and 217 of 323 queries have every gold doc at a
single grade.** With a median of 16 gold docs and no signal to rank them, "the
top-k gold documents" is not a defined object — so milestone 4b's oracle context,
the whole point of which is to be the ceiling, cannot be constructed here without
an arbitrary choice. That is a structural reason, not an anecdote, and it is the
one that went into the plan.

**The corpus-size problem dissolved once the constraint was stated precisely.**
Full-wiki HotpotQA is 5.2M documents — a 654MB download against NFCorpus's 2.4MB
— which drags in a vector index the scope discipline explicitly defers. But the
dev distractor split already pools ten paragraphs per question: 73,700 slots,
**66,581 unique after deduping by title**, which is FiQA-sized. Every one of the
7,405 questions keeps exactly 2 gold docs and not one gold title is missing from
the pool. Brute-force cosine survives; the corpus is adversarial by construction
because the distractors were TF-IDF-mined per question. It is a custom reduction,
so it carries a distinct id — `hotpotqa-distractor-pool`, never `hotpotqa` — and
its numbers are not comparable to published BEIR results.

**A gold answer is not a faithfulness label.** The tempting inference from
HotpotQA shipping answers is that the ~50 hand labels become unnecessary. They do
not. Correctness asks whether the answer is right; faithfulness asks whether it
came from the passages. The probe produced the cell where these diverge: an
answer citing "[1]" for the claim that 7th Sea is fantasy-themed, which passage
[1] never says. Correctness marks that wrong for the wrong reason and would have
marked a lucky-guess fabrication right. What gold answers actually buy is
stratification — drawing the 50 labels across correct and incorrect answers, so
the class imbalance that makes κ meaningless cannot take hold.

## 2026-08-19 — the sentinel the model won't say, and reasoning that isn't a rationale

Two design questions about Phase 4a's output contract both looked like taste and
both turned out to have measurements behind them.

**A refusal marker that reads like a control token gets treated like one.**
`REFUSAL = "__REFUSED__"` was already in `generate.py`, so the obvious move was
to ask the model for that literal string. Eight unanswerable questions against a
two-passage context, three candidate sentinels, `qwen3:8b` at temperature 0:

```
__REFUSED__            exact 5/8   tagless 3
INSUFFICIENT_CONTEXT   exact 8/8   tagless 0
NOT IN PASSAGES        exact 8/8   tagless 0
```

The failure is not what it looks like. The model reproduced `__REFUSED__`
correctly every time — it **dropped the `<answer>` tags** and emitted the bare
sentinel. The double underscores appear to read as a directive about the whole
reply rather than as a value for a field, so the model skips the formatting it
was otherwise happy to follow. The natural-language sentinels stay inside the
frame because they read as things one would say.

**Why 3-in-8 matters more than it sounds.** Trace it: a strict `parse_answer`
finds no `<answer>` tag, returns `("", raw)`; a model-side `should_refuse`
keying off that sees an empty string and returns `False`; and `validate()`'s
`not g.refused and not g.answer.strip()` then counts a *correct refusal* as
parser drift. Refusal rate is the metric D11 kept, and this wiring would have
moved a third of it into the parse-miss column — the same shape as
`NDCG@10 = 0.0000` meaning wrong doc ids, one layer up.

Resolved by keeping two strings rather than one: the model emits a
natural-language sentinel, `should_refuse` recognises it, and the pipeline
normalises to `REFUSAL`. That preserves a split the dataclass already pays for —
`raw` is what the model said, `answer` is what the pipeline concluded — and
without it "the model declined" and "my rule classified this as a decline" are
the same bytes, which is exactly the model-side/pipeline-side distinction the
stub's docstring calls not interchangeable. n=8 on one prompt wording, and the
rule stated the sentinel bare; phrasing it as `<answer>__REFUSED__</answer>`
might well close the gap and was not tested.

**Thinking mode is not a free rationale.** `think: True` looks like it hands you
the supporting text without asking the model for it. Three things say otherwise.
Ollama returns it as a separate `thinking` field, so it never reaches `raw` and
the transport's return type would have to change. What it contains is
meta-commentary about the task, not evidence — a probe returned *"the question
is asking for a brief answer... let me confirm there's no trick here"* — and it
carries **no citation markers**, so the ungrounded-citation failure mode the
dataset probe caught would be invisible in it. And chain-of-thought is famously
not a faithful record of what produced the answer, so judging it measures
whether the *narration* is grounded, not whether the answer is. It stays worth
measuring as a correctness ablation; it is not the rationale. Note that `think`
is absent from `generation_cache_path`, so running it as a config today would
silently collide with the `think: False` file.

**Three facts that constrain the prompt, from the data rather than from taste.**
458 of 7,405 gold answers are literally `yes` or `no` (6.2%) — a prompt asking
only for a span floors token-F1 at zero on one question in sixteen. Gold answers
run 2 words at the median, 4 at p90, so the length instruction is calibratable
and "a sentence" is wrong. And context is not scarce: ten docs is roughly 1,200
tokens against a 40,960 window, verified by a 9,038-token prompt coming back
with `prompt_eval_count: 9038` rather than a silent truncation.

## 2026-08-19 — the adverb was innocent, and four of six wrong answers weren't wrong

The grounding rule reads *"Only answer if the passages fully support it."* The
worry was that HotpotQA makes that unsatisfiable: the answer lives in two gold
passages jointly and in neither alone, so a model reading "fully" strictly
should refuse on exactly the questions the dataset is built from. The wording
would then set the refusal rate — the metric D11 kept — and it would read as a
retrieval signal when it was an adverb.

**Feeding the model gold-only context turns that into a measurement.** Retrieval
is perfect by construction, so every refusal is caused by wording alone. Fifteen
questions, three wordings differing in one rule line, `qwen3:8b` at temperature 0:

```
1-current (fully support)   false-refusals 1/15 | tags 15/15 | exact-match 9/15
2-  + "may combine"         false-refusals 1/15 | tags 15/15 | exact-match 8/15
3-softened (support)        false-refusals 0/15 | tags 15/15 | exact-match 8/15
```

**The prediction was wrong.** The model combines across passages without being
told it may, and the licence to combine buys nothing. The exact-match spread is
noise at this n — it says no variant loses, not that the current one wins. The
wording stands as written.

The single refusal survives inspection as genuine rather than as annotation
noise: the linking fact the question turns on is present in the full gold text,
and the model's own rationale *found the answer* — "the reform opera that came
before Orfeo ed Euridice is Alceste, which is mentioned in passage [2]" — before
refusing anyway, tangled in the question's nested phrasing rather than in the
grounding rule. The softened wording rescued that one case, which is a hint at
n=1 and nothing more.

**Format adherence held 45/45 under rationale-first ordering**, closing a caveat
the earlier sentinel probe left open — those 8/8 runs were answer-first, so the
ordering actually shipped had never been exercised. Scope it honestly though:
these were two-passage prompts, and production is ten passages at roughly 1,200
tokens. Adherence at that length is still unmeasured, with `validate()`'s 20%
parse-miss gate as the net on the first real batch.

**The finding with downstream weight is not about the prompt at all.** Of the six
answers that missed exact match, four were not wrong:

```
gold 'the George Washington Bridge'   model 'George Washington Bridge'
gold 'an Otto Dix painting'           model "Otto Dix's painting"
gold 'genus of plants'                model 'plant genera'
gold '"Alceste"'                      model 'INSUFFICIENT_CONTEXT'
gold 'Sesame Street'                  model 'AOL'
gold 'Helen Elizabeth Hunt'           model 'yes'
```

An article, a possessive, a pluralisation — and gold answers carry literal
punctuation, `'"Alceste"'` quotes included. **The correctness scorer's
normaliser is load-bearing, not a detail**, and a normalisation gap gets read as
a weak generator — same genre as `NDCG@10 = 0.0000` meaning wrong doc ids, on
code nobody has written yet.

**Corrected the same evening, by actually running the metric.** This paragraph
first claimed normalisation "dissolves all of them", moving accuracy from 9/15
to ~13/15 — 60% to ~87%. Implementing the SQuAD/HotpotQA normaliser and scoring
the pairs says otherwise:

```
model 'George Washington Bridge'  gold 'the George Washington Bridge'  F1 1.000
model "Otto Dix's painting"       gold 'an Otto Dix painting'          F1 0.667
model 'plant genera'              gold 'genus of plants'               F1 0.000
model 'INSUFFICIENT_CONTEXT'      gold '"Alceste"'                     F1 0.000
model 'AOL'                       gold 'Sesame Street'                 F1 0.000
model 'yes'                       gold 'Helen Elizabeth Hunt'          F1 0.000

exact-match 0.600   |   mean token-F1 0.711
```

Only the article case dissolves completely. The possessive survives normalisation
as the token `dixs` and earns partial credit, not full. `plant genera` against
`genus of plants` scores **zero** — token-F1 is surface overlap and knows nothing
about morphology, so two ways of saying the same thing share no tokens. And
HotpotQA's own scorer makes yes/no all-or-nothing, so the `yes` miss gets no
partial credit either.

So normalisation is worth **0.600 → 0.711**, not 0.600 → 0.87. The 87% was a
*human* read of which answers were semantically fine, silently substituted for
what the metric reports. Which is the exact failure this file keeps cataloguing:
a well-formed, plausible number that nobody computed. The lesson survives — the
normaliser earns 11 points and must be tested — but the number it earns is the
one the metric gives, not the one the reader would.

**One thing to watch when a real batch runs.** The `'yes'` miss came from *"Who
has won more awards, Dan Schneider or Helen Hunt?"* — a comparison-shaped
question wanting a name, where the prompt's yes/no rule plausibly over-fired.
Only 6.2% of gold answers are actually yes or no, so a batch returning
materially more than that has one obvious suspect.

Method note worth keeping: gold-only context isolates the prompt from retrieval
completely, and it is milestone 4b's oracle run pointed at a different question.
The ceiling experiment and the prompt calibration are the same setup.

## 2026-08-22 — the first real batch, where a third of the refusals are the metric working

Fifteen questions, ten retrieved passages each, `qwen3:8b` at temperature 0,
prompt v1. The first generation that reads real retrieval output rather than
gold context or a two-passage probe.

```
generated      : 15
refused        :  5  (33.3%)
no short answer:  0
yes/no answers :  0  (base rate predicts ~1)
```

**Two caveats the probes left open both closed.** Format adherence held 15/15 at
roughly 1,200 tokens of context, which the 45/45 result could not speak to — it
was measured on two-passage prompts. And the yes/no rule, suspected of
over-firing after a comparison-shaped question came back `yes` against a name,
did not fire once. Both are n=15 and neither is a strong claim; they are the
absence of a specific worry, not evidence of a virtue.

**33.3% against 6.7% on gold context looks alarming and mostly is not.** HotpotQA
answers are supported by two gold passages jointly, so the honest question is
whether the model refused when the evidence was actually present. Splitting the
five refusals on how much gold retrieval delivered:

```
refused with 2/2 gold in context   1     over-refusal
refused with 1/2 gold in context   4     correctly calibrated
```

Four of the five declined because the evidence genuinely was not there. That is
the first direct evidence for the thing the refusal branch exists to measure —
refusal rate moving with retrieval quality — and it is the safe failure profile.

**The one over-refusal is not a grounding failure, and calling it one would
corrupt the metric.** For *"In which year was the choreographer for 'Best Foot
Forward' born?"* both gold passages were retrieved:

```
[ 2] Best Foot Forward (musical)   <-- GOLD
[ 9] Gene Kelly                    <-- GOLD
```

The answer requires linking the musical to its choreographer and then the
choreographer to a birth year — ranks 2 to 9, seven distractors in between. The
model's rationale reads *"The passage does not provide information about the
birth year of the choreographer"*, singular. It did not fail to trust the
passages; it failed to connect two of them. Retrieval did its job here, so this
refusal carries no retrieval signal at all, and reading refusal rate as a pure
retrieval measurement quietly folds multi-hop linking failures into it. At n=15
that is 1 in 5 of the refusals.

**The opposite failure showed up in the same batch, and it is the more dangerous
one.** For the Vienna reform-opera question retrieval returned *zero* gold
passages. The model answered anyway:

```
rationale: Gluck's reform opera in Vienna, such as "Orfeo ed Euridice" and
           "Alceste", came before the opera "Romolo ed Ersilia" ... [1], [7]
answer   : Orfeo ed Euridice          gold: Paride ed Elena
```

Fluent, cited, wrong. It reasoned off Gluck's biography page — a passage
genuinely in context, which is why the grounding rule did not fire — and
produced an answer the passages do not support. A correctness-only metric scores
this 0 and reads it as a weak generator, when what happened is a retriever that
missed both gold documents and a model that filled the gap from an adjacent
page. The distinction only exists because refusal is measured separately, which
is the argument D14 settled from the other direction.

**What the batch hands to 4c.** Of the ten answered questions, five were produced
from incomplete evidence:

```
                full gold   partial   none
answered (10)       5          4        1
refused   (5)       1          4        0
```

Those five are where faithfulness and correctness can disagree, and they are the
reason a judge is worth building rather than scoring token-F1 alone. The Vienna
answer is a particularly good fixture case: its citations are specific, so a
faithfulness judge has something checkable to rule on rather than a vibe.

**Cost note.** The retrieval cache miss over 66,581 documents took about two
minutes on an M1 Pro with MPS — roughly 720 docs/sec — and the fifteen
generations a few minutes more. That cache is now on disk and every later Phase 4
run skips straight to the generator. The generation cache key covers neither
`--n-queries` nor `--seed`, so this batch's file collides with any later n; the
post-hoc count check catches it and `--refresh` clears it, but the trial's
generations do not carry forward. Its deliverables were the two measurements.

## 2026-08-23 — the refusal gate is calibrated, and the split is the finding

One hundred questions, same config as the fifteen: ten retrieved passages,
`qwen3:8b` at temperature 0, prompt v1. Everything the trial measured
replicated, and the number that looked alarming resolved into a result.

```
                 n=15     n=100
refusal rate     33.3%    34.0%     95% CI [25.5%, 43.7%]
parse misses     0/15     0/100
yes/no answers   0/15     7/100  (7.0% against a 6.2% base rate)
```

Format adherence at ten passages is now 115/115 across both batches, which
closes the caveat the two-passage probes left open for good. The yes/no rule
came back clean: `0/15` had looked like the rule under-firing, and at n=100 it
lands within a point of the base rate, so it was small-n and nothing else. The
CI narrowed from 43 points to 18, as the sizing arithmetic predicted.

**Splitting refusals by how much gold retrieval actually delivered turns a flat
number into a curve.**

```
refusal rate by gold passages in context (HotpotQA has exactly 2 per question)
  full gold (2/2)    5/51  =   9.8%   [ 4.3%, 21.0%]
  partial  (1/2)    22/41  =  53.7%   [38.7%, 67.9%]
  none     (0/2)     7/8   =  87.5%   [52.9%, 97.8%]
```

Monotonic, and the full-gold and partial-gold intervals do not overlap. This is
the first Phase 4 claim with real separation behind it: **as retrieval degrades
this generator declines rather than invents**, which is the safe half of the
failure space and the exact question the refusal branch was kept to answer.

**The claim is descriptive, not causal, and the distinction is worth being
strict about.** Those three buckets contain *different queries*. Questions whose
gold docs retrieval missed may also be vaguer or harder questions, so
"worse retrieval → more refusal" is confounded with "harder question → more
refusal" in this analysis and cannot be untangled within it. The clean test is
the paired one the plan already specifies — the same fixed query sample under
four retrieval configs plus the ceiling — where question difficulty is held
constant by construction and a monotonic dose-response across configs cannot be
explained by difficulty. **The stratified split earns its place as a diagnostic
of where refusals concentrate inside one config; the causal reading has to wait
for the paired run.** Designing that run has its own trap, recorded as an
amendment on 4b in `docs/plan.md` (2026-08-23) rather than repeated here.

**The rarest failure stayed rare, and the reason is the result.** Only one of the
hundred answers was produced with zero gold passages in context:

```
            full gold   partial   none   | total
  answered       46        19        1   |   66
  refused         5        22        7   |   34
```

The pure hallucination case is thin *because* the model refuses 87.5% of the
time when it has nothing to work from. Good calibration is bad fixture supply,
and the 4c.1 fixture has to be built around that rather than wish it away: the
19 answers produced from partial evidence are where faithfulness is genuinely at
stake, since the model committed to an answer while holding half the chain, and
the 5 over-refusals are the other interesting stratum. Sampling the
"answered with nothing" profile richly would take a much larger batch, or a
config chosen to retrieve badly on purpose.

**A note on what the n=15 batch was worth.** Every headline number it produced
survived the 7x scale-up, including the two it got by luck — `0/15` yes/no and
`0/15` parse misses were both consistent with the true rates rather than
evidence for them. What it could not do was carry a claim: a refusal rate of
"somewhere between 15% and 58%" is not a measurement, and two of its five
fixture strata held a single example. The trial's value was catching whether the
pipeline worked at all, at ten passages, on real retrieved context — and it did
that in four minutes.

## 2026-08-23 — the prompt leaks its own instructions, and a refusal that isn't

Two findings, both surfaced by hand-labelling rather than by a script, which is
itself the argument for hand-labelling before automating the judge.

**A format template is text the model will copy.** `build_prompt` v1 ended with

```
Format your reply exactly as:
    <rationale>... cite passages as [1], [2] ...</rationale>
```

and roughly one rationale in nine came back with `Cite passages as [1], [2]`
appended verbatim — 19 of 175 generations across four batches. The instruction
sat *inside* the block the model was told to reproduce "exactly", and it
complied more literally than intended. Moving the citation rule up into `Rules:`
— where the length and grounding instructions already lived without ever being
echoed — took the leak to 0/60 on regeneration. The tell was there all along:
instructions in `Rules:` are followed, instructions inside the format template
are *transcribed*.

The reason this is worth more than a formatting note: the leaked string injects
`[1], [2]` into rationales that cited nothing. A faithfulness judge reading such
a row sees citation-shaped tokens attached to an uncited claim, which is the
exact confusion 4c exists to resolve. It was a contaminant on the measurement
axis, not a cosmetic blemish, and it would have been judged rather than caught.

**Refusals split into two mechanisms, and the gold-stratified curve cannot see
it.** The first row of the seed-1 fixture asks which of two men worked in both
film and photography. Passage [2] says of one: "collage, film, video, drawing,
**photography** and installation". The model's rationale asserts he works "in
film, but not photography", eliminates him, concludes the other man, notices
that passage does not support *that*, and refuses.

That is not a context failure. Both gold passages were present and the answer
was plainly derivable; the model misread one and refused as a consequence. The
refusal curve stratified by gold-in-context (9.8% / 53.7% / 87.5% for 2 / 1 / 0
gold docs, n=100) would file this row under "2 gold docs, refused" and read it
as conservatism. It is a comprehension error wearing a refusal's clothes, and
the two have opposite implications: one says the model knows what it doesn't
know, the other says it doesn't reliably read what it has been given.

**Refusal rate is noisier across draws than the n=100 CI suggests.** The same
config, same prompt, differing only in which questions were sampled: 34.0%
(n=100, v1), 53.3% (n=30 seed 1, v1), 43.3% (n=30 seed 2, v1), 46.7% (n=30
seed 1, v2). The n=30 intervals are ~20 points wide either side, so none of this
contradicts the n=100 estimate — but it is a standing caution against reading a
single small batch's refusal rate as a property of the configuration.

**What the fixture rubric ended up being, and the one thing it cannot count.**
Two binary axes: `grounded` for answered rows (every clause supported by the
passages it cites — strict, so an answer that is *factually* right via
parametric knowledge is still `false`), and `refusal_ok` for refused rows, since
a refusal makes no claims and would score trivially "grounded" otherwise. The
over-refusal test is deliberately narrow: `false` only when the rationale itself
names the answer correctly and the model refused anyway. That is checkable by
two raters, which is what κ needs — at the cost that a refusal whose rationale
never names the answer scores `true` even when the answer was derivable. **Any
over-refusal rate from this axis is therefore a lower bound.** The misread case
above sits exactly on that boundary and is the reason the caveat is written into
`fixture.py` rather than remembered.

"Correctly" means what the ten passages *establish* — not what HotpotQA's gold
says. On a row where retrieval delivered no gold passage the dataset's answer is
unreachable from the context the model saw, so a refusal there is calibrated
regardless of an answer existing in the world. And the bar is establish, not
suggest: the Vienna case's passages point toward an answer without carrying it,
and reading "points toward" as "was derivable" would convert calibrated refusals
into over-refusals wholesale.

**A judge prompt can be rubric-perfect and still measure nothing, because
STRUCTURE decides whether the model grades or answers.** The first judge prompt
led with the task, then ten long passages, then a tail reading `QUESTION:` /
`PASSAGES:` / `ANSWER:` / `RATIONALE:` — which is, structurally, the *generator's*
prompt. `gemma3:4b` did the nearest task it recognised and answered the question:
`"Rhodesia...true"`, `"Mumbai"`. Parse rate 0/4. Moving the instruction *after*
the content and adding an explicit "you are a GRADER, do not answer the question
yourself" took the same model, same rubric, same rows to 4/4; `mistral-small`
went 0/2 → 4/4 alongside it. Two smaller causes rode along: models copy an `...`
placeholder into the tag verbatim (`<judge_answer>...false` parses as nothing),
and they drop closing tags unless asked for them. None of this is visible in a
unit test — every input there is a literal, and 42 of them stayed green against a
prompt no model could follow. `--smoke` exists for exactly this: N rows, live
models, prints the raws, writes nothing.

**Two rubrics of record in one file is a trap the amendment convention creates.**
`fixture.py` carries the 2026-08-23 rubric above `LABEL_VALUES` and the 2026-08-26
"THE BOUNDARIES" block below it. They disagree on `refusal_ok` — the older one
rules on whether the *rationale* names the answer, the newer on whether the
*passages* support one — and `ce989db` amended `AXIS_QUESTIONS` to the passages
rule. Keeping the superseded text is deliberate and right (`CLAUDE.md`: a
reversal should say so rather than overwrite silently), but it means "the rubric
of record" is no longer answerable by position in the file, and `judge.py`'s
docstring pointed at the older block. v1 of the judge prompt was built from the
superseded rule. Nothing raises — a judge graded against labels it was given a
different rubric for produces a perfectly well-formed κ that measures our own
edit. Worth noting the seed-1 labels *cannot* adjudicate this: both
`refusal_ok=false` rows have rationales that name the answer, so they satisfy
either rule. **When a rubric is amended, the pointer to it has to move in the
same commit as the amendment.**

## 2026-08-28 — the judge that parsed everything and agreed with nothing

The 4c.2 bake-off ran against the labelled seed-1 fixture under rubric v2. Both
candidates had cleared `--smoke` 4/4 the same evening, so the format question was
settled going in and the only open question was agreement.

```
judge            kappa      n    95% CI          agree    miss%   sec/call
mistral-small   +0.586     16    [+0.09, +1.00]  13/16    0.0%     42.45
gemma3:4b       -0.257     16    [-0.67, +0.14]   5/16    0.0%      6.77
```

**A judge can pass every structural check available and be anti-correlated with
the truth.** `gemma3:4b` emitted well-formed, parseable rulings on 30/30 rows and
landed below chance. The error is systematic, not noisy — it ruled `false` on 8
of the 11 rows labelled `true`:

```
human -> judge, grounded axis
  gemma3:4b       TT 2 · FF 3 · TF 8 · FT 3
  mistral-small   TT 9 · FF 4 · FT 2 · TF 1
```

It has the shape of a strict groundedness ruling without the discrimination: it
fails nearly everything, including rationales that are in fact supported. This is
the same class of finding as the 0.0000 NDCG and the leaked format template —
well-formed output, wrong meaning, nothing raises — and it is the one that most
directly threatens 4c, because parse rate is the cheap monitor one would actually
automate. `--smoke` cannot see it. Only labels can.

**The bake-off does not isolate size, and the write-up must not imply it does.**
This is the caveat most likely to decay into "size matters" on retelling:

```
gemma3:4b       family=gemma3  params=4.3B   quant=Q4_K_M   (Google)
mistral-small   family=llama   params=23.6B  quant=Q4_K_M   (Mistral)
```

Quantization is matched; vendor, architecture and training corpus are not. Two
points differing on four axes support a *selection* result — use `mistral-small`,
digest `8039dd90c113` — and not a causal one. A size ablation needs two sizes
inside one family (`gemma3:4b` vs `gemma3:27b`) and was not run. The CI on the
winner, `[+0.09, +1.00]`, clears zero and little else: the ranking is secure at
n=16, the value of kappa is not.

Worth recording that the issue-body premise was half right. "Judging is
classification against a rubric, and everything it rules on is in the prompt, so
parameter count may buy little" — the premise holds and the conclusion does not
follow. The 4B model had the passages, the rationale and the rule in front of it
and could not apply them. **What is in the prompt bounds what a judge could know,
not what it can do with it.**

**The refusal curve moved under generator prompt v2, and only in the middle
bucket.** `n100 seed0` was regenerated so the whole Phase 4 story sits on one
generator (the fixture had been v2 since seed 1; the curve was still quoting v1):

```
                      v1              v2
overall refusal      34.0%           38.0%   [29.1%, 47.8%]
  gold 2/2            9.8%            9.8%   [ 4.3%, 21.0%]
  gold 1/2           53.7%           63.4%   [48.1%, 76.4%]
  gold 0/2           87.5%           87.5%   [52.9%, 97.8%]
leaked template     5/100           0/100
```

Recomputing the v1 batch with the same script reproduces 2026-08-23 exactly
(34.0%, 9.8 / 53.7 / 87.5), so the shift is the prompt change and not a
difference in how the curve is computed. The full-gold and zero-gold buckets are
unchanged row-for-row; only partial gold moved, and its two intervals overlap
heavily, so **do not claim v2 refuses more** — claim the curve is still monotonic
and the 2/2 and 1/2 intervals still do not overlap, which is the property the
separation argument actually needs.

**A worked example can dissolve under regeneration, and the labels notice while
the prose does not.** `fixture.py` and the 2026-08-23 entry above both name
seed-1 row 1 (Edmund Mortimer) as the type case for over-refusal. Under v2 that
row answers rather than refusing — v1 named the answer and then talked itself out
of it, v2 stops at the answer. The `raw_sha` anchoring did its job and the labels
invalidated loudly; the *comments* citing that row went stale silently and are
still stale. Anchoring artefacts is cheap and standard. Anchoring the prose that
discusses them is neither, and the current type case for over-refusal is the
Hund's-rule row (`5ae24b16…`, gold 2/2), whose rationale names Friedrich Hund and
then refuses on the question's exact phrasing.
