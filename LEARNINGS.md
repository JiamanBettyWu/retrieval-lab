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
