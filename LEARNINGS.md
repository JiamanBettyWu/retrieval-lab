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
