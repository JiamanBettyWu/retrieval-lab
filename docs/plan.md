# Retrieval Lab — plan

> The working design doc for the repo. Read this before proposing
> architectural changes; `TODO.md` carries current state and open decisions.

A **retrieval-quality lab**: build a two-stage RAG retriever, LoRA-fine-tune its
encoder, and **prove** each improvement on a labeled benchmark (BEIR) with
metrics traced in Weave — then demo the validated pipeline as an "ask my second
brain" front door over a personal wiki.

## The pitch (why this project)

These concepts are studied but never built: **RAG, reranking, learning-to-rank,
embeddings, LoRA, fine-tuning.** This project applies **all six at once**, and
does the one thing most RAG projects skip: it **measures** whether the fancy
parts (reranker, fine-tuned encoder) actually help. That measurement is the
career story — *classic-ML rigor (ranking metrics, headroom, generalization)
applied to a GenAI stack*, exactly the classic-ML→GenAI pivot. Portfolio-
shareable: **everything is public by construction.**

> **Amended 2026-08-14.** This paragraph used to end by flagging the private
> wiki as an open sharing question (`TODO.md` D6). Phase 3 is retired and Phase
> 4 runs over NFCorpus [**2026-08-28: over `hotpotqa-distractor-pool` since
> 2026-08-17 — also public, so the conclusion is unchanged**], so no private
> corpus is involved anywhere and D6 dissolves. The amendment also sharpens the pivot claim: as of Phase 2 the repo
> contained **no LLM at all** — every phase was retrieval, which is classic IR
> rather than GenAI. Phase 4 is what actually closes that gap, and it does so by
> pointing the same measurement discipline at a target where almost nobody
> measures: the LLM judge itself.

## Concepts exercised

**Will apply:** RAG (retrieve → rerank → generate), embeddings (the bi-encoder
retriever), reranking (bi-encoder → cross-encoder, *measured* for lift),
learning-to-rank (NDCG@k / MRR as the eval lens), LoRA / fine-tuning
(LoRA-fine-tune the encoder on MS MARCO triples), W&B Weave (tracing +
`weave.Evaluation`), deliberate-practice (build-to-learn).

**Promoted from stretch 2026-08-14 — now Phase 4, the next real phase:**
**LLM-evaluation** (grade the *generation* half for faithfulness/groundedness
via LLM-as-judge, **with the judge itself validated against human labels and a
κ published** — the measurement, not just the method).

> **Amended again 2026-08-14 (same day): LangGraph is out.** It was briefly
> listed here, placed at the pipeline's one genuine branch. Dropped because the
> skill is already demonstrated in other projects, so carrying a framework
> dependency to re-demonstrate it buys nothing. The **branch itself survives as
> a plain conditional** — what mattered was never the graph, it was the refusal
> *rate* as a per-config metric. See D11 in the journal.

## Core design decisions (locked)

1. **Fine-tune, never train from scratch.** Start from a pre-trained encoder
   (`all-MiniLM-L6-v2` or a BGE base); LoRA-adapt it (nudge low-rank adapters,
   not all weights).
2. **Evaluate ≠ demo.** *Evaluate* (compute NDCG/MRR) on a **labeled BEIR set**
   — needs qrels + headroom. *Demo* on the **wiki** — no labels needed; it's a
   qualitative zero-shot-generalization showcase.
3. **Zero-shot generalization is the headline claim.** Fine-tune on MS MARCO
   (general), evaluate on an *unseen* BEIR domain. "A retriever that generalizes"
   beats "I improved my own test split."

   > **Amended 2026-08-14 — decision 3 was measured and refuted.** Kept above as
   > written rather than rewritten, because the reasoning that produced it is
   > what the result argues against. Phase 2 fine-tuned on MS MARCO and the
   > retriever generalized **worse**, on all three BEIR sets tried: NFCorpus
   > `0.3159 -> 0.2725`, SciFact `0.6451 -> 0.5736`, FiQA `0.3687 -> 0.2784`.
   > The off-the-shelf `msmarco-MiniLM-L6-cos-v5` control loses harder still on
   > every one, so this is not a flaw in the recipe — narrowing a model trained
   > on 1.17B diverse pairs down toward ~500k MS MARCO triples costs retrieval
   > quality everywhere. A pre-registered prediction that FiQA would reverse the
   > ordering (committed in `66e798c`, before the run) was also refuted. See
   > `LEARNINGS.md` 2026-08-14 and the README's Phase 2 section.
   >
   > **The headline claim is now the measurement itself**: "a fine-tune that
   > lost, proven not to be a bug, with its mechanism tested across three
   > datasets" — which is what decision 4 was always really asking for.

4. **Every phase ships a number and a working artifact.** The deliverable is a
   growing ablation table with *movement* in it. **Movement includes downward**
   — Phase 2's row is a regression and it stays in the table, because a table
   that only fills in when a phase wins cannot be trusted when it says a phase
   won.

## Pipeline

```
query
  │
  ▼
[bi-encoder]  embed query + corpus → cosine top-100        ← Phase 0 (+ Phase 2 fine-tunes this)
  │
  ▼
[cross-encoder rerank]  re-score top-100 jointly → top-10   ← Phase 1
  │
  ▼
[generate]  Claude answers from top-k with citations        ← Phase 4 (was Phase 3)
  │
  ▼
[judge]  a second model scores faithfulness + relevance      ← Phase 4
```

**Amended 2026-08-14:** `[generate]` moved from "Phase 3, wiki demo only" to
Phase 4, and is now an *evaluated* stage rather than a qualitative showcase. The
`[judge]` stage is new. Reasoning in the amended milestones below.

> **Amended 2026-08-28 — the diagram's model names are wrong, and so are the
> judge's axes.** `[generate]` is **`qwen3:8b` run locally via Ollama**, not
> Claude (see the tech-stack amendment below), and it either answers from the
> top-10 or emits an `INSUFFICIENT_CONTEXT` sentinel. `[judge]` does not score
> "faithfulness + relevance": relevance was never built, answer-*correctness*
> was ruled out (2026-08-14, amended 2026-08-17), and the judge rules on **two
> disjoint binary axes** — `grounded` for answered rows, `refusal_ok` for
> refused ones. They are disjoint because a refusal makes no claims and would
> score trivially grounded otherwise.

### Models — the bi-encoder ≠ cross-encoder trap

They share the `MiniLM-L6` **backbone** but are different model *types* and are
**not interchangeable** (the name collision is the classic trap):

| Role | Model | Type | Outputs |
|---|---|---|---|
| **Phase 0 retriever** | `sentence-transformers/all-MiniLM-L6-v2` | **bi-encoder** | an embedding per text (index the corpus once, cosine top-k) |
| **Phase 1 reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **cross-encoder** | one relevance score per (query, doc) pair — cannot do first-stage retrieval |
| *(optional reference)* | `sentence-transformers/msmarco-MiniLM-L6-cos-v5` | bi-encoder | an MS-MARCO-tuned "upper-bound" comparison |

**Baseline = the *general* `all-MiniLM-L6-v2`, deliberately not the MS-MARCO-tuned
bi-encoder** — Phase 2 fine-tunes on MS MARCO, so starting from an
already-MS-MARCO-specialized model would flatten the ablation. General baseline
= headroom for the LoRA fine-tune to visibly help. (A stronger modern baseline
like `BGE-small-en` also works — just less headroom.)

## Milestones (each independently shippable, each de-risks the next)

**Phase 0 — Baseline retriever + eval harness (the foundation).**
- Pick one small BEIR set with qrels — **NFCorpus** (~3.6K docs, medical, hard)
  or **SciFact** (~5K, scientific claims). Load via the `beir` library.
- Off-the-shelf bi-encoder → embed corpus + queries → cosine top-k.
- Compute **NDCG@10, MRR, Recall@100** (BEIR's `EvaluateRetrieval` /
  `pytrec_eval`). Trace in Weave.
- **Deliverable:** the baseline number *and* the headroom check — confirm
  NDCG@10 is *not* already ~1.0 (else pick a harder set).

**Phase 1 — Cross-encoder reranker (first lift).**
- Retrieve top-100 (Phase 0), rerank with a pre-trained cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) → top-10.
- Re-measure. **Deliverable:** before/after table (bi-encoder vs +rerank).

**Phase 2 — LoRA-fine-tune the bi-encoder (the depth move).**
- Fine-tune the encoder on **MS MARCO** `(query, positive, negative)` triples,
  contrastively (sentence-transformers `MultipleNegativesRankingLoss`, in-batch
  + optional hard negatives), with a **LoRA adapter** on the backbone (`peft`).
- Evaluate **zero-shot** on the *same untouched* BEIR set.
- **Deliverable:** the full ablation — off-the-shelf → +rerank → LoRA-encoder →
  LoRA-encoder+rerank. Zero-shot generalization, proven.
- **Detailed design: [Phase 2 design](#phase-2-design--lora-fine-tune-the-bi-encoder)
  below** (written 2026-08-12). It extends this milestone; nothing here is
  reversed.

**Phase 3 — The wiki demo ("ask my second brain"). ~~Planned~~ RETIRED
2026-08-14.** Kept here rather than deleted, since why it was dropped matters
more than that it existed.

- ~~Point the validated pipeline at the wiki's markdown; thin Gradio/Streamlit
  UI; qualitative zero-shot showcase — a MS-MARCO-trained retriever answering
  over a personal domain. The README front door.~~
- **Why it's retired:** its stated deliverable was *"a MS-MARCO-trained
  retriever answering over a personal domain"* — and Phase 2 measured that
  retriever to be the **worse** one on all three datasets tried. Building it
  would mean demoing the model the evidence says not to ship. The deliverable
  died with decision 3, not with the UI.
- **What replaces it:** generation moves into Phase 4 as a *measured* stage
  rather than a qualitative showcase. A Gradio front door remains optional
  later, over the best-measured config (Phase 1) rather than the fine-tuned one.
- **Knock-on:** **D6 dissolves** — it asked whether the demo is hosted and over
  what corpus, and with no wiki demo there is no private-corpus exposure to
  decide about. Phase 4 runs over NFCorpus, which is public and already cached.
  **D2 is deferred**, not resolved: if a UI is ever built it is a thin front
  door over Phase 1, and Gradio remains the recommendation.

  > **Amended 2026-08-17 — Phase 4 does *not* run over NFCorpus.** The sentence
  > above picked NFCorpus by inheritance from Phases 0–2, before anyone had read
  > a generated answer. A 20-question probe showed NFCorpus queries are not
  > questions (`turnips`, `folic acid`, "To Snack or Not to Snack?" — they are
  > NutritionFacts.org video titles), and that **67% of its queries have every
  > gold doc at a single relevance grade** (95% of all labels are grade 1). With
  > a median of 16 gold docs and no grade signal, "the top-k gold docs" is an
  > arbitrary draw — so **milestone 4b's oracle context is ill-defined here**.
  > Phase 4 evaluates on **`hotpotqa-distractor-pool`** instead: the HotpotQA
  > dev distractor paragraphs pooled into a 66,581-doc corpus (deduped from
  > 73,700 slots), 7,405 questions, exactly 2 gold docs each. It is **not BEIR
  > HotpotQA** and its numbers are not comparable to published BEIR results —
  > hence the distinct dataset id. Full reasoning in `TODO.md` D12.
  > **The public-corpus point stands**: HotpotQA is Wikipedia, so D6 stays
  > dissolved. **NFCorpus is unaffected as a *retrieval* benchmark** — Phases
  > 0–2 are untouched; document relevance is exactly what its qrels label.

**Phase 4 — Generation + LLM-as-judge (promoted from stretch to the phase).**

> Amended 2026-08-14. Was one line of "answer-quality scorers, stretch". It is
> now the project's next real phase, because the repo currently contains no LLM
> at all — every phase to date is retrieval, which is a classic-IR discipline.

**The question:** every phase so far optimised **NDCG@10** on the assumption that
it proxies answer quality. That assumption has never been tested, and this repo's
posture is that untested assumptions get measured. Phase 1 and 2 left behind
**four retrieval configurations with measured, spread-out quality and cached
candidates for each** (LoRA `0.2725`, baseline `0.3159`, LoRA+rerank `0.3407`,
baseline+rerank `0.3412`), which is an unusually clean setup for asking whether
answer quality *tracks* the metric.

The likely answer is "barely" — Phase 1's cross-encoder already absorbed 99% of
Phase 2's retrieval regression, and a generator handed ten passages may absorb
the rest. **That is the interesting outcome, not the disappointing one**, and it
is the natural next chapter of the Phase 2 finding rather than a new topic.

- **4a · generation, cached.** `generate.py`: top-10 → prompt → answer with
  citations. Cached per `(dataset, retrieval_config, generator, prompt_version)`
  — same reason `cache.py` exists, and more load-bearing here, since generation
  is nondeterministic even at temperature 0. **`prompt_version` belongs in the
  key**: editing a prompt invalidates results exactly as silently as editing
  `retrieve.py` does.
- **4b · the ceiling, before the thing.** The `oracle.py` move one level up:
  generate from the **qrel-perfect context** (the actually-relevant docs, read
  straight from the labels). That is the best this generator can do given
  perfect retrieval, and every real config is read against it rather than
  against 1.0. If oracle-context answers barely beat baseline-context answers,
  **retrieval was never the bottleneck** — and the whole ablation table has been
  optimising the wrong end of the pipeline. Uncomfortable, and worth knowing.

  > **Amended 2026-08-23 — 4b's context is padded, and the ceiling run splits in
  > two.** Read literally, "generate from the qrel-perfect context" hands the
  > model HotpotQA's 2 gold docs while every real config gets 10 retrieved
  > passages. That contrast changes three things at once: whether the evidence is
  > present (the intended variable), the context length (2 passages against
  > ~1,200 tokens), and the distractor count (0 against 8–10). "Refuses less with
  > gold context" would then be indistinguishable from "refuses less with a
  > short, clean prompt", and the paired design's whole advantage — same queries,
  > so question difficulty is held constant — would be spent on a comparison that
  > is confounded on a different axis.
  >
  > There is direct reason to care about distractor count specifically: the one
  > over-refusal in the 15-query trial batch had **both** gold passages in
  > context, at ranks 2 and 9 with seven distractors between them, and failed to
  > link them (`LEARNINGS.md` 2026-08-23).
  >
  > So 4b runs **two** context builders, reported separately:
  >   - **gold-padded** — the gold docs, padded to `n_context` with non-gold docs
  >     from that query's retrieved list. Length and distractor count are held at
  >     the real configs' values, so only evidence presence varies. **This is the
  >     one the config comparison is read against.**
  >   - **gold-only** — the gold docs alone. The true upper bound, and not a
  >     controlled contrast; quote it as a ceiling, never as the counterfactual.
  >
  > The same constraint applies to any later config comparison: hold `n_context`
  > fixed across configs, or the refusal-rate column measures prompt length.
  > Note this is 4b's *ceiling* licence to read `qrels` — `oracle.py`'s rule that
  > an oracle may only reorder what retrieval found still binds everywhere else,
  > and `context_for` must keep reading `results`.

  > **Amended 2026-08-29 — 4b reports THREE measurements, and one of them does
  > not exist yet.** The bullet above says a real config is read against the
  > ceiling if it "barely beats" it, and never defines *beats*. 4d (added
  > 2026-08-28) does define its columns, so 4b was silently inheriting them.
  > Naming them here, because the two must match or the comparison 4b exists to
  > enable is not a comparison:
  >
  >   - **refusal rate** — did it answer at all. Free, no judge, no scorer, and
  >     the signal already known to move hardest with evidence (9.8 / 63.4 /
  >     87.5% across `gold_in_context` 2/1/0).
  >   - **grounded rate** — is the rationale supported by the passages it cites.
  >     The judge, `mistral-small` digest `8039dd90c113`, rubric v2.
  >   - **correctness** — token-F1 of the short answer against HotpotQA's gold,
  >     over non-refused rows only, **per D14**.
  >
  > **Why all three rather than the cheapest one.** They answer different
  > questions and are free to disagree: a model can be grounded and wrong
  > (faithfully reasoning from a passage that does not carry the answer — the
  > Vienna case), or correct and ungrounded (right answer, invented
  > justification). Collapsing them hides exactly the behaviour 4b is asking
  > about. **Where they diverge is a finding, not noise.**
  >
  > **D14 is load-bearing here specifically.** Refusals leave the correctness
  > denominator, with refusal rate published beside it. That convention matters
  > more in 4b than anywhere else in the project, because 4b *deliberately moves
  > refusal rate* — it changes what evidence is present. Let refusals score 0
  > and correctness stops being a third measurement and becomes a noisy restatement
  > of the first.
  >
  > **Prerequisite, and it is real: `token_f1` is not implemented.**
  > `generate.py:71` reserves the field and `hotpot_pool.py:214` supplies the gold
  > answers, but nothing scores it. Write and TEST the scorer before running 4b —
  > its normaliser is load-bearing, not cosmetic (article and punctuation
  > stripping moved the trial figure 0.600 → 0.711; `LEARNINGS.md` 2026-08-19),
  > and a normaliser checked by eye is the same class of well-formed-but-wrong
  > output as a leaked `corpus_id`. It needs **no new hand labels** — the gold
  > answers are already in the pool, and the fixture sheet stays blind to them by
  > design.

- **4c · validate the judge before trusting it.** The part that makes this the
  same project rather than a tutorial, in three checks:
  1. **A known-answer fixture** — the analogue of the five-doc `NDCG@10 = 1.0`
     test. Grounded condition (answer from qrel-relevant docs) vs ungrounded
     (answer from randomly sampled irrelevant docs). **A judge that cannot
     separate those is broken**, and this finds out in seconds rather than after
     scoring 500 answers. Fixture first, full run second.
  2. **Human calibration** — ~50 hand-labelled (context, answer) pairs, report
     **Cohen's κ**. Raw accuracy lies under class imbalance: if 90% of answers
     are faithful, a judge that always says "faithful" scores 90% and knows
     nothing. Publishing κ is the point — *"we used LLM-as-judge"* is a method,
     *"our judge agrees with human labels at κ = 0.71 on 50 examples"* is a
     measurement.
  3. **Cross-model judging** to dodge self-bias — and *measure* the gap on a
     sample rather than assume it.

  > **Amended 2026-08-28 — how the three checks actually went.**
  >
  > **Check 1 (known-answer separability fixture) was never run**, and was
  > overtaken rather than skipped. Its job was a seconds-fast "can this judge
  > tell grounded from ungrounded at all" gate built from *synthetic* conditions
  > (answers from qrel-relevant docs vs from randomly sampled irrelevant ones).
  > Check 2's hand-labelled fixture answers the same question on **real
  > generations**, which is strictly harder and strictly more informative: the
  > synthetic contrast is easy by construction, so passing it would not have
  > predicted `gemma3:4b`'s failure on real rows. What the plan lost is the
  > *cheapness* — check 1 was meant to cost seconds, and hand-labelling 30 rows
  > cost an evening. If a third judge is ever added, build check 1 then.
  >
  > **Check 2 ran at n=30, not ~50** — and, because the two axes are disjoint,
  > the certifying axis is **n=16** (`grounded`; the other 14 rows are refusals
  > scored on `refusal_ok`). The consequence is exactly what this line worried
  > about: the winner's bootstrap CI is `[+0.09, +1.00]`, which clears zero and
  > little else. **This supports a ranking, not a value for κ.** The sizing
  > heuristic below ("decide by labelling 20 first") was followed in spirit —
  > seed 1 was labelled, read back, and the rubric sharpened as a result.
  >
  > **Check 3 half ran.** Two candidate judges were scored against the human
  > labels (`mistral-small` κ 0.586, `gemma3:4b` κ −0.257 — issue #2), which
  > settles *selection* and honours the no-self-bias constraint. The **gap
  > between two judges was not measured**, because only one judge was kept. That
  > half is parked as `TODO.md` D10, and the open question there is not "can we"
  > but "what is a judge–judge agreement column *for*" — as a claim it is weak
  > (two judges sharing a prompt and rubric can agree and both be wrong, which
  > `gemma3:4b` demonstrates: perfectly consistent, anti-correlated with truth),
  > as a diagnostic for routing contested rows to a human it is useful.
  >
  > **The unplanned finding, recorded because no check would have caught it.**
  > Both candidates emitted well-formed, parseable rulings on 30/30 rows and
  > passed a 4/4 format smoke test; one of them still landed *below chance*.
  > Parse rate is the cheap monitor one would actually automate and it cannot
  > separate these two models. Only labels can. Same class as `NDCG@10 = 0.0000`
  > from leaked `corpus_id`s — well-formed output, wrong meaning, nothing
  > raises.
- **4d · the config comparison — the milestone this phase exists for.**
  **Added 2026-08-28**, because it was only ever implied by the Scope and
  Deliverable bullets below and never given a design. Score the four retrieval
  configs Phases 1–2 left behind (LoRA `0.2725`, baseline `0.3159`, LoRA+rerank
  `0.3407`, baseline+rerank `0.3412`) with the validated judge, and answer the
  headline question: **does answer quality track NDCG@10?**

  - **Run 4b first, and read it before running 4d.** If gold-padded context
    barely beats baseline context, retrieval was never the bottleneck — and the
    expected effect size across four configs spanning `0.2725`–`0.3412` is then
    *smaller than that*, because those configs differ by less than the gap to
    perfect evidence. Knowing the ceiling first tells you whether 4d is a
    measurement or an underpowered one, and it is cheaper to learn that from two
    runs than from five.
  - **Hold `n_context` fixed across every config**, per the 4b amendment of
    2026-08-23. A refusal-rate column that varies with prompt length measures
    prompt length.
  - **Report per config:** grounded rate (n and CI), refusal rate, **correctness
    (token-F1 over non-refused rows, per D14)**, and refusal rate stratified by
    `gold_in_context`. The strata are the diagnostic; the headline is whether the
    grounded column moves monotonically with the NDCG column.
    *(Correctness added 2026-08-29 with the 4b amendment above — 4b and 4d must
    report the same columns, or the ceiling 4b establishes is not one 4d can be
    read against.)*
  - **Judge:** `mistral-small`, digest `8039dd90c113`, rubric `v2`. Pin both in
    the cache key — a judge swapped or a rubric edited mid-sweep makes the
    configs incomparable, which is the same failure the retrieval cache exists
    to prevent.
  - **Both outcomes are results.** "Grounded rate tracks NDCG" validates four
    phases of optimisation. "It does not" is the more interesting finding and
    the one the phase predicts, and it says the ablation table has been
    optimising a proxy. The failure case is neither of those: it is a comparison
    too underpowered to distinguish them, which is what the 4b-first ordering and
    the label-widening note are there to avoid.

- **The second judge is deferred — decided 2026-08-28.** D10's remaining half
  ("two judges, so their agreement becomes a result") is **not being built now**,
  and the reasoning is worth keeping because the number is more attractive than
  it is useful. **Judge–judge agreement is not evidence of correctness**: two
  judges sharing a prompt, a rubric and the same passages can agree with each
  other and both be wrong. This repo has the demonstration in hand — `gemma3:4b`
  was perfectly *consistent* (30/30 well-formed rulings, a stable strict-sounding
  policy) and *anti-correlated with the truth*. Two such judges would have
  produced a reassuring agreement column and no information.

  - **What to spend on instead:** widening the label set. The certifying axis is
    n=16 with CI `[+0.09, +1.00]`; that interval, not the absence of a second
    judge, is what limits every claim Phase 4 can make.
  - **If a second judge is ever added, it is a diagnostic, not a claim** — use
    disagreement to route contested rows to a human, and **do not filter them
    out.** Judges disagree on the hard, ambiguous rows, which is where the
    interesting failures live; dropping them raises the measured grounded rate
    by deleting its counterexamples. Worse for 4d specifically: disagreement rate
    almost certainly varies across configs (worse retrieval → messier context →
    more ambiguous rows), so filtering would drop *more* rows from the *worse*
    configs and flatter them, compressing the very gap 4d measures. A
    denominator that shrinks differently per config is not comparable across
    configs — the same error as reading `NDCG@100` against `NDCG@10`. Report
    contested rows as a stratum, with `n` given both ways.

- **Scored axes:** **faithfulness/groundedness** (every claim supported by the
  retrieved passages — needs no external ground truth) and **answer relevance**.
  **Correctness is deliberately excluded**: NFCorpus qrels label *document
  relevance*, not answer correctness, so scoring it would mean inventing ground
  truth — the thing decision 2 exists to prevent.

  > **Amended 2026-08-28 — what is actually scored is two axes, and neither is
  > relevance.** As shipped, the judge rules on **`grounded`** (answered rows:
  > every clause supported by the passage it cites) and **`refusal_ok`** (refused
  > rows: was declining right given these passages). They are disjoint because a
  > refusal makes no claims and would score trivially grounded otherwise — which
  > is also why the certifying axis is n=16 of 30 labelled rows.
  >
  > **Answer relevance was never built** and no longer has a champion: it was
  > listed here by analogy with RAGAS-style axis sets, and nothing downstream
  > asks for it. Treat it as dropped unless someone argues it back in.
  >
  > **Correctness has been excluded, reinstated, and is still not built.** The
  > 2026-08-17 amendment (below, in the scope-discipline section) correctly
  > reinstated it once the dataset moved to `hotpotqa-distractor-pool`, which
  > *does* ship gold short answers — but the scorer does not exist yet. It is
  > carried in `TODO.md` with two constraints worth not re-deriving: quote it as
  > **0.7619 over 14 with a 6.7% refusal rate** under the settled convention
  > (refusals leave the denominator), and its normaliser is load-bearing
  > (0.600 → 0.711 on the gold-context probe), so test it rather than eyeball
  > it.
- **The refusal gate — a measurement, not a framework.** If the retrieved
  context does not support an answer, refuse rather than hallucinate, and
  **report refusal rate per retrieval configuration**: does worse retrieval
  cause more refusals? That feeds the headline question directly, and it is the
  one place the generation half can register retrieval quality even if
  faithfulness scores come out flat.

  > **D11 resolved 2026-08-14 — C, skip LangGraph.** This bullet briefly
  > justified a LangGraph dependency on the grounds that the gate is a genuine
  > conditional edge, unlike the otherwise-linear `retrieve → rerank →
  > generate`. That reasoning was sound and the conclusion is still no: the
  > skill is demonstrated in other projects, so a framework carried here would
  > be re-demonstrating it at the cost of a dependency. **The gate ships as a
  > plain conditional.** Its value was always the refusal *rate*; the graph was
  > only ever one way to spell the `if`. A judge-driven re-retrieval loop —
  > the only genuine *cycle*, and the one thing that would actually have
  > justified a graph library — stays out of scope unless the gate proves
  > insufficient.
  >
  > ~~**Still open:** whether the gate ships *at all*. If NFCorpus answers turn
  > out trivially groundable it never fires and the metric is constant. Decide
  > by reading ten generated answers, per the risks below.~~
  >
  > **Settled 2026-08-28 — it ships and it fires.** Moot on the NFCorpus premise
  > (the dataset moved; see the scope amendment below) and moot on the worry:
  > refusal rate is **38.0%** over n=100 and is anything but constant — it tracks
  > how much gold retrieval delivered (9.8% / 63.4% / 87.5% at 2 / 1 / 0 gold
  > passages in context). The "read ten answers first" instruction was followed
  > and paid for itself twice, finding the prompt-template leak and a refusal
  > that was a comprehension failure in disguise (`LEARNINGS.md` 2026-08-23).
- ~~**Scope:** ~100 NFCorpus queries (sampled once, seeded, fixed), 4 retrieval
  configs + the oracle-context ceiling, so ~500 generations and ~600 judge
  calls. Cheap on a small fast model.~~ **The ~50 hand labels are the real cost
  and the only part that cannot be automated — which is exactly why they are
  what makes the result credible.**

  > **Amended 2026-08-28 — the dataset is `hotpotqa-distractor-pool`, not
  > NFCorpus, and the cost line was wrong in both directions.** The dataset moved
  > on 2026-08-17 (D12, amended in the milestone above); this bullet was missed
  > and kept saying NFCorpus. The shape of the scope survives — ~100 queries
  > sampled once and seeded, 4 retrieval configs plus the ceiling — with three
  > corrections:
  >
  > - **The ceiling is two runs, not one.** 4b splits into **gold-padded** (the
  >   controlled contrast the config comparison is read against) and
  >   **gold-only** (a true upper bound, never a counterfactual). See the 4b
  >   amendment of 2026-08-23. `n_context` must be held fixed across every
  >   config, or the refusal-rate column measures prompt length.
  > - **"Cheap on a small fast model" did not survive contact.** The judge that
  >   passed the fixture runs at **42.45 s/call**, so 100 queries is ~71 min per
  >   config and ~5.9 h for four configs plus the ceiling. That is affordable
  >   unattended, but it is an overnight run, not a coffee break — and the cheap
  >   fast candidate was rejected on agreement, not on speed (issue #2).
  > - **The hand labels came in at 30, not ~50** (16 on the certifying axis).
  >   The bullet's instinct was right that they are the real cost; what it
  >   understated is the consequence of stopping short — CI `[+0.09, +1.00]`.
  >   **Widening the label set is the highest-value remaining spend on judge
  >   trust**, ahead of adding a second judge.
- **Known risks, recorded up front.** The judge may be too coarse to
  discriminate (the ungrounded fixture catches this *before* the spend).
  NFCorpus makes awkward RAG — biomedical abstracts, terse queries — so
  **generate ten answers by hand and read them before building anything**. And
  the finding may be "nothing tracks anything", which is still a result.

  > **Amended 2026-08-28 — both named risks fired, and the mitigation worked
  > both times.** "NFCorpus makes awkward RAG" was correct and is *why* Phase 4
  > left it: a 20-question probe found the queries are not answerable questions
  > (D12, 2026-08-17). Recorded here rather than struck, because a risk that was
  > written down and then actually caught is the argument for writing them down.
  >
  > "The judge may be too coarse to discriminate" also fired — `gemma3:4b` landed
  > at κ −0.257 — but **not through the mechanism this bullet expected.** The
  > parenthetical trusts the ungrounded fixture to catch it before the spend;
  > that fixture was never built (see the 4c amendment), and the failure would
  > have slipped past a format check regardless, since the model parsed 30/30
  > rows cleanly. What caught it was hand labels. Read together: the risk
  > register was right about *what* could go wrong and wrong about *what would
  > notice*.
- **Deliverable:** an answer-quality column beside the retrieval column in the
  ablation table, a published judge-vs-human κ, and a stated answer to whether
  NDCG@10 bought anything downstream.

## Phase 2 design — LoRA fine-tune the bi-encoder

> Written 2026-08-12, before any Phase 2 code. Extends the Phase 2 milestone
> above. Every library API named here was verified against the *installed*
> versions (`sentence-transformers` 5.7.0, `transformers` 5.14.1, `torch` 2.13.0)
> and the live HF hub, not from recall — the versions are in `uv.lock`.

### What Phase 2 has to prove

Phase 1 proved a *pre-trained* reranker helps. Phase 2's claim is stronger and
easier to fake: **a retriever fine-tuned on MS MARCO retrieves better on a
domain it never saw.** The number that supports it is NDCG@10 on NFCorpus — the
same set Phases 0 and 1 used, which means NFCorpus is the *test* set and must
stay untouched by every training decision, not just by the training data.

That is the whole difficulty. The training data never touches NFCorpus
(design decision 3 already covers it), but *hyperparameter choice* is a second,
quieter channel from the test set into the model. The rules below close it.

### The measurement rules (pre-committed, before any number exists)

These are Phase 2's equivalent of "an oracle may only reorder what retrieval
found, never add to it." They constrain the experiment, not the code.

- **R1 — No hyperparameter is chosen by looking at NFCorpus.** LoRA rank, alpha,
  learning rate, batch size, epochs and checkpoint step are all selected on a
  held-out **MS MARCO dev slice** only. If NFCorpus NDCG picks the config, the
  result stops being zero-shot and becomes a test-set fit, and the headline
  claim dies quietly — nothing raises, the table just gets a number it hasn't
  earned. Same failure shape as a leaked `corpus_id`: well-formed and wrong.
- **R2 — Every NFCorpus number obtained gets reported.** No running three
  configs and tabling the best. If more than one config is ever evaluated on
  NFCorpus, all of them appear in `LEARNINGS.md`, with the R1-selected one
  marked as the config the table quotes.
- **R3 — The ceiling moves, so it gets re-measured.** A fine-tuned encoder
  returns a *different top-100*, so the 0.6263 oracle ceiling no longer applies
  to it. `oracle.py` is re-run on the LoRA candidate set and **both ceilings are
  reported.** Reusing the old one would compare Phase 2's gain against a bound
  computed for someone else's candidates.
- **R4 — The fine-tune is allowed to lose.** MS MARCO tuning making an
  out-of-domain medical set *worse* is a real and publishable outcome (it is
  what "zero-shot generalization" is testing for). The row ships either way; a
  negative result gets the same README treatment Phase 1's dense-query
  regression got.

### Training data

**`sentence-transformers/msmarco-bm25`, config `triplet`** — 502,931 rows,
~0.23 GB, columns exactly `query` / `positive` / `negative` (verified via the
HF datasets-server; the negatives are BM25-mined, i.e. hard-ish rather than
random). This is the shape `MultipleNegativesRankingLoss` consumes directly:
column order is (anchor, positive, negative), no collation code of our own.

- A `triplet-hard` config exists at 19.1M rows / 3.2 GB. **Out of scope** — it
  is a bigger download than the whole rest of the project and Phase 2 is not a
  "how good can MS MARCO training get" study.
- **How many triples we actually train on is not decided here**, deliberately.
  It is a wall-clock question about MPS throughput, and a guessed number in a
  plan is worth less than a measured one. The *procedure*: the smoke run below
  reports steps/sec, and the subsample is sized to a **training budget of
  roughly 30–60 minutes**, recorded in `LEARNINGS.md` with the throughput that
  justified it.
- **Dev slice for R1:** a held-out shard of the same `triplet` config, never
  trained on, scored with an in-training ST evaluator. It is the *only* signal
  allowed to choose a config.

### The LoRA setup

Confirmed present in the installed `sentence-transformers` 5.7.0:

- `SentenceTransformer.add_adapter(...)` — the PEFT entry point, via
  `base/peft_mixin.py`, which delegates to the transformers
  `PeftAdapterMixin`. Takes a `peft.LoraConfig`. Sibling methods
  `set_adapter` / `disable_adapters` / `enable_adapters` exist and matter for
  testing (see below).
- `SentenceTransformerTrainer` + `SentenceTransformerTrainingArguments`
  (exported from the package root) — these sit on the HF `Trainer`, which is
  why `accelerate` becomes a real dependency.
- `MultipleNegativesRankingLoss` at
  `sentence_transformers.sentence_transformer.losses`.
  `CachedMultipleNegativesRankingLoss` is also available and is the fallback if
  MPS memory caps the batch size — MNRL's signal quality scales with batch size
  (more in-batch negatives), so a small batch is a real quality cost, and the
  cached variant buys a large effective batch back.

**Adapt the backbone, not the pooling.** LoRA targets the transformer's
attention projections; the pooling layer is parameter-free and unchanged. Rank
and target modules are R1-governed — chosen on the dev slice.

### Checkpoint naming, and the cache-key trap wearing a new hat

`cache.py` keys on `(dataset, model_name, top_k)` and slugs `/` to `__`, so a
local checkpoint path works mechanically. But **two LoRA runs with different
hyperparameters written to the same directory produce the same cache key** —
the second run silently reads the first run's retrieval results. This is
exactly the "cache key does not cover `retrieve.py`" trap in new clothing, and
it is worse here because the stale value looks like a legitimate metric.

**Convention:** checkpoints live at `models/lora-<tag>/` where `<tag>` encodes
the config that was varied (e.g. `lora-r16-lr2e5-v1`), and that path is what is
passed as `model_name`. A config change means a new directory, therefore a new
cache key, therefore no silent reuse. `models/` is gitignored (added 2026-08-12
alongside this design) — adapters are rebuildable, and the lockfile plus the
recorded config is what makes them reproducible.

**Why the checkpoint name is a sufficient fix, checked rather than assumed:**
retrieval is the *only* cached stage. `rerank.py` and `oracle.py` both
recompute from the `results` they are handed on every run
(`rerank.py:356` re-scores unconditionally), so there is no second stale-value
channel downstream — get the retrieval key right and the whole chain is honest.
Two consequences: reranking the LoRA candidates costs full cross-encoder
inference every time, which is a wall-clock cost to expect but not a
correctness risk; and the stray `cache/reranked_nfcorpus.json` on disk is an
**orphan from an earlier iteration that nothing reads or writes** — delete it
rather than reasoning about it later.

### The ablation rows Phase 2 lands

| Row | What it isolates |
|---|---|
| Phase 2 · LoRA encoder | the fine-tune alone, vs Phase 0's `0.3159` |
| Phase 2 · LoRA + rerank | the full stack, vs Phase 1's `0.3412` |
| _reference_ · `msmarco-MiniLM-L6-cos-v5` | a **fully** MS-MARCO-specialized bi-encoder, off the shelf, zero training cost |
| _ceiling_ · oracle over LoRA candidates | R3 — the bound that actually applies to the new candidate set |

The reference row is the cheapest high-value row available and it is already
anticipated in the model table above. Without it, Phase 2 can only say "LoRA
moved the number." With it, Phase 2 says **"LoRA captured X% of what full
MS MARCO specialization achieves, at adapter cost"** — which is the actual
question anyone reading a LoRA result has. Verified on the hub as
`sentence-transformers/msmarco-MiniLM-L6-cos-v5`.

Both Phase 2 rows must be reported, not just the better one: the fine-tune and
the reranker can overlap (both were trained toward MS MARCO relevance), so
`LoRA + rerank` may well be *less* than the sum of the two gains. That
interaction is a finding, not a disappointment.

### Two separate checks, deliberately not conflated

A single NFCorpus number cannot distinguish "the training loop is broken" from
"training worked and did not transfer." So Phase 2 reports both:

1. **Did training work?** Falling train loss + improving MS MARCO dev-slice
   score. In-domain, expected to move, and diagnostic only.
2. **Did it generalize?** NFCorpus NDCG@10. The headline. Out-of-domain, and
   under R4 allowed to go either way.

Check 1 passing while check 2 fails is the *interesting* result. Check 1
failing means there is a bug, and no NFCorpus number should be quoted at all.

### D5 (dense-query blending) as an ablation axis

`TODO.md` D5 — the cross-encoder subtracts 1.06 NDCG points on dense queries —
carries into Phase 2 as a **blend weight between bi-encoder and cross-encoder
scores**, and it applies over *both* candidate sets, base and LoRA. It is
independent of the fine-tune and needs no training, so it can land before,
during or after. Worth noting it may partly dissolve on its own: a better
first-stage encoder changes which queries are dense in the top-100.

### What pins Phase 2 in tests

Mirroring how `retrieve.py` and `rerank.py` are pinned — fixture-only, no
download, no training:

- **Triple shape** — the training rows reach the loss as (anchor, positive,
  negative) in that column order. A silent column swap trains the model to
  rank negatives up, and the loss curve still looks fine.
- **The adapter actually changes embeddings** — encode a fixture sentence with
  adapters disabled and enabled; the vectors must differ. Guards the failure
  where LoRA attaches to nothing, training "succeeds," and the fine-tuned model
  is byte-identical to the baseline.
- **Checkpoint path → distinct cache key** — two tags must not collide, per the
  trap above.

### Build order — smoke run before any config is committed

1. **Smoke run first.** A few hundred steps on a tiny triple subset on MPS,
   just to prove the ST-trainer-on-`accelerate`-on-MPS path runs and to
   *measure* throughput. MNRL on MPS is exactly the kind of thing that fails in
   ways a plan cannot predict, and the subsample size depends on the number it
   produces.
2. Size the real run from that throughput (30–60 min budget), train, save to
   `models/lora-<tag>/`.
3. Retrieve on NFCorpus with the checkpoint → `evaluate.py` (new cache key,
   so no `--refresh` needed — but check the log says MISS).
4. Re-run `oracle.py` on the new candidates (R3), and `rerank.py` for the
   LoRA + rerank row.
5. Add the reference-model row — no training, just a second `evaluate.py` run.
6. README table + `LEARNINGS.md`, including every number R2 requires.

`finetune.py` is the new module (already anticipated in the layout below);
`evaluate.py`, `oracle.py` and `rerank.py` need a `--model` argument rather
than any structural change, since all three already take `results` and key off
a model name.

### Dependencies and the lockfile

Phase 2 adds **`peft`**, **`accelerate`**, and **`datasets`** — the last is
currently installed only transitively, which Phase 2 must stop relying on and
declare. They go in a **`train` extra**, so an eval-only install stays light and
the training stack is opt-in like `tracing` already is.

Because the repo's position is that its numbers are version-dependent, the
re-lock lands in **a commit that touches nothing else**, so any metric drift
afterwards is attributable to the dependency change rather than tangled with
code. Re-run `pytest` and re-check the Phase 0/1 numbers against the table
immediately after locking; if they moved, that is the finding to record before
any training starts.

## Repo layout (actual — installable `src/` package)

```
retrieval-lab/
├── README.md              # pitch + the ablation table (the money shot)
├── LEARNINGS.md           # devlog — findings recorded while they're fresh
├── pyproject.toml         # deps + editable install (`uv pip install -e '.[dev]'`)
├── docs/plan.md           # this file
├── src/retrieval_lab/
│   ├── observability.py   # weave init + @op shim (from the mise pattern)
│   ├── data.py            # BEIR load (ships the qrels labels)
│   ├── retrieve.py        # Phase 0 · bi-encoder embed + top-k
│   ├── cache.py           # on-disk retrieval cache (pins candidates across phases)
│   ├── evaluate.py        # Phase 0 · entrypoint: load → retrieve → metrics → headroom
│   └── oracle.py          # Phase 1 · entrypoint: the perfect-rerank ceiling
├── tests/                 # fixture suites (pytest, no download)
├── data/                  # BEIR downloads (git-ignored)
├── cache/                 # cached retrieval runs (git-ignored)
└── reports/               # dated ablation tables
```

**Amends the original flat-layout decision** (see SESSIONS.md 2026-07-15, which
chose root-level files to dodge package-import traps). Reversed 2026-08-07 once
the module count reached seven: `pyproject.toml` + an editable install makes
imports unambiguous rather than fragile, which was the actual concern. Cost is
one setup step and `python -m retrieval_lab.<entrypoint>` invocations.

Planned additions as phases land: ~~`rerank.py` (Phase 1), `finetune.py`
(Phase 2), `app/` (Phase 3).~~ **Updated 2026-08-14** — `rerank.py` and
`finetune.py` landed; `app/` is retired with Phase 3. Phase 4 adds
`generate.py` (retrieve → prompt → cited answer, cached on
`prompt_version` among the rest) and `judge.py` (the LLM-as-judge scorers *and*
the κ calibration against hand labels — the calibration is not a notebook, for
the same reason `rerank.py --breakdown` is not: every figure quoted in the
README has to be printed by code someone else can re-run). Hand labels live in
`data/labels/` and are **committed**, not gitignored — they are the one input
here that cannot be regenerated.

> **Amended 2026-08-28 — Phase 4 landed as four modules, not two.**
> `hotpot_pool.py` builds the `hotpotqa-distractor-pool` corpus (the dataset
> move of 2026-08-17 needed a corpus that did not exist); `generate.py` is as
> described; **`fixture.py`** draws, labels and verifies the hand-labelling
> sheet, which the original line folded into `judge.py`; and `judge.py` keeps
> the judging and the κ calibration (`--smoke`, `--bakeoff`). The split is
> load-bearing: `fixture.py` owns the *rubric* and the label format, `judge.py`
> consumes them, and a judge graded against labels it was handed a different
> rubric for measures rubric disagreement rather than judge quality. The
> committed-labels rule held — `data/labels/` is tracked.

## Tech stack

- **`sentence-transformers`** — bi-encoder, cross-encoder, MS MARCO training
  losses (MNRL). Recent versions integrate `peft` for LoRA adapters.
- **`beir`** — datasets (with qrels) + `EvaluateRetrieval` (NDCG/MAP/Recall/
  Precision via `pytrec_eval`).
- **`peft`** — the LoRA adapter on the encoder backbone.
- **`weave`** — tracing + `weave.Evaluation` (reuse the `observability.py` shim
  from the mise project).
- ~~**Anthropic Claude** — the Phase 4 answer generator, and a *different* model
  as the Phase 4 judge (cross-model, to dodge self-bias — with the gap measured
  on a sample rather than assumed).~~
  **Amended 2026-08-28 — Phase 4 runs entirely on local models via Ollama.**
  Generator **`qwen3:8b`**; judge **`mistral-small`** (23.6B, digest
  `8039dd90c113`), chosen over `gemma3:4b` on the labelled fixture rather than on
  a spec sheet (issue #2). The cross-model principle survives intact and is
  *why* the judge is not a qwen3 model — a judge sharing the generator's
  pretraining corpus is the self-bias this line was written to dodge. What
  changed is the vendor, not the reasoning: local models keep the phase free,
  reproducible offline, and pinnable by digest. An API judge (Sonnet 5) is still
  a live option as a *second* judge; see `TODO.md` D10.
- ~~**`langgraph`** — Phase 4 only, for the refusal gate's conditional edge.~~
  **Dropped 2026-08-14 (D11-C).** The skill is demonstrated in other projects;
  the refusal gate ships as a plain conditional instead. No orchestration
  framework in this repo — the pipeline is a function chain and stays one.
- Vector search: brute-force in-memory cosine is fine at BEIR-small scale; add
  FAISS only if a corpus grows.

## Scope discipline — explicitly OUT (v1)

- No training from scratch; **LoRA only**, never full fine-tuning.
- No full-scale MS MARCO *evaluation* — small BEIR set for eval, MS MARCO only
  as *training* data.
- No production/multi-user app — this is a lab. (**Amended 2026-08-14:** and no
  longer a demo either; Phase 3 is retired.)
- No chunking-strategy rabbit hole — BEIR docs are pre-chunked, and with the
  wiki demo retired there is no unchunked corpus left in scope. **Amended
  2026-08-14**, previously "revisit only for the Phase 3 wiki demo."
- **Added 2026-08-14:** no answer-*correctness* scoring in Phase 4. NFCorpus
  qrels label document relevance, not answer correctness; scoring it would mean
  inventing ground truth, which is what decision 2 exists to prevent.

  > **Amended 2026-08-17 — correctness is back IN, because its exclusion was
  > never about correctness.** Read the rule above: it bars correctness scoring
  > *because NFCorpus ships no answer labels*, so scoring it would mean
  > inventing ground truth. With Phase 4 moved to `hotpotqa-distractor-pool`
  > (D12), the ground truth is **shipped with the dataset** — a gold short
  > answer per question. Nothing is invented, so decision 2 is satisfied rather
  > than bent, and correctness joins faithfulness and answer relevance as a
  > scored axis.
  >
  > **This does not shrink 4c.2's ~50 hand labels.** Correctness and
  > faithfulness are orthogonal: an answer can be right while citing passages it
  > never used (the probe produced exactly this — a "fantasy" answer citing a
  > claim absent from the cited passage). Gold answers grade the *answer*; the
  > κ grades the *judge*, whose axis is groundedness, and no dataset ships a
  > label for that. What the gold answers do buy is **stratified sampling**:
  > draw the ~50 labels across correct and incorrect answers rather than at
  > random, so the class imbalance 4c.2 warns about cannot hollow out κ.
  >
  > **Scoring wrinkle:** gold answers are short spans ("Nelson County") and the
  > generator emits paragraphs. Naive exact-match scores 0 on everything — use
  > normalized token-F1 over a short answer, prompted or extracted.

## Open decisions

- ~~**Eval dataset:** NFCorpus vs SciFact.~~ **Settled 2026-08-14 — both, plus
  FiQA.** NFCorpus was the primary throughout; SciFact and FiQA were added to
  test whether the Phase 2 finding replicated. It did, on all three.
- ~~**Demo UI:** Gradio vs FastAPI+React.~~ **Deferred 2026-08-14** with Phase 3
  (`TODO.md` D2). If a UI is ever built it is a thin front door over the
  best-*measured* config (Phase 1), and Gradio remains the recommendation.

~~**Live for Phase 4:**~~ **All three settled 2026-08-28** — kept for the
reasoning, struck because none is open:

- ~~**Which judge model, and which generator?**~~ **Settled:** generator
  `qwen3:8b`, judge `mistral-small` (digest `8039dd90c113`), different families
  so the self-bias constraint holds. The "judge should be the stronger of the
  two" instinct was right but for an unmeasured reason — see the 4c amendment
  above; the 4B candidate failed on *discrimination*, not on format.
- ~~**How many hand labels?**~~ **Settled at 30 (16 on the certifying axis)**,
  and the worry was justified: CI `[+0.09, +1.00]`. Widening the label set is
  now the highest-value spend on judge trust, ahead of adding a second judge.
- ~~**Does the refusal gate ship?**~~ **Yes, and it fires.** The premise is moot
  twice over: Phase 4 left NFCorpus for `hotpotqa-distractor-pool` (amended
  2026-08-17), and the gate is not a constant column — refusal rate is 38.0%
  over n=100 and tracks how much gold retrieval delivered (9.8% / 63.4% / 87.5%
  at 2 / 1 / 0 gold passages in context). The "read ten answers first" advice
  was taken and paid for itself: reading them is what found the prompt leak and
  the misread-context refusal (`LEARNINGS.md` 2026-08-23).

## Learning resources to line up

- Sentence-BERT paper — the bi/cross-encoder foundation.
- `sentence-transformers` MS MARCO training + `beir` eval docs.
- W&B "LLM Apps Evaluation" course — for Phase 4.
