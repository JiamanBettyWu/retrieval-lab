# Session log

Reverse-chronological journal — one dated entry per working session: what got
done, what was decided and why. History only; for what to do next see
[TODO.md](TODO.md).

Older entries archived in [`sessions/`](sessions/) — 2026-07-15 through
2026-08-08 (Phase 0 scaffold through Phase 1 landing) live in
[`sessions/2026-07-15--2026-08-08.md`](sessions/2026-07-15--2026-08-08.md).

---

## 2026-08-14 (later — three datasets, a refuted prediction, and Phase 3 retired)

Continues the entry below, which closed at the first handoff (`aa0935b`). What
follows started from a question rather than a task: **the Phase 2 results were
unflattering, so was Phase 3 still worth doing, or was the write-up enough?**
The answer turned out to be neither.

**Replication, and a prediction committed before the run.** The Phase 2 finding
was n=1 dataset, and it had a competitor it could not rule out — NFCorpus is a
known weak spot for MS MARCO-trained models, so the "specialisation axis" might
belong to the dataset rather than to specialisation. The discriminating test
needed a BEIR set whose *query distribution* resembles MS MARCO's. FiQA fits;
SciFact was added as a cheap second point. **The prediction and its falsifier
were written into `LEARNINGS.md` and committed as `66e798c` before either ran**,
so neither could be adjusted afterwards — and R2 was explicitly extended past
NFCorpus, because evaluating on new datasets until one flatters the fine-tune is
exactly the search this repo argues against.

SciFact reproduced the ordering. **FiQA refuted the prediction** (`5575509`): the
control lost there by the largest margin of the three, on the dataset chosen
because it should have reversed. Numbers and the replacement mechanism are in
`LEARNINGS.md` 2026-08-14 ("the prediction was wrong, and the mechanism is
breadth, not domain"). The short version is that `all-MiniLM-L6-v2`'s card
records 1,170,060,424 training pairs against the control's MS MARCO alone, so
this is the cost of **narrowing a training distribution**, not of crossing a
domain boundary — which also explains the saturation across a 50x lr range
better than the domain story did, and strengthens D8 rather than competing
with it.

**Two measurement facts fell out of writing a docstring.** `dev_loss()` was
scaffolded (left `NotImplementedError` per the working split) for the half of
the Phase 2 claim that is still unverified: the base model's dev loss was never
measured, so "traded out-of-domain quality for in-domain gain" rests on nothing
— if the base model already scores below 0.3129 there was no gain to trade.
Writing its docstring surfaced that **`per_device_eval_batch_size` defaults to 8
and does not inherit from the train batch size**, confirmed against
`training_args.bin`. Every Phase 2 dev loss was therefore measured with 7
in-batch negatives rather than 31 — so the comparison must use 8, the
random-guess floor is `ln(8) = 2.079` rather than `ln(32)`, and the selector was
lower-resolution than intended, which is a second independent reason to distrust
the `0.0051` margin that picked 1e-3 over 5e-4. Also checked and recorded: warmup
did run, despite the args object reading back as `warmup_ratio=None,
warmup_steps=0.1`.

**Phase 3 retired, Phase 4 promoted (`b52c50e`) — the real decision of the
session.** Reading `docs/plan.md` to answer the Gradio question surfaced that
**Phase 3's stated deliverable was already dead**: "a MS-MARCO-trained retriever
answering over a personal domain" is the model Phase 2 measured to be the worse
one, so building it would mean demoing what the evidence says not to ship. Its
deliverable died with locked design decision 3, not with the UI — and decision 3
itself was measured and refuted, so it now carries an amendment note rather than
a rewrite, per the repo convention that reversals say so explicitly.

The counter-argument to simply stopping was that **the repo contained no LLM at
all.** Every phase to date is retrieval, which is a classic-IR discipline, and
the stated career goal is a classic-ML→GenAI pivot; `[generate]` first appeared
in the retired phase. So Phase 4 moved from a one-line stretch goal to the next
real phase, keeping the existing discipline rather than bolting on a demo: it
asks **whether answer quality tracks NDCG@10** across the four retrieval
configurations Phases 1-2 already left cached, measures a **qrel-perfect-context
ceiling before building anything** (the `oracle.py` move one level up), and
**validates the judge before trusting it** — a grounded/ungrounded fixture, ~50
human labels, and a published Cohen's κ. The likely finding is "barely tracks",
which is the interesting outcome given Phase 1's cross-encoder already absorbed
99% of Phase 2's regression.

LangGraph was placed deliberately rather than shoehorned: `retrieve → rerank →
generate` is linear, and wrapping a linear pipeline in a graph framework for a
résumé line is resume-driven development. It earns its place at a **refusal
gate** — a real conditional edge whose refusal rate is itself a per-config
metric feeding the headline question. **The plan says outright that if the
pipeline stays linear, LangGraph does not get used**, which makes the first
Phase 4 action "generate ten answers and read them" rather than "write code".

**D6 dissolved and D2 left the file.** D6 asked whether the demo is hosted and
over what corpus; with no wiki demo there is no private-corpus exposure to
decide, and Phase 4 runs over public NFCorpus. D2 (which UI framework) is
answered in the plan for the hypothetical future case — a thin Gradio front door
over the best-*measured* config, Phase 1, not the fine-tuned one — so it is
recorded there rather than carried as live state. Also settled in the amendment:
the long-open eval-dataset question (all three used), `generate.py`/`judge.py`
added to the planned layout with **hand labels committed rather than gitignored**
(the one input here that cannot be regenerated), and answer *correctness*
explicitly excluded from scope, since NFCorpus qrels label document relevance and
inventing ground truth is what design decision 2 exists to prevent.

**Two decisions resolved at the end of the session.** **D8 -> B**: no Phase 2.5
capacity ablation; Phase 2 stays closed. The capacity-ceiling explanation for the
saturation floor remains an inference and the README says so, which is the honest
state — measuring it would deepen a mechanism rather than move the project, and
Phase 4 is what closes the GenAI gap. **D9 follows it to B**: with no further
close calls coming, a noise floor would inform nothing.

**D11 -> C: LangGraph is out entirely.** The reasoning that put it in was sound
— `retrieve -> rerank -> generate` is linear, so the framework needed a genuine
branch to justify it, and the refusal gate is one. The conclusion is still no,
for a reason the sketch had not accounted for: **the skill is already
demonstrated in other projects**, so carrying a dependency here would
re-demonstrate it at a cost. The distinction worth keeping is that **skipping the
framework did not mean skipping the branch** — the gate's value was always the
refusal *rate* as a per-config metric feeding the headline question, and a plain
conditional spells that `if` just as well. What actually leaves scope is the
judge-driven re-retrieval *loop*, the only genuine cycle and the one thing that
would have justified a graph library. `docs/plan.md` records both as dated
amendments rather than deletions, so the tech stack now reads "no orchestration
framework in this repo — the pipeline is a function chain and stays one".

Still open for Phase 4 and unchanged by either: whether the gate ships at all,
which is a question about NFCorpus rather than about architecture. If its answers
are trivially groundable the gate never fires and refusal rate is a constant
column. Ten hand-read answers settle it, which is why Phase 4 starts with reading
rather than coding.

## 2026-08-14 (Phase 2 lands — the fine-tune loses, and a control turns that into a finding)

Phase 2 completed end to end and **the fine-tune lost**: NFCorpus NDCG@10
`0.3159 -> 0.2725`, −13.7% relative. Every number and its mechanism is recorded
in `LEARNINGS.md` (2026-08-14, "the fine-tune lost, and the control is what made
that a finding") and in the new Phase 2 section of `README.md`; what follows is
the order things happened in and why each call was made.

**Reading the overnight run, and resolving D7.** The 2e-5 run finished clean —
16 eval points, monotone `0.6263 -> 0.3960`, no upturn, so `load_best_model_at_end`
had nothing to rescue. Train loss opened at ~0.70 rather than `ln(32) = 3.47`,
which said the base encoder was already competent at MNRL and this was refinement
rather than learning. The flat tail was **confounded**: the linear scheduler
decays to ~0 over `max_steps`, so "converged" and "out of learning rate" predict
the same curve. That confound is what promoted D7 from a suspicion about the
inherited `2e-5` default into a question the data could not answer, and it was
resolved by running 1e-4 and 5e-4 (dev loss `0.3443`, `0.3180`), then 1e-3
(`0.3129`) when 5e-4 won *at the edge of the grid*. A boundary winner is a wall,
not an optimum — the search only stopped once returns fell to `−0.005` per
doubling and the dev curve went non-monotone, which is the stability edge showing
up as variance rather than as divergence. **D7 resolves to lr 1e-3**, selected on
MS MARCO dev loss alone per R1. Corrected cost along the way: the smoke run's
"~30 min per config" ignored evaluation, and real throughput was ~0.85 steps/sec,
so each config was ~60 min.

**The `--model` blocker (`5f20446`, pushed).** `evaluate.py`, `oracle.py` and
`rerank.py` all hardcoded `BI_ENCODER`, so nothing could evaluate an adapter.
Each now takes `--model` (hub id or local path) threaded to `cached_retrieval`
and `load_encoder`, with `MODEL_HELP` in `retrieve.py` documenting the cache-key
caveat once. The check that mattered was not the CLI but the adapter being *live*
at inference — 22,860,672 params (base + 147,456) and cosine 0.9625 to base,
because a silently-unloaded adapter would have produced a Phase 2 number
identical to 0.3159 and an hour of blaming LoRA. The default path was verified
unchanged: same cache file, `0.3159` reproduced exactly. Two fixes rode along —
`@op` on `train()` was serialising both Datasets as call inputs and killing the
trace at 413 on an 80MB payload (`postprocess_inputs` now substitutes row count
and column order, and column *order* is the more useful record since MNRL reads
columns positionally), and `rerank.py`'s result columns read "Phase 0 / Phase 1",
true only while `--model` was hardcoded, now "retrieved / reranked".

**The sequencing constraint that shaped the whole session.** D7 option A said
NFCorpus stays untouched until a winner is picked, while the old "Pick up here"
said evaluate the adapter once. Those conflict, and resolving D7 first is what
kept R1 intact. Everything after this point spent NFCorpus looks deliberately,
with the count stated before each one — seven in total, all reported per R2.

**The hypothesis that was wrong, and what replaced it.** The predicted shape of
the damage was a slope monotone in learning rate (more MS MARCO specialisation,
worse biomedical transfer). Running all four adapters refuted it: dev loss spans
21%, NFCorpus spans 1.5%, and all four sit at ~0.272 despite embedding drift from
base growing cleanly with lr. `checkpoint-200` of the 2e-5 run (cosine 0.9998 to
base, still inside warmup) scored baseline-within-noise, bounding the other end —
so the damage accrues between step 200 and one epoch and then **saturates**. The
per-query and per-bucket diff came free from the cached runs and ruled out the D5
dense-query story: the degradation is diffuse, every bucket down, recall included.

**The control was the decision that mattered.** Six looks in, the results still
supported two incompatible readings: *the fine-tune is broken* versus *this is
what MS MARCO specialisation costs on biomedical text*. One evaluation of
`msmarco-MiniLM-L6-cos-v5` (`0.2584`, **below** the fine-tune) settled it — the
three models order along one specialisation axis and the fine-tune paid 75% of
the full penalty. Without it the write-up would have claimed a broken recipe,
which was the wrong claim. It also relocated the saturation floor: the adapters
converge at 0.272 but the axis runs to 0.258, making adapter **capacity** the
likely explanation rather than training dynamics — which is what the proposed D8
tests.

**Written up in the same session, deliberately.** `README.md` gained both Phase 2
rows, the reference row, **both ceilings** (a ceiling belongs to a candidate set,
not a dataset — reusing 0.6263 would have scored Phase 2 against a bound it never
had), and a Phase 2 section that states the method costs outright: no config was
run twice, so every number is n=1, and the `5e-4 -> 1e-3` gap of `0.0051` that
decided the winner is exactly the size where that might matter. The repo intro
was reworded from "prove each improvement" to "measure every change" — the old
phrasing only made sense while every phase won.

## 2026-08-14 (Phase 2 trains — finetune.py hand-written, first real run launched)

The Phase 2 stubs got filled in — by Betty, per the working mode set on
2026-08-12 — and the first real 100k-triple run went out overnight in her own
terminal. **No NFCorpus number exists yet, deliberately.** Commit `8a64055`,
pushed to `origin/main`; it is the only code commit of the session.

**The config, and why each knob is what it is.** `r=16, alpha=32, dropout=0.1,
target_modules=["query", "value"], bias="none"` → **147,456 trainable params,
0.65% of 22,713,216**. The count was predicted from `r x (d_in + d_out)` per
targeted Linear *before* running, and matched the model exactly — the same move
`arithmetic_ceiling()` makes in `oracle.py`, deriving the answer independently
of the thing under test so agreement means something.

`key` was dropped from `target_modules` after discussion: attention scores
depend on the product `q·k`, so adapting both gives two routes to the same
rescaling for 50% more parameters (221,184 vs 147,456). Not a correctness call
— it is an ablation axis, and `--tag q-k-v` against this one is a cheap row if
Phase 2 has budget for it.

The suffix trap in the scaffold's docstring turned out to *understate* itself:
PEFT matches `target_modules` by name suffix, and this backbone has a fourth
`dense` the docstring missed — `model.pooler.dense`, 147,840 params. It is dead
weight here (the ST `Transformer` reads `last_hidden_state` and mean-pools it,
so the sentence embedding never passes through the BERT pooler), which makes it
the sharpest illustration of the trap: `["dense"]` would have attached adapters
to a layer that **cannot affect the output**, and nothing would have raised.
The parameterless `Pooling` module at index 1 is a different object entirely and
has no weights to adapt at all.

**The bug of the session, and the reason the smoke run couldn't catch it.**
Recorded in full in `LEARNINGS.md` (2026-08-14) — dev slice taken from the
unshuffled dataset while training came from a seed-42 shuffle, giving 20% dev
contamination at n=100k but only ~1% at smoke scale. What belongs in the
narrative is the sequencing: the smoke run had already passed twice before this
surfaced, and it passed *correctly* — proving the path executes is the whole of
what `--smoke` claims. The correction also produced a stronger property than
was asked for, measured rather than assumed: every row in `msmarco-bm25/triplet`
carries a unique query, so the split is query-disjoint and MNRL's
false-negative failure mode (the loose thread from 2026-08-12) cannot occur on
this config.

**How Phase 2 satisfies R1, concretely.** The plan said hyperparameters get
selected on a held-out MS MARCO dev slice; that slice did not exist until
today. `load_triples` now returns `(train, dev)` — first `n` and last `k` of the
same shuffle, guarded by an explicit `n + k` check that raises rather than
`assert`s (a bare `assert` is stripped under `python -O`, which is the wrong
property for a guard against silent data leakage). `eval_strategy="steps"` plus
`load_best_model_at_end` extends R1 one level further, making *which checkpoint*
a dev-selected decision rather than "whatever step 3,125 happened to produce".

A conceptual point worth keeping, since it shaped the plan: hyperparameters are
**searched, not tuned**. Each config needs its own complete training run from a
fresh adapter — weights from run A cannot be carried into run B without
confounding "B is better" with "B trained twice as long" — and the winning run's
checkpoint *is* the final model. There is no final retrain step. The classic-ML
refit-on-train+dev habit was considered and rejected: it would reclaim ~10k of
110k triples and ship a model no dev number was ever measured on.

**Sizing, from measurement.** The smoke run reported **1.83 steps/sec** at batch
32 on MPS (109.8s wall clock vs 109.3s of training — startup overhead is ~0.5s,
so the extrapolation is trustworthy). That afforded ~105,344 triples in 30
minutes; rounded to **100,000** because the triple count is a recorded
hyperparameter and a round number is legible in a table where `105,344` invites
"why?". 3,125 steps, ~31 min at the real run's observed ~1.65 it/s — slightly
under smoke throughput, as expected when streaming 100k unique rows instead of
recycling 2,000.

**Three library behaviours checked against installed source rather than
recalled**, each of which fails silently: `bias="lora_only"` unfreezing base-layer
biases, `metric_for_best_model` inferring `greater_is_better` from the name, and
`fp16` no longer being CUDA-only. All three are written up in `LEARNINGS.md`.
The fp16 one is a straight correction of an assertion made earlier in this same
session — `accelerate/accelerator.py:565` lists `mps` as supported for
torch >= 2.5.0, so the scaffold's "fp16 is a CUDA thing" hint was stale. fp32
stays, now for a stated reason (GradScaler skips overflow steps, muddying the
steps/sec the smoke run exists to produce) rather than an inherited one.

**Also fixed, all in the same file:** `max_steps` needed `-1` at the *argparse*
layer, not a `-1` default on `train()` — a default only fires when the caller
omits the argument, and `main()` was passing `None` explicitly. `main()` now
calls `train()` by keyword, after a signature reorder came within one run of
swapping `epochs` and `max_steps` positionally (200 epochs on 2,000 triples,
capped at 1 step, no exception, wrong throughput number). And `--dropout` was
wired through, having been accepted by `build_lora_model` but reachable only by
editing the file.

**A gap found while sweeping, which is tomorrow's first task.** `docs/plan.md`
and the `finetune.py` module docstring both asserted that the existing
entrypoints take `--model <checkpoint path>`. They do not: `evaluate.py:31`,
`oracle.py:99` and `rerank.py:344` all hardcode `BI_ENCODER`, and none of the
three exposes such a flag. The underlying plumbing *is* ready — `cache.py:22`
already keys on `model_name`, and `load_encoder("models/smoke")` was verified to
load a saved checkpoint correctly (22,860,672 params = backbone + adapter, 384-d
output) — so this is CLI wiring, not design. The docstring has been corrected to
say so.

## 2026-08-12 (Phase 2 designed, deps landed, finetune.py scaffolded for hand-writing)

Phase 2 opens. Nothing trained yet and that is the point — the session went
design → dependencies → scaffold, in that order, because each step's value
depends on the previous one being settled.

**The design (`docs/plan.md`, commit 8cc388d).** Wrote the Phase 2 section as an
*extension* of the existing milestone rather than a rewrite, so nothing locked
got reversed. The load-bearing part is four pre-committed measurement rules,
written before any number exists:

- **R1 — no hyperparameter may be chosen by looking at NFCorpus.** LoRA rank,
  alpha, lr, batch size, epochs and checkpoint step are selected on a held-out
  MS MARCO dev slice only. This is the whole difficulty of Phase 2 and the
  reason it needed a design pass. The *training data* never touches NFCorpus —
  design decision 3 covered that from the start — but **hyperparameter choice is
  a second, quieter channel from test set into model.** Leave it open and
  "zero-shot generalization" degrades into a test-set fit, with nothing raising
  and the table simply acquiring a number it has not earned. Same failure class
  as a leaked `corpus_id`: well-formed and wrong.
- **R2** — every NFCorpus number obtained gets reported; no running three
  configs and tabling the best.
- **R3** — the 0.6263 ceiling does *not* carry over. A fine-tuned encoder
  returns a different top-100, so `oracle.py` gets re-run on the new candidate
  set and both ceilings are reported. Reusing the old bound would measure
  Phase 2's gain against a ceiling computed for someone else's candidates.
- **R4** — the fine-tune is allowed to lose. MS MARCO tuning making an
  out-of-domain medical set worse is exactly what "zero-shot" is testing for,
  and the row ships either way.

Also settled: the **checkpoint-naming convention** (`models/lora-<tag>/`, where
the directory name flows into the retrieval cache key, so two configs can never
silently share results — the "cache key doesn't cover `retrieve.py`" trap
wearing new clothes); the **`msmarco-MiniLM-L6-cos-v5` reference row**, which
costs zero training and converts "LoRA moved the number" into "LoRA captured X%
of what full MS MARCO specialization gets"; and **two separate success checks**,
because one NFCorpus number cannot distinguish "training loop is broken" from
"training worked and did not transfer" (train loss + MS MARCO dev answers the
first, NFCorpus answers the second; check-1-passes-while-check-2-fails is the
interesting outcome).

Deliberately **not** decided: how many triples to train on. That is a wall-clock
question about MPS throughput, and a guessed number in a plan is worth less than
a measured one — so the plan carries the *sizing procedure* (smoke run reports
steps/sec, size to a 30–60 min budget) instead.

Every library API and dataset name in the design was verified against the
installed versions and the live HF hub rather than recalled. That mattered:
`sentence-transformers` 5.7.0 moved the losses to
`sentence_transformers.sentence_transformer.losses`, which is not where an
older memory would look.

**The dependency commit (97800d1), deliberately isolated.** `peft`,
`accelerate` and `datasets` in a new `train` extra — opt-in like `tracing`, so
an eval-only install stays light. `datasets` was already present transitively;
`base/trainer.py:176` raises an explicit "you need accelerate and datasets"
error, which confirmed it had to be declared rather than leaned on.

The commit touches nothing but `pyproject.toml` and `uv.lock`, for the reason
the lockfile is committed at all: if a lock bump arrives tangled with new code,
a metric shift has two candidate causes and no cheap way to separate them.
Three checks established zero drift, and the third is the one worth repeating:

1. `uv lock` added exactly `peft` 0.20.0, `accelerate` 1.14.0 and `psutil`; no
   existing package changed version. Read the diff for `version =` lines
   specifically — the other ~100 lines are dependency *edges* regrouping.
2. A full `--refresh` recompute reproduced all seven published figures to four
   decimals (0.3159 / 0.3412 / 0.5046 → 0.5675 / 0.6263 / 8.1%). A cached run
   would have proved nothing, since the cache was the thing under suspicion.
3. Copied the cached top-100 out before refreshing and `cmp`'d it after:
   **byte-for-byte identical, floats included.** Matching metrics make identical
   candidates near-certain; this makes them a fact — the same distinction
   `cache.py` exists to enforce one level down. Two encode passes on MPS gave
   bit-identical cosine scores, so MPS non-determinism needs no hedge in
   Phase 2. Recorded in `LEARNINGS.md` (commit f1c2713).

**The scaffold, and a change in how this repo gets built.** Betty asked to hand-
write Phase 2's ML rather than review generated code, so `finetune.py` was
scaffolded to a chosen split: **plumbing written (argparse, logging, paths, the
trainable-params sanity check, the throughput→triple-count arithmetic), the ML
left as `NotImplementedError` with docstrings that explain the reasoning and
name the traps.** Coaching mode is *explain the concept, then she writes it*.
This is a working-mode preference, not a one-off — expect it to hold for the
rest of Phase 2 and probably beyond.

`load_triples` is done and verified (returns a `Dataset` of exactly `n` rows,
column order preserved, seed reproduces and varies). Two bugs in the first
draft, both instructive: `dataset[:n]` returns a **plain dict of lists**, not a
`Dataset` — the general `datasets` rule is that bracket access materializes
data while `.select()`/`.filter()`/`.shuffle()` return a view — and an off-by-one
`n+1`. Fixed to `.select(range(n))`.

**A teaching hint that didn't survive contact with the data, worth recording
because the correction is the better lesson.** The scaffold asked "why shuffle
before taking n?", hinting that MS MARCO's file order encodes something. Betty
reasonably guessed popularity ordering. Checking the actual dataset showed **100
distinct queries in the first 100 rows, zero adjacent duplicates, topics fully
interleaved** — the file looks already shuffled at build time, and the hint had
pointed at an unverified story. The honest reason to shuffle survives on
different ground: 100 rows is 0.02% of 502,931, nothing documents the ordering
as random, and one method call removes the need to care. Same move as the `cmp`
above — don't reason about whether a property holds, make it hold.

The trap the hint was *reaching* for is real elsewhere and worth keeping in
mind if `triplet-hard` ever comes into play: if consecutive rows shared a query,
a batch would contain several rows for the same question, and MNRL would treat
those rows' positives as negatives for each other — actively training the model
to push apart two correct answers. **False negatives are MNRL's characteristic
failure, and batch composition controls it.**

Two smaller findings. `uv sync --all-extras` uninstalled an ad-hoc ipykernel
tree (documented behaviour — sync *matches* the lock rather than adding to it —
but the first time it has actually taken something away here; `uv pip install
ipykernel` restores it). And `cache/reranked_nfcorpus.json` turned out to be an
orphan: `rerank.py:356` re-scores unconditionally and nothing reads or writes
that file. Deleted. The consequence is load-bearing for Phase 2 — **retrieval is
the only cached stage**, so getting the checkpoint into the cache key is a
sufficient fix rather than a partial one; there is no second stale-value channel
downstream.

D6's leftover was also cleared: `docs/plan.md:19` had asserted "public benchmark
+ public wiki", which D6 has not decided. It now states the benchmark half is
public and points at D6 for the rest. D6 itself remains open.
