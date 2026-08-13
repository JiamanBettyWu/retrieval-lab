# Session log

Reverse-chronological journal — one dated entry per working session: what got
done, what was decided and why. History only; for what to do next see
[TODO.md](TODO.md).

Older entries archived in [`sessions/`](sessions/) — 2026-07-15 through
2026-08-08 (Phase 0 scaffold through Phase 1 landing) live in
[`sessions/2026-07-15--2026-08-08.md`](sessions/2026-07-15--2026-08-08.md).

---

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
