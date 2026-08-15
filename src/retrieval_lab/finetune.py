"""Phase 2: LoRA-fine-tune the bi-encoder on MS MARCO triples.

    python -m retrieval_lab.finetune --smoke                  # measure throughput, prove the path runs
    python -m retrieval_lab.finetune --tag r16-lr2e5-v1 --triples 50000

Trains the Phase 0 bi-encoder (`all-MiniLM-L6-v2`) contrastively on
(query, positive, negative) triples with a LoRA adapter on the backbone, then
saves the adapter to `models/lora-<tag>/`. Evaluation happens elsewhere.

**That evaluation path is not wired up yet** (verified 2026-08-14):
`cache.py:22` already keys on `model_name`, and `load_encoder(name)` loads a
checkpoint directory correctly — `load_encoder("models/smoke")` returns
22,860,672 params, i.e. backbone + adapter. But `evaluate.py`, `oracle.py` and
`rerank.py` all hardcode `BI_ENCODER` at their `cached_retrieval` call and
expose no `--model` flag. Adding it is what makes the checkpoint name flow into
the cache key, so two configs cannot silently share retrieval results.

See `docs/plan.md` "Phase 2 design" for the rules this file operates under. The
one to keep in mind while editing: **no hyperparameter here may be chosen by
looking at NFCorpus.** Config selection happens on the MS MARCO dev slice only.
"""
import argparse
import logging
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from peft import LoraConfig

from datasets import Dataset, load_dataset

from .observability import init_weave, op
from .retrieve import BI_ENCODER

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("retrieval_lab")

TRIPLES_DATASET = "sentence-transformers/msmarco-bm25"
TRIPLES_CONFIG = "triplet"   # 502,931 rows of query/positive/negative
MODELS_DIR = Path("models")

# --smoke defaults: small enough to finish in a couple of minutes, large enough
# that steps/sec means something. Never write these into the ablation table.
SMOKE_TRIPLES = 2_000
SMOKE_MAX_STEPS = 200
SMOKE_N_EVAL = 100


# ---------------------------------------------------------------------------
# YOURS — the training data
# ---------------------------------------------------------------------------
def load_triples(n: int, k: int, seed: int = 42) -> tuple[Dataset, Dataset]:
    """Load and subsample the MS MARCO BM25 triples.

    `load_dataset(TRIPLES_DATASET, TRIPLES_CONFIG, split="train")` gives you
    502,931 rows with columns `query`, `positive`, `negative` — already the
    shape MultipleNegativesRankingLoss wants, so there is no collation code to
    write. Your job is to take `n` of them, reproducibly.

    **Column ORDER is the contract, not column names.** MNRL reads the dataset's
    columns positionally as (anchor, positive, negative). Reorder them and the
    loss still falls smoothly — you would just be training the model to rank
    negatives *up*. `tests/test_finetune.py` pins this for the same reason
    `test_oracle.py` pins the oracle's candidate source.

    **Train is the first `n` rows of the seed-`seed` shuffle; dev is the last
    `k` of the SAME shuffle.** Slicing dev off the unshuffled dataset instead
    puts ~`n/502,931` of it back into training — 20% at n=100k, and only ~1% at
    --smoke scale, so the smoke run cannot catch it. Disjointness has to be a
    property of the definition, not a coincidence between two row counts; the
    guard below keeps it that way as `n` grows.

    Measured 2026-08-14, and stronger than row-disjointness: every row in this
    dataset carries a **unique query** (100,000 distinct queries in 100,000
    rows), so the split is query-disjoint and dev loss measures generalization
    to unseen questions. It also means MNRL's characteristic false-negative
    failure — two rows for one query landing in a batch, training the model to
    push apart two correct answers — cannot occur here. That would not hold on
    `triplet-hard`.

    The shuffle itself is insurance rather than a fix for a known ordering: the
    file looks already shuffled at build time (100 distinct, fully interleaved
    queries in the first 100 rows), but nothing documents that as guaranteed,
    and one method call removes the need to care.
    """

    dataset = load_dataset(TRIPLES_DATASET, TRIPLES_CONFIG, split="train")
    dataset_shuffled = dataset.shuffle(seed=seed)
    total_rows = dataset.num_rows

    if n + k > total_rows:
        raise ValueError(f"Sum of n and k must not exceed {total_rows}.")

    # use first n rows as the training data, use last k rows as the eval data
    return (dataset_shuffled.select(range(n)), dataset_shuffled.select(range(total_rows-k, total_rows)))

# ---------------------------------------------------------------------------
# YOURS — the model
# ---------------------------------------------------------------------------
def build_lora_model(rank: int, alpha: int, dropout: float = 0.1):
    """Load the Phase 0 bi-encoder and attach a fresh LoRA adapter.

    Two calls:
        model = SentenceTransformer(BI_ENCODER)
        model.add_adapter(LoraConfig(...))       # peft.LoraConfig
        return model

    `add_adapter` is real on this version — `base/peft_mixin.py`, delegating to
    the transformers PeftAdapterMixin. `LoraConfig` accepts `r`, `lora_alpha`,
    `lora_dropout`, `target_modules`, `bias`, `task_type`.

    **The trap, and it is a real one here.** PEFT matches `target_modules` by
    *name suffix*. The backbone's Linear layers, per block, are:

        attention.self.query          (384, 384)
        attention.self.key            (384, 384)
        attention.self.value          (384, 384)
        attention.output.dense        (384, 384)
        intermediate.dense           (1536, 384)
        output.dense                 (384, 1536)

    So `target_modules=["dense"]` silently matches *three different things* of
    two different shapes. Prefer names that mean one thing.

    Sanity check when you're done — the numbers should shock you a little:

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        # backbone is 22,713,216 params

    If `trainable` comes back equal to `total`, the adapter didn't attach and
    you are full-fine-tuning. If it comes back 0, nothing will train and the
    loss will sit flat — which looks identical to a bad learning rate.
    """
    model = SentenceTransformer(BI_ENCODER)

    lora_config = LoraConfig(
        r = rank,
        lora_alpha=alpha,
        target_modules=["query", "value"],
        lora_dropout=dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )

    model.add_adapter(adapter_config=lora_config)

    return model


def _summarize_datasets(inputs: dict) -> dict:
    """Replace the Dataset args with a one-line shape before Weave uploads them.

    Without this the 2026-08-14 run's trace was an 80,466,260-byte payload —
    weave serializes `train_dataset`/`eval_dataset` as call inputs, so 110k rows
    of MS MARCO text went up with the call and the server rejected it at 413
    (limit 67,108,864). The run itself was unaffected; only the trace was lost,
    which is the worst failure shape for observability code: silent until the
    moment you want the record.

    What replaces it is strictly more useful than the rows anyway — the row
    count and column ORDER are what a later reader needs, because MNRL reads
    columns positionally as (anchor, positive, negative).
    """
    out = dict(inputs)
    for key in ("train_dataset", "eval_dataset"):
        ds = out.get(key)
        if ds is not None:
            out[key] = {"rows": getattr(ds, "num_rows", None),
                        "columns": list(getattr(ds, "column_names", []) or [])}
    return out


# ---------------------------------------------------------------------------
# YOURS — the training loop
# ---------------------------------------------------------------------------
@op(postprocess_inputs=_summarize_datasets)  # traced when init_weave() ran; a no-op otherwise
def train(model, train_dataset: Dataset, eval_dataset: Dataset, out_dir: Path, lr: float,
          batch_size: int,  epochs: float, max_steps: int = -1, seed: int = 42) -> float:
    """Fit the adapter with MultipleNegativesRankingLoss. Returns steps/sec.

    The pieces, all confirmed present on the installed versions:

        from sentence_transformers import (SentenceTransformerTrainer,
                                           SentenceTransformerTrainingArguments)
        from sentence_transformers.sentence_transformer.losses import (
            MultipleNegativesRankingLoss)

    `SentenceTransformerTrainingArguments` accepts (among others):
        output_dir, per_device_train_batch_size, learning_rate, num_train_epochs,
        max_steps, warmup_ratio, logging_steps, seed, report_to, bf16, fp16

    `save_pretrained` writes the ADAPTER, not 22M weights — measured at 593,072
    bytes for r=16 on query+value (147,456 params x 4B + header), which is why
    `models/` can stay gitignored as rebuildable.

    Selection happens here too: `eval_strategy` + `load_best_model_at_end` make
    "which checkpoint step" a hyperparameter chosen on the MS MARCO dev slice,
    per R1. `metric_for_best_model="eval_loss"` sets `greater_is_better=False`
    on its own (`training_args.py:1534` keys off the name ending in "loss") —
    had it defaulted True, the run would silently keep the WORST checkpoint.

    Decisions made deliberately, not by copying a tutorial:

      - **batch_size is a quality knob, not just a speed knob.** MNRL treats
        every *other* document in the batch as a negative. Batch 16 gives each
        query 15 wrong answers to push away from; batch 128 gives it 127. If MPS
        memory forces you small, `CachedMultipleNegativesRankingLoss` buys the
        large effective batch back at the cost of extra forward passes.
      - **Precision: neither flag, i.e. fp32.** `fp16` used to mean "CUDA only";
        it no longer does — `accelerate/accelerator.py:565` lists `mps` as
        supported for torch >= 2.5.0, so `fp16=True` would really run AMP here.
        It stays off anyway: the model is 22M params so memory is not the
        constraint, `GradScaler` skips overflow steps (muddying the steps/sec
        this function returns), and fp32 on MPS was measured bit-reproducible
        in the 2026-08-12 drift check. Revisit by *measuring* it with --smoke,
        not by argument.
      - **`report_to="none"`, explicitly.** `transformers` auto-detects logging
        integrations, so leaving the default makes behaviour depend on whether
        `weave`/`wandb` happen to be importable — an environment-dependent code
        path in a repo whose whole thesis is against those. Weave tracing here
        comes from the `@op` above, not from the trainer.
    """

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    
    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        max_steps=max_steps,
        report_to="none",
        seed=seed,
        logging_steps=20,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        eval_steps=200,
        save_steps=200
        # fp16=True,
    )

    loss = MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset,
                                         loss=loss)
    output = trainer.train()
    model.save_pretrained(out_dir)

    return output.metrics["train_steps_per_second"]

# ---------------------------------------------------------------------------
# YOURS — the missing baseline
# ---------------------------------------------------------------------------
@op
def dev_loss(model, eval_dataset: Dataset, out_dir: Path,
             batch_size: int = 32, seed: int = 42) -> float:
    """MNRL loss for ANY model over the dev slice. Returns `eval_loss`.

    Why this exists: the four Phase 2 configs produced dev losses of 0.3960,
    0.3443, 0.3180 and 0.3129, and those numbers are only comparable **to each
    other**. The base model's dev loss was never measured, so "the fine-tune
    traded out-of-domain quality for in-domain gain" is currently unverified —
    if the base model already scores below 0.3129, there was no gain to trade
    and Phase 2 is damage in both directions. One number decides which.

    **The trap: MNRL loss is a property of a BATCH, not of an example.** Every
    triple's loss depends on the other `batch_size - 1` triples sharing its
    batch, because those are its in-batch negatives. A loop that computes this
    at a different batch size, or over a reshuffled dev set, returns a
    well-formed number on a different scale — plausible, comparable-looking,
    and wrong. Same failure family as a leaked `corpus_id`.

    So do NOT hand-roll the eval loop. Rebuild the same trainer and let it do
    the batching:

        args = SentenceTransformerTrainingArguments(
            output_dir=..., per_device_eval_batch_size=batch_size,
            eval_strategy="no", report_to="none", seed=seed)
        loss = MultipleNegativesRankingLoss(model)
        trainer = SentenceTransformerTrainer(model=model, args=args,
                                             eval_dataset=eval_dataset, loss=loss)
        return trainer.evaluate()["eval_loss"]

    Two things worth getting right:

      - **`MultipleNegativesRankingLoss(model)` binds to the model you hand it.**
        Reusing an instance built around a different model would score the wrong
        weights while raising nothing.
      - **`eval_strategy="no"`.** You are calling `evaluate()` directly rather
        than training, so nothing should be scheduling evals of its own.

    Note `per_device_EVAL_batch_size` here, against `per_device_TRAIN_batch_size`
    in `train()` — and **the default is 8, not the train batch size.** Confirmed
    against `checkpoint-3125/training_args.bin`: every Phase 2 run trained at 32
    and evaluated at 8. So `0.3960 / 0.3443 / 0.3180 / 0.3129` were all measured
    with **7 in-batch negatives, not 31**, and `batch_size=8` is what makes this
    function's output comparable to them. Passing 32 here would produce a number
    on a harder task and a different scale.

    Two consequences worth carrying forward. The random-guess floor for those
    dev losses is `ln(8) = 2.079`, not `ln(32) = 3.466` — the latter applies to
    the *train* loss, which really did run at 32. And **the selector was weaker
    than intended**: discriminating between 8 candidates is an easier task than
    between 32, so the loss scale is compressed and configs sit closer together
    than they would have. That is a plausible contributor to the `5e-4 -> 1e-3`
    gap being only 0.0051 — the margin that decided the winner.
    """
    raise NotImplementedError("Betty writes this one.")


# ---------------------------------------------------------------------------
# MINE — plumbing. Read it, but you shouldn't need to change it.
# ---------------------------------------------------------------------------
def resolve_out_dir(tag: str, smoke: bool) -> Path:
    """Where the adapter lands. `models/lora-<tag>/`, or a throwaway for smoke.

    Smoke checkpoints deliberately do NOT land beside real ones — a directory
    full of `lora-smoke3` is how you end up unable to say which adapter produced
    a table row. See docs/plan.md on the checkpoint-naming convention: the
    directory name flows into the retrieval cache key, so it is identity, not
    decoration.
    """
    return MODELS_DIR / ("smoke" if smoke else f"lora-{tag}")


def main(tag: str, n_triples: int, k: int, smoke: bool, rank: int, alpha: int,
         dropout: float, lr: float, batch_size: int, epochs: float,
         max_steps: int | None) -> None:
    init_weave()

    if smoke:
        n_triples, max_steps, k = SMOKE_TRIPLES, SMOKE_MAX_STEPS, SMOKE_N_EVAL
        log.warning("--smoke: %d triples, %d steps max. This measures THROUGHPUT "
                    "and proves the path runs. The resulting adapter is not a "
                    "trained model and its numbers belong in no table.",
                    n_triples, max_steps)

    out_dir = resolve_out_dir(tag, smoke)
    log.info("Loading %s [%s] — %d triples (seed 42) ...",
             TRIPLES_DATASET, TRIPLES_CONFIG, n_triples)
    train_dataset, eval_dataset = load_triples(n_triples, k)

    log.info("Building %s + LoRA (r=%d, alpha=%d, dropout=%.2f) ...",
             BI_ENCODER, rank, alpha, dropout)
    model = build_lora_model(rank, alpha, dropout)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("trainable %s / %s params (%.2f%%)",
             f"{trainable:,}", f"{total:,}", 100 * trainable / max(total, 1))
    if trainable == total:
        log.warning("trainable == total — the adapter did not attach; this is a "
                    "FULL fine-tune, which Phase 2 explicitly excludes.")
    elif trainable == 0:
        log.warning("nothing is trainable — the loss will not move, and that "
                    "looks exactly like a bad learning rate. Check the adapter.")

    started = time.perf_counter()
    # Keyword args deliberately: train()'s parameter ORDER is yours to change, and
    # a positional call would silently swap epochs and max_steps the moment it does.
    steps_per_sec = train(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset, out_dir=out_dir, lr=lr,
                          batch_size=batch_size, epochs=epochs, max_steps=max_steps)
    elapsed = time.perf_counter() - started

    log.info("\n=== Phase 2 %s ===", "smoke run" if smoke else f"train ({tag})")
    log.info("adapter       : %s", out_dir)
    log.info("wall clock    : %.1fs", elapsed)
    log.info("throughput    : %.2f steps/sec", steps_per_sec or 0.0)

    if smoke and steps_per_sec:
        # The smoke run's actual job: turn "how many triples?" into arithmetic.
        for budget_min in (30, 60):
            affordable = int(steps_per_sec * budget_min * 60) * batch_size
            log.info("  a %d-min budget affords ~%s triples (1 epoch, batch %d)",
                     budget_min, f"{affordable:,}", batch_size)
        log.info("Record the throughput in LEARNINGS.md alongside the size you pick.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="dev",
                   help="checkpoint identity -> models/lora-<tag>/. Change it whenever "
                        "a hyperparameter changes, or the retrieval cache serves the "
                        "previous config's results under the new config's name.")
    p.add_argument("--triples", type=int, default=50_000)
    p.add_argument("--k", type=int, default=5_000)
    p.add_argument("--smoke", action="store_true",
                   help="tiny subset, few steps: prove the path runs and measure steps/sec")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1,
                   help="LoRA dropout — applied inside the adapter only; the frozen "
                        "path is untouched. Recorded in the checkpoint's "
                        "adapter_config.json, so a run stays identifiable after the fact.")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="-1 means 'unset' — transformers' sentinel, not None. A positive "
                        "value OVERRIDES --epochs, which is what --smoke relies on and "
                        "what a real run must never do by accident.")
    args = p.parse_args()
    main(args.tag, args.triples, args.k, args.smoke, args.rank, args.alpha,
         args.dropout, args.lr, args.batch_size, args.epochs, args.max_steps)
