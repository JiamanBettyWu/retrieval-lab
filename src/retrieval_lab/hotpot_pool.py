"""Build `hotpotqa-distractor-pool`: Phase 4's evaluation corpus.

    python -m retrieval_lab.hotpot_pool

**This is NOT BEIR HotpotQA and its numbers do not compare to published BEIR
results.** BEIR's HotpotQA is the full-wiki setting: ~5.2M paragraphs, a 654MB
download, ~8GB of resident embeddings, and a vector index that `docs/plan.md`
explicitly defers ("brute-force in-memory cosine is fine at BEIR-small scale").

This builds the *distractor* setting instead. HotpotQA ships 10 paragraphs per
dev question — 2 gold plus 8 distractors that were TF-IDF-mined against that
specific question. Pooling every question's ten into one deduplicated corpus
gives ~66.5k documents: FiQA-sized, so brute-force cosine survives and the
scope discipline holds. The pooling is what makes it a retrieval task at all —
per-question the answer sits in a 10-doc haystack, but pooled, each question
competes against 66k paragraphs that were *selected for looking relevant*. The
corpus is adversarial by construction, and that is a feature.

Because it is a custom reduction it carries a distinct dataset id, which flows
into the retrieval cache key and the README row. Calling it `hotpotqa` would
invite exactly the comparison that is invalid (D12, `TODO.md`).

Written in BEIR's on-disk shape so `data.load_beir` reads it with no special
casing, plus one file BEIR has no slot for:

    data/hotpotqa-distractor-pool/
      corpus.jsonl        {"_id", "title", "text"}
      queries.jsonl       {"_id", "text"}
      qrels/test.tsv      query-id  corpus-id  score      (score is always 1)
      answers.jsonl       {"_id", "answer"}   <- gold short answers, Phase 4 only

`answers.jsonl` is the file that changed a scope decision. BEIR has no answer
field because BEIR grades *retrieval*; HotpotQA ships a gold short answer per
question, which is why `docs/plan.md`'s ban on correctness scoring was amended
rather than bent — nothing is invented, the labels come with the dataset.
"""
import argparse
import hashlib
import json
import logging
import re
import urllib.request
from pathlib import Path

log = logging.getLogger("retrieval_lab")

DATASET = "hotpotqa-distractor-pool"

# The HuggingFace mirror, not the original CMU host (curtis.ml.cmu.edu is dead
# as of 2026-08). 27MB of parquet against BEIR-hotpotqa's 654MB zip.
PARQUET_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)

# HotpotQA's dev set is the only split with public supporting_facts (test is
# blind), so "test" here means "HotpotQA dev" — the split name is BEIR's
# convention for `GenericDataLoader(...).load(split=...)`, not a claim about
# HotpotQA's own splits.
SPLIT = "test"


def doc_id(title: str) -> str:
    """A content-derived id for a paragraph, keyed on its Wikipedia title.

    Deliberately NOT an integer or an enumeration index. The failure mode this
    repo keeps tripping over is a positional index leaking into a doc id slot
    (`semantic_search` returns `corpus_id`, an offset into the encode order) —
    and the reason it is expensive is that it never raises, it just scores
    NDCG@10 = 0.0000 against qrels it shares no ids with. An id that cannot be
    confused with a position makes that class of bug loud instead of silent.
    """
    return "hp-" + hashlib.sha1(title.encode("utf-8")).hexdigest()[:12]


def pool_paragraphs(rows):
    """Pool per-question paragraph sets into one corpus. Pure — no I/O.

    `rows` is an iterable of dicts shaped like the HotpotQA distractor split:
        {"id", "question", "answer",
         "supporting_facts": {"title": [...]},
         "context": {"title": [...], "sentences": [[...], ...]}}

    Returns (corpus, queries, qrels, answers) in BEIR's in-memory shapes.

    **Deduplication is by title, keeping the longest text.** 61 of the ~66.5k
    titles carry slightly different paragraph text across the questions that
    cite them (differing extractions of the same article). Keeping the longest
    is deterministic and loses no sentence that a shorter variant had; keeping
    both would split one Wikipedia article across two doc ids, and a gold title
    that resolved to two ids would quietly halve that question's recall.
    """
    corpus: dict[str, dict] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    answers: dict[str, str] = {}
    slots = 0
    conflicts = 0

    for row in rows:
        qid = str(row["id"])
        queries[qid] = row["question"]
        answers[qid] = row["answer"]

        ctx = row["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            slots += 1
            did = doc_id(title)
            text = "".join(sentences).strip()
            prev = corpus.get(did)
            if prev is None:
                corpus[did] = {"title": title, "text": text}
            elif prev["title"] != title:
                # Two distinct titles hashing to one id. Vanishingly unlikely at
                # 66k docs, and it would otherwise land in the branch below and
                # be *counted as a text conflict* — silently merging two
                # Wikipedia articles into one document. Loud beats lucky.
                raise AssertionError(
                    f"doc_id collision: {title!r} and {prev['title']!r} both -> {did}"
                )
            elif prev["text"] != text:
                conflicts += 1
                if len(text) > len(prev["text"]):
                    corpus[did] = {"title": title, "text": text}

        # Gold labels come from supporting_facts, which names the titles whose
        # sentences the annotator actually used. Binary — HotpotQA has no
        # relevance grades, so unlike NFCorpus there is no flat-tie problem
        # when milestone 4b needs "the gold context": it is exactly these two.
        gold = dict.fromkeys(str(t) for t in row["supporting_facts"]["title"])
        qrels[qid] = {doc_id(t): 1 for t in gold}

    log.info("pooled %d paragraph slots -> %d unique docs (%d text conflicts "
             "resolved by longest-wins)", slots, len(corpus), conflicts)
    return corpus, queries, qrels, answers


def check(corpus, queries, qrels, answers) -> None:
    """Sanity checks that raise. A silently wrong corpus is the whole risk here.

    Every one of these failing produces a well-formed dataset that scores badly
    for a reason no metric would name — the same class of bug as a leaked
    `corpus_id`, which is why they raise rather than warn.
    """
    if not corpus or not queries:
        raise AssertionError("empty corpus or queries — the parquet read wrong")

    positional = [d for d in corpus if re.fullmatch(r"\d+", d)]
    if positional:
        raise AssertionError(
            f"{len(positional)} doc ids are bare integers (e.g. {positional[:3]}) — "
            "that is the positional-index leak, not a doc id"
        )

    missing = {d for docs in qrels.values() for d in docs} - set(corpus)
    if missing:
        raise AssertionError(
            f"{len(missing)} gold doc ids are absent from the corpus "
            f"(e.g. {sorted(missing)[:3]}) — every config's recall would be "
            "silently deflated by an unreachable ceiling"
        )

    unlabelled = set(queries) - set(qrels)
    if unlabelled:
        raise AssertionError(f"{len(unlabelled)} queries have no qrels entry")

    # HotpotQA is 2-hop by construction: exactly two supporting paragraphs.
    # A question with one means the multi-hop premise broke for it, and its
    # oracle context would be a different task from every other question's.
    wrong = {q: len(d) for q, d in qrels.items() if len(d) != 2}
    if wrong:
        raise AssertionError(
            f"{len(wrong)} questions do not have exactly 2 gold docs "
            f"(e.g. {dict(list(wrong.items())[:3])}) — not 2-hop"
        )

    no_answer = [q for q, a in answers.items() if not str(a).strip()]
    if no_answer:
        raise AssertionError(
            f"{len(no_answer)} questions have an empty gold answer — correctness "
            "scoring would silently count those as failures"
        )


def write_beir(corpus, queries, qrels, answers, out_dir: Path) -> None:
    """Write BEIR's on-disk layout so `load_beir` needs no special casing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "qrels").mkdir(exist_ok=True)

    with (out_dir / "corpus.jsonl").open("w") as f:
        for did, doc in corpus.items():
            f.write(json.dumps({"_id": did, **doc}) + "\n")

    with (out_dir / "queries.jsonl").open("w") as f:
        for qid, text in queries.items():
            f.write(json.dumps({"_id": qid, "text": text}) + "\n")

    # BEIR's loader expects a header row and tab separation; ids are hex/hash
    # and questions never reach this file, so nothing here can contain a tab.
    with (out_dir / "qrels" / f"{SPLIT}.tsv").open("w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for qid, docs in qrels.items():
            for did, score in docs.items():
                f.write(f"{qid}\t{did}\t{score}\n")

    with (out_dir / "answers.jsonl").open("w") as f:
        for qid, answer in answers.items():
            f.write(json.dumps({"_id": qid, "answer": answer}) + "\n")

    log.info("wrote %s (%d docs, %d queries)", out_dir, len(corpus), len(queries))


def load_answers(dataset: str = DATASET, out_dir: str = "data") -> dict[str, str]:
    """{query_id: gold short answer} — the Phase 4 correctness labels."""
    path = Path(out_dir) / dataset / "answers.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python -m retrieval_lab.hotpot_pool` first"
        )
    with path.open() as f:
        return {r["_id"]: r["answer"] for r in map(json.loads, f)}


def main(out_dir: str = "data", refresh: bool = False) -> None:
    import pandas as pd

    dest = Path(out_dir) / DATASET
    if (dest / "corpus.jsonl").exists() and not refresh:
        log.info("%s already built — pass --refresh to rebuild", dest)
        return

    raw = Path(out_dir) / "_raw" / "hotpot_dev_distractor.parquet"
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading HotpotQA dev distractor split (~27MB) ...")
        urllib.request.urlretrieve(PARQUET_URL, raw)

    df = pd.read_parquet(raw)
    log.info("read %d dev questions from %s", len(df), raw)

    built = pool_paragraphs(row for _, row in df.iterrows())
    check(*built)
    write_beir(*built, dest)

    corpus, queries, _, _ = built
    log.info("\n=== %s ready ===", DATASET)
    log.info("corpus  : %d paragraphs (deduped by title)", len(corpus))
    log.info("queries : %d, each with exactly 2 gold docs", len(queries))
    log.info("NOTE    : this is the distractor pool, NOT BEIR full-wiki HotpotQA. "
             "Its numbers are not comparable to published BEIR results.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data")
    p.add_argument("--refresh", action="store_true",
                   help="rebuild even if the dataset is already on disk")
    args = p.parse_args()
    main(args.out_dir, args.refresh)
