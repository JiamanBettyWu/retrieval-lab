"""BEIR dataset loading (download + parse).

No labels are created here — BEIR *ships* the qrels (query relevance
judgments), which is the whole reason we evaluate on a benchmark rather than
on an unlabeled corpus like the wiki.
"""
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def load_beir(dataset: str = "nfcorpus", split: str = "test", out_dir: str = "data"):
    """Download (once) and load a BEIR dataset.

    Returns (corpus, queries, qrels):
      corpus  : {doc_id:   {"title": str, "text": str}}
      queries : {query_id: str}
      qrels   : {query_id: {doc_id: relevance:int}}   ← the ground-truth labels
    """
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    # A dataset already on disk is loaded as-is, never re-fetched. This is what
    # lets locally-*built* datasets be first-class: `hotpotqa-distractor-pool`
    # (Phase 4, see hotpot_pool.py) has no BEIR zip to download, and without
    # this short-circuit beir's util would 404 chasing one. Real BEIR datasets
    # hit the same path once downloaded, so nothing about them changes.
    local = out / dataset
    if (local / "corpus.jsonl").exists():
        data_path = str(local)
    else:
        data_path = util.download_and_unzip(BEIR_URL.format(name=dataset), str(out))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    return corpus, queries, qrels
