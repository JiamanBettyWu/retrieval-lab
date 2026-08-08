"""retrieval-lab — a two-stage RAG retriever, measured at every step.

Entrypoints are modules, run from the repo root:

    python -m retrieval_lab.evaluate --dataset nfcorpus   # baseline
    python -m retrieval_lab.oracle   --dataset nfcorpus   # reranking ceiling

Both write/read `data/` and `cache/` relative to the CURRENT WORKING DIRECTORY,
so run them from the repo root or they'll re-download BEIR somewhere else.
"""
