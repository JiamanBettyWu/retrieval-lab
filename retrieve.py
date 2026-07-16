"""Phase 0 retrieval: an off-the-shelf BI-ENCODER over the whole corpus.

The bi-encoder embeds each document once and each query once, then ranks by
cosine similarity — cheap, because doc embeddings are precomputed. (Contrast
the cross-encoder reranker in Phase 1, which scores query+doc jointly and so
can't scale to a full corpus.)
"""
from sentence_transformers import SentenceTransformer, util as st_util

from observability import op

BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def load_encoder(name: str = BI_ENCODER) -> SentenceTransformer:
    return SentenceTransformer(name)


def doc_text(doc: dict) -> str:
    """BEIR docs are {'title', 'text'} — join into one string to embed."""
    return (doc.get("title", "") + " " + doc.get("text", "")).strip()


@op  # traced in Weave when init_weave() ran; a no-op otherwise
def retrieve(model: SentenceTransformer, corpus: dict, queries: dict, top_k: int = 100) -> dict:
    """Rank the corpus for every query with the bi-encoder.

    Return BEIR's expected shape:  {query_id: {doc_id: score}}  (top_k docs/query).

    TODO(human): implement the bi-encoder retrieval. Four steps:
      1. Build parallel id/text lists:
           doc_ids   = list(corpus);   doc_texts   = [doc_text(corpus[d]) for d in doc_ids]
           query_ids = list(queries);  query_texts = [queries[q] for q in query_ids]
      2. Embed both with model.encode(...). Useful kwargs:
           normalize_embeddings=True   (so cosine == dot product)
           convert_to_tensor=True
           show_progress_bar=True      (nice for the ~3.6k-doc corpus)
      3. Get top_k per query:
           hits = st_util.semantic_search(query_emb, doc_emb, top_k=top_k)
         -> hits[i] is a list of {"corpus_id": int, "score": float} for query i.
      4. Map row indices back to real ids and build the nested dict:
           results[query_ids[i]][doc_ids[hit["corpus_id"]]] = float(hit["score"])
      Return `results`.
    """
    doc_ids = list(corpus)
    doc_texts = [doc_text(corpus[d]) for d in doc_ids]
    query_ids = list(queries)
    query_texts = [queries[q] for q in query_ids]

    model.encode()

    raise NotImplementedError("retrieve() — your turn (steps are in the docstring above)")
