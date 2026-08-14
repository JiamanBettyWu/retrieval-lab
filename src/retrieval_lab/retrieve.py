"""Phase 0 retrieval: an off-the-shelf BI-ENCODER over the whole corpus.

The bi-encoder embeds each document once and each query once, then ranks by
cosine similarity — cheap, because doc embeddings are precomputed. (Contrast
the cross-encoder reranker in Phase 1, which scores query+doc jointly and so
can't scale to a full corpus.)
"""
from sentence_transformers import SentenceTransformer, util as st_util

from .observability import op

BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Shared by every entrypoint's --model flag. The cache-key note is the load-bearing
# part: cache_path() keys on this string, so a local adapter path and the hub
# baseline never collide, but two DIFFERENT adapters written to the same path
# would. Give each fine-tune run its own --tag, and the paths stay distinct.
MODEL_HELP = (
    "bi-encoder to retrieve with: a hub id, or a path to a fine-tuned adapter "
    "(e.g. models/lora-r16-a32-lr2e5-100k). Keys the retrieval cache."
)


def load_encoder(name: str = BI_ENCODER) -> SentenceTransformer:
    return SentenceTransformer(name)


def doc_text(doc: dict) -> str:
    """BEIR docs are {'title', 'text'} — join into one string to embed."""
    return (doc.get("title", "") + " " + doc.get("text", "")).strip()


@op  # traced in Weave when init_weave() ran; a no-op otherwise
def retrieve(model: SentenceTransformer, corpus: dict, queries: dict, top_k: int = 100) -> dict:
    """Rank the corpus for every query with the bi-encoder.

    Return BEIR's expected shape:  {query_id: {doc_id: score}}  (top_k docs/query).

    Embeddings are L2-normalized so cosine similarity == dot product, which is
    what semantic_search ranks by. It returns row indices (`corpus_id`) into the
    order docs were encoded in, so the last step maps those back to real doc_ids.

    Covered by test_retrieve.py.
    """
    doc_ids = list(corpus)
    doc_texts = [doc_text(corpus[d]) for d in doc_ids]
    query_ids = list(queries)
    query_texts = [queries[q] for q in query_ids]

    doc_emb = model.encode(doc_texts, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
    query_emb = model.encode(query_texts, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)

    hits = st_util.semantic_search(query_emb, doc_emb, top_k=top_k)

    results: dict[str, dict[str, float]] = {}

    for i, q in enumerate(query_ids):
        # corpus_id is a ROW INDEX into doc_texts — map it back to the real
        # BEIR doc_id, or pytrec_eval finds no overlap with qrels and scores 0.
        results[q] = {doc_ids[hit["corpus_id"]]: float(hit["score"]) for hit in hits[i]}

    return results