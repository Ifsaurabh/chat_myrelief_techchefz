# CrossEncoder reranking of retrieved chunks, plus the quality-check
# signal that drives the agentic retry loop.

from llama_index.core.postprocessor import SentenceTransformerRerank

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_AFTER_RERANK = 5
RERANK_MIN_SCORE = -5.0

_reranker = SentenceTransformerRerank(model=RERANKER_MODEL, top_n=TOP_K_AFTER_RERANK)


# Rerank candidates, return (nodes, is_quality_sufficient)
def rerank(query: str, nodes: list) -> tuple[list, bool]:
    if not nodes:
        return [], False

    reranked = _reranker.postprocess_nodes(nodes, query_str=query)

    best_score = reranked[0].score if reranked else float("-inf")
    is_quality_sufficient = best_score >= RERANK_MIN_SCORE

    return reranked, is_quality_sufficient
