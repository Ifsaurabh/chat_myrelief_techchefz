# Loads the existing index, retrieves top candidates from PGVector via
# hybrid search, reranks with CrossEncoder, returns final top chunks.

import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from rerank import rerank

# config
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K_RETRIEVAL = 5


# Loading index
def load_index() -> VectorStoreIndex:
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    vector_store = PGVectorStore.from_params(
        database=DB_NAME,
        host=DB_HOST,
        password=DB_PASSWORD,
        port=DB_PORT,
        user=DB_USER,
        table_name="policy_chunks",
        embed_dim=EMBEDDING_DIM,
        hybrid_search=True,
        text_search_config="english",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, embed_model=embed_model
    )


# Metadata filters
def build_metadata_filters(tier: str = None, doc_type: str = None) -> MetadataFilters | None:
    filters = []
    if tier:
        filters.append(MetadataFilter(key="tier", value=tier))
    if doc_type:
        filters.append(MetadataFilter(key="doc_type", value=doc_type))
    return MetadataFilters(filters=filters) if filters else None


# Retrieval with reranking, retry-if-weak safety net included
def retrieve(query: str, index: VectorStoreIndex, tier: str = None, doc_type: str = None) -> tuple[list, bool]:
    filters = build_metadata_filters(tier, doc_type)
    retriever = index.as_retriever(
        vector_store_query_mode="hybrid",
        similarity_top_k=TOP_K_RETRIEVAL,
        filters=filters,
    )
    candidates = retriever.retrieve(query)
    reranked, ok = rerank(query, candidates)

    if not ok and (tier or doc_type):
        retriever = index.as_retriever(
            vector_store_query_mode="hybrid",
            similarity_top_k=TOP_K_RETRIEVAL,
            filters=None,
        )
        candidates = retriever.retrieve(query)
        reranked, ok = rerank(query, candidates)

    return reranked, ok
