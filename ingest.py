# Windows torch.compile workaround- see DECISIONS_LOG.md
import os
import platform

if platform.system() == "Windows":
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("PYTORCH_JIT", "0")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_CPP_WRAPPER", "0")

import torch
if platform.system() == "Windows":
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.reset()

# ---------------------------------------------------------------------

# imports
import json
from dotenv import load_dotenv
from pathlib import Path
from docling.document_converter import DocumentConverter
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from observability import start_phoenix, init_tracing, register_prompts, get_prompt

# Config
load_dotenv()
RAW_PDF_DIR = Path("data/raw_pdfs")
converter = DocumentConverter()
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Checkpoint file - saves contextually-enriched nodes to disk so a
# failure downstream (e.g. DB schema issue during storage) doesn't
# require redoing the slow, 89-call contextual enrichment step.
# Plain JSON (not pickle) - avoids deserializing arbitrary objects,
# and this checkpoint only ever needs each node's id, text, and metadata.
CHECKPOINT_FILE = Path("data/enriched_nodes.json")


def save_checkpoint(nodes: list, path: Path):
    serializable = [
        {"id_": node.id_, "text": node.text, "metadata": node.metadata}
        for node in nodes
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def load_checkpoint(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [TextNode(id_=item["id_"], text=item["text"], metadata=item["metadata"]) for item in raw]


# Docling PDF to markdown
def convert_all_pdfs() -> dict[str, str]:
    results = {}
    for pdf_path in sorted(RAW_PDF_DIR.glob("*.pdf")):
        print(f"Converting {pdf_path.name}...")
        result = converter.convert(str(pdf_path))
        markdown_text = result.document.export_to_markdown()
        results[pdf_path.name] = markdown_text
        print(f"  -> {len(markdown_text)} characters")
    return results


# Metadata map
DOC_METADATA_MAP = {
    "myrelief_basic_policy.pdf": {"tier": "basic", "doc_type": "policy"},
    "myrelief_standard_policy.pdf": {"tier": "standard", "doc_type": "policy"},
    "myrelief_premium_policy.pdf": {"tier": "premium", "doc_type": "policy"},
    "myrelief_claims_sop.pdf": {"tier": "all", "doc_type": "sop"},
}


# Markdown to LlamaIndex Document
def markdown_to_document(markdown_text: str, source_filename: str) -> Document:
    metadata = DOC_METADATA_MAP.get(source_filename, {"tier": "unknown", "doc_type": "unknown"})
    metadata["source_file"] = source_filename
    return Document(text=markdown_text, metadata=metadata)


# Chunking - splits along markdown heading boundaries
def chunk_documents(documents: list[Document]) -> list:
    parser = MarkdownNodeParser()
    return parser.get_nodes_from_documents(documents)


# Adding context to the chunks (Anthropic-style Contextual Retrieval)
def add_context_to_chunks(nodes: list, documents: list[Document]) -> list:
    context_prompt = get_prompt("contextual_chunk_enrichment")
    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, request_timeout=500.0)

    doc_text_by_file = {doc.metadata["source_file"]: doc.text for doc in documents}

    total = len(nodes)
    for i, node in enumerate(nodes):
        source_file = node.metadata.get("source_file")
        doc_text = doc_text_by_file.get(source_file, "")

        prompt = context_prompt.format(doc_text=doc_text, chunk_text=node.get_content())
        context = llm.complete(prompt).text.strip()

        node.text = f"{context}\n\n{node.get_content()}"
        print(f"  [{i+1}/{total}] context added ({source_file})")

    return nodes


# Embed and store
def embed_and_store(nodes: list) -> VectorStoreIndex:
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
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


# Run the ingestion pipeline
if __name__ == "__main__":
    start_phoenix()
    init_tracing()
    register_prompts()

    if CHECKPOINT_FILE.exists():
        print(f"Found checkpoint at {CHECKPOINT_FILE} - skipping conversion, chunking, and contextual enrichment.")
        nodes = load_checkpoint(CHECKPOINT_FILE)
        print(f"Loaded {len(nodes)} already-enriched chunks from checkpoint.")
    else:
        all_markdown = convert_all_pdfs()

        documents = [
            markdown_to_document(md_text, filename)
            for filename, md_text in all_markdown.items()
        ]

        nodes = chunk_documents(documents)
        print(f"\nTotal chunks across all documents: {len(nodes)}")
        print("\nGenerating contextual blurbs for each chunk (this may take a while)...")
        nodes = add_context_to_chunks(nodes, documents)

        save_checkpoint(nodes, CHECKPOINT_FILE)
        print(f"Checkpoint saved to {CHECKPOINT_FILE} ({len(nodes)} enriched chunks).")

    print("\nEmbedding and storing chunks in PGVector...")
    index = embed_and_store(nodes)
    print("Done — chunks embedded and stored in policy_chunks table.")
    print("\n--- Sample chunk ---")
    print(nodes[0].get_content()[:500])
    print("\nMetadata:", nodes[0].metadata)
