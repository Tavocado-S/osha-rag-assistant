"""
Embed the chunks from chunking.py and persist them in a local
Chroma vector store. Run once after ingest.py; the API just loads the
persisted store afterwards (no re-embedding on every request).

Run:
    python -m src.embed_store
"""
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.chunking import build_documents
from src.config import settings


def get_embeddings():
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        chunk_size=500,  # texts per API request — keeps us under OpenAI's 300k-token-per-request limit
    )


def build_vector_store() -> Chroma:
    documents = build_documents()
    embeddings = get_embeddings()

    print(f"Embedding {len(documents)} chunks with {settings.embedding_model}...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_persist_dir,
    )
    print(f"Persisted vector store to {settings.chroma_persist_dir}")
    return vector_store


def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )


if __name__ == "__main__":
    build_vector_store()
