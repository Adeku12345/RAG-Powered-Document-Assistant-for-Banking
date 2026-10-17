"""Step 3: embeddings and the Chroma vector store."""
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from . import config


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)


def _open_store(embeddings) -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Create the store from scratch. Any previous collection is dropped, so re-runs never duplicate chunks."""
    embeddings = get_embeddings()
    _open_store(embeddings).delete_collection()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=str(config.CHROMA_DIR),
    )


def load_vectorstore() -> Chroma:
    """Open the saved store without re-embedding anything."""
    if not config.CHROMA_DIR.exists():
        raise FileNotFoundError(f"No vector store at {config.CHROMA_DIR}. Run `python ingest.py` first.")
    return _open_store(get_embeddings())
