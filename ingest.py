"""Build the vector store: PDFs -> pages -> chunks -> embeddings -> Chroma.

Run from the project root:  python ingest.py
"""
from collections import Counter

from src import config
from src.chunker import split_documents
from src.loader import load_documents
from src.vectorstore import build_vectorstore


def main() -> None:
    documents = load_documents()
    print(f"Loaded {len(documents)} pages from {config.PDF_DIR}")

    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    for source, n in sorted(Counter(c.metadata["source"] for c in chunks).items()):
        print(f"  {source}: {n} chunks")

    build_vectorstore(chunks)
    print(f"Vector store saved to {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
