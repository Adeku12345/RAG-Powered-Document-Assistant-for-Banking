"""Step 2: split pages into one chunk per clause, using the Markdown headings."""
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from . import config


def split_documents(
    documents: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    min_chars: int = config.MIN_CHUNK_CHARS,
) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "doc_title"), ("##", "section")],
        strip_headers=False,  # keep the heading inside the chunk text
    )

    sections = []
    for doc in documents:
        for piece in header_splitter.split_text(doc.page_content):
            piece.metadata.update(doc.metadata)  # source + page
            # Prefix the filename so similar clauses in different documents stay distinguishable
            piece.page_content = f"[{doc.metadata['source']}]\n{piece.page_content}"
            sections.append(piece)

    # Only clauses longer than chunk_size get split further
    fallback = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = fallback.split_documents(sections)

    return [c for c in chunks if len(c.page_content) > min_chars]
