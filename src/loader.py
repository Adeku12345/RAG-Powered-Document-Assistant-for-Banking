"""Step 1: load PDFs into LangChain Documents (one per page), with tables kept as Markdown."""
import re
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document

from . import config


def clean_text(text: str, noise_patterns=config.NOISE_PATTERNS) -> str:
    """Remove repeated headers/footers and bold markers, and collapse blank lines."""
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    text = text.replace("**", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_pdf(path: Path) -> list[Document]:
    """Load one PDF. Metadata: filename and 1-indexed page number."""
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    docs = []
    for i, page in enumerate(pages):
        text = clean_text(page["text"])
        if text:  # skip empty pages
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": i + 1}))
    return docs


def load_documents(pdf_dir: Path = config.PDF_DIR) -> list[Document]:
    """Load every PDF under pdf_dir."""
    pdfs = sorted(Path(pdf_dir).glob("**/*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    documents = []
    for pdf in pdfs:
        documents.extend(load_pdf(pdf))
    return documents
