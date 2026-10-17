"""Step 4: retrieve relevant chunks and generate a cited answer."""
from dataclasses import dataclass

from . import config
from .prompts import RAG_PROMPT


@dataclass
class Source:
    source: str
    page: int
    text: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[Source]


def format_context(docs) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata['source']}, p. {d.metadata['page']}]\n{d.page_content}" for d in docs
    )


def extract_text(content) -> str:
    """Some models return a list of blocks (thinking + text). Keep only the text."""
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if b.get("type") == "text")
    return content


def answer_question(question: str, vectorstore, llm, k: int = config.TOP_K) -> RAGResponse:
    docs = vectorstore.similarity_search(question, k=k)
    response = (RAG_PROMPT | llm).invoke({"context": format_context(docs), "question": question})
    sources = [Source(d.metadata["source"], d.metadata["page"], d.page_content) for d in docs]
    return RAGResponse(answer=extract_text(response.content), sources=sources)
