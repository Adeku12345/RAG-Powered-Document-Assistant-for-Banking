"""Prompt templates."""
from langchain_core.prompts import ChatPromptTemplate

REFUSAL_MESSAGE = "I don't have that information in the available documents."

SYSTEM_PROMPT = (
    "You are a banking support assistant. Answer ONLY from the context below. "
    'If the context does not contain the answer, reply only: "' + REFUSAL_MESSAGE + '" '
    "Do not describe what the context covers or suggest other products. "
    "Cite sources as (filename, p. N) after each fact.\n\nContext:\n{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])
