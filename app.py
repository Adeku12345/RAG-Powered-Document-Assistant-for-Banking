"""Streamlit UI.  Run from the project root:  streamlit run app.py"""
import streamlit as st

from src import config
from src.llm import get_llm
from src.rag import answer_question
from src.vectorstore import load_vectorstore

EXAMPLES = [
    "What is the ATM withdrawal limit?",
    "What is the fee for cash withdrawals abroad?",
    "What should I do if my card is stolen?",
    "What is the monthly account fee?",
    "Do you offer mortgages?",
]


@st.cache_resource
def _vectorstore():
    return load_vectorstore()


@st.cache_resource
def _llm():
    return get_llm()


def render_sources(sources) -> None:
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**{s.source}, p. {s.page}**")
            st.text(s.text)


st.set_page_config(page_title="Banking Document Q&A", page_icon="🏦")
st.title("🏦 Banking Document Q&A")
st.caption("Ask questions about the Harborview Bank sample documents. Answers cite their sources.")

if not config.CHROMA_DIR.exists():
    st.error("Vector store not found. Run `python ingest.py` in the project root first.")
    st.stop()

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.header("Settings")
    k = st.slider("Chunks to retrieve (k)", 1, 8, config.TOP_K)
    st.markdown("**Try asking**")
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state["pending"] = example
    if st.button("Clear chat"):
        st.session_state["history"] = []
        st.rerun()

for question, response in st.session_state["history"]:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.markdown(response.answer)
        render_sources(response.sources)

question = st.chat_input("Ask about accounts, fees, cards, savings...") or st.session_state.pop("pending", None)

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                response = answer_question(question, _vectorstore(), _llm(), k)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.markdown(response.answer)
        render_sources(response.sources)
    st.session_state["history"].append((question, response))