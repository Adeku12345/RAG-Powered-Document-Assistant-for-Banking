# RAG Banking Document Q&A Platform

Ask questions about banking PDFs (terms, savings guide, credit card agreement, fee schedule) and get answers with source citations. Answers come only from the documents; out-of-scope questions are refused.

## Structure

```
.
├── app.py              # Streamlit UI
├── ingest.py           # build the vector store from data/pdf
├── evaluate.py         # score the pipeline against data/eval_set.json
├── src/
│   ├── config.py       # all settings (paths, models, chunking, top-k)
│   ├── loader.py       # PDF -> cleaned Markdown pages
│   ├── chunker.py      # pages -> one chunk per clause
│   ├── vectorstore.py  # embeddings + Chroma
│   ├── prompts.py      # prompt templates
│   ├── llm.py          # chat model
│   ├── rag.py          # retrieve + generate
│   └── evaluation.py   # evaluation logic
├── data/
│   ├── pdf/            # source documents
│   └── eval_set.json   # test questions
└── notebook/           # exploration only
```

## Setup

```
pip install -r requirements.txt
cp .env.example .env     # then add your Anthropic API key
```
