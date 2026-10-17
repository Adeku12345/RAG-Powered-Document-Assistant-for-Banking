"""Run a question set through the pipeline and score it."""
import json
from pathlib import Path

from . import config
from .rag import answer_question


def _norm(text: str) -> str:
    return text.replace("\u2019", "'").lower()  # treat curly and straight apostrophes the same


def load_eval_set(path: Path = config.EVAL_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(eval_set, vectorstore, llm, k: int = config.TOP_K) -> list[dict]:
    """Each item: {"question", "expected", "type": "answer" | "refusal"}.

    retrieval_ok: the expected text was in the retrieved chunks (skipped for refusals).
    answer_ok:    the expected text was in the final answer.
    Comparing the two shows whether a failure came from retrieval or from the LLM.
    """
    results = []
    for item in eval_set:
        resp = answer_question(item["question"], vectorstore, llm, k)
        expected = _norm(item["expected"])
        retrieval_ok = None
        if item.get("type", "answer") == "answer":
            retrieval_ok = any(expected in _norm(s.text) for s in resp.sources)
        results.append({
            "question": item["question"],
            "answer": resp.answer,
            "retrieval_ok": retrieval_ok,
            "answer_ok": expected in _norm(resp.answer),
        })
    return results
