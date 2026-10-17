"""Score the pipeline against data/eval_set.json.

Run from the project root:  python evaluate.py
"""
from src.evaluation import load_eval_set, run_evaluation
from src.llm import get_llm
from src.vectorstore import load_vectorstore


def main() -> None:
    results = run_evaluation(load_eval_set(), load_vectorstore(), get_llm())

    for r in results:
        status = "PASS" if r["answer_ok"] else "FAIL"
        retrieval = {True: "hit", False: "MISS", None: "n/a"}[r["retrieval_ok"]]
        print(f"{status} | retrieval: {retrieval} | {r['question']}")

    answered = sum(r["answer_ok"] for r in results)
    graded = [r for r in results if r["retrieval_ok"] is not None]
    hits = sum(r["retrieval_ok"] for r in graded)
    print(f"\nAnswers correct: {answered}/{len(results)}")
    print(f"Retrieval hits:  {hits}/{len(graded)}")


if __name__ == "__main__":
    main()
