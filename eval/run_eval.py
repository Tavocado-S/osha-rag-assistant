"""
Basic prompt/retrieval evaluation harness.

This deliberately stays simple rather than pulling in a full eval
framework (e.g. RAGAS) for a first project — the goal is to demonstrate
you understand WHAT to measure, not to maximize tooling:

  - Retrieval hit-rate: did the correct section appear in the top-k
    retrieved chunks? This isolates retrieval quality from generation
    quality — if hit-rate is low, the fix is chunking/embeddings, not
    prompt tweaking.
  - Manual answer spot-check: the generated answer + citations are
    printed for human review, since faithfulness scoring properly
    needs either a second LLM-as-judge call or human eyes — both are
    reasonable "next iteration" additions to mention in an interview.

Run:
    python -m eval.run_eval
"""
import json
from pathlib import Path

from src.embed_store import load_vector_store
from src.rag_chain import answer_question
from src.config import settings

EVAL_SET_PATH = Path(__file__).parent / "eval_qa.json"


def run_retrieval_eval():
    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retrieval_k})

    hits = 0
    print(f"Running retrieval eval on {len(eval_set)} questions (k={settings.retrieval_k})...\n")

    for item in eval_set:
        docs = retriever.invoke(item["question"])
        retrieved_sections = [d.metadata.get("section_id", "") for d in docs]
        hit = any(item["expected_section"] in s for s in retrieved_sections)
        hits += hit

        status = "HIT " if hit else "MISS"
        print(f"[{status}] Q: {item['question']}")
        print(f"        expected: {item['expected_section']} | retrieved: {retrieved_sections}\n")

    hit_rate = hits / len(eval_set)
    print(f"Retrieval hit-rate: {hit_rate:.0%} ({hits}/{len(eval_set)})")
    return hit_rate


def run_answer_spotcheck():
    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    print("\n--- Answer spot-check (review manually) ---\n")
    for item in eval_set[:3]:  # keep API cost down — full generation on a subset
        result = answer_question(item["question"])
        print(f"Q: {item['question']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {[s['section_id'] for s in result['sources']]}\n")


if __name__ == "__main__":
    run_retrieval_eval()
    run_answer_spotcheck()
