import argparse
import json
from pathlib import Path

from generate import generate_answer
from retrieve import retrieve_chunks


def answer_question(query: str) -> dict:
    """Run end-to-end RAG: retrieve relevant chunks and generate an answer."""
    retrieval_result = retrieve_chunks(query)
    generation_result = generate_answer(
        query=query,
        chunks=retrieval_result["retrieved_chunks"],
    )
    return {
        **generation_result,
        "retrieval": retrieval_result,
    }


def run_test_suite(
    questions_path: str = "test_questions.json",
    output_path: str = "pipeline_outputs.json",
) -> None:
    """Run the pipeline over all test questions and save pipeline_outputs.json."""
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    outputs = []

    for i, item in enumerate(questions, start=1):
        qid = item["id"]
        query = item["question"]
        print(f"[{i}/{len(questions)}] {qid}: {query}")

        result = answer_question(query)
        retrieval = result["retrieval"]
        chunks = retrieval["retrieved_chunks"]

        row = {
            "id": qid,
            "question": query,
            "expected_answer": item["expected_answer"],
            "actual_answer": result["answer"],
            "retrieved_context": [c["content"] for c in chunks],
            "context_sources": [
                {
                    "source": c["source"],
                    "page": c["page"],
                    "score": c["relevance_score"],
                }
                for c in chunks
            ],
            "model": result["model"],
            "thinking": result["thinking"],
            "finish_reason": result["finish_reason"],
            "embedding_model": retrieval["embedding_model"],
            "prompt_tokens": result["prompt_tokens"],
            "tokens_generated": result["tokens_generated"],
            "retrieval_time_ms": retrieval["retrieval_time_ms"],
            "generation_time_ms": result["generation_time_ms"],
            "passed": None,
            "rating_note": "",
        }
        outputs.append(row)
        print(
            f"  finish={row['finish_reason']} | prompt_tokens={row['prompt_tokens']} | "
            f"gen={row['generation_time_ms']} ms"
        )

    Path(output_path).write_text(
        json.dumps(outputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(outputs)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline over test_questions.json.")
    parser.add_argument(
        "--output",
        "--test-set",
        dest="output_path",
        default="pipeline_outputs.json",
        help="Path to write pipeline outputs JSON (default: pipeline_outputs.json)",
    )
    args = parser.parse_args()

    if Path("test_questions.json").exists():
        run_test_suite(output_path=args.output_path)
    else:
        print("test_questions.json not found yet — running quick smoke test:\n")
        for sample_q in [
            "What is the maximum number of remote work days per week?",
            "Compare the Professional and Enterprise support tiers.",
        ]:
            res = answer_question(sample_q)
            print(f"Q: {sample_q}")
            print(f"A: {res['answer']}")
            print(
                f"   finish={res['finish_reason']} | prompt_tokens={res['prompt_tokens']} | "
                f"tokens_generated={res['tokens_generated']} | "
                f"retrieval={res['retrieval']['retrieval_time_ms']} ms | "
                f"gen={res['generation_time_ms']} ms\n"
            )
