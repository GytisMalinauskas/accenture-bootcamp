import json
import time
from openai import OpenAI

MODELS = ["qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M"]
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
REASONING_EFFORT = "none"
MAX_TOKENS = 1024
TEMPERATURE = 0.7

PROMPTS = [
    {
        "category": "factual",
        "prompt": "What causes tides on Earth? Answer in 2-3 sentences.",
    },
    {
        "category": "reasoning",
        "prompt": (
            "If a train travels 120km in 1.5 hours, and then 80km in 1 hour, "
            "what is the average speed for the entire journey? Show your steps."
        ),
    },
    {
        "category": "summarization",
        "prompt": (
            "Summarize the following paragraph in 2 sentences:\n\n"
            "Local large language models allow organizations in regulated industries "
            "to keep sensitive documents on-premise, avoiding third-party cloud APIs. "
            "However, running models locally introduces hardware constraints: smaller "
            "quantized models use less RAM and respond faster, while larger models "
            "generally follow complex instructions more reliably. Choosing the right "
            "model requires benchmarking speed, memory footprint, and answer quality "
            "on representative tasks."
        ),
    },
    {
        "category": "structured",
        "prompt": (
            "List 3 European capitals. Respond ONLY with valid JSON: "
            '[{"city": "...", "country": "..."}]'
        ),
    },
    {
        "category": "code",
        "prompt": (
            "Write a Python function `chunk_text(text: str, size: int, overlap: int) -> list[str]` "
            "that splits a string into overlapping chunks of length `size`. "
            "Include a brief docstring and handle invalid inputs."
        ),
    },
    {
        "category": "instruction_following",
        "prompt": (
            "Give 3 tips for writing good RAG test questions. "
            "Format as a numbered list (1., 2., 3.), each item under 12 words, "
            "and do not include an intro or outro sentence."
        ),
    },
]


def run_benchmark(client: OpenAI, model: str, prompt_item: dict) -> dict:
    """Run a single prompt against a local Ollama model and collect metrics."""
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_item["prompt"]}],
        temperature=TEMPERATURE,
        reasoning_effort=REASONING_EFFORT,
        max_tokens=MAX_TOKENS,
    )
    elapsed = time.time() - start
    choice = response.choices[0]
    usage = response.usage
    content = choice.message.content or ""

    if not content.strip():
        print(
            f"WARNING: Empty response for model={model}, "
            f"category={prompt_item['category']}, finish_reason={choice.finish_reason}"
        )

    completion_tokens = usage.completion_tokens if usage else None
    tokens_per_sec = (
        round(completion_tokens / elapsed, 1)
        if completion_tokens and elapsed > 0
        else None
    )

    return {
        "model": model,
        "category": prompt_item["category"],
        "prompt": prompt_item["prompt"],
        "response": content,
        "thinking": REASONING_EFFORT != "none",
        "finish_reason": choice.finish_reason,
        "time_seconds": round(elapsed, 2),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_sec,
        "quality_rating": None,
        "rating_note": "",
    }


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    results = []

    for model in MODELS:
        print(f"\n=== Benchmarking {model} ===")
        for item in PROMPTS:
            print(f"  Running [{item['category']}]...", end=" ", flush=True)
            row = run_benchmark(client, model, item)
            results.append(row)
            print(
                f"{row['time_seconds']}s | {row['tokens_per_second']} tok/s | "
                f"finish={row['finish_reason']}"
            )

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} rows to benchmark_results.json")


if __name__ == "__main__":
    main()
