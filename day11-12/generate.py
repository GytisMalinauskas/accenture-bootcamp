import time

from openai import OpenAI

import config


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with source and page headers for the LLM prompt."""
    parts = []
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")
        content = chunk.get("content", "")
        parts.append(f"[Source: {source}, Page: {page}]\n{content}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Generate an answer from retrieved chunks using the configured LLM."""
    context = format_context(chunks)
    system_message = config.SYSTEM_PROMPT.format(context=context)

    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ],
        temperature=config.TEMPERATURE,
        reasoning_effort=config.REASONING_EFFORT,
        **{config.TOKEN_LIMIT_PARAM: config.MAX_TOKENS},
    )
    generation_time_ms = round((time.perf_counter() - start) * 1000)

    choice = response.choices[0]
    answer = choice.message.content or ""
    finish_reason = choice.finish_reason
    usage = response.usage

    if not answer.strip():
        print(
            f"WARNING: Empty answer returned for query={query!r} "
            f"(finish_reason={finish_reason})"
        )

    return {
        "query": query,
        "answer": answer,
        "context_used": context,
        "model": config.LLM_MODEL,
        "thinking": config.REASONING_EFFORT != "none",
        "finish_reason": finish_reason,
        "generation_time_ms": generation_time_ms,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "tokens_generated": usage.completion_tokens if usage else None,
    }


if __name__ == "__main__":
    sample_chunks = [
        {
            "content": "Employees may work remotely up to 3 days per week with manager approval.",
            "source": "company_handbook.pdf",
            "page": 1,
        }
    ]
    res = generate_answer("What is the maximum number of remote work days per week?", sample_chunks)
    print(f"Answer: {res['answer']}")
    print(
        f"model={res['model']} | finish={res['finish_reason']} | "
        f"prompt_tokens={res['prompt_tokens']} | tokens_generated={res['tokens_generated']} | "
        f"time={res['generation_time_ms']} ms"
    )
