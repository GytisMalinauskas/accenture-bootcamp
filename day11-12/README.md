# Local Document Q&A Service (Day 11–12 Capstone)

A local Retrieval-Augmented Generation (RAG) pipeline that answers questions over internal company PDFs using Ollama and ChromaDB, with zero cloud API calls.

## Project Structure

- `config.py` — Central configuration for LLM, embedding model, chunking, and retrieval (supports swapping to cloud via environment variables).
- `ingest.py` — Extracts PDF text (`pypdf` layout mode), chunks text (`1024` chars, `150` overlap), embeds with `nomic-embed-text`, and stores vectors in `chroma_db/` (cosine space).
- `retrieve.py` — Embeds queries locally and retrieves top-`K` (`8`) chunks with cosine relevance scores and metadata.
- `generate.py` — Prompts the local LLM (`qwen3.5:4b-q4_K_M`, `reasoning_effort="none"`) using only retrieved context.
- `pipeline.py` — End-to-end RAG runner over `test_questions.json` -> `pipeline_outputs.json`.
- `benchmark.py` — Compares `qwen3.5:4b-q4_K_M` and `qwen3.5:2b-q4_K_M` -> `benchmark_results.json`.
- `decision_log.md` — Architecture decisions, benchmark comparisons, and observations.

## Setup

1. Install [Ollama](https://ollama.com/download) and set context length to `8192` (`OLLAMA_CONTEXT_LENGTH=8192`).
2. Pull the required local models:
   ```powershell
   ollama pull qwen3.5:4b-q4_K_M
   ollama pull qwen3.5:2b-q4_K_M
   ollama pull nomic-embed-text
   ```
3. Install Python dependencies with `uv`:
   ```powershell
   uv sync
   ```

## How to Run

Rebuild the vector database (`chroma_db/`):
```powershell
uv run python ingest.py
```

Run the end-to-end pipeline over `test_questions.json`:
```powershell
uv run python pipeline.py
```

Run individual checks:
```powershell
uv run python test_api.py
uv run python benchmark.py
uv run python retrieve.py
uv run python generate.py
```

## AI Assistant Disclosure

Built with assistance from **Antigravity** (Gemini 4 Hydrogen).
