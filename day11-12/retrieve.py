import time

import chromadb
from openai import OpenAI

import config


def embed_query(query: str) -> list[float]:
    """Embed a single query string using the configured local embedding model."""
    client = OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key="ollama")
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


def retrieve_chunks(query: str, top_k: int | None = None) -> dict:
    """Query ChromaDB with a local embedding and return ranked chunks with metadata."""
    k = top_k if top_k is not None else config.TOP_K
    start = time.perf_counter()

    query_embedding = embed_query(query)
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = chroma_client.get_collection(name=config.COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000)

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    retrieved_chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved_chunks.append(
            {
                "content": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
                "relevance_score": round(1.0 - dist, 4),
            }
        )

    return {
        "query": query,
        "retrieved_chunks": retrieved_chunks,
        "retrieval_time_ms": elapsed_ms,
        "embedding_model": config.EMBEDDING_MODEL,
    }


if __name__ == "__main__":
    sample_queries = [
        "What is the maximum number of remote work days per week?",
        "Compare the Professional and Enterprise support tiers.",
    ]
    for q in sample_queries:
        print(f"\nQuery: {q}")
        res = retrieve_chunks(q)
        print(
            f"Retrieved {len(res['retrieved_chunks'])} chunks in {res['retrieval_time_ms']} ms ({res['embedding_model']})"
        )
        for i, chunk in enumerate(res["retrieved_chunks"], start=1):
            preview = chunk["content"].replace("\n", " ")
            print(
                f"  {i}. [{chunk['source']} p.{chunk['page']}] score={chunk['relevance_score']} | {preview}..."
            )
