from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader

import config


def load_pdf_pages(documents_dir: str) -> list[dict]:
    """Extract text per page from all PDFs in documents_dir using layout mode."""
    pdf_paths = sorted(Path(documents_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {documents_dir}/")

    pages = []
    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text(extraction_mode="layout") or ""
            if text.strip():
                pages.append(
                    {
                        "source": pdf_path.name,
                        "page": page_num,
                        "text": text,
                    }
                )
        print(f"  Loaded {pdf_path.name} ({len(reader.pages)} pages)")
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split extracted pages into character-sized overlapping chunks with stable IDs."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = []
    for page_item in pages:
        splits = splitter.split_text(page_item["text"])
        for idx, chunk_text in enumerate(splits):
            chunk_id = f"{page_item['source']}::p{page_item['page']}::c{idx}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": page_item["source"],
                        "page": page_item["page"],
                        "chunk_index": idx,
                    },
                }
            )
    return chunks


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Compute local embeddings via Ollama's OpenAI-compatible embeddings endpoint."""
    client = OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key="ollama")
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


def ingest() -> None:
    print(f"Loading PDFs from {config.DOCUMENTS_DIR}/...")
    pages = load_pdf_pages(config.DOCUMENTS_DIR)

    print(
        f"Chunking {len(pages)} non-empty pages "
        f"(size={config.CHUNK_SIZE} chars, overlap={config.CHUNK_OVERLAP} chars)..."
    )
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks.")

    print(f"Generating embeddings with {config.EMBEDDING_MODEL}...")
    embeddings = embed_texts([c["text"] for c in chunks])
    print(f"Embedding dimensions: {len(embeddings[0])}")

    chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        chroma_client.delete_collection(name=config.COLLECTION_NAME)
    except NotFoundError:
        pass

    collection = chroma_client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
    )
    print(
        f"Stored {collection.count()} chunks in '{config.COLLECTION_NAME}' "
        f"at {config.CHROMA_DIR}/ (cosine space)."
    )


if __name__ == "__main__":
    ingest()
