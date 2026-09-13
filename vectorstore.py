from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from config import NVIDIA_API_KEY, EMBEDDING_MODEL, FAISS_INDEX_DIR, RETRIEVER_K


def get_embeddings() -> NVIDIAEmbeddings:
    """
    Initializes and returns the NVIDIA Embeddings instance.
    """
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY not found. Please set it in your .env file.")
    return NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=NVIDIA_API_KEY,
    )


def get_vectorstore(
    chunks: Optional[List[Document]] = None,
    persist_dir: Path = FAISS_INDEX_DIR,
) -> FAISS:
    """
    Retrieves the FAISS vector store. If a saved index exists locally, it loads it.
    Otherwise, it builds a new vector store from document chunks and saves it locally.
    """
    embeddings = get_embeddings()

    # Check if a cached FAISS index already exists on disk
    index_file = persist_dir / "index.faiss"
    if index_file.exists():
        print(f"Loading existing FAISS index from: {persist_dir}...")
        return FAISS.load_local(
            str(persist_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )

    # Build new index if chunks provided
    if not chunks:
        raise ValueError(
            f"No existing index found at {persist_dir}, and no document chunks were provided to build one."
        )

    print(f"Generating embeddings and building FAISS vector store for {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(embedding=embeddings, documents=chunks)

    # Cache index locally for instant future startups
    persist_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(persist_dir))
    print(f"Saved FAISS index to: {persist_dir}")

    return vectorstore


def get_retriever(vectorstore: FAISS, k: int = RETRIEVER_K):
    """
    Returns a retriever configured with the specified top-k neighbors.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})
