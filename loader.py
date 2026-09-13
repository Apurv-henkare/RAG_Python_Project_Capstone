from pathlib import Path
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf_documents(folder_path: Path = DOCUMENTS_DIR) -> List[Document]:
    """
    Loads all PDF files from the specified directory and extracts text into Document objects.
    """
    documents: List[Document] = []
    pdf_files = list(folder_path.glob("*.pdf"))

    if not pdf_files:
        print(f"Warning: No PDF files found in {folder_path}")
        return documents

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(pdf_path.name),
                                "page": page_number + 1,
                            },
                        )
                    )
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits loaded documents into smaller overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def get_document_chunks(folder_path: Path = DOCUMENTS_DIR) -> List[Document]:
    """
    Convenience function that loads and splits all documents in one step.
    """
    print(f"Loading documents from: {folder_path}...")
    docs = load_pdf_documents(folder_path)
    print(f"Loaded {len(docs)} pages. Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")
    return chunks
