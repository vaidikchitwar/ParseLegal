# ingest.py
"""
Ingest a legal document into the ChromaDB vector store.
Usage:
    python ingest.py                      # uses legal_doc.pdf (default)
    python ingest.py my_contract.pdf      # uses a custom file
    python ingest.py agreement.docx
    python ingest.py notice.txt
"""
import os
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED = {".pdf", ".txt", ".md", ".docx"}


def get_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {SUPPORTED}")


def create_vector_db(file_path: str = "legal_doc.pdf"):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    print(f"📄 Loading document: {file_path}")
    loader = get_loader(file_path)
    docs = loader.load()
    print(f"   → Loaded {len(docs)} page(s)")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"🧩 Split into {len(chunks)} chunks")

    print("⚙️  Creating embeddings (this may take a moment)...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("💾 Saving to ./chroma_db ...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory="./chroma_db",
    )
    print(f"✅ Done! {len(chunks)} chunks indexed in ./chroma_db")
    return vector_db


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "legal_doc.pdf"
    create_vector_db(file_path)