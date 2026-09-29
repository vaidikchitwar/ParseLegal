import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DB_PATH = "./chroma_db"

app = FastAPI(title="ParseLegal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
app.state.rag_chain = None
app.state.db_ready = False
app.state.doc_name = None

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_rag_chain(vector_db):
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
    )
    system_prompt = """You are ParseLegal, an expert Indian Legal Assistant.

Analyse the provided document excerpts and answer the user's question thoroughly, referencing:
- Relevant sections and provisions from the document
- Applicable Indian statutes (IPC, CPC, CrPC, Constitution of India, Contract Act 1872, Transfer of Property Act, Consumer Protection Act, IT Act, Companies Act, etc.)
- Landmark Supreme Court or High Court judgements where relevant
- Legal principles under Indian jurisprudence

GUIDELINES:
1. Ground your answer primarily in the retrieved document context.
2. Where the document is silent, cite the relevant Indian law that would apply.
3. If the answer cannot be determined, say so clearly.
4. Use precise legal language. Cite specific sections (e.g., "Section 17 of the Indian Contract Act, 1872").
5. Structure: direct answer → supporting reasoning → legal basis.
6. End with: "⚠️ This is for informational purposes only and does not constitute legal advice."

---
Retrieved document context:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        {"context": lambda x: retriever.invoke(x["input"]), "input": lambda x: x["input"]}
    ).assign(answer=rag_chain_from_docs)

def ingest_file_logic(file_bytes: bytes, filename: str) -> Chroma:
    embeddings = get_embeddings()
    suffix = "." + filename.split(".")[-1]

    if not file_bytes:
        raise ValueError("The uploaded file appears to be empty (0 bytes).")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        ext = suffix.lower()
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
        elif ext in (".txt", ".md"):
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(tmp_path, encoding="utf-8")
        elif ext == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        raise ValueError("No text could be extracted from the document.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    if not chunks:
        raise ValueError("Document was loaded but produced no usable text chunks.")

    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def load_existing_db() -> Chroma:
    return Chroma(persist_directory=DB_PATH, embedding_function=get_embeddings())


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]

@app.post("/api/ingest")
async def api_ingest(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        vector_db = ingest_file_logic(file_bytes, file.filename)
        app.state.rag_chain = build_rag_chain(vector_db)
        app.state.db_ready = True
        app.state.doc_name = file.filename
        return {"status": "success", "message": f"Ready: {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/load_existing")
def api_load_existing():
    try:
        vector_db = load_existing_db()
        count = vector_db._collection.count()
        if count == 0:
            raise ValueError("DB is empty. Upload a document first.")
        
        app.state.rag_chain = build_rag_chain(vector_db)
        app.state.db_ready = True
        app.state.doc_name = "legal_doc.pdf (pre-loaded)"
        return {"status": "success", "message": f"Loaded {count} chunks"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest):
    if not app.state.db_ready or app.state.rag_chain is None:
        raise HTTPException(status_code=400, detail="Database is not ready. Please ingest or load a document first.")
    
    try:
        response = app.state.rag_chain.invoke({"input": req.message})
        answer = response["answer"]
        sources = []
        for src in response.get("context", []):
            sources.append({
                "page": src.metadata.get("page", "—"),
                "content": src.page_content
            })
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
def api_status():
    return {
        "db_ready": app.state.db_ready,
        "doc_name": app.state.doc_name
    }
