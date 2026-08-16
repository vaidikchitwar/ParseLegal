import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ──────────────────────────────────────────────
# Config & Secrets
# ──────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DB_PATH = "./chroma_db"

# ──────────────────────────────────────────────
# Page Setup
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Regal Mind – Indian Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0f;
    color: #e8e0d5;
}
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #12100e 0%, #1a1612 100%);
    border-right: 1px solid #2a2420;
}
[data-testid="stSidebar"] h2 {
    font-family: 'EB Garamond', serif;
    color: #c9a96e;
    font-size: 1.4rem;
}
.main .block-container { background-color: #0d0d0f; padding-top: 1.5rem; }
.regal-title {
    font-family: 'EB Garamond', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #c9a96e;
    letter-spacing: 0.05em;
    margin-bottom: 0;
}
.regal-sub {
    font-size: 0.82rem;
    color: #7a6e64;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.source-card {
    background: #18160f;
    border: 1px solid #2e2820;
    border-left: 3px solid #c9a96e;
    border-radius: 6px;
    padding: 0.65rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.8rem;
    color: #a09080;
    line-height: 1.55;
}
.source-label {
    font-weight: 600;
    color: #c9a96e;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.25rem;
}
.stButton > button {
    background: #c9a96e;
    color: #0d0d0f;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 0.45rem 1.2rem;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #d4b87e;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(201,169,110,0.25);
}
[data-testid="stFileUploader"] {
    background: #15130e;
    border: 1px dashed #3a3020;
    border-radius: 8px;
    padding: 0.5rem;
}
.status-pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.status-ready   { background: #1a2e1a; color: #5db85d; border: 1px solid #2d4a2d; }
.status-pending { background: #2e2010; color: #c9a96e; border: 1px solid #4a3010; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "db_ready" not in st.session_state:
    st.session_state.db_ready = False
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def build_rag_chain(vector_db):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",

        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
    )
    system_prompt = """You are Regal Mind, an expert Indian Legal Assistant.

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
    qa_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    return create_retrieval_chain(retriever, qa_chain)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def ingest_file(uploaded_file) -> Chroma:
    embeddings = get_embeddings()
    suffix = "." + uploaded_file.name.split(".")[-1]

    # Use getvalue() — safe regardless of Streamlit's internal file-pointer position
    file_bytes = uploaded_file.getvalue()
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
        os.unlink(tmp_path)  # always clean up temp file

    # Filter out pages with no meaningful text
    docs = [d for d in docs if d.page_content and d.page_content.strip()]

    if not docs:
        raise ValueError(
            "No text could be extracted from the document.\n\n"
            "This usually means the PDF contains scanned images instead of "
            "selectable text. Please use a text-based PDF, or convert the "
            "scanned PDF to text first (e.g., using Adobe Acrobat OCR or an "
            "online OCR tool)."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Guard: filter empty chunks
    chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
    if not chunks:
        raise ValueError("Document was loaded but produced no usable text chunks. Please try a different file.")

    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def load_existing_db() -> Chroma:
    return Chroma(persist_directory=DB_PATH, embedding_function=get_embeddings())


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Regal Mind")
    st.markdown("*AI Legal Assistant — Indian Law*")
    st.divider()

    st.markdown("### 📄 Document Source")
    tab1, tab2 = st.tabs(["Upload New", "Use Existing"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload a legal document",
            type=["pdf", "txt", "md", "docx"],
            label_visibility="collapsed",
        )
        if uploaded:
            if st.button("⚡ Analyse Document", use_container_width=True):
                with st.spinner("🔍 Reading & indexing document..."):
                    try:
                        vector_db = ingest_file(uploaded)
                        st.session_state.rag_chain = build_rag_chain(vector_db)
                        st.session_state.db_ready = True
                        st.session_state.doc_name = uploaded.name
                        st.session_state.messages = []
                        st.success(f"✅ Ready: **{uploaded.name}**")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    with tab2:
        st.caption("Load the pre-ingested `legal_doc.pdf` database")
        if st.button("📂 Load Existing DB", use_container_width=True):
            with st.spinner("Loading..."):
                try:
                    vector_db = load_existing_db()
                    count = vector_db._collection.count()
                    if count == 0:
                        st.warning("DB is empty. Upload a document first.")
                    else:
                        st.session_state.rag_chain = build_rag_chain(vector_db)
                        st.session_state.db_ready = True
                        st.session_state.doc_name = "legal_doc.pdf (pre-loaded)"
                        st.session_state.messages = []
                        st.success(f"✅ Loaded {count} chunks")
                except Exception as e:
                    st.error(f"❌ {e}")

    st.divider()

    if st.session_state.db_ready:
        st.markdown('<span class="status-pill status-ready">● Active</span>', unsafe_allow_html=True)
        st.markdown(f"**Document:** {st.session_state.doc_name}")
    else:
        st.markdown('<span class="status-pill status-pending">○ No document loaded</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 💡 Suggested Questions")
    suggestions = [
        "What are the key obligations in this document?",
        "What penalties or consequences are mentioned?",
        "What rights does this document grant?",
        "Are there any dispute resolution clauses?",
        "What Indian law applies to this agreement?",
        "What are the termination conditions?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s[:20]}", use_container_width=True):
            st.session_state["prefill"] = s

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Gemini 1.5 Flash · LangChain · ChromaDB")


# ──────────────────────────────────────────────
# MAIN CHAT AREA
# ──────────────────────────────────────────────
st.markdown('<div class="regal-title">⚖️ Regal Mind</div>', unsafe_allow_html=True)
st.markdown('<div class="regal-sub">Indian Legal Document Assistant</div>', unsafe_allow_html=True)

if not st.session_state.db_ready:
    st.info(
        "👈 **Get started**: Upload any legal document (PDF, DOCX, TXT) in the sidebar, "
        "or load the pre-ingested document. Then ask any question — Regal Mind will "
        "answer with reference to Indian law."
    )
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Source References", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.metadata.get("page", "—")
                        snippet = src.page_content[:320].strip().replace("\n", " ")
                        st.markdown(
                            f'<div class="source-card">'
                            f'<div class="source-label">Excerpt {i} · Page {page}</div>'
                            f'{snippet}…'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Pre-fill from suggestion buttons
    prefill = st.session_state.pop("prefill", None)
    user_input = st.chat_input("Ask a legal question based on your document…")
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("⚖️ Consulting Indian law…"):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    sources = response.get("context", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 Source References", expanded=False):
                            for i, src in enumerate(sources, 1):
                                page = src.metadata.get("page", "—")
                                snippet = src.page_content[:320].strip().replace("\n", " ")
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<div class="source-label">Excerpt {i} · Page {page}</div>'
                                    f'{snippet}…'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    err = f"❌ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})