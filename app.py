import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

# ──────────────────────────────────────────────
# Config & Secrets
# ──────────────────────────────────────────────
load_dotenv()
def get_hf_token():
    try:
        if "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
        if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
            return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except Exception:
        pass
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

HF_TOKEN = get_hf_token()
DB_PATH = "./chroma_db"

# ──────────────────────────────────────────────
# Page Setup
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ParseLegal — AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — Full Design System
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #09090B;
    color: #FAFAFA;
}
.main .block-container {
    background-color: #09090B;
    max-width: 800px;
    padding-top: 2rem;
    padding-bottom: 6rem;
}

/* ── Sidebar (Ingestion Zone) ── */
[data-testid="stSidebar"] {
    background-color: #18181B;
    border-right: 1px solid #27272A;
    width: 300px !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #A1A1AA;
    font-size: 0.85rem;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Inter', sans-serif;
    color: #FAFAFA;
    font-weight: 600;
}

/* ── App Title (Sidebar Header) ── */
.app-header {
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #FAFAFA;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.app-header-accent {
    color: #2563EB;
}
.app-subtitle {
    font-size: 0.78rem;
    color: #A1A1AA;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Main Area Title ── */
.main-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: #FAFAFA;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}
.main-title-accent {
    color: #2563EB;
}
.main-sub {
    font-size: 0.82rem;
    color: #A1A1AA;
    letter-spacing: 0.04em;
    margin-bottom: 1.5rem;
}

/* ── Buttons — Primary (Royal Blue) ── */
.stButton > button {
    background-color: #2563EB;
    color: #FAFAFA;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    transition: background-color 150ms ease-in-out;
    box-shadow: none;
}
.stButton > button:hover {
    background-color: #3B82F6;
    box-shadow: none;
    transform: none;
}
.stButton > button:active {
    background-color: #1D4ED8;
}

/* ── File Uploader (Dropzone) ── */
[data-testid="stFileUploader"] {
    background-color: #09090B;
    border: 2px dashed #27272A;
    border-radius: 8px;
    padding: 0.75rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3B82F6;
}
[data-testid="stFileUploader"] label {
    color: #A1A1AA !important;
}

/* ── Status Indicators ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.status-indexed {
    background-color: #064E3B;
    color: #34D399;
    border: 1px solid #065F46;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #10B981;
    display: inline-block;
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.status-waiting {
    background-color: #18181B;
    color: #A1A1AA;
    border: 1px solid #27272A;
}

/* ── Source / Citation Cards ── */
.source-card {
    background-color: #18181B;
    border: 1px solid #27272A;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #A1A1AA;
    line-height: 1.6;
}
.source-label {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.citation-badge {
    display: inline-block;
    background-color: #064E3B;
    color: #34D399;
    border-radius: 9999px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 500;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background-color: transparent;
    border: none;
    padding: 0.75rem 0;
}
/* Assistant messages get the elevated zinc surface */
[data-testid="stChatMessage"][aria-label="assistant"] {
    background-color: #18181B;
    border: 1px solid #27272A;
    border-radius: 8px;
    padding: 1rem;
}
/* User messages get left blue border */
[data-testid="stChatMessage"][aria-label="user"] {
    border-left: 2px solid #2563EB;
    padding-left: 1rem;
    border-radius: 0;
}

/* ── Chat Input (Fixed Bottom) ── */
[data-testid="stChatInput"] {
    background-color: #18181B;
    border: 1px solid #27272A;
    border-radius: 8px;
}
[data-testid="stChatInput"] textarea {
    background-color: #18181B !important;
    color: #FAFAFA !important;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
}
[data-testid="stChatInput"] button {
    color: #2563EB !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #27272A;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    color: #A1A1AA;
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 0.5rem 1rem;
    transition: all 150ms ease-in-out;
}
.stTabs [aria-selected="true"] {
    color: #FAFAFA !important;
    border-bottom-color: #2563EB !important;
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #FAFAFA;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #A1A1AA;
    background-color: transparent;
}
[data-testid="stExpander"] {
    border: 1px solid #27272A;
    border-radius: 8px;
    background-color: #09090B;
}

/* ── Divider ── */
hr {
    border-color: #27272A;
}

/* ── Suggestion Buttons (ghost style) ── */
.suggestion-btn button {
    background-color: transparent !important;
    border: 1px solid #27272A !important;
    color: #A1A1AA !important;
    font-weight: 400 !important;
    font-size: 0.8rem !important;
    text-align: left !important;
}
.suggestion-btn button:hover {
    border-color: #2563EB !important;
    color: #FAFAFA !important;
    background-color: rgba(37, 99, 235, 0.08) !important;
}

/* ── Info Box ── */
[data-testid="stAlert"] {
    background-color: #18181B;
    border: 1px solid #27272A;
    border-radius: 8px;
    color: #A1A1AA;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #2563EB !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #09090B; }
::-webkit-scrollbar-thumb { background: #27272A; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3F3F46; }

/* ── Caption / Footer text ── */
.stCaption, [data-testid="stCaption"] {
    color: #52525B !important;
    font-size: 0.72rem !important;
}
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
    import requests as _requests

    HF_MODEL = "google/gemma-2-2b-it"
    HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"

    def call_hf_llm(prompt_value):
        """Call Hugging Face Inference API directly via HTTP, bypassing provider routing."""
        messages = []
        for msg in prompt_value.to_messages():
            role = "user" if msg.type == "human" else ("system" if msg.type == "system" else "assistant")
            messages.append({"role": role, "content": msg.content})

        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = _requests.post(
                    HF_API_URL,
                    headers={
                        "Authorization": f"Bearer {HF_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": HF_MODEL,
                        "messages": messages,
                        "max_tokens": 1024,
                        "temperature": 0.3,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except _requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)

    llm = RunnableLambda(call_hf_llm)
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

    # LCEL chain: retrieve docs, format context, run LLM
    # Returns dict with keys: "context" (list of docs), "input" (str), "answer" (str)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableParallel(
        {"context": lambda x: retriever.invoke(x["input"]), "input": lambda x: x["input"]}
    ).assign(answer=rag_chain_from_docs)


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
# SIDEBAR (Ingestion Zone)
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="app-header">Parse<span class="app-header-accent">Legal</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="app-subtitle">AI-Powered Legal Document Analysis</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Document Source")
    tab1, tab2 = st.tabs(["Upload New", "Use Existing"])

    with tab1:
        uploaded = st.file_uploader(
            "Drag & Drop PDF Agreement",
            type=["pdf", "txt", "md", "docx"],
            label_visibility="collapsed",
        )
        if uploaded:
            if st.button("Analyse Document", use_container_width=True):
                with st.spinner("Indexing document..."):
                    try:
                        vector_db = ingest_file(uploaded)
                        st.session_state.rag_chain = build_rag_chain(vector_db)
                        st.session_state.db_ready = True
                        st.session_state.doc_name = uploaded.name
                        st.session_state.messages = []
                        st.success(f"Ready: **{uploaded.name}**")
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab2:
        st.caption("Load the pre-ingested `legal_doc.pdf` database")
        if st.button("Load Existing DB", use_container_width=True):
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
                        st.success(f"Loaded {count} chunks")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # Status indicator
    if st.session_state.db_ready:
        st.markdown(
            '<div class="status-badge status-indexed">'
            '<span class="status-dot"></span> Status: Indexed'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Document:** {st.session_state.doc_name}")
    else:
        st.markdown(
            '<div class="status-badge status-waiting">'
            '○ No document loaded'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown("### Suggested Queries")
    suggestions = [
        "What are the key obligations in this document?",
        "What penalties or consequences are mentioned?",
        "What rights does this document grant?",
        "Are there any dispute resolution clauses?",
        "What Indian law applies to this agreement?",
        "What are the termination conditions?",
    ]
    for s in suggestions:
        st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
        if st.button(s, key=f"sug_{s[:20]}", use_container_width=True):
            st.session_state["prefill"] = s
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by Hugging Face · LangChain · ChromaDB")


# ──────────────────────────────────────────────
# MAIN CONTENT AREA (Query Zone)
# ──────────────────────────────────────────────
st.markdown(
    '<div class="main-title">Parse<span class="main-title-accent">Legal</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-sub">Indian Legal Document Assistant</div>',
    unsafe_allow_html=True,
)

if not st.session_state.db_ready:
    st.info(
        "**Get started** — Upload any legal document (PDF, DOCX, TXT) in the sidebar, "
        "or load the pre-ingested document. Then ask any question and ParseLegal will "
        "answer with reference to Indian law."
    )
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Source References", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.metadata.get("page", "—")
                        snippet = src.page_content[:320].strip().replace("\n", " ")
                        st.markdown(
                            f'<div class="source-card">'
                            f'<div class="source-label">'
                            f'<span class="citation-badge">Page {page}</span> Excerpt {i}'
                            f'</div>'
                            f'{snippet}…'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Pre-fill from suggestion buttons
    prefill = st.session_state.pop("prefill", None)
    user_input = st.chat_input("Ask a legal question about your document…")
    if prefill and not user_input:
        user_input = prefill

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analysing document…"):
                try:
                    response = st.session_state.rag_chain.invoke({"input": user_input})
                    answer = response["answer"]
                    sources = response.get("context", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("Source References", expanded=False):
                            for i, src in enumerate(sources, 1):
                                page = src.metadata.get("page", "—")
                                snippet = src.page_content[:320].strip().replace("\n", " ")
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<div class="source-label">'
                                    f'<span class="citation-badge">Page {page}</span> Excerpt {i}'
                                    f'</div>'
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
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})