# query.py
import os
import sys
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # <--- NEW
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: GOOGLE_API_KEY not found in .env")
    sys.exit(1)

# 1. Setup LOCAL Embeddings (Must match ingest.py)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Load DB
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function
)

# 3. Setup Gemini (Only for the final answer)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",

    temperature=0.3,
    google_api_key=api_key
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

if __name__ == "__main__":
    print("🤖 Regal Mind (Hybrid: Local Embeddings + Gemini LLM)")
    print("-" * 50)
    while True:
        query = input("\nUse Query: ")
        if query.lower() in ["exit", "quit"]: break
        try:
            response = qa_chain.invoke({"query": query})
            print(f"\nAnswer: {response['result']}")
        except Exception as e:
            print(f"Error: {e}")