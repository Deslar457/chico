# app.py

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import os

# === Load Groq API Key from Streamlit secrets ===
api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY"))

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

# === Page Layout ===
st.set_page_config(page_title="Michele Research Assistant", layout="centered")
st.title("📘 Michele: PhD Tool")
st.markdown("Ask a question about your PhD.")

# === Load Vector Store ===
@st.cache_resource
def load_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("vector_store", embedding_model, allow_dangerous_deserialization=True)
    return vector_store.as_retriever()

retriever = load_retriever()

# === Generate Answer ===
def generate_answer(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs[:3]])

    messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. Only use the provided context to answer the question. "
            "Do not use any external knowledge or assumptions. "
            "If the answer is not in the context, say 'I don't know based on the provided document.'"
        )
    },
    {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    }
]

git c
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )

    return response.choices[0].message.content.strip(), docs

# === Interface ===
query = st.text_input("💬 Ask your question:")
if query:
    with st.spinner("Thinking..."):
        answer, docs = generate_answer(query)

    st.markdown("### ✅ Answer")
    st.write(answer)

    st.markdown("### 🔍 Source Chunks")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        st.markdown(f"**Chunk {i+1}:** `{source}`")
        st.markdown(doc.page_content[:400] + "...")


# === Footer ===
st.markdown("""---""")
st.markdown(
    "Michele app uses RAG (Retrieval-Augmented Generation)  to answer questions. Built with Streamlit, FAISS, HuggingFace, LangChain, and Groq."
)
