# ingest.py

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# === Step 1: Load PDFs =======
pdf_folder = "data"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

documents = []
for file in pdf_files:
    loader = UnstructuredPDFLoader(os.path.join(pdf_folder, file))
    documents.extend(loader.load())

# === Step 2: Chunk the text ===
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# === Step 3: Embed with HuggingFace model ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Step 4: Store in FAISS vector store ===
vector_store = FAISS.from_documents(chunks, embedding_model)
vector_store.save_local("vector_store")

print("✅ Ingestion complete. Vector store saved to ./vector_store")
