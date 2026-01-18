import os
from rag.ingestion import load_documents, chunk_documents
from rag.embeddings import get_embedding_model
from langchain_community.vectorstores import FAISS

INDEX_PATH = "data/index"

def build_faiss_index():
    print("Loading documents...")
    docs = load_documents()

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    if len(chunks) == 0:
        print("[WARN] No chunks were created. Check document loading and chunking.")
        return

    embeddings = get_embedding_model()

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)

    print("FAISS index saved successfully at:" , INDEX_PATH)

if __name__ == "__main__":
    build_faiss_index()