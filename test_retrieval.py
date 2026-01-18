from langchain_community.vectorstores import FAISS
from rag.embeddings import get_embedding_model

db = FAISS.load_local("data/index", get_embedding_model(), allow_dangerous_deserialization=True)

results = db.similarity_search("How is AI used in medical products?", k=3)

for r in results:
    print(f"Source: {r.metadata['source']}, Page: {r.metadata.get('page','N/A')}")
    print(r.page_content[:300], "...\n")