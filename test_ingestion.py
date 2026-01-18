from rag.ingestion import load_documents, chunk_documents

docs = load_documents()
chunks = chunk_documents(docs)

print(f"Loaded docs: {len(docs)}")
print(f"Generated chunks: {len(chunks)}")
print(chunks[0])
