from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os

DATA_DIR = "data/raw"

def load_documents() -> List[Document]:
    documents = []

    for file in os.listdir(DATA_DIR):
        if not file.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(DATA_DIR, file)

        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = file
                d.metadata["page"] = d.metadata.get("page", "unknown")
            documents.extend(docs)

        except Exception as e:
            print(f"[WARN] Skipping {file}: {e}")

    return documents

def clean_text(text:str) -> str:
    text = text.replace("\n"," ")
    text = " ".join(text.split())
    return text.strip()

def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 150,
        separators = ["\n\n","\n","."," ",""]
    )

    chunks = []
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        splits = splitter.split_documents([doc])
        chunks.extend(splits)

    chunk_counter = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")

        if source not in chunk_counter:
            chunk_counter[source] = 0

        chunk.metadata["chunk_id"] = chunk_counter[source]
        chunk_counter[source] += 1

    return chunks

