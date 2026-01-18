import os
from typing import Tuple, List
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from rag.embeddings import get_embedding_model

VECTOR_PATH = "data/index"
TOP_K = 4                    
MAX_CONTEXT_CHARS = 3500

def load_faiss_index(vector_path: str = VECTOR_PATH) -> FAISS:
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"FAISS index not found at {vector_path}")
    return FAISS.load_local(
        vector_path,
        get_embedding_model(),
        allow_dangerous_deserialization=True
    )

def get_llm() -> OllamaLLM:
    return OllamaLLM(
        model="llama3.2:1b", #tinyllama
        temperature=0.1,
        top_p=0.9,
        base_url="http://127.0.0.1:11434"
    )


def safe_qa(
        query: str,
        db: FAISS,
        llm: OllamaLLM,
        chat_history: List[dict] | None = None
        ) -> Tuple[str, List[str]]:
    
    injection_phrases = [
    "ignore previous",
    "ignore all",
    "system prompt",
    "developer message",
    "you are now",
    "act as",
    "bypass",
    "follow these instructions",
    "override",
    ]

    if any(p in query.lower() for p in injection_phrases):
        return "I don't know based on the documents.", []
    
    results = db.similarity_search_with_score(query, k=TOP_K)

    if not results:
        return "I don't know based on the documents.", []

    MAX_DISTANCE = 0.78
    filtered = [(doc, score) for doc, score in results if score <= MAX_DISTANCE]

    if len(filtered) == 0:
        return "I don't know based on the documents.", []

    context_chunks = []
    citations = []

    for doc, score in filtered:
        meta = doc.metadata
        clean_text = doc.page_content.strip()

        blocked_doc_phrases = [
            "ignore previous instructions",
            "follow these instructions",
            "system prompt",
            "act as",
        ]

        if any(p in clean_text.lower() for p in blocked_doc_phrases):
            continue
        if clean_text not in context_chunks:
            context_chunks.append(clean_text)

        citations.append({
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", "N/A"),
            "chunk": meta.get("chunk_id", "N/A"),
            "score": float(score)
        })

    if not context_chunks:
            return "I don't know based on the documents.", []
    
    unique = {}
    for c in citations:
        key = (c["source"], c["page"], c["chunk"])  
        unique[key] = c

    citations = list(unique.values())

    context_text = "\n\n".join(context_chunks)
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated]"

    conversation_context = ""
    if chat_history:
        for turn in chat_history[-3:]:  
            conversation_context += (
                f"User: {turn['question']}\n"
                f"Assistant: {turn['answer']}\n"
            )

    prompt = f"""
You are a factual question-answering assistant.

Answer the question using ONLY the information provided.
Do NOT hallucinate or invent facts.
If the question asks for names, lists, or examples:
- Return a comma-separated list
- Do NOT explain
- Do NOT summarize
- Use exact terms from the Information

If the answer is not present, say exactly:
I don't know based on the documents.

Use conversation history only for context, not as a source.
Conversation history (for context only):
{conversation_context}

Information:
{context_text}

Question:
{query}

Answer format (MANDATORY):

Answer:
<one concise paragraph>
"""
    
    answer = llm.invoke(prompt).strip()

    if not answer or not context_chunks:
        return "I don't know based on the documents.", []

    if any(w in query.lower() for w in ["list", "name", "names", "examples"]):
        if len(answer.strip()) < 2:  
            return "I don't know based on the documents.", []

    def is_relevant_answer(question: str, answer: str) -> bool:
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())

        return len(q_words & a_words) >= 1 or len(answer.split()) >= 12
    
    if not is_relevant_answer(query, answer):
        return "I don't know based on the documents.", []

    if not answer or answer.strip() == "":
        return "I don't know based on the documents.", []

    if "I don't know based on the documents." not in answer and not context_chunks:
        return "I don't know based on the documents.", []

    return answer, citations

chat_history = []
def main():
    print("Loading FAISS index and models...")
    db = load_faiss_index()
    llm = get_llm()
    print("RAG Chatbot ready! Type your question or 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not query:
            print("Please enter a question.\n")
            continue

        answer, citations = safe_qa(query, db, llm, chat_history)
        print(f"\nAnswer: {answer}")
        print(f"Sources: {citations}\n")

        chat_history.append({
            "question": query,
            "answer": answer
        })

if __name__ == "__main__":
    main()
