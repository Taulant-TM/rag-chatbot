import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from rag.qa import safe_qa, load_faiss_index, get_llm

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.caption("Retrieval-Augmented Generation - Healthcare Knowledge Only")


@st.cache_resource(show_spinner="Loading vector database and LLM...")
def load_resources():
    db = load_faiss_index()
    llm = get_llm()
    return db, llm

db, llm = load_resources()

if "history" not in st.session_state:
    st.session_state.history = []


query = st.text_input(
    "Ask a question:",
    placeholder="e.g. What is RAG and why is it useful?"
)

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        try:
            answer, sources = safe_qa(
                query,
                db,
                llm,
                chat_history=st.session_state.history
            )

            st.session_state.history.append({
                "question": query,
                "answer": answer,
                "sources": sources
            })

        except Exception as e:
            st.error("Something went wrong while generating the answer.")
            st.exception(e)

st.divider()
st.subheader("Conversation")

for chat in reversed(st.session_state.history):
    st.markdown(f"**🧑 You:** {chat['question']}")
    st.markdown(f"**🤖 Answer:** {chat['answer']}")

    if chat["sources"]:
        with st.expander("📚 Sources"):
            for src in chat["sources"]:
                if isinstance(src, dict):
                    st.markdown(
                        f"- {src['source']} (page {src['page']}, chunk {src['chunk']}, score {src['score']:.2f})"
                    )
                else:
                    st.markdown(f"- {src}")

    st.divider()

