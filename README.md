# 🤖 RAG Chatbot

**Retrieval-Augmented Generation (RAG) Chatbot** focused on **Healthcare Knowledge**.

This project is a question-answering chatbot that retrieves relevant information from a document database using **FAISS** and generates answers with **OpenAI’s LLM**. It is deployed using **Streamlit**.

**Live Demo:** [Open Chatbot](https://taulant-ragchatbot.streamlit.app/)

## Features

- Retrieval-Augmented Generation (RAG) using FAISS vector store.
- AI-powered answers using OpenAI GPT-4o-mini.
- Handles healthcare-related queries based on uploaded documents.
- Shows sources and citations for answers.
- Prevents prompt injection and hallucinatory responses.
- Simple, interactive web interface via Streamlit.

---

## Technologies Used

- Python 3.11
- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- PDF parsing: `pypdf`

---

## Setup / Installation 

1. **Clone the repository:**
```bash
git clone https://github.com/Taulant-TM/rag-chatbot.git
cd rag-chatbot
```

2. **Create a virtual env and activate it:**
```bash
python -m venv venv311
# Windows
venv311\Scripts\activate
# macOS/Linux
source venv311/bin/activate
```

3. **Install requirements:**
```bash
pip install -r requirements.txt
```

4. **Set your OpenAI API key:**
```bash
# Windows
setx OPENAI_API_KEY "your_openai_api_key_here"
# macOS/Linux
export OPENAI_API_KEY="your_openai_api_key_here"
```

5. **File structure & run order:**
- rag/ingestion.py – Processes documents and prepares embeddings.
- rag/embeddings.py – Embedding model for vectorization.
- rag/index.py – Creates the FAISS vector store.
- rag/qa.py – Core question-answering logic using FAISS and LLM.
- app/app.py – Streamlit frontend to interact with the chatbot.

6. **Run locally:**
```bash
python -m rag.ingestion 
python -m rag.index

streamlit run app/app.py
```

## Usage

1. Open the app in your browser.

2. Type a healthcare-related question in the input box.

3. Click Ask.

4. View the answer along with sources (if available) in the conversation history.


## Notes

Ensure the data/index folder (FAISS index) is present or the app will fail to load.

The chatbot only answers questions based on the uploaded documents.

Conversation history is maintained in the session for context but not used as a source.