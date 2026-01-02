# 🤖 Multi-Utility AI Chatbot

A **stateful, multi-threaded AI chatbot** built using **LangGraph**, **LLMs**, **RAG**, and **tool calling**, capable of answering questions from uploaded PDFs, performing web searches, and invoking external tools via **MCP servers** — all within an interactive **Streamlit UI**.

---

## 🚀 Features

- 📄 **PDF-Based RAG (Retrieval-Augmented Generation)**
  - Upload PDFs per chat thread
  - Ask contextual questions from documents using FAISS + HuggingFace embeddings

- 🧠 **Stateful Multi-Threaded Conversations**
  - Each chat thread maintains independent memory
  - Conversations are persisted using SQLite checkpointer

- 🔧 **Tool Calling & Agentic Behavior**
  - Web search via DuckDuckGo
  - Custom tools via **MCP (Model Context Protocol) servers**
  - Automatic tool routing using LangGraph

- 🌐 **Asynchronous & Scalable Backend**
  - Dedicated async event loop
  - Non-blocking tool execution
  - Streaming AI responses

- 💬 **Interactive Streamlit UI**
  - Chat-style interface
  - Thread switching
  - Live tool usage status
  - PDF upload per chat

---

## 🛠️ Tech Stack

- **LLM**: ChatGroq (LLaMA 3.3 70B)
- **Framework**: LangGraph, LangChain
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store**: FAISS
- **Tools**: DuckDuckGo Search, MCP Servers
- **Backend**: Async Python, SQLite
- **Frontend**: Streamlit
- **Persistence**: Async SQLite Checkpointer

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Atanu-Manna2003/Multi-Utility-AI-Chatbot.git
cd multi-utility-ai-chatbot

### Create Virtual Environment
python -m venv venv
venv\Scripts\Activate

### install dependencies
pip install -r requirements.txt

### Environment Variables setup
create a .env file and add these 

GROQ_API_KEY=api key here
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT='https://api.smith.langchain.com'
LANGSMITH_API_KEY=langsmith api key
LANGSMITH_PROJECT= your project name

### Run the Application
first run -> python mcp_server.py
then run -> streamlit run frontend_rag.py



