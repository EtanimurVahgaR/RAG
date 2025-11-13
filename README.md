# RAG App

This iteration adds a simple Retrieval-Augmented Generation (RAG) app to:

- Upload files (PDF, DOCX, TXT, images) with on-screen preview  
- Save raw files to MongoDB GridFS  
- Save file chunks and metadata to MongoDB  
- Chunk and index text into **Chroma** (Cloud if configured, local otherwise)  
- Run simple RAG queries against the indexed chunks  

---

## Structure

- `frontend.py` — Streamlit UI: upload, preview, save, chunk+index, ask questions  
- `services/chunking.py` — Uses `RecursiveCharacterTextSplitter` with chunk size 1000, overlap 200  
- `services/rag.py` — Handles vector store, embeddings, indexing, and simple RAG chain  
- `data_access/mongo.py` — MongoDB client, GridFS, and chunk persistence  
- `config.py` — Central configuration for Chroma, MongoDB, and model settings  
- `index.py` — One-time indexing script (unchanged)  
- `main.py` — Conversational CLI flow (unchanged)  

---

## Environment

Provide these in a `.env` file or via environment variables:

- `MONGO_CONNECTION_STRING` — MongoDB connection string  
- `MONGO_CONNECTION_NAME` — Name of the Mongo collection  
- `MONGO_DB_NAME` — Database for files/chunks *(default: EY_HACKATHON)*  
- `HF_TOKEN` — Hugging Face API token  
- `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` — For Chroma Cloud  

---

## How to Run

0. **Clone and navigate to the project directory:**
```bash
git clone https://github.com/EtanimurVahgaR/RAG.git
cd RAG
````

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Update environment variables as per**[Environment Variables](#environment)

3. **Start the backend:**

```bash
python main.py
```

4. **Start the frontend:**

```bash
streamlit run frontend.py
```

Then follow the steps in the UI:
**Upload → Preview → Save → Chunk+Index → Ask.**

