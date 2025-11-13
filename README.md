# RAG App (6th Implementation)

This iteration adds a simple app to:

- Upload files (PDF, DOCX, TXT, images) with on-screen preview
- Save raw files to MongoDB GridFS
- Save file chunks/metadata to MongoDB
- Chunk and index text into Chroma (Cloud if configured, local otherwise)
- Run a simple RAG query against the indexed chunks

## Structure

- `frontend.py` — Streamlit UI: upload, preview, save, chunk+index, ask questions
- `services/chunking.py` — Reuses `RecursiveCharacterTextSplitter` with 1000/200
- `services/rag.py` — Vector store, embeddings, indexing, and simple RAG chain
- `data_access/mongo.py` — MongoDB client, GridFS, and chunk persistence
- `config.py` — Central configuration for Chroma, MongoDB, and model settings
- `index.py` — Existing one-time indexing script (unchanged)
- `main.py` — Existing conversational CLI flow (unchanged)

## Environment

Provide these in a `.env` file or environment variables:

- `MONGO_CONNECTION_STRING` — MongoDB connection string
- `MONGO_DB_NAME` — Database for files/chunks (default: EY_HACKATHON)
- `MONGO_RAW_FILES_BUCKET` — GridFS bucket (default: raw_files)
- `MONGO_CHUNKS_COLLECTION` — Chunks collection (default: file_chunks)
- `HF_TOKEN` — Hugging Face API token
- `HF_CHAT_MODEL` — Chat model (default: inclusionAI/Ling-1T)
- `EMBEDDING_MODEL` — Embeddings model (default: hkunlp/instructor-large)
- `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` — for Chroma Cloud
- `CHROMA_COLLECTION` — Collection name (default: rag_collection)

## How to run

1. Install requirements

2. Start the app

```powershell
streamlit run 1_naive_rag/6_Ai_implementation/frontend.py
```

Then follow the steps in the UI: Upload -> Preview -> Save -> Chunk+Index -> Ask.

## Notes

- Order of operations follows the 6th implementation: chunking settings and Chroma/LLM usage retained.
- For images or binary files, we store raw bytes and index a placeholder chunk describing the file.
- The CLI chat `main.py` remains available and unchanged.
