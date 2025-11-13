import os
from dataclasses import dataclass

# Centralized configuration for the 6th implementation

@dataclass(frozen=True)
class ChromaConfig:
    collection_name: str = os.environ.get("CHROMA_COLLECTION", "rag_collection")
    db_path: str = os.environ.get("CHROMA_DB_PATH", "./my_chroma_db_client")
    api_key: str | None = os.environ.get("CHROMA_API_KEY")
    tenant: str | None = os.environ.get("CHROMA_TENANT")
    database: str | None = os.environ.get("CHROMA_DATABASE")


@dataclass(frozen=True)
class MongoConfig:
    uri: str = os.environ.get("MONGO_CONNECTION_STRING", "")
    db_name: str = os.environ.get("MONGO_DB_NAME", "EY_HACKATHON")
    raw_files_bucket: str = os.environ.get("MONGO_RAW_FILES_BUCKET", "raw_files")
    chunks_collection: str = os.environ.get("MONGO_CHUNKS_COLLECTION", "file_chunks")
    chat_db_name: str = os.environ.get("MONGO_CHAT_DB_NAME", "EY_HACKATHON")
    chat_collection: str = os.environ.get("MONGO_CHAT_COLLECTION", "RAG")
    file_notes_collection: str = os.environ.get("MONGO_FILE_NOTES_COLLECTION", "file_notes")


@dataclass(frozen=True)
class ModelConfig:
    hf_token: str | None = os.environ.get("HF_TOKEN")
    model_name: str = os.environ.get("HF_CHAT_MODEL", "inclusionAI/Ling-1T")
    embedding_model_name: str = os.environ.get("EMBEDDING_MODEL", "hkunlp/instructor-large")


chroma_cfg = ChromaConfig()
mongo_cfg = MongoConfig()
model_cfg = ModelConfig()
