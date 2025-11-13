from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any, List
from bson import ObjectId
from pymongo import MongoClient
import gridfs
from datetime import datetime

from config import mongo_cfg


@dataclass
class StoredFile:
    id: ObjectId
    filename: str
    content_type: Optional[str]
    length: int


def get_client(uri: Optional[str] = None) -> MongoClient:
    uri = uri or mongo_cfg.uri
    if not uri:
        raise ValueError("MONGO_CONNECTION_STRING not set. Set it in environment or .env")
    return MongoClient(uri)


def get_db(client: MongoClient, name: Optional[str] = None):
    return client[name or mongo_cfg.db_name]


def get_gridfs(db) -> gridfs.GridFS:
    # Use GridFS bucket name from config
    return gridfs.GridFS(db, collection=mongo_cfg.raw_files_bucket)


def save_raw_file(filename: str, content: bytes, content_type: Optional[str] = None) -> StoredFile:
    client = get_client()
    db = get_db(client)
    fs = get_gridfs(db)
    file_id = fs.put(content, filename=filename, content_type=content_type)
    # Retrieve file metadata
    file_doc = fs.get(file_id)
    return StoredFile(
        id=file_id,
        filename=file_doc.filename,
        content_type=file_doc.content_type,
        length=file_doc.length,
    )


def fetch_raw_file(file_id: ObjectId) -> bytes:
    client = get_client()
    db = get_db(client)
    fs = get_gridfs(db)
    grid_out = fs.get(file_id)
    return grid_out.read()


def save_chunks(
    file_id: ObjectId,
    filename: str,
    chunks: Iterable[str],
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[ObjectId]:
    client = get_client()
    db = get_db(client)
    coll = db[mongo_cfg.chunks_collection]
    docs = []
    for idx, text in enumerate(chunks):
        doc = {
            "file_id": file_id,
            "filename": filename,
            "chunk_index": idx,
            "text": text,
        }
        if extra_metadata:
            doc.update(extra_metadata)
        docs.append(doc)
    if not docs:
        return []
    result = coll.insert_many(docs)
    return result.inserted_ids


def get_chunks_for_file(file_id: ObjectId) -> List[Dict[str, Any]]:
    client = get_client()
    db = get_db(client)
    coll = db[mongo_cfg.chunks_collection]
    return list(coll.find({"file_id": file_id}).sort("chunk_index", 1))


def get_chunk_count_for_file(file_id: ObjectId) -> int:
    client = get_client()
    db = get_db(client)
    coll = db[mongo_cfg.chunks_collection]
    return coll.count_documents({"file_id": file_id})


def get_file_note(file_id: ObjectId) -> Optional[Dict[str, Any]]:
    client = get_client()
    db = get_db(client)
    coll = db[mongo_cfg.file_notes_collection]
    return coll.find_one({"file_id": file_id})


def upsert_file_note(
    file_id: ObjectId,
    filename: Optional[str],
    note: str,
    suggested_k: Optional[int] = None,
    chunk_count: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    client = get_client()
    db = get_db(client)
    coll = db[mongo_cfg.file_notes_collection]
    doc: Dict[str, Any] = {
        "file_id": file_id,
        "filename": filename,
        "note": note,
        "suggested_k": suggested_k,
        "chunk_count": chunk_count,
        "updated_at": datetime.utcnow(),
    }
    if extra:
        doc.update(extra)
    coll.update_one({"file_id": file_id}, {"$set": doc}, upsert=True)
