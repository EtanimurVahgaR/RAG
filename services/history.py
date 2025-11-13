from __future__ import annotations
from typing import Tuple, List, Optional
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from pymongo import MongoClient
import json

from config import mongo_cfg


def load_chat_history(session_id: str, mongo_uri: str) -> Tuple[Optional[MongoDBChatMessageHistory], List, Optional[MongoClient]]:
    """
    Returns (chat_history_adapter, loaded_messages, client) where loaded_messages
    is a list of BaseMessage. It also supports a legacy document shape with
    fields { SessionId: <id>, History: "<json>" }.
    """
    chat_history = None
    loaded = []
    client = None

    if not mongo_uri:
        return None, [], None

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        client.server_info()
        chat_history = MongoDBChatMessageHistory(
            connection_string=mongo_uri,
            session_id=session_id,
            database_name=mongo_cfg.chat_db_name,
            collection_name=mongo_cfg.chat_collection,
        )
        loaded = list(chat_history.messages)
        if not loaded:
            # Try legacy schema fallback
            try:
                legacy_coll = client[mongo_cfg.chat_db_name][mongo_cfg.chat_collection]
                legacy_doc = legacy_coll.find_one({"SessionId": session_id})
                if legacy_doc and isinstance(legacy_doc.get("History"), str):
                    raw = json.loads(legacy_doc["History"]) or []
                    for item in raw:
                        typ = item.get("type")
                        data = item.get("data", {})
                        content = data.get("content", "")
                        if typ == "human":
                            loaded.append(HumanMessage(content=content))
                        elif typ == "ai":
                            loaded.append(AIMessage(content=content))
            except Exception:
                pass
    except Exception:
        return None, [], None

    return chat_history, loaded, client
