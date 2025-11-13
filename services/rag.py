from __future__ import annotations
import os
from typing import Iterable, Optional, List, Dict, Any
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from huggingface_hub import InferenceClient

from config import chroma_cfg, model_cfg


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=model_cfg.embedding_model_name)


def get_chroma_client():
    if chroma_cfg.api_key:
        return chromadb.CloudClient(
            api_key=chroma_cfg.api_key,
            tenant=chroma_cfg.tenant,
            database=chroma_cfg.database,
        )
    # Fallback to local persistent client for development
    return chromadb.PersistentClient(path=chroma_cfg.db_path)


def get_vector_store():
    client = get_chroma_client()
    embeddings = get_embeddings()
    return Chroma(
        client=client,
        collection_name=chroma_cfg.collection_name,
        embedding_function=embeddings,
    )


def index_text_chunks(
    chunks: Iterable[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
    batch_size: int = 128,
):
    """
    Index chunks into Chroma using add_texts in batches (faster and avoids re-creating the collection).
    Provide metadatas and ids when possible to help with deduplication and filtering later.
    """
    vs = get_vector_store()
    texts = list(chunks)
    total = len(texts)
    if metadatas is not None and len(metadatas) != total:
        raise ValueError("metadatas length must match number of chunks")
    if ids is not None and len(ids) != total:
        raise ValueError("ids length must match number of chunks")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_metas = metadatas[start:end] if metadatas is not None else None
        batch_ids = ids[start:end] if ids is not None else None
        vs.add_texts(texts=batch_texts, metadatas=batch_metas, ids=batch_ids)


def get_llm_runnable():
    client = InferenceClient(api_key=model_cfg.hf_token)
    model_name = model_cfg.model_name

    def invoke_chat_model(prompt_value, client, model_name):
        content = prompt_value.to_string()
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
        )
        return completion.choices[0].message.content

    return RunnableLambda(invoke_chat_model).bind(client=client, model_name=model_name)


def estimate_k_ai(user_query: str, min_k: int = 3, max_k: int = 40, default_k: int = 7) -> int:
    """
    Use a tiny LLM prompt to classify how broad the request is and return an integer k.
    The model should output ONLY a number. We clamp to [min_k, max_k]. If anything fails,
    we return default_k.
    """
    try:
        # Build a direct client to minimize any prompt-format overhead
        client = InferenceClient(api_key=model_cfg.hf_token)
        model_name = model_cfg.model_name
        prompt = (
            "You are a classifier. Given a user request, choose how many relevant chunks to retrieve "
            f"from a vector database. Respond with ONLY a single integer between {min_k} and {max_k}.\n\n"
            "Guidelines:\n"
            "- Focused or specific question -> smaller (3-5)\n"
            "- Moderate breadth -> medium (7-10)\n"
            "- Broad or 'all/across/for each' aggregation -> large (15-40)\n\n"
            f"User request: {user_query}\n"
            "Answer:"
        )
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = completion.choices[0].message.content.strip()
        # Extract first integer in the response
        import re
        m = re.search(r"\d+", raw)
        if not m:
            return default_k
        k = int(m.group(0))
        k = max(min_k, min(max_k, k))
        return k
    except Exception:
        return default_k


def build_simple_rag(retriever, llm_runnable):
    qa_template = """
    You are a helpful assistant. Use the following pieces of context to answer the user's question.
    If you don't know the answer from the context, just say that you don't know.

    Context:
    {context}

    Question:
    {query}

    Helpful Answer:
    """
    qa_prompt = PromptTemplate.from_template(qa_template)

    def retrieve(query: str):
        docs = retriever.invoke(query)
        return {"context": docs, "query": query}

    return RunnableLambda(lambda x: retrieve(x["input"])) | qa_prompt | llm_runnable | StrOutputParser()


def format_chat_history(history: list) -> str:
    """Formats chat history (LangChain messages) to a simple string for prompts."""
    buffer = []
    for msg in history or []:
        if isinstance(msg, HumanMessage):
            buffer.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            buffer.append(f"AI: {msg.content}")
    return "\n".join(buffer)


def build_conversational_rag(retriever, llm_runnable):
    """
    Conversational RAG chain that:
    1) Rewrites the user's question based on chat history (contextualization)
    2) Retrieves docs using the rewritten question
    3) Answers with the retrieved context
    Input format: {"input": str, "chat_history": list[BaseMessage]}
    """
    contextualize_q_prompt_template = """
    Given the following chat history and a new user question, rephrase the
    user question to be a standalone question that can be understood without
    the chat history. Do NOT answer the question, just reformulate it.

    Chat History:
    {chat_history}

    User Question:
    {input}

    Standalone Question:
    """
    contextualize_q_prompt = PromptTemplate.from_template(contextualize_q_prompt_template)

    contextualizer_chain = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: format_chat_history(x.get("chat_history", [])),
        }
        | contextualize_q_prompt
        | llm_runnable
        | StrOutputParser()
    )

    qa_template = """
    You are a helpful assistant. Use the following pieces of context to answer the user's question.
    If you don't know the answer from the context, just say that you don't know.

    Context:
    {context}

    Question:
    {query}

    Helpful Answer:
    """
    qa_prompt = PromptTemplate.from_template(qa_template)

    def contextualized_retrieval(input_dict):
        if input_dict.get("chat_history"):
            standalone = contextualizer_chain.invoke(input_dict)
        else:
            standalone = input_dict["input"]
        docs = retriever.invoke(standalone)
        return {"context": docs, "query": standalone}

    return RunnableLambda(contextualized_retrieval) | qa_prompt | llm_runnable | StrOutputParser()


# k estimation helpers have been moved to services.estimation for better modularity.
