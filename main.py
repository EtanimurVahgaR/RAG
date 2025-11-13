import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# --- NEW IMPORTS for Memory ---
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# --- Helper Function for InferenceClient (UNCHANGED) ---
def invoke_chat_model(prompt_value, client, model_name):
    # ... (same as your original code) ...
    content = prompt_value.to_string()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
    )
    return completion.choices[0].message.content

# --- Define paths (must match index.py) ---
CHROMA_DB_PATH = "./my_chroma_db_client"
COLLECTION_NAME = "rag_collection"

# --- MongoDB Constants ---
# MONGO_DB_NAME = "rag_chat_db"
# MONGO_COLLECTION_NAME = "chat_history"
MONGO_DB_NAME = "EY_HACKATHON"
MONGO_COLLECTION_NAME = "RAG"

# --- Module 1: Indexing (UNCHANGED) ---
def module_indexing():
    """
    Loads the vector store from disk using the Chroma client.
    """
    print("--- [Indexing Module] ---")
    
    # 1. Load the embedding model (must be the same one)
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large"
    )
    print("Loaded embedding model.")
    
    # 2. Initialize the persistent client
    chroma_api_key = os.environ.get("CHROMA_API_KEY")
    chroma_tenant = os.environ.get("CHROMA_TENANT")
    chroma_database = os.environ.get("CHROMA_DATABASE")

    if chroma_api_key:
        print("Initializing Chroma Cloud client...")
        client = chromadb.CloudClient(
            api_key=chroma_api_key,
            tenant=chroma_tenant,
            database=chroma_database,
        )
    else:
        print(f"Loading Chroma client from '{CHROMA_DB_PATH}'... (local PersistentClient)")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # 3. Load the persistent vector store from the collection
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    
    print(f"Loaded vector store from collection: '{COLLECTION_NAME}'.")
    print("--- [Indexing Complete] ---")
    return vector_store

# --- Module 2: Retrieval (UNCHANGED) ---
def module_retrieval(vector_store, k=3):
    """
    Creates a retriever component from the vector store.
    """
    print("\n--- [Retrieval Module] ---")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    print("Created retriever.")
    return retriever

# --- NEW: Helper for formatting chat history ---
def format_chat_history(history):
    """Formats chat history into a simple string."""
    buffer = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            buffer.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            buffer.append(f"AI: {msg.content}")
    return "\n".join(buffer)

# --- Module 3: Generation (UPDATED FOR CONVERSATION) ---
def create_conversational_rag_chain(retriever, llm_runnable):
    """
    Creates the full *conversational* RAG chain.
    """
    print("\n--- [Generation Module] ---")

    # 1. Contextualizer Chain: Rephrases the user's question
    #    Takes (chat_history, input) -> standalone_question
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
            # Pass the user's 'input'
            "input": lambda x: x["input"],
            # Format the 'chat_history' list of messages into a string
            "chat_history": lambda x: format_chat_history(x["chat_history"])
        }
        | contextualize_q_prompt
        | llm_runnable
        | StrOutputParser()
    )

    # 2. RAG (Answerer) Chain: Answers the standalone question
    #    Takes (context, query) -> answer
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
        """
        Custom runnable.
        1. Checks for history. If history, runs contextualizer_chain.
        2. If no history, uses the original input as the query.
        3. Retrieves documents based on the (potentially new) query.
        4. Returns a dict with "context" (docs) and "query" (standalone_question).
        """
        if input_dict.get("chat_history"):
            # If history exists, create a standalone question
            standalone_question = contextualizer_chain.invoke(input_dict)
            print(f"[Debug] Standalone Question: {standalone_question}")
        else:
            # If no history, use the original input
            standalone_question = input_dict["input"]
            print(f"[Debug] Standalone Question (original): {standalone_question}")
        
        # Retrieve documents
        docs = retriever.invoke(standalone_question)
        # Return the context and query for the final QA chain
        return {"context": docs, "query": standalone_question}

    # 3. Full Conversational Chain
    #    The input to this chain will be a dict: {"input": str, "chat_history": list}
    full_rag_chain = (
        # This custom lambda runs first, handling history and retrieval
        RunnableLambda(contextualized_retrieval)
        # The output {"context": ..., "query": ...} is passed to the qa_prompt
        | qa_prompt
        | llm_runnable
        | StrOutputParser()
    )
    
    print("Created conversational LCEL chain with memory.")
    print("--- [Generation Complete] ---")
    return full_rag_chain

# --- Main RAG Pipeline (UPDATED to an interactive chat loop) ---
def run_chat_session():
    """
    Executes an interactive chat session with stateful RAG.
    """
    
    # --- 1. Load MongoDB Config ---
    MONGO_CONN_STR = os.environ.get("MONGO_CONNECTION_STRING")
    if not MONGO_CONN_STR:
        raise ValueError("MONGO_CONNECTION_STRING not set in .env file")
    
    # --- 2. Define 'Frozen LLM' ---
    model_name = "inclusionAI/Ling-1T"
    client = InferenceClient(
        api_key=os.environ["HF_TOKEN"],
    )
    # Bind the LLM settings into a runnable
    llm_runnable = RunnableLambda(invoke_chat_model).bind(
        client=client, 
        model_name=model_name
    )

    # --- 3. Setup RAG Components ---
    vector_store = module_indexing()
    retriever = module_retrieval(vector_store)
    
    # Create the new conversational chain
    rag_chain = create_conversational_rag_chain(retriever, llm_runnable)

    # --- 4. Setup Chat History ---
    # We'll use a fixed session_id for this example.
    # In a real app, this would be unique per user/session.
    SESSION_ID = "main_chat_session" 
    print(f"\n--- Starting Chat Session (ID: {SESSION_ID}) ---")
    print(f"Connecting to MongoDB (DB: {MONGO_DB_NAME}, Collection: {MONGO_COLLECTION_NAME})...")
    
    try:
        chat_history = MongoDBChatMessageHistory(
            connection_string=MONGO_CONN_STR,
            session_id=SESSION_ID,
            database_name=MONGO_DB_NAME,
            collection_name=MONGO_COLLECTION_NAME,
        )
        # Do NOT fetch previous messages for generation. We still connect to
        # MongoDB to allow storing new user/AI messages, but we won't retrieve
        # history to influence model outputs.
        print("Connected to MongoDB. Previous messages will not be fetched for response generation.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        print("Please ensure MongoDB is running and MONGO_CONNECTION_STRING is correct.")
        return

    print("Type 'exit' or 'quit' to end the chat.")
    print("=======================================================")

    # --- 5. Run Interactive Chat Loop ---
    while True:
        try:
            user_query = input("You: ")
            if user_query.lower() in ["exit", "quit"]:
                print("Ending chat session. Goodbye!")
                break
            
            if not user_query.strip():
                continue

            # Invoke chain WITHOUT retrieving previous messages. We only send
            # the current user input to the RAG chain; previous messages are
            # still saved to MongoDB after the response but are not used to
            # form the model's output.
            print("\nAI is thinking...")
            chain_input = {"input": user_query}
            answer = rag_chain.invoke(chain_input)
            
            # 3. Print answer
            print(f"\nAI: {answer}")
            
            # 4. Save the new messages to MongoDB
            chat_history.add_user_message(user_query)
            chat_history.add_ai_message(answer)
            print("-------------------------------------------------------")

        except KeyboardInterrupt:
            print("\nEnding chat session. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    # The main script now runs the interactive chat session
    run_chat_session()