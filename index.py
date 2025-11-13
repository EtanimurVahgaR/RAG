import os
import chromadb  # <-- NEW IMPORT
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# --- Define the path for your persistent client database ---
CHROMA_DB_PATH = "./my_chroma_db_client"
COLLECTION_NAME = "rag_collection"

def create_and_persist_index():
    """
    One-time script to create and save the vector store using a Chroma client.
    """
    print("--- [Starting Indexing] ---")
    
    # 1. Define your documents
    documents = [
        "The hippocampus is a major component of the brains of humans and other vertebrates. It has a distinctive, curved shape that has been likened to the sea-horse.",
        "Humans and other mammals have two hippocampi, one in each side of the brain. The hippocampus is part of the limbic system, and plays important roles in the consolidation of information from short-term memory to long-term memory, and in spatial navigation.",
        "The amygdala is an almond-shaped set of neurons located deep in the brain's medial temporal lobe. It plays a primary role in the processing of memory, decision-making, and emotional responses (including fear, anxiety, and aggression).",
        "The cerebellum is located at the back of the brain and is crucial for coordinating voluntary movements such as posture, balance, coordination, and speech."
    ]
    documents_text = "\n\n".join(documents)
    print("Loaded documents.")
    
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(documents_text)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large"
    )
    print("Loaded embedding model.")

    # 4. Indexing and Persisting (UPDATED)
    # This client persists data to disk at the specified path
    # Initialize a Chroma client. Prefer CloudClient when environment
    # variables are present, otherwise fall back to a local PersistentClient.
    # This script now requires using Chroma Cloud. Attempt to initialize
    # a CloudClient and surface any errors explicitly so the failure is easy
    # to debug when cloud credentials/configuration are incorrect.
    chroma_api_key = os.environ.get("CHROMA_API_KEY")
    chroma_tenant = os.environ.get("CHROMA_TENANT")
    chroma_database = os.environ.get("CHROMA_DATABASE")

    if not chroma_api_key:
        raise RuntimeError(
            "CHROMA_API_KEY is required for this script. Remove the local "
            "fallback and provide a valid Chroma Cloud API key in the "
            "CHROMA_API_KEY environment variable."
        )

    try:
        print("Initializing Chroma Cloud client...")
        client = chromadb.CloudClient(
            api_key=chroma_api_key,
            tenant=chroma_tenant,
            database=chroma_database,
        )
    except Exception as e:
        # Print a helpful message and the original exception to make the
        # cloud failure obvious to the user, then re-raise so callers can
        # handle or exit with a clear traceback.
        print("Failed to initialize Chroma Cloud client. Please check your\n"
              "CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE settings.")
        print(f"Error: {e!r}")
        raise
    
    print(f"Creating/loading vector store in collection: '{COLLECTION_NAME}'...")
    # This will create the collection if it doesn't exist and add the documents.
    vector_store = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        collection_name=COLLECTION_NAME,  # <-- Specify collection
        client=client                     # <-- Pass the client
        # 'persist_directory' is no longer used here
    )
    
    print("--- [Indexing Complete] ---")
    print(f"Vector store client database saved in '{CHROMA_DB_PATH}'")

if __name__ == "__main__":
    # This will create a new directory `my_chroma_db_client`
    # containing your vector index, managed by the client.
    create_and_persist_index()
