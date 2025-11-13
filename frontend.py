import streamlit as st
from io import BytesIO
from pathlib import Path
import tempfile
import os
import sys
from pathlib import Path as _Path
from dotenv import load_dotenv, find_dotenv

# Ensure local packages are importable when running via Streamlit
_BASE = _Path(__file__).resolve().parent
if str(_BASE) not in sys.path:
	sys.path.append(str(_BASE))

# Load environment variables BEFORE importing modules that read env at import time
_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
	load_dotenv(_dotenv_path, override=False)
else:
	# Fallback to default search in current working dir
	load_dotenv()

# Local modules (these import config which reads from os.environ)
from services.chunking import split_text
from services.rag import (
	get_vector_store,
	get_llm_runnable,
	build_simple_rag,
	build_conversational_rag,
	index_text_chunks,
)
from services.ingestion import extract_text, read_pdf, read_docx, read_txt
from data_access.mongo import save_raw_file, save_chunks
from services.history import load_chat_history
from services.estimation import estimate_k_ai
from config import model_cfg
from langchain_mongodb import MongoDBChatMessageHistory
from pymongo import MongoClient
from config import mongo_cfg
from langchain_core.messages import HumanMessage, AIMessage
import json

try:
	from PyPDF2 import PdfReader
except Exception:
	PdfReader = None

try:
	import docx
except Exception:
	docx = None

# Optional improved PDF extraction libraries. We will try these in order to
# preserve as much of the original PDF layout as possible (line breaks,
# columns, etc). If not installed we gracefully fall back to PyPDF2.
try:
	import pdfplumber
except Exception:
	pdfplumber = None

try:
	import fitz  # PyMuPDF
except Exception:
	fitz = None

load_dotenv()

st.set_page_config(page_title="RAG App (6th Implementation)", layout="wide")

st.title("RAG App — Upload, Store, Chunk, Ask")
st.write("Upload a document (PDF, TXT, DOCX, or image), preview it, store it in MongoDB, chunk and index it into Chroma, then ask questions.")

with st.sidebar:
	st.subheader("Model")
	st.caption(f"HF model: {model_cfg.model_name}")
	st.caption("Embeddings: hkunlp/instructor-large")
	st.markdown("---")
	st.subheader("Retrieval (Auto k)")
	auto_k = st.toggle("Auto-select k (AI)", value=True)
	min_k = st.number_input("Min k", min_value=1, max_value=100, value=3)
	max_k = st.number_input("Max k", min_value=1, max_value=200, value=40)
	default_k = st.number_input("Default k", min_value=1, max_value=100, value=7)
	st.caption("The app will infer how many chunks to retrieve based on the question. Clamp via min/max.")
	st.markdown("---")
	st.subheader("Chat Session")
	default_session = st.session_state.get("chat_session_id", "streamlit_session")
	chat_session_id = st.text_input("Session ID", value=default_session, key="chat_session_id")
	st.caption("Messages will be loaded from and saved to MongoDB in this session.")
	st.markdown("---")
	st.subheader("Chunking")
	chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
	chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)
	st.caption("Tip: larger chunks and smaller overlap reduce the number of embeddings -> faster indexing")
	st.markdown("---")
	st.subheader("Mode")
	skip_to_chat = st.toggle("Skip upload and go to Chat", value=False, help="Use existing indexed data; don't upload or store new data.")
	st.markdown("---")
	st.subheader("Steps")
	st.write("1) Upload file")
	st.write("2) Preview")
	st.write("3) Save to MongoDB")
	st.write("4) Chunk + Index")
	st.write("5) Ask questions")

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "docx", "png", "jpg", "jpeg"])

def read_pdf(file_bytes: bytes) -> str:
	if PdfReader is None:
		# If PyPDF2 isn't present we may still be able to use pdfplumber or fitz
		if pdfplumber is None and fitz is None:
			return "No PDF libraries (pdfplumber/PyMuPDF/PyPDF2) installed; cannot preview PDF text."
		# otherwise continue and rely on other available libraries
	# 1) Try pdfplumber (best at preserving layout/newlines)
	if pdfplumber is not None:
		try:
			texts = []
			with pdfplumber.open(BytesIO(file_bytes)) as pdf:
				for page in pdf.pages:
					# extract_text preserves a lot of layout; fallback to extract_words if empty
					text = page.extract_text() or ""
					texts.append(text)
			return "\n\n".join(texts).strip()
		except Exception:
			# fall through to next method
			pass

	# 2) Try PyMuPDF (fitz) which can preserve line breaks and has good control
 
	if fitz is not None:
		try:
			doc = fitz.open(stream=file_bytes, filetype="pdf")
			texts = []
			for page in doc:
				# use get_text with "text" option to keep line breaks
				texts.append(page.get_text("text") or "")
			return "\n\n".join(texts).strip()
		except Exception:
			pass

	# 3) Fallback to PyPDF2 if available
	if PdfReader is not None:
		try:
			reader = PdfReader(BytesIO(file_bytes))
			texts = []
			for page in reader.pages:
				try:
					texts.append(page.extract_text() or "")
				except Exception:
					texts.append("")
			return "\n\n".join(texts).strip()
		except Exception as e:
			return f"Error reading PDF: {e}"

	# If we reached here no library succeeded
	return "Unable to extract text from PDF with available libraries."

def read_docx(file_bytes: bytes) -> str:
	if docx is None:
		return "python-docx not installed; cannot preview DOCX text."
	try:
		with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
			tmp.write(file_bytes)
			tmp_path = tmp.name
		document = docx.Document(tmp_path)
		paragraphs = [p.text for p in document.paragraphs]
		try:
			os.remove(tmp_path)
		except Exception:
			pass
		return "\n\n".join(paragraphs).strip()
	except Exception as e:
		return f"Error reading DOCX: {e}"

def read_txt(file_bytes: bytes) -> str:
	try:
		return file_bytes.decode("utf-8", errors="replace")
	except Exception as e:
		return f"Error reading text file: {e}"

if uploaded_file is not None:
	file_details = {
		"filename": uploaded_file.name,
		"type": uploaded_file.type,
		"size": uploaded_file.size,
	}
	st.sidebar.subheader("File details")
	st.sidebar.write(file_details)

	file_bytes = uploaded_file.read()

	ext = Path(uploaded_file.name).suffix.lower()

	if ext == ".pdf":
		st.subheader("PDF Preview")
		text = read_pdf(file_bytes)
		if text:
			st.text_area("Extracted text", value=text, height=400)
		else:
			st.info("No extractable text found in PDF. You can still download the file below.")
		st.download_button("Download PDF", data=file_bytes, file_name=uploaded_file.name)

	elif ext == ".docx":
		st.subheader("DOCX Preview")
		text = read_docx(file_bytes)
		st.text_area("Extracted text", value=text, height=400)
		st.download_button("Download DOCX", data=file_bytes, file_name=uploaded_file.name)

	elif ext == ".txt":
		st.subheader("Text Preview")
		text = read_txt(file_bytes)
		st.text_area("Contents", value=text, height=400)
		st.download_button("Download TXT", data=file_bytes, file_name=uploaded_file.name)

	elif ext in [".png", ".jpg", ".jpeg"]:
		st.subheader("Image Preview")
		st.image(file_bytes)
		st.download_button("Download Image", data=file_bytes, file_name=uploaded_file.name)

	else:
		st.warning("Unsupported file type for inline preview. You can still download the uploaded file.")
		st.download_button("Download File", data=file_bytes, file_name=uploaded_file.name)

	# Optional: show a simple summary (size, number of characters)
	try:
		char_count = len(file_bytes)
		st.sidebar.markdown(f"**Byte size:** {char_count}")
	except Exception:
		pass

	# TODO: Integrate with vector store / indexing pipeline (index.py) if desired.

	# Save button row
	st.markdown("---")
	col1, col2, col3 = st.columns(3)
	with col1:
		do_save = st.button("Save to MongoDB")
	with col2:
		do_chunk = st.button("Chunk + Index to Chroma")
	with col3:
		st.write("")

	# States for saving results
	if "last_saved_file_id" not in st.session_state:
		st.session_state.last_saved_file_id = None
	if "last_chunks_count" not in st.session_state:
		st.session_state.last_chunks_count = 0

	if do_save:
		try:
			stored = save_raw_file(filename=uploaded_file.name, content=file_bytes, content_type=uploaded_file.type)
			st.session_state.last_saved_file_id = str(stored.id)
			st.success(f"Saved file to MongoDB GridFS with id: {stored.id}")
		except Exception as e:
			st.error(f"Failed to save file: {e}")

	if do_chunk:
		# Derive text for chunking where possible
		if ext in [".pdf", ".docx", ".txt"]:
			# Use preview text if present, else re-extract
			if ext == ".pdf":
				text = read_pdf(file_bytes)
			elif ext == ".docx":
				text = read_docx(file_bytes)
			else:
				text = read_txt(file_bytes)
			chunks = split_text(text, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
		else:
			# For images or other types, store raw bytes as a single chunk placeholder
			chunks = [f"[Binary file {uploaded_file.name} of {len(file_bytes)} bytes]"]
		try:
			# Save chunks metadata/text to MongoDB
			from bson import ObjectId
			file_id = st.session_state.last_saved_file_id
			oid = ObjectId(file_id) if file_id else None
			inserted_ids = save_chunks(
				file_id=oid,
				filename=uploaded_file.name,
				chunks=chunks,
				extra_metadata={"content_type": uploaded_file.type},
			)
			# Index into Chroma for retrieval (batch for speed and supply metadata)
			metadatas = []
			ids = []
			for idx, _ in enumerate(chunks):
				metadatas.append({
					"filename": uploaded_file.name,
					"content_type": uploaded_file.type,
					"chunk_index": idx,
					"file_id": file_id or "",
				})
				# Use a deterministic id if we have a file_id, else leave None and let Chroma assign
				ids.append(f"{file_id}_{idx}" if file_id else None)

			with st.spinner("Indexing into Chroma (this may take a while on first run due to model download)..."):
				# Filter out None IDs if some are None
				try:
					index_text_chunks(chunks, metadatas=metadatas, ids=ids, batch_size=128)
				except Exception:
					# Retry without custom ids if duplicates cause issues
					index_text_chunks(chunks, metadatas=metadatas, ids=None, batch_size=128)
			st.session_state.last_chunks_count = len(chunks)
			st.success(f"Chunked ({len(chunks)}) and indexed. Mongo inserted: {len(inserted_ids)}")
		except Exception as e:
			st.error(f"Failed to chunk/index: {e}")

	# Chat is handled outside this upload block now

else:
	if not skip_to_chat:
		st.info("No file uploaded yet. Use the uploader above to select a document, or toggle 'Skip upload and go to Chat' in the sidebar.")

# ---- Chat Section (available when file uploaded OR user chooses to skip) ----
if (uploaded_file is not None) or ("skip_to_chat" in locals() and skip_to_chat):
	st.markdown("---")
	st.subheader("Chat with your data")
	if 'skip_to_chat' in locals() and skip_to_chat and uploaded_file is None:
		st.caption("Chatting with previously indexed data. No new content will be uploaded or stored.")

	# Prepare chat history backed by MongoDB
	chat_history = None
	mongo_uri = os.environ.get("MONGO_CONNECTION_STRING")
	if not mongo_uri:
		st.warning("MONGO_CONNECTION_STRING is not set; chat history will be disabled.")
	else:
		try:
			# Ensure DB connectivity first (avoid chat history crash on init)
			_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
			_client.server_info()
			chat_history = MongoDBChatMessageHistory(
				connection_string=mongo_uri,
				session_id=chat_session_id,
				database_name=mongo_cfg.chat_db_name,
				collection_name=mongo_cfg.chat_collection,
			)
		except Exception as e:
			st.warning(f"MongoDB not reachable for chat history: {e}")

	vector_store = get_vector_store()
	llm = get_llm_runnable()

	# Display previous messages
	loaded_history_msgs = []
	if chat_history:
		loaded_history_msgs = list(chat_history.messages)
		# Fallback loader for legacy schema: { SessionId: <id>, History: "<json>" }
		if not loaded_history_msgs:
			try:
				legacy_coll = _client[mongo_cfg.chat_db_name][mongo_cfg.chat_collection]
				legacy_doc = legacy_coll.find_one({"SessionId": chat_session_id})
				if legacy_doc and isinstance(legacy_doc.get("History"), str):
					raw = json.loads(legacy_doc["History"]) or []
					for item in raw:
						typ = item.get("type")
						data = item.get("data", {})
						content = data.get("content", "")
						if typ == "human":
							loaded_history_msgs.append(HumanMessage(content=content))
						elif typ == "ai":
							loaded_history_msgs.append(AIMessage(content=content))
			except Exception:
				pass

	if loaded_history_msgs:
		for msg in loaded_history_msgs:
			role = "user" if msg.type == "human" else "assistant"
			with st.chat_message(role):
				st.markdown(msg.content)

	# Chat input
	user_msg = st.chat_input("Ask a question about the uploaded/indexed data...")
	if user_msg:
		# Echo user message in UI
		with st.chat_message("user"):
			st.markdown(user_msg)

		# Build inputs for chain
		inputs = {"input": user_msg}
		if loaded_history_msgs:
			inputs["chat_history"] = loaded_history_msgs

		# Generate AI reply with informed dynamic k
		try:
			# Choose k using the new multi-step AI estimator
			chosen_k = int(default_k)
			if auto_k:
				try:
					chosen_k = int(
						estimate_k_ai(
							user_query=user_msg,
							min_k=int(min_k),
							max_k=int(max_k),
							default_k=int(default_k),
							vector_store=vector_store,
						)
					)
				except Exception:
					chosen_k = int(default_k)

			retriever = vector_store.as_retriever(search_kwargs={"k": int(chosen_k)})
			conv_chain = build_conversational_rag(retriever, llm)

			with st.chat_message("assistant"):
				with st.spinner(f"Thinking... (k={chosen_k})"):
					answer = conv_chain.invoke(inputs)
				st.markdown(answer)
			if chat_history:
				chat_history.add_user_message(user_msg)
				chat_history.add_ai_message(answer)
		except Exception as e:
			st.error(f"Failed to run conversational RAG: {e}")

