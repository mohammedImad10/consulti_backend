import asyncio
import os
import json
import re
import hashlib
import numpy as np
import pandas as pd
from pypdf import PdfReader
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openpyxl import load_workbook
from diskcache import Cache
from typing import List, Optional
import openai
from openai import OpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
import websockets

# Global DEBUG flag: set to True to show debug logs
DEBUG = True

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("Please set your OPENAI_API_KEY in the environment variables.")

# WebSocket server setup
user_responses = {}

async def websocket_handler(websocket, path):
    print("WebSocket connection established")
    try:
        while True:
            # Wait for a message from the frontend (e.g., user response)
            message = await websocket.recv()
            data = json.loads(message)
            if data["type"] == "response":
                # Store the user's response
                user_responses["default"] = data["message"]
    except websockets.ConnectionClosed:
        print("WebSocket connection closed")

async def start_websocket_server():
    async with websockets.serve(websocket_handler, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

# Function to send messages to the frontend
async def send_message(message, message_type):
    async with websockets.connect("ws://localhost:8765") as websocket:
        await websocket.send(json.dumps({"message": message, "type": message_type}))

# Replace print statements with WebSocket messages
def log_message(message, message_type="log"):
    asyncio.run(send_message(message, message_type))
    print(f"[{message_type.upper()}] {message}")

# Replace user input with WebSocket request
async def get_user_input(question):
    await send_message(question, "question")
    while "default" not in user_responses:
        await asyncio.sleep(1)  # Wait for the user to respond
    user_val = user_responses["default"]
    del user_responses["default"]  # Clear the response
    return user_val

# =================== PERSISTENT CACHE CLASS ===================
class PersistentCache:
    """A wrapper around diskcache for persistent caching with TTL."""
    def __init__(self, cache_dir: str = "./cache", ttl: int = 3600):
        self.cache = Cache(cache_dir)
        self.ttl = ttl

    def get(self, key: str):
        return self.cache.get(key)

    def set(self, key: str, value, ttl: Optional[int] = None):
        self.cache.set(key, value, expire=ttl or self.ttl)

    def clear(self):
        self.cache.clear()

    def __del__(self):
        self.cache.close()

# =================== AGENTIC RAG CLASS ===================
class AgenticRAG:
    def __init__(self, collection_name: str, is_client: bool):
        self.openai_key = openai_key
        self.client = OpenAI(api_key=self.openai_key)
        self.llm_model_name = os.getenv('LLM_MODEL', 'gpt-4o-mini')
        self.model = OpenAIModel(self.llm_model_name)
        # For global docs (yellow) use is_client=False; for customer docs (pink) use is_client=True.
        self.db_path = "./clients_db_storage" if is_client else "./chromadb_storage"
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=OpenAIEmbeddingFunction(api_key=openai_key)
        )
        self.system_prompt = (
            "You are an AI assistant that helps process documents by extracting content, generating embeddings, "
            "and retrieving relevant information. You may retrieve, generate, or refine responses iteratively based on the query."
        )
        self.knowledge_agent = Agent(self.model, system_prompt=self.system_prompt, retries=2)
        self.cache = PersistentCache(cache_dir="./cache", ttl=3600)
        self.response_cache = PersistentCache(cache_dir="./response_cache", ttl=3600)
        # Ingestion metadata to avoid reprocessing unchanged files.
        self.metadata_path = f"ingestion_metadata_{collection_name}.json"
        self.ingestion_metadata = self.load_ingestion_metadata()

    def load_ingestion_metadata(self) -> dict:
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                log_message(f"Loading metadata failed: {e}", "error")
        return {}

    def save_ingestion_metadata(self):
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.ingestion_metadata, f)
        except Exception as e:
            log_message(f"Saving metadata failed: {e}", "error")

    # ------------------ DOCUMENT INGESTION METHODS ------------------
    async def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        try:
            reader = PdfReader(pdf_path)
            text_chunks = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    chunks = [chunk.strip() for chunk in page_text.split("\n\n") if chunk.strip()]
                    text_chunks.extend(chunks)
            log_message(f"Extracted {len(text_chunks)} chunks from PDF: {pdf_path}", "log")
            return text_chunks
        except Exception as e:
            log_message(f"Failed to extract PDF {pdf_path}: {e}", "error")
            return []

    async def ingest_pdf(self, pdf_path: str) -> None:
        mod_time = os.path.getmtime(pdf_path)
        if pdf_path in self.ingestion_metadata and self.ingestion_metadata[pdf_path] >= mod_time:
            log_message(f"PDF unchanged: {pdf_path}", "log")
            return
        text_chunks = await self.extract_text_from_pdf(pdf_path)
        if not text_chunks:
            log_message(f"No chunks found in PDF: {pdf_path}", "warning")
            return
        for i, chunk in enumerate(text_chunks):
            self.collection.add(ids=[f"pdf_{os.path.basename(pdf_path)}_{i}"], documents=[chunk])
        self.ingestion_metadata[pdf_path] = mod_time
        self.save_ingestion_metadata()
        self.cache.clear()
        self.response_cache.clear()
        log_message(f"Ingested PDF: {pdf_path} with {len(text_chunks)} chunks.", "log")

    async def ingest_directory(self, directory: str) -> None:
        log_message(f"Ingesting from directory: {directory}", "log")
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                ext = filename.lower().split(".")[-1]
                if ext == "pdf":
                    await self.ingest_pdf(file_path)
                elif ext in ["xlsx", "xls"]:
                    await self.ingest_excel(file_path)
                elif ext == "txt":
                    await self.ingest_txt(file_path)
                else:
                    log_message(f"Skipped unsupported file: {filename}", "warning")
        log_message(f"Finished ingesting from directory: {directory}", "log")

    # ------------------ QUERY & RESPONSE METHODS ------------------
    async def query_data(self, context: RunContext, user_query: str, n_results: int = 5) -> list:
        try:
            query_hash = hashlib.md5(user_query.encode()).hexdigest()
            cached = self.cache.get(query_hash)
            if cached:
                if DEBUG:
                    log_message(f"Using cached chunks for query: {user_query}", "debug")
                return cached
            results = self.collection.query(query_texts=[user_query], n_results=n_results)
            retrieved = results["documents"][0] if results.get("documents") else []
            self.cache.set(query_hash, retrieved)
            if DEBUG:
                log_message(f"Found {len(retrieved)} chunks for query: {user_query}", "debug")
            return retrieved
        except Exception as e:
            log_message(f"Error retrieving data for '{user_query}': {e}", "error")
            return []

    async def generate_response(self, query: str, context_chunks: Optional[list], searching_web: bool = False) -> str:
        query_hash = hashlib.md5(query.encode()).hexdigest()
        context_text = "\n\n".join(context_chunks) if context_chunks else "No context provided."
        web_note = ""
        if not context_chunks and searching_web:
            web_note = "No local data found. Searching the web for typical values and references."
        prompt = (
            "You are an assistant for question-answering tasks. Use the following context to answer the question. "
            "If the context contains multiple numerical values for the same attribute (e.g., area values), sum them and provide the total. "
            "Then, on a new line, write FINAL_ANSWER: followed by the exact final value (or 'N/A').\n\n"
            f"{web_note}\n"
            f"Context:\n{context_text}\n\nQuestion:\n{query}"
        )
        if DEBUG:
            log_message(f"Prompt length: {len(prompt)} chars", "debug")
        response = self.client.chat.completions.create(
            model=self.llm_model_name,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": query}],
            temperature=0.3
        )
        answer = response.choices[0].message.content
        log_message(f"Response from model is: {answer}", "log")
        self.response_cache.set(query_hash, answer)
        return answer

    async def agentic_rag(self, query: str, is_client: bool, unique_id: Optional[str] = None, max_iterations: int = 1) -> str:
        combined_query = f"{unique_id}:{query}" if unique_id else query
        run_context = RunContext(deps={}, model=self.model, usage={}, prompt=self.system_prompt)

        # Query local knowledge base
        retrieved_chunks = await self.query_data(run_context, query, n_results=5)

        # Simulate searching the web if no local data & is pink cell
        searching_web = (not is_client)

        # Generate the final response
        response = await self.generate_response(combined_query, retrieved_chunks, searching_web=searching_web)
        if DEBUG:
            log_message(f"agentic_rag final response:\n{response}", "debug")
        return response

# =================== MAIN FUNCTION ===================
async def main():
    global_docs_dir = "global_docs"          # Global (book) knowledge
    customer_docs_dir = "customer_data_docs"   # Customer-specific documents

    log_message("Ingesting global docs...", "log")
    agentic_rag_global = AgenticRAG(collection_name="global_agent_knowledge", is_client=False)
    await agentic_rag_global.ingest_directory(global_docs_dir)
    log_message("Finished ingesting global docs.", "log")

    log_message("Ingesting customer docs...", "log")
    agentic_rag_customer = AgenticRAG(collection_name="customer_data_knowledge", is_client=True)
    await agentic_rag_customer.ingest_directory(customer_docs_dir)
    log_message("Finished ingesting customer docs.", "log")

    system_prompt = (
        "You are a financial analyst assistant. You only return the final answer or 'N/A'. "
        "If searching the web, include references. No extra commentary is needed."
    )
    llm_client = LLMClient(api_key=openai_key, system_prompt=system_prompt, model="gpt-4o-mini", temperature=0.3)

    processor = ExcelTemplateProcessor("financial_sheet.xlsx", llm_client, agentic_rag_customer, agentic_rag_global)
    # processor.fill_yellow_cells_from_docs(max_fields=6)
    await processor.fill_pink_cells_from_docs(max_fields=60)
    processor.save("processed_sheet.xlsx")
    log_message("Starting chat interface...", "log")
    chat_interface()

if __name__ == "__main__":
    asyncio.run(main())