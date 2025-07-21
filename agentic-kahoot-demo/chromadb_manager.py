"""
chromadb_manager.py

Docling + ChromaDB Knowledge Base Manager.

Environment Variables:
- OPENAI_API_KEY: API key for OpenAI embeddings (required for embedding functions)
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions

import fitz  # PyMuPDF
import concurrent.futures
import hashlib
import re

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Used for embedding function

class PDFChromaProcessor:
    """
    A processor that uses PyMuPDF (fitz) for PDF parsing and ChromaDB for storage.
    Handles document conversion, chunking, and embedding.
    """
    def __init__(self,
                 openai_api_key: str = None,
                 chroma_db_path: str = "./chroma_db",
                 embedding_model: str = "text-embedding-ada-002"):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.chroma_db_path = chroma_db_path
        self.embedding_model = embedding_model
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
        self.embedding_function = self._setup_embedding_function()

    def _setup_embedding_function(self):
        if not self.openai_api_key:
            print("❌ OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        try:
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name=self.embedding_model
            )
        except Exception as e:
            print(f"❌ Failed to initialize OpenAIEmbeddingFunction: {e}")
            raise

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of the file for deduplication."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def extract_text_with_fitz(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and metadata from a PDF using PyMuPDF (fitz), parallelizing page extraction.
        """
        try:
            doc = fitz.open(file_path)
            def get_page_text(page):
                return page.get_text()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                texts = list(executor.map(get_page_text, doc))
            text = "\n".join(texts)
            metadata = {
                "source": file_path,
                "title": Path(file_path).name,
                "page_count": doc.page_count,
                "format": "plain_text",
                "file_hash": self._compute_file_hash(file_path),
            }
            return {
                "content": text,
                "metadata": metadata,
                "page_texts": texts,
                "document_object": None
            }
        except Exception as e:
            print(f"❌ PDF parsing failed for '{file_path}': {e}. Please check the file format and try again.")
            return None

    def simple_sent_tokenize(self, text):
        """Split text into sentences using regex."""
        return [s.strip() for s in re.split(r'(?<=[.!?]) +(?=[A-Z])', text) if s.strip()]

    def smart_chunk_content(self, page_texts: list, chunk_size: int = 1000, overlap: int = 100) -> list:
        """
        Chunk by sentences, keeping track of page numbers. Returns list of (chunk, page_start, page_end).
        """
        chunks = []
        current_chunk = []
        current_len = 0
        page_start = 0
        for page_num, page_text in enumerate(page_texts):
            sentences = self.simple_sent_tokenize(page_text)
            for sent in sentences:
                if current_len + len(sent) > chunk_size and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append((chunk_text, page_start, page_num))
                    # Overlap by sentences
                    overlap_sents = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        overlap_len += len(s)
                        overlap_sents.insert(0, s)
                        if overlap_len >= overlap:
                            break
                    current_chunk = overlap_sents[:]
                    current_len = sum(len(s) for s in current_chunk)
                    page_start = page_num
                current_chunk.append(sent)
                current_len += len(sent)
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, page_start, page_num))
        return chunks

    def _batch_embed(self, texts: list, batch_size: int = 16) -> list:
        """
        Batch embed using OpenAI API for speed.
        """
        import openai
        openai.api_key = self.openai_api_key
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = openai.Embedding.create(input=batch, model=self.embedding_model)
            batch_embeds = [d['embedding'] for d in resp['data']]
            embeddings.extend(batch_embeds)
        return embeddings

    def create_or_get_collection(self, collection_name: str):
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        print(f"Collection '{collection_name}' is ready (created or retrieved).")
        return collection

    def embed_document(self, collection_name: str, file_path: str,
                      chunk_size: int = 1000, overlap: int = 100) -> bool:
        if not os.path.isfile(file_path):
            print(f"❌ File not found: {file_path}. Please provide a valid file path.")
            return False
        collection = self.create_or_get_collection(collection_name)
        file_hash = self._compute_file_hash(file_path)
        existing_docs = collection.get(where={"file_hash": file_hash})
        if existing_docs['ids']:
            print(f"ℹ️ Document with hash '{file_hash}' already exists in collection '{collection_name}'. Skipping embedding.")
            return False
        extracted_data = self.extract_text_with_fitz(file_path)
        if not extracted_data:
            print(f"❌ Failed to extract content from '{file_path}'. Ensure the file is a supported format and try again.")
            return False
        page_texts = extracted_data['page_texts']
        metadata = extracted_data['metadata']
        chunk_tuples = self.smart_chunk_content(page_texts, chunk_size, overlap)
        documents = [chunk for chunk, _, _ in chunk_tuples]
        ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunk_tuples))]
        metadatas = []
        for i, (chunk, page_start, page_end) in enumerate(chunk_tuples):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunk_tuples),
                "page_start": page_start,
                "page_end": page_end
            })
            metadatas.append(chunk_metadata)
        # Batch embed for speed (optional, can fallback to Chroma's embedding_function if needed)
        # embeddings = self._batch_embed(documents)  # Uncomment if you want to store embeddings manually
        try:
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            print(f"✅ Successfully embedded {len(chunk_tuples)} chunks from '{file_path}' into collection '{collection_name}'.")
            return True
        except Exception as e:
            print(f"❌ Failed to add document to collection '{collection_name}': {e}")
            return False

    def query_collection(self, collection_name: str, query: str,
                        n_results: int = 5, filter_dict: Dict = None) -> Dict:
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            query_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            if filter_dict:
                query_params["where"] = filter_dict
            results = collection.query(**query_params)
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i],
                        "id": results['ids'][0][i]
                    })
            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
        except Exception as e:
            print(f"Query failed: {e}")
            return {"query": query, "results": [], "total_results": 0}

    def get_collection_stats(self, collection_name: str) -> Dict:
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            all_docs = collection.get()
            stats = {
                "total_documents": len(all_docs['ids']),
                "unique_sources": len(set(
                    meta.get('source', 'unknown')
                    for meta in all_docs['metadatas']
                )),
                "document_types": self._count_document_types(all_docs['metadatas']),
                "avg_chunk_size": self._calculate_avg_chunk_size(all_docs['documents']),
                "collection_name": collection_name
            }
            return stats
        except Exception as e:
            print(f"Failed to get collection stats: {e}")
            return {}

    def _count_document_types(self, metadatas: List[Dict]) -> Dict:
        type_counts = {}
        for meta in metadatas:
            doc_type = meta.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        return type_counts

    def _calculate_avg_chunk_size(self, documents: List[str]) -> float:
        if not documents:
            return 0
        return sum(len(doc) for doc in documents) / len(documents)

class InteractiveKnowledgeBase:
    def __init__(self):
        self.processor = PDFChromaProcessor()
        self.current_collection = None # Initialize current_collection to None
    
    def run(self):
        """Run interactive knowledge base manager"""
        print("=== Docling + ChromaDB Knowledge Base Manager ===")
        
        while True:
            print("\nOptions:")
            print("1. List collections")
            print("2. Create/Select collection")
            print("3. Add document")
            print("4. Query collection")
            print("5. View collection stats")
            print("6. Delete collection")
            print("7. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == "1":
                self.list_collections()
            elif choice == "2":
                self.create_or_select_collection()
            elif choice == "3":
                self.add_document()
            elif choice == "4":
                self.query_collection()
            elif choice == "5":
                self.view_stats()
            elif choice == "6":
                self.delete_collection()
            elif choice == "7":
                print("Goodbye!")
                break
            else:
                print("Invalid option")
    
    def list_collections(self):
        """List all collections"""
        collections = self.processor.chroma_client.list_collections()
        if collections:
            print("\nAvailable collections:")
            for i, collection in enumerate(collections, 1):
                print(f"{i}. {collection.name}")
        else:
            print("No collections found")
    
    def create_or_select_collection(self):
        """Create or select a collection"""
        name = input("Enter collection name: ").strip()
        if name:
            try:
                # Try to get the collection, if it exists, select it
                self.processor.create_or_get_collection(name)
                self.current_collection = name
                print(f"Selected collection: {name}")
            except Exception as e:
                print(f"Error creating/selecting collection: {e}")
    
    def add_document(self):
        """Add a document to current collection"""
        if not self.current_collection:
            print("Please select a collection first")
            return
        
        file_path = input("Enter file path: ").strip()
        if os.path.isfile(file_path):
            success = self.processor.embed_document(
                collection_name=self.current_collection,
                file_path=file_path
            )
            if success:
                print("Document added successfully!")
            else:
                print("Failed to add document")
        else:
            print("File not found")
    
    def query_collection(self):
        """Query the current collection"""
        if not self.current_collection:
            print("Please select a collection first")
            return
        
        query = input("Enter your query: ").strip()
        if query:
            results = self.processor.query_collection(
                collection_name=self.current_collection,
                query=query,
                n_results=3
            )
            
            print(f"\nFound {results['total_results']} results:")
            for i, result in enumerate(results['results'], 1):
                print(f"\n--- Result {i} ---")
                print(f"Content: {result['content'][:300]}...")
                print(f"Source: {result['metadata'].get('source', 'unknown')}")
                print(f"Type: {result['metadata'].get('type', 'unknown')}")
                print(f"Similarity: {1 - result['distance']:.3f}")
    
    def view_stats(self):
        """View collection statistics"""
        if not self.current_collection:
            print("Please select a collection first")
            return
        
        stats = self.processor.get_collection_stats(self.current_collection)
        print(f"\n=== Collection Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    def delete_collection(self):
        """Delete a collection"""
        self.list_collections()
        name = input("Enter collection name to delete: ").strip()
        if name:
            confirm = input(f"Are you sure you want to delete '{name}'? (y/N): ")
            if confirm.lower() == 'y':
                try:
                    self.processor.chroma_client.delete_collection(name)
                    print(f"Collection '{name}' deleted")
                    if self.current_collection == name:
                        self.current_collection = None # Deselect if current collection is deleted
                except Exception as e:
                    print(f"Error deleting collection: {e}")

if __name__ == "__main__":
    manager = InteractiveKnowledgeBase()
    manager.run()
