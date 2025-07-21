import chromadb
from typing import List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import os
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

class RAGToolInput(BaseModel):
    query: str = Field(..., description="The query to search for relevant knowledge in the specified ChromaDB collection.")
    n_results: int = Field(3, description="Number of relevant documents to retrieve.")
    collection_name: str = Field(..., description="Name of the ChromaDB collection to query.")

class RAGTool(BaseTool):
    name: str = "ChromaDB RAG Tool"
    description: str = (
        "A Retrieval-Augmented Generation (RAG) tool for CrewAI. "
        "Queries a specified ChromaDB SQLite collection for relevant knowledge. "
        "Use the 'query' argument to search for information and 'collection_name' to specify the collection."
    )
    args_schema = RAGToolInput

    def __init__(self, collection_name: str, db_path: str = "chroma.sqlite3"):
        super().__init__()
        persist_dir = os.path.dirname(os.path.abspath(db_path)) or "."
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        # Ensure client and embedding_function are attached to the instance using object.__setattr__
        # to avoid potential issues with BaseTool's __setattr__ or Pydantic interference.
        object.__setattr__(self, "_chroma_client", chromadb.PersistentClient(path=persist_dir))
        
        # Use ChromaDB's built-in OpenAI embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-ada-002"
        )
        object.__setattr__(self, "_embedding_function", openai_ef)

        # Initialize collection_name and collection attribute
        object.__setattr__(self, "collection_name", collection_name)
        try:
            object.__setattr__(self, "_collection", self._chroma_client.get_collection(
                name=collection_name,
                embedding_function=self._embedding_function
            ))
        except Exception as e:
            print(f"Warning: Could not get collection '{collection_name}': {e}")
            object.__setattr__(self, "_collection", None)

    def _run(self, query: str, n_results: int = 3, collection_name: str = None) -> List[str]:
        """
        Query the ChromaDB collection for relevant documents.
        """
        collection_to_query = self._collection

        if not collection_to_query:
            return ["❌ Error: No collection selected for query. Please select a collection at the start of the session."]

        try:
            results = collection_to_query.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas']  # Ensure metadatas are included
            )
        except Exception as e:
            import traceback
            error_type = type(e).__name__
            tb = traceback.format_exc()
            return [
                f"❌ Error querying ChromaDB: {e}",
                f"Exception type: {error_type}",
                f"Traceback:\n{tb}"
            ]
        docs_with_metadata = []
        if results.get("documents") and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                doc_content = results["documents"][0][i]
                doc_metadata = results["metadatas"][0][i]
                # Assuming 'page' is in metadata, if not, adjust or handle gracefully
                page_info = doc_metadata.get('page', 'N/A')
                source_info = doc_metadata.get('source', 'unknown')
                docs_with_metadata.append(f"Content: {doc_content}\nSource: {source_info}\nPage: {page_info}")

        if not docs_with_metadata:
            return ["No relevant documents found."]
        return docs_with_metadata
