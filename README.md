Agentic RAG: Optimized ChromaDB 

Overview

This project provides an interactive, menu-driven interface for managing ChromaDB collections and documents, with a focus on fast and accurate PDF ingestion and semantic search. The core is the optimized `chromadb_manager.py`, which enables efficient document chunking, embedding, and retrieval for Retrieval-Augmented Generation (RAG) workflows.

##Key Features

- **Fast PDF Ingestion:**  
  - Parallelized PDF page extraction using PyMuPDF (fitz) and Python's `concurrent.futures`.
  - Efficient file hashing to avoid redundant embeddings.
- **Accurate Chunking:**  
  - Sentence-based chunking using a custom regex splitter (no NLTK dependency).
  - Smart overlap between chunks for better context retention.
- **Metadata Enrichment:**  
  - Each chunk is stored with metadata including file hash, page numbers, chunk size, and source.
- **Deduplication:**  
  - Uses file hash to prevent duplicate document ingestion, even if the file path changes.
- **Batch Embedding Ready:**  
  - Utility for batch embedding chunks (for use with APIs like OpenAI).
- **Interactive CLI:**  
  - List, create/select, add documents, query, view stats, and delete collections from a simple menu.

## Optimizations Explained

### 1. **Speed**
- **Parallel PDF Extraction:**  
  Pages are extracted in parallel, making ingestion of large PDFs much faster.
- **Batch Embedding (Optional):**  
  Chunks can be embedded in batches for faster API calls (if enabled).

### 2. **Accuracy**
- **Regex Sentence Chunking:**  
  Replaces NLTK with a lightweight regex-based splitter, avoiding SSL/certificate issues and external dependencies.  
  Chunks are created at sentence boundaries, improving semantic search quality.
- **Smart Overlap:**  
  Overlapping sentences between chunks ensures context is preserved for downstream retrieval.

### 3. **Robustness**
- **File Hash Deduplication:**  
  Prevents re-embedding the same document, even if the filename changes.
- **Rich Metadata:**  
  Each chunk is tagged with source, page range, and chunk info for better traceability and filtering.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Agentic-RAG.git
   cd Agentic-RAG
   ```

2. **Create and activate a Python environment:**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r agentic-kahoot-demo/requirements.txt
   ```

4. **Set up your `.env` file:**
   - Create a `.env` file in `agentic-kahoot-demo/` with your API keys:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

5. **Run the ChromaDB Manager:**
   ```bash
   cd agentic-kahoot-demo
   python chromadb_manager.py
   ```

## Usage

- **List collections**
- **Create/select a collection**
- **Add a document (PDF)**
- **Query a collection**
- **View collection stats**
- **Delete a collection**

All actions are performed via the interactive menu.

## Troubleshooting
- If you see warnings about submodules when pushing to git, ensure you remove any `.git` directories inside subfolders you want to track as regular files.
- No NLTK or SSL issues: The project uses a regex-based sentence splitter for chunking.

## License
MIT

---

Let me know if you want to add more sections (e.g., Gradio deployment, API usage, or contribution guidelines)!
