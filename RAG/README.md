# RAG — Retrieval Augmented Generation

A from-scratch RAG pipeline built with LangChain, SentenceTransformers, and ChromaDB.

---

## What is RAG?

RAG = **Retrieve** relevant context from a knowledge base, then **Generate** an answer using an LLM grounded in that context. This project covers the full retrieval pipeline (load → chunk → embed → store → query).

---

## Project Structure

```
RAG/
├── data/
│   ├── text_files/        # .txt source documents
│   ├── pdf_files/         # .pdf source documents
│   └── chroma_db/         # ChromaDB persistent storage (auto-created)
├── notebook/
│   └── document.ipynb     # Step-by-step walkthrough of the pipeline
├── src/
│   └── __init__.py        # Modular pipeline classes (use this in code)
├── requirements.txt
└── main.py
```

---

## Pipeline Stages

```
Files → DocumentLoader → TextChunker → EmbeddingManager → ChromaVectorStore
                                                                  ↑
                                                         query(question)
                                                                  ↓
                                                         ranked context chunks
```

### 1. `DocumentLoader`
Loads `.txt` files (via `DirectoryLoader`) and `.pdf` files (via `PyPDFLoader`) into LangChain `Document` objects with metadata.

### 2. `TextChunker`
Splits documents into overlapping chunks using `RecursiveCharacterTextSplitter`.
- Default: `chunk_size=1000`, `chunk_overlap=200`
- Separators: `\n\n` → `\n` → ` ` → `""` (tries to keep paragraphs intact)

### 3. `EmbeddingManager`
Encodes text chunks into 384-dimensional vectors using the `all-MiniLM-L6-v2` SentenceTransformer model (fast, no API key needed).

### 4. `ChromaVectorStore`
Stores chunk embeddings in ChromaDB with cosine similarity space. Supports:
- **Persistent** storage (survives restarts) via `PersistentClient`
- **In-memory** storage for quick experiments via `Client()`
- Batched inserts and idempotent adds (skips if collection already populated)

### 5. `RAGPipeline`
Orchestrates all stages behind two methods:
- `ingest(text_dir, pdf_dir)` — load, chunk, embed, and store
- `query(question)` — embed question → retrieve top-k chunks → return context string

---

## Quick Start

```python
from src import RAGPipeline

pipeline = RAGPipeline(
    collection_name="my_docs",
    persist_path="data/chroma_db",   # omit for in-memory
)

# Index your documents (safe to call again — skips if already indexed)
pipeline.ingest(text_dir="data/text_files", pdf_dir="data/pdf_files")

# Retrieve relevant context for a question
context = pipeline.query("What is the highest rank in sumo?", n_results=3)

# context is a plain string — pass it to any LLM as part of your prompt
```

---

## Notebook Walkthrough (`notebook/document.ipynb`)

The notebook walks through each concept step by step:

| Section | What it covers |
|---|---|
| Data Ingestion | `Document` datastructure, `TextLoader`, `DirectoryLoader`, `PyPDFLoader` |
| Chunking | `RecursiveCharacterTextSplitter`, chunk size & overlap |
| `EmbeddingManager` | SentenceTransformer class, `encode()`, embedding matrix shape |
| Cosine Similarity | Manual sanity check before using ChromaDB |
| ChromaDB Setup | `PersistentClient`, collection with cosine space |
| Add to ChromaDB | Insert embedded docs with ids, texts, embeddings, metadata |
| Query ChromaDB | Embed a question → `collection.query()` → similarity scores |
| `rag_query` function | Reusable function wrapping embed → retrieve → format |
| Demo | Three sample questions run through the full pipeline |

---

## Dependencies

```
langchain
langchain-core
langchain-community
pypdf
pymupdf
chromadb
sentence-transformers
```

Install: `pip install -r requirements.txt`

The embedding model (`all-MiniLM-L6-v2`) is downloaded automatically from HuggingFace on first run. No API keys required for retrieval.

---

## Extending to Full RAG (adding generation)

The `query()` method returns a context string. To add LLM-based answer generation, pass it to any model:

```python
context = pipeline.query("What food do sumo wrestlers eat?")

prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: What food do sumo wrestlers eat?
Answer:"""

# Pass `prompt` to OpenAI, Anthropic, or any LLM of your choice.
```
