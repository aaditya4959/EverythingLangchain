"""
RAG Pipeline — modular end-to-end Retrieval Augmented Generation.

Stages:
    DocumentLoader   → load text / PDF files into LangChain Documents
    TextChunker      → split documents into overlapping chunks
    EmbeddingManager → encode chunks with SentenceTransformer
    ChromaVectorStore→ persist and query embeddings via ChromaDB
    RAGPipeline      → orchestrate all stages with a single interface
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import chromadb
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

class DocumentLoader:
    """Loads documents from text and/or PDF files into LangChain Documents."""

    def load_text_files(self, directory: str) -> List[Document]:
        loader = DirectoryLoader(
            directory,
            loader_cls=TextLoader,
            glob="*.txt",
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        docs = loader.load()
        print(f"[DocumentLoader] Loaded {len(docs)} text document(s) from '{directory}'.")
        return docs

    def load_pdf_files(self, directory: str) -> List[Document]:
        docs: List[Document] = []
        pdf_paths = list(Path(directory).glob("**/*.pdf"))
        print(f"[DocumentLoader] Found {len(pdf_paths)} PDF file(s) in '{directory}'.")
        for path in pdf_paths:
            pages = PyPDFLoader(str(path)).load()
            for page in pages:
                page.metadata["source_file"] = path.name
                page.metadata["file_type"] = "pdf"
            docs.extend(pages)
            print(f"  ↳ {path.name}: {len(pages)} page(s)")
        return docs

    def load(
        self,
        text_dir: Optional[str] = None,
        pdf_dir: Optional[str] = None,
    ) -> List[Document]:
        """Load from either or both source directories."""
        docs: List[Document] = []
        if text_dir:
            docs.extend(self.load_text_files(text_dir))
        if pdf_dir:
            docs.extend(self.load_pdf_files(pdf_dir))
        print(f"[DocumentLoader] Total documents loaded: {len(docs)}")
        return docs


# ---------------------------------------------------------------------------
# TextChunker
# ---------------------------------------------------------------------------

class TextChunker:
    """Splits LangChain Documents into smaller, overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        print(f"[TextChunker] {len(documents)} document(s) → {len(chunks)} chunk(s).")
        return chunks


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------

class EmbeddingManager:
    """Encodes text with a SentenceTransformer model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        print(
            f"[EmbeddingManager] Model '{model_name}' loaded "
            f"(dim={self.model.get_embedding_dimension()})."
        )

    def embed(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=show_progress)


# ---------------------------------------------------------------------------
# ChromaVectorStore
# ---------------------------------------------------------------------------

class ChromaVectorStore:
    """ChromaDB-backed vector store with cosine similarity."""

    def __init__(
        self,
        collection_name: str = "rag_docs",
        persist_path: Optional[str] = None,
    ):
        if persist_path:
            self.client = chromadb.PersistentClient(path=persist_path)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"[ChromaVectorStore] Collection '{collection_name}' ready "
            f"(existing docs: {self.collection.count()})."
        )

    def add_documents(
        self,
        chunks: List[Document],
        embedding_manager: EmbeddingManager,
        batch_size: int = 100,
    ) -> None:
        """Embed and store chunks; skips insert if collection is already populated."""
        if self.collection.count() > 0:
            print(
                f"[ChromaVectorStore] Collection already has "
                f"{self.collection.count()} docs — skipping insert."
            )
            return

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        print(f"[ChromaVectorStore] Embedding {len(texts)} chunk(s)…")
        embeddings = embedding_manager.embed(texts, show_progress=True)

        # Insert in batches to avoid large single payloads
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            self.collection.add(
                ids=[str(i) for i in range(start, end)],
                embeddings=embeddings[start:end].tolist(),
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

        print(f"[ChromaVectorStore] Stored {self.collection.count()} chunk(s).")

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return top-k results as [{text, metadata, similarity}]."""
        raw = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "text": doc,
                "metadata": meta,
                "similarity": round(1 - dist, 4),
            }
            for doc, meta, dist in zip(
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
            )
        ]

    @property
    def count(self) -> int:
        return self.collection.count()


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Orchestrates the full RAG pipeline: load → chunk → embed → store → retrieve."""

    def __init__(
        self,
        collection_name: str = "rag_docs",
        persist_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size, chunk_overlap)
        self.embedder = EmbeddingManager(embedding_model)
        self.store = ChromaVectorStore(collection_name, persist_path)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        text_dir: Optional[str] = None,
        pdf_dir: Optional[str] = None,
    ) -> List[Document]:
        """Load documents, chunk them, and store embeddings in ChromaDB."""
        docs = self.loader.load(text_dir=text_dir, pdf_dir=pdf_dir)
        if not docs:
            print("[RAGPipeline] No documents loaded — nothing to ingest.")
            return []
        chunks = self.chunker.split(docs)
        self.store.add_documents(chunks, self.embedder)
        return chunks

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Return the top-k most relevant chunks for *query*."""
        query_emb = self.embedder.embed([query])
        hits = self.store.query(query_emb, n_results)
        return hits

    def query(self, question: str, n_results: int = 3) -> str:
        """Retrieve relevant chunks and return them as a single context string.

        The returned context string is ready to be injected into an LLM prompt.
        """
        hits = self.retrieve(question, n_results)

        print(f"\n[RAGPipeline] Question: {question}")
        print(f"[RAGPipeline] Top {n_results} retrieved chunk(s):")
        for i, hit in enumerate(hits, 1):
            print(f"  [{i}] similarity={hit['similarity']}  {hit['text'][:100]}…")

        return "\n\n---\n\n".join(h["text"] for h in hits)


# ---------------------------------------------------------------------------
# Quick demo when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base = Path(__file__).parent.parent  # RAG/

    pipeline = RAGPipeline(
        collection_name="demo_sumo",
        persist_path=str(base / "data" / "chroma_db"),
    )

    pipeline.ingest(text_dir=str(base / "data" / "text_files"))

    for q in [
        "What is the highest rank in sumo?",
        "Who is considered the greatest sumo wrestler?",
        "How long does a sumo tournament last?",
        "What food do sumo wrestlers eat?",
    ]:
        context = pipeline.query(q, n_results=2)
        print()
