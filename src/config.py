"""Configuration management for the RAG system.

This module provides centralized configuration classes for all components
of the RAG (Retrieval-Augmented Generation) system, allowing for easy
customization through environment variables or direct instantiation.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    """Configuration for the RAG system."""

    # Document processing
    docs_dir: str = "data/chunks"
    persist_directory: str = "data/chroma_db"
    batch_size: int = 64

    # Retrieval settings
    top_k_documents: int = 5
    bm25_portion: float = 0.03

    # Text processing
    chunk_size: int = 512
    chunk_overlap: int = 75

    # LLM settings
    temperature: float = 0.1
    max_tokens: int = 1000

    # Chunking settings
    keyphrase_top_n: int = 3
    keyphrase_ngram_range: tuple = (1, 1)

    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create config from environment variables."""
        return cls(
            docs_dir=os.getenv("RAG_DOCS_DIR", "data/chunks"),
            persist_directory=os.getenv("RAG_PERSIST_DIR", "data/chroma_db"),
            batch_size=int(os.getenv("RAG_BATCH_SIZE", "64")),
            top_k_documents=int(os.getenv("RAG_TOP_K", "5")),
            bm25_portion=float(os.getenv("RAG_BM25_PORTION", "0.03")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "75")),
            temperature=float(os.getenv("RAG_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("RAG_MAX_TOKENS", "1000")),
            keyphrase_top_n=int(os.getenv("RAG_KEYPHRASE_TOP_N", "3")),
        )


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""

    persist_directory: str
    batch_size: int = 64
    collection_name: Optional[str] = None

    def __post_init__(self):
        if self.collection_name is None:
            self.collection_name = (
                "medivocate-"
                + os.getenv("HF_MODEL", "default_model").split(":")[0].split("/")[-1]
            )


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    chunk_size: int = 512
    chunk_overlap: int = 75
    top_n: int = 3
    keyphrase_ngram_range: tuple = (1, 1)
