"""Medivocate RAG System - A Retrieval-Augmented Generation system for document Q&A."""

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Main API
from .config import ChunkingConfig, RAGConfig, VectorStoreConfig
from .factory import create_rag_system, create_vector_store_manager
from .rag_pipeline.rag_system import RAGSystem
from .vector_store.vector_store import VectorStoreManager

__all__ = [
    # Configuration
    "RAGConfig",
    "VectorStoreConfig",
    "ChunkingConfig",
    # Main classes
    "RAGSystem",
    "VectorStoreManager",
    # Factory functions
    "create_rag_system",
    "create_vector_store_manager",
]
