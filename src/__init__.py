"""Medivocate RAG System - A Retrieval-Augmented Generation system for document Q&A."""

from loguru import logger

logger.add(
    lambda msg: print(msg, end=""),
    level="WARNING",
    format="{time:YYYY-MM-DD HH:mm:ss} - {name} - {level} - {message}",
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
