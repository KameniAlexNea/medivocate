"""Factory functions for creating RAG system components."""

from typing import Optional

from .config import RAGConfig
from .rag_pipeline.rag_system import RAGSystem
from .vector_store.vector_store import VectorStoreManager


def create_rag_system(
    config: Optional[RAGConfig] = None, auto_initialize: bool = False
) -> RAGSystem:
    """Create and optionally initialize a RAG system.

    Args:
        config: RAG configuration. If None, uses environment variables.
        auto_initialize: Whether to automatically load documents and initialize vector store.

    Returns:
        Configured RAG system instance.
    """
    config = config or RAGConfig.from_env()
    rag_system = RAGSystem(config)

    if auto_initialize:
        documents = rag_system.load_documents()
        rag_system.initialize_vector_store(documents)

    return rag_system


def create_vector_store_manager(
    persist_directory: str, batch_size: int = 64
) -> VectorStoreManager:
    """Create a vector store manager instance.

    Args:
        persist_directory: Directory to persist the vector store.
        batch_size: Batch size for document processing.

    Returns:
        Configured vector store manager.
    """
    return VectorStoreManager(persist_directory, batch_size)
