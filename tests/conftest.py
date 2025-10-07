"""Test configuration and fixtures."""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.config import RAGConfig, VectorStoreConfig, ChunkingConfig


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_config():
    """Create a sample RAG configuration for testing."""
    return RAGConfig(
        docs_dir="test_docs",
        persist_directory="test_db",
        n_documents=3,
        temperature=0.2,
        max_tokens=500
    )


@pytest.fixture
def custom_chunking_config():
    """Create a custom chunking configuration."""
    return ChunkingConfig(
        chunk_size=300,
        overlap=50,
        chunking_strategy="fixed"
    )


@pytest.fixture
def vector_store_config():
    """Create a vector store configuration."""
    return VectorStoreConfig(
        persist_directory="test_chroma",
        collection_name="test_collection",
        batch_size=16
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="This is a sample document about artificial intelligence and machine learning.",
            metadata={"source": "ai.txt", "page": 1}
        ),
        Document(
            page_content="Natural language processing is a subfield of AI that focuses on language understanding.",
            metadata={"source": "nlp.txt", "page": 1}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers to solve complex problems.",
            metadata={"source": "deep_learning.txt", "page": 1}
        )
    ]


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="This is a mock response from the LLM.")
    return mock


@pytest.fixture
def mock_vector_store_manager():
    """Create a mock vector store manager."""
    mock = MagicMock()
    mock.initialize_vector_store.return_value = None
    mock.create_retriever.return_value = MagicMock()
    return mock