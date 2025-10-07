"""Tests for factory functions."""

from unittest.mock import MagicMock, patch

from src.config import RAGConfig
from src.factory import create_rag_system, create_vector_store_manager


class TestFactoryFunctions:
    """Test factory functions for creating system components."""

    @patch("src.factory.RAGSystem")
    def test_create_rag_system_default(self, mock_rag_system):
        """Test creating a RAG system with default config."""
        mock_instance = MagicMock()
        mock_rag_system.return_value = mock_instance

        rag_system = create_rag_system()

        # Should use RAGConfig.from_env() when no config provided
        mock_rag_system.assert_called_once()
        assert rag_system == mock_instance

    @patch("src.factory.RAGSystem")
    def test_create_rag_system_custom_config(self, mock_rag_system):
        """Test creating a RAG system with custom config."""
        mock_instance = MagicMock()
        mock_rag_system.return_value = mock_instance

        config = RAGConfig(docs_dir="custom/docs")
        rag_system = create_rag_system(config)

        mock_rag_system.assert_called_once_with(config)
        assert rag_system == mock_instance

    @patch("src.factory.RAGSystem")
    def test_create_rag_system_auto_initialize(self, mock_rag_system):
        """Test creating a RAG system with auto-initialization."""
        mock_instance = MagicMock()
        mock_documents = [MagicMock()]
        mock_instance.load_documents.return_value = mock_documents
        mock_rag_system.return_value = mock_instance

        config = RAGConfig()
        rag_system = create_rag_system(config, auto_initialize=True)

        # Should call load_documents and initialize_vector_store
        mock_instance.load_documents.assert_called_once()
        mock_instance.initialize_vector_store.assert_called_once_with(mock_documents)
        assert rag_system == mock_instance

    @patch("src.factory.VectorStoreManager")
    def test_create_vector_store_manager(self, mock_vector_store):
        """Test creating vector store manager."""
        mock_instance = MagicMock()
        mock_vector_store.return_value = mock_instance

        manager = create_vector_store_manager("test/dir", batch_size=32)

        mock_vector_store.assert_called_once_with("test/dir", 32)
        assert manager == mock_instance

    @patch("src.factory.VectorStoreManager")
    def test_create_vector_store_manager_default_batch_size(self, mock_vector_store):
        """Test creating vector store manager with default batch size."""
        mock_instance = MagicMock()
        mock_vector_store.return_value = mock_instance

        manager = create_vector_store_manager("test/dir")

        mock_vector_store.assert_called_once_with("test/dir", 64)
        assert manager == mock_instance
