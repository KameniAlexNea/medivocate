"""Tests for vector store managers."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.vector_store.base_vector_store import BaseVectorStoreManager
from src.vector_store.bivector_store import EnsembleVectorStoreManager
from src.vector_store.vector_store import VectorStoreManager


class TestBaseVectorStoreManager:
    """Test base vector store manager functionality."""

    def test_abstract_methods(self):
        """Test that base class defines abstract methods."""
        # Base class should not be instantiable directly
        with pytest.raises(TypeError):
            BaseVectorStoreManager("test_dir")

    def test_initialization(self):
        """Test BaseVectorStoreManager initialization."""

        class ConcreteVectorStoreManager(BaseVectorStoreManager):
            def __init__(self, persist_directory, batch_size):
                # Mock the initialization to avoid actual config/embedding loading
                self.persist_directory = persist_directory
                self.batch_size = batch_size
                self.collection_name = "medivocate-nomic-embed-text-v2-moe"
                self.embeddings = MagicMock()
                self.vs_initialized = False

            def _initialize_chroma_store(self, documents):
                pass

            def _add_chroma_documents(self, documents):
                pass

            def initialize_vector_store(self, documents=None):
                pass

            def create_retriever(self, llm, n_documents, bm25_portion=0.8):
                pass

        manager = ConcreteVectorStoreManager("test_dir", batch_size=32)

        assert manager.persist_directory == "test_dir"
        assert manager.batch_size == 32
        assert manager.collection_name == "medivocate-nomic-embed-text-v2-moe"
        assert manager.vs_initialized is False


class TestVectorStoreManager:
    """Test single vector store manager."""

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.vector_store.vector_store.Chroma")
    def test_initialization(self, mock_chroma, mock_embedding):
        """Test VectorStoreManager initialization."""
        mock_embedding.return_value = MagicMock()

        manager = VectorStoreManager("test_dir", batch_size=32)

        assert manager.persist_directory == "test_dir"
        assert manager.batch_size == 32
        assert isinstance(manager.vector_stores, dict)
        assert "chroma" in manager.vector_stores

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.vector_store.vector_store.Chroma")
    def test_initialize_chroma_store(self, mock_chroma, mock_embedding):
        """Test Chroma store initialization."""
        mock_embedding.return_value = MagicMock()
        mock_chroma_instance = MagicMock()
        mock_chroma.from_documents.return_value = mock_chroma_instance

        manager = VectorStoreManager("test_dir")
        documents = [Document(page_content="test", metadata={})]

        manager._initialize_chroma_store(documents)

        mock_chroma.from_documents.assert_called_once_with(
            collection_name="medivocate-nomic-embed-text-v2-moe",
            documents=documents,
            embedding=manager.embeddings,
            persist_directory="test_dir",
        )
        assert manager.vector_stores["chroma"] == mock_chroma_instance

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.vector_store.vector_store.Chroma")
    def test_add_chroma_documents(self, mock_chroma, mock_embedding):
        """Test adding documents to Chroma store."""
        mock_embedding.return_value = MagicMock()
        mock_chroma_instance = MagicMock()

        manager = VectorStoreManager("test_dir")
        manager.vector_stores["chroma"] = mock_chroma_instance
        documents = [Document(page_content="test", metadata={})]

        manager._add_chroma_documents(documents)

        mock_chroma_instance.add_documents.assert_called_once_with(documents)

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.vector_store.vector_store.Chroma")
    def test_initialize_vector_store_with_documents(self, mock_chroma, mock_embedding):
        """Test initializing vector store with documents."""
        mock_embedding.return_value = MagicMock()

        manager = VectorStoreManager("test_dir")
        documents = [Document(page_content="test", metadata={})]

        with patch.object(manager, "_batch_process_chroma_documents") as mock_batch:
            manager.initialize_vector_store(documents)

            mock_batch.assert_called_once_with(documents)
            assert manager.vs_initialized is True

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.vector_store.vector_store.Chroma")
    def test_initialize_vector_store_without_documents(
        self, mock_chroma, mock_embedding
    ):
        """Test initializing vector store without documents."""
        mock_embedding.return_value = MagicMock()
        mock_chroma_instance = MagicMock()
        mock_chroma.return_value = mock_chroma_instance

        manager = VectorStoreManager("test_dir")

        manager.initialize_vector_store()

        mock_chroma.assert_called_once_with(
            collection_name="medivocate-nomic-embed-text-v2-moe",
            persist_directory="test_dir",
            embedding_function=manager.embeddings,
        )
        assert manager.vector_stores["chroma"] == mock_chroma_instance
        assert manager.vs_initialized is True


class TestEnsembleVectorStoreManager:
    """Test ensemble vector store manager."""

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.vector_store.bivector_store.AutoTokenizer")
    def test_initialization(self, mock_tokenizer, mock_embedding):
        """Test EnsembleVectorStoreManager initialization."""
        mock_embedding.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        manager = EnsembleVectorStoreManager("test_dir", batch_size=32)

        assert manager.persist_directory == "test_dir"
        assert manager.batch_size == 32
        assert isinstance(manager.vector_stores, dict)
        assert "chroma" in manager.vector_stores
        assert "bm25" in manager.vector_stores
        assert hasattr(manager, "tokenizer")

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.vector_store.bivector_store.AutoTokenizer")
    @patch("src.vector_store.bivector_store.BM25Retriever")
    def test_batch_process_documents(self, mock_bm25, mock_tokenizer, mock_embedding):
        """Test batch processing documents for ensemble."""
        mock_embedding.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_bm25_instance = MagicMock()
        mock_bm25.from_documents.return_value = mock_bm25_instance

        manager = EnsembleVectorStoreManager("test_dir")
        documents = [Document(page_content="test", metadata={})]

        with patch.object(manager, "_batch_process_chroma_documents") as mock_batch:
            manager._batch_process_documents(documents)

            mock_batch.assert_called_once_with(documents)
            mock_bm25.from_documents.assert_called_once_with(
                documents, tokenizer=mock_tokenizer.from_pretrained.return_value
            )
            assert manager.vector_stores["bm25"] == mock_bm25_instance

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.vector_store.bivector_store.AutoTokenizer")
    @patch("src.vector_store.bivector_store.Chroma")
    @patch("src.vector_store.bivector_store.BM25Retriever")
    def test_initialize_vector_store_without_documents(
        self, mock_bm25, mock_chroma, mock_tokenizer, mock_embedding
    ):
        """Test initializing ensemble vector store without documents."""
        mock_embedding.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_chroma_instance = MagicMock()
        mock_bm25_instance = MagicMock()

        mock_chroma.return_value = mock_chroma_instance
        mock_bm25.from_documents.return_value = mock_bm25_instance

        # Mock the get method to return sample data
        mock_chroma_instance.get.return_value = {
            "documents": ["doc1", "doc2"],
            "ids": ["id1", "id2"],
            "metadatas": [{"key": "value1"}, {"key": "value2"}],
        }

        manager = EnsembleVectorStoreManager("test_dir")

        manager.initialize_vector_store()

        # Verify Chroma was initialized
        mock_chroma.assert_called_once_with(
            collection_name="medivocate-nomic-embed-text-v2-moe",
            persist_directory="test_dir",
            embedding_function=manager.embeddings,
        )

        # Verify BM25 was initialized with reconstructed documents
        mock_bm25.from_documents.assert_called_once()
        assert manager.vs_initialized is True
