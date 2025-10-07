"""Tests for RAG system functionality."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.config import RAGConfig
from src.rag_pipeline.rag_system import RAGSystem


class TestRAGSystem:
    """T        config = RAGConfig()
    rag_system = RAGSystem(config)

    # Mock the chain and its stream method
    mock_chain = MagicMock()
    mock_chain.stream.return_value = [{"answer": "Test"}, {"answer": " response"}]
    rag_system.chain = mock_chain

    result = list(rag_system.query("Test question"))

    assert result == ["Test", " response"]
    mock_chain.stream.assert_called_once_with({
        "input": "Test question",
        "chat_history": []
    })em functionality."""

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_initialization(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test RAG system initialization."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig(
            docs_dir="test_docs",
            persist_directory="test_db",
            batch_size=32,
            top_k_documents=5,
            temperature=0.1,
            max_tokens=1000,
        )

        rag_system = RAGSystem(config)

        assert rag_system.config == config
        assert rag_system.docs_dir == "test_docs"
        assert rag_system.vector_store_management == mock_vs_instance

        # Check that get_llm_model_chat was called with config values
        mock_get_llm.assert_called_once_with(temperature=0.1, max_tokens=1000)
        # Check that VectorStoreManager was initialized with config values
        mock_vs_manager.assert_called_once_with("test_db", 32)

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_load_documents_success(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test successful document loading."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance
        mock_documents = [Document(page_content="test doc", metadata={})]
        mock_vs_instance.load_and_process_documents.return_value = mock_documents

        config = RAGConfig(docs_dir="test_docs")
        rag_system = RAGSystem(config)

        result = rag_system.load_documents()

        assert result == mock_documents
        mock_vs_instance.load_and_process_documents.assert_called_once_with("test_docs")

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_load_documents_failure(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test document loading failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance
        mock_vs_instance.load_and_process_documents.side_effect = Exception(
            "Load failed"
        )

        config = RAGConfig(docs_dir="test_docs")
        rag_system = RAGSystem(config)

        with pytest.raises(Exception, match="Load failed"):
            rag_system.load_documents()

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_initialize_vector_store_success(
        self, mock_vs_manager, mock_get_llm, mock_get_emb
    ):
        """Test successful vector store initialization."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)
        documents = [Document(page_content="test", metadata={})]

        rag_system.initialize_vector_store(documents)

        mock_vs_instance.initialize_vector_store.assert_called_once_with(documents)

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_initialize_vector_store_failure(
        self, mock_vs_manager, mock_get_llm, mock_get_emb
    ):
        """Test vector store initialization failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance
        mock_vs_instance.initialize_vector_store.side_effect = Exception(
            "VS init failed"
        )

        config = RAGConfig()
        rag_system = RAGSystem(config)
        documents = [Document(page_content="test", metadata={})]

        with pytest.raises(Exception, match="VS init failed"):
            rag_system.initialize_vector_store(documents)

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_query_success(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test successful query execution."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)

        # Mock the chain and its stream method
        mock_chain = MagicMock()
        mock_chain.stream.return_value = [{"answer": "Test"}, {"answer": " response"}]
        rag_system.chain = mock_chain

        result = list(rag_system.query("Test question"))

        assert result == ["Test", " response"]
        mock_chain.stream.assert_called_once_with(
            {"input": "Test question", "chat_history": []}
        )

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_query_with_history(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test query execution with chat history."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)
        rag_system.chain = MagicMock()
        rag_system.chain.invoke.return_value = {"answer": "Test response"}

        history = [("Human: Hello", "Assistant: Hi")]
        result = rag_system.query("Test question", history)

        assert result == "Test response"
        rag_system.chain.invoke.assert_called_once_with(
            {"input": "Test question", "chat_history": history}
        )

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_query_failure(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test query execution failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)
        rag_system.chain = MagicMock()
        rag_system.chain.invoke.side_effect = Exception("Query failed")

        with pytest.raises(Exception, match="Query failed"):
            rag_system.query("Test question")

    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_initialize_vector_store_success(self, mock_vs_manager, mock_get_llm):
        """Test successful vector store initialization."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)
        documents = [Document(page_content="test", metadata={})]

        rag_system.initialize_vector_store(documents)

        mock_vs_instance.initialize_vector_store.assert_called_once_with(documents)

    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_initialize_vector_store_failure(self, mock_vs_manager, mock_get_llm):
        """Test vector store initialization failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance
        mock_vs_instance.initialize_vector_store.side_effect = Exception(
            "VS init failed"
        )

        config = RAGConfig()
        rag_system = RAGSystem(config)
        documents = [Document(page_content="test", metadata={})]

        with pytest.raises(Exception, match="VS init failed"):
            rag_system.initialize_vector_store(documents)

    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_query_success(self, mock_vs_manager, mock_get_llm):
        """Test successful query execution."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)
        rag_system.chain = MagicMock()
        rag_system.chain.invoke.return_value = {"answer": "Test response"}

        result = rag_system.query("Test question")

        assert result == "Test response"
        rag_system.chain.invoke.assert_called_once_with(
            {"input": "Test question", "chat_history": []}
        )

    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_query_with_history(self, mock_vs_manager, mock_get_llm):
        """Test query execution with chat history."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)
        rag_system.chain = MagicMock()
        rag_system.chain.invoke.return_value = {"answer": "Test response"}

        history = [("Human: Hello", "Assistant: Hi")]
        result = rag_system.query("Test question", history)

        assert result == "Test response"
        rag_system.chain.invoke.assert_called_once_with(
            {"input": "Test question", "chat_history": history}
        )

    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_query_failure(self, mock_vs_manager, mock_get_llm):
        """Test query execution failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)
        rag_system.chain = MagicMock()
        rag_system.chain.invoke.side_effect = Exception("Query failed")

        with pytest.raises(Exception, match="Query failed"):
            rag_system.query("Test question")

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_initialization(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test RAG system initialization."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig(
            docs_dir="test_docs",
            persist_directory="test_db",
            batch_size=32,
            top_k_documents=5,
            temperature=0.1,
            max_tokens=1000,
        )

        rag_system = RAGSystem(config)

        assert rag_system.config == config
        assert rag_system.docs_dir == "test_docs"
        assert rag_system.vector_store_management == mock_vs_instance

        # Check that VectorStoreManager was initialized with config values
        mock_vs_manager.assert_called_once_with("test_db", 32)

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_load_documents_success(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test successful document loading."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance
        mock_documents = [Document(page_content="test doc", metadata={})]
        mock_vs_instance.load_and_process_documents.return_value = mock_documents

        config = RAGConfig(docs_dir="test_docs")
        rag_system = RAGSystem(config)

        result = rag_system.load_documents()

        assert result == mock_documents
        mock_vs_instance.load_and_process_documents.assert_called_once_with("test_docs")

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_load_documents_failure(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test document loading failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance
        mock_vs_instance.load_and_process_documents.side_effect = Exception(
            "Load failed"
        )

        config = RAGConfig(docs_dir="test_docs")
        rag_system = RAGSystem(config)

        with pytest.raises(Exception, match="Load failed"):
            rag_system.load_documents()

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_initialize_vector_store_success(
        self, mock_vs_manager, mock_get_llm, mock_get_emb
    ):
        """Test successful vector store initialization."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)
        documents = [Document(page_content="test", metadata={})]

        rag_system.initialize_vector_store(documents)

        mock_vs_instance.initialize_vector_store.assert_called_once_with(documents)

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_initialize_vector_store_failure(
        self, mock_vs_manager, mock_get_llm, mock_get_emb
    ):
        """Test vector store initialization failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance
        mock_vs_instance.initialize_vector_store.side_effect = Exception(
            "VS init failed"
        )

        config = RAGConfig()
        rag_system = RAGSystem(config)
        documents = [Document(page_content="test", metadata={})]

        with pytest.raises(Exception, match="VS init failed"):
            rag_system.initialize_vector_store(documents)

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_query_success(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test successful query execution."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)

        # Mock the chain and its stream method
        mock_chain = MagicMock()
        mock_chain.stream.return_value = [{"answer": "Test"}, {"answer": " response"}]
        rag_system.chain = mock_chain

        result = list(rag_system.query("Test question"))

        assert result == ["Test", " response"]
        mock_chain.stream.assert_called_once_with(
            {"input": "Test question", "chat_history": []}
        )

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_query_with_history(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test query execution with chat history."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)

        # Mock the chain and its stream method
        mock_chain = MagicMock()
        mock_chain.stream.return_value = [{"answer": "Test"}, {"answer": " response"}]
        rag_system.chain = mock_chain

        history = [("Human: Hello", "Assistant: Hi")]
        result = list(rag_system.query("Test question", history))

        assert result == ["Test", " response"]
        mock_chain.stream.assert_called_once_with(
            {"input": "Test question", "chat_history": history}
        )

    @patch("src.vector_store.base_vector_store.get_llm_model_embedding")
    @patch("src.rag_pipeline.rag_system.get_llm_model_chat")
    @patch("src.rag_pipeline.rag_system.VectorStoreManager")
    def test_query_failure(self, mock_vs_manager, mock_get_llm, mock_get_emb):
        """Test query execution failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_vs_instance = MagicMock()
        mock_vs_manager.return_value = mock_vs_instance

        config = RAGConfig()
        rag_system = RAGSystem(config)

        # Mock the chain to raise an exception
        mock_chain = MagicMock()
        mock_chain.stream.side_effect = Exception("Query failed")
        rag_system.chain = mock_chain

        with pytest.raises(Exception, match="Query failed"):
            list(rag_system.query("Test question"))
