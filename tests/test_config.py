"""Tests for configuration classes."""
import pytest
from unittest.mock import patch

from src.config import ChunkingConfig, RAGConfig, VectorStoreConfig


class TestChunkingConfig:
    """Test chunking configuration."""

    def test_default_chunking_config(self):
        """Test default chunking configuration values."""
        config = ChunkingConfig()
        assert config.chunk_size == 512
        assert config.chunk_overlap == 75
        assert config.top_n == 3
        assert config.keyphrase_ngram_range == (1, 1)

    def test_custom_chunking_config(self):
        """Test custom chunking configuration."""
        config = ChunkingConfig(
            chunk_size=300,
            chunk_overlap=50,
            top_n=5,
            keyphrase_ngram_range=(1, 2)
        )
        assert config.chunk_size == 300
        assert config.chunk_overlap == 50
        assert config.top_n == 5
        assert config.keyphrase_ngram_range == (1, 2)


class TestVectorStoreConfig:
    """Test vector store configuration."""

    def test_vector_store_config_creation(self):
        """Test vector store configuration creation."""
        config = VectorStoreConfig(persist_directory="test/db")
        assert config.persist_directory == "test/db"
        assert config.batch_size == 64
        assert config.collection_name is not None  # Should be auto-generated

    def test_custom_vector_store_config(self):
        """Test custom vector store configuration."""
        config = VectorStoreConfig(
            persist_directory="custom/db",
            batch_size=32,
            collection_name="test_collection"
        )
        assert config.persist_directory == "custom/db"
        assert config.batch_size == 32
        assert config.collection_name == "test_collection"

    def test_collection_name_auto_generation(self):
        """Test automatic collection name generation."""
        config = VectorStoreConfig(persist_directory="test/db")
        # Should generate a name based on HF_MODEL env var
        assert isinstance(config.collection_name, str)
        assert len(config.collection_name) > 0


class TestRAGConfig:
    """Test RAG configuration."""

    def test_default_rag_config(self):
        """Test default RAG configuration."""
        config = RAGConfig()
        assert config.docs_dir == "data/chunks"
        assert config.persist_directory == "data/chroma_db"
        assert config.batch_size == 64
        assert config.top_k_documents == 5
        assert config.bm25_portion == 0.03
        assert config.chunk_size == 512
        assert config.chunk_overlap == 75
        assert config.temperature == 0.1
        assert config.max_tokens == 1000
        assert config.keyphrase_top_n == 3
        assert config.keyphrase_ngram_range == (1, 1)

    def test_custom_rag_config(self):
        """Test custom RAG configuration."""
        config = RAGConfig(
            docs_dir="custom/docs",
            persist_directory="custom/db",
            batch_size=128,
            top_k_documents=10,
            bm25_portion=0.1,
            chunk_size=300,
            chunk_overlap=50,
            temperature=0.5,
            max_tokens=500,
            keyphrase_top_n=5
        )

        assert config.docs_dir == "custom/docs"
        assert config.persist_directory == "custom/db"
        assert config.batch_size == 128
        assert config.top_k_documents == 10
        assert config.bm25_portion == 0.1
        assert config.chunk_size == 300
        assert config.chunk_overlap == 50
        assert config.temperature == 0.5
        assert config.max_tokens == 500
        assert config.keyphrase_top_n == 5

    def test_rag_config_from_env(self):
        """Test creating RAG config from environment variables."""
        import os

        env_vars = {
            "RAG_DOCS_DIR": "env/docs",
            "RAG_PERSIST_DIR": "env/db",
            "RAG_BATCH_SIZE": "128",
            "RAG_TOP_K": "10",
            "RAG_BM25_PORTION": "0.1",
            "RAG_CHUNK_SIZE": "300",
            "RAG_CHUNK_OVERLAP": "50",
            "RAG_TEMPERATURE": "0.5",
            "RAG_MAX_TOKENS": "500",
            "RAG_KEYPHRASE_TOP_N": "5"
        }

        with patch.dict(os.environ, env_vars):
            config = RAGConfig.from_env()

            assert config.docs_dir == "env/docs"
            assert config.persist_directory == "env/db"
            assert config.batch_size == 128
            assert config.top_k_documents == 10
            assert config.bm25_portion == 0.1
            assert config.chunk_size == 300
            assert config.chunk_overlap == 50
            assert config.temperature == 0.5
            assert config.max_tokens == 500
            assert config.keyphrase_top_n == 5