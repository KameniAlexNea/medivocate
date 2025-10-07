"""Tests for utility functions."""
import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.utilities.load_data import download_and_prepare_data
from src.utilities.embedding import CustomEmbedding


class TestLoadData:
    """Test data loading utilities."""

    @patch('src.utilities.load_data.os.path.exists')
    @patch('src.utilities.load_data.os.makedirs')
    @patch('src.utilities.load_data.shutil.move')
    @patch('src.utilities.load_data.shutil.rmtree')
    @patch('src.utilities.load_data.os.remove')
    @patch('src.utilities.load_data.zipfile.ZipFile')
    @patch('src.utilities.load_data.subprocess.run')
    def test_download_and_prepare_data_success(self, mock_subprocess, mock_zipfile, mock_remove, mock_rmtree, mock_move,
                                             mock_makedirs, mock_exists):
        """Test successful data download and preparation."""
        # Setup mocks
        mock_exists.return_value = False  # Target folder doesn't exist initially

        with tempfile.TemporaryDirectory() as temp_dir:
            gdrive_url = "https://drive.google.com/uc?id=test_id"
            zip_filename = "test.zip"
            extract_to = os.path.join(temp_dir, "extract")
            target_folder = os.path.join(temp_dir, "target")

            download_and_prepare_data(gdrive_url, zip_filename, extract_to, target_folder)

            # Verify subprocess.run was called for gdown
            mock_subprocess.assert_called_once_with(["gdown", gdrive_url, "-O", zip_filename], check=True)

            # Verify zip extraction
            mock_zipfile.assert_called_once_with(zip_filename, 'r')
            mock_zipfile.return_value.__enter__.return_value.extractall.assert_called_once_with(extract_to)

            # Verify directory operations
            mock_move.assert_called_once()
            mock_remove.assert_called_once_with(zip_filename)

    @patch('src.utilities.load_data.os.path.exists')
    def test_skip_existing_data(self, mock_exists):
        """Test skipping download when data already exists."""
        mock_exists.return_value = True  # Target folder exists

        with tempfile.TemporaryDirectory() as temp_dir:
            target_folder = os.path.join(temp_dir, "existing")

            with patch('src.utilities.load_data.logger') as mock_logger:
                download_and_prepare_data(
                    "test_url", "test.zip", temp_dir, target_folder
                )

                # Should log that data exists and skip download
                mock_logger.info.assert_called_with(f"Data already exists in {target_folder}")

    @patch('src.utilities.load_data.subprocess.run')
    @patch('src.utilities.load_data.os.path.exists')
    def test_download_failure(self, mock_exists, mock_subprocess):
        """Test handling download failure."""
        mock_exists.return_value = False
        mock_subprocess.side_effect = Exception("Download failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.utilities.load_data.logger') as mock_logger:
                download_and_prepare_data(
                    "test_url", "test.zip", temp_dir, temp_dir
                )

                # Should log the error
                mock_logger.info.assert_called_with("An error occurred: Download failed")


class TestEmbedding:
    """Test CustomEmbedding functionality."""

    @patch('src.utilities.embedding.HuggingFaceEmbeddings')
    def test_initialization(self, mock_hf_embeddings):
        """Test CustomEmbedding initialization."""
        mock_model = MagicMock()
        mock_hf_embeddings.return_value = mock_model

        embedding = CustomEmbedding(matryoshka_dim=512)

        assert embedding.matryoshka_dim == 512
        assert embedding.cpu_embedding == mock_model
        mock_hf_embeddings.assert_called_once()

    @patch('src.utilities.embedding.HuggingFaceEmbeddings')
    def test_embed_documents(self, mock_hf_embeddings):
        """Test document embedding."""
        mock_model = MagicMock()
        mock_model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_hf_embeddings.return_value = mock_model

        embedding = CustomEmbedding()
        texts = ["Test document 1", "Test document 2"]
        result = embedding.embed_documents(texts)

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.embed_documents.assert_called_once_with(texts)

    @patch('src.utilities.embedding.HuggingFaceEmbeddings')
    def test_embed_query(self, mock_hf_embeddings):
        """Test query embedding."""
        mock_model = MagicMock()
        mock_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_hf_embeddings.return_value = mock_model

        embedding = CustomEmbedding()
        result = embedding.embed_query("Test query")

        assert result == [0.1, 0.2, 0.3]
        mock_model.embed_query.assert_called_once_with("Test query")

    def test_get_instruction(self):
        """Test instruction retrieval."""
        embedding = CustomEmbedding()
        instruction = embedding.get_instruction()
        
        assert isinstance(instruction, str)
        assert len(instruction) > 0