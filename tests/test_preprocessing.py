"""Tests for preprocessing components."""

from src.config import ChunkingConfig
from src.preprocessing.processor import Processor


class TestProcessor:
    """Test document processing functionality."""

    def test_initialization(self):
        """Test Processor initialization."""
        processor = Processor(chunk_size=500, chunk_overlap=50)

        assert processor.text_splitter is not None
        assert processor.text_splitter._chunk_size == 500
        assert processor.text_splitter._chunk_overlap == 50

    def test_from_config(self):
        """Test creating Processor from config."""
        config = ChunkingConfig(chunk_size=600, chunk_overlap=100)
        processor = Processor.from_config(config)

        assert processor.text_splitter is not None
        assert processor.text_splitter._chunk_size == 600
        assert processor.text_splitter._chunk_overlap == 100

    def test_merge_sentences(self):
        """Test sentence merging functionality."""
        text = "This is a test-\nline that should be merged.\nThis is another line."
        result = Processor.merge_sentences(text)
        expected = "This is a testline that should be merged.\nThis is another line."
        assert result == expected

    def test_is_potential_title(self):
        """Test title detection."""
        assert Processor.is_potential_title("ALL CAPS TITLE") == True
        assert (
            Processor.is_potential_title("Normal sentence") == True
        )  # Starts with uppercase, short, no ending punctuation
        assert (
            Processor.is_potential_title("lowercase sentence.") == False
        )  # Ends with punctuation
        assert (
            Processor.is_potential_title(
                "This is a very long sentence that exceeds the word limit for titles"
            )
            == False
        )

    def test_is_valid_file(self):
        """Test file validation."""
        # Valid: 15-40 lines of content without too many titles/citations
        valid_content = "\n".join(
            [f"This is a normal line of text number {i}." for i in range(15)]
        )
        assert Processor.is_valid_file(valid_content) == True

        # Invalid: too few lines
        assert Processor.is_valid_file("valid content") == False

        # Invalid: empty
        assert Processor.is_valid_file("") == False
        assert Processor.is_valid_file("   ") == False

    def test_split_text_into_large_chunks(self):
        """Test large chunk splitting."""
        text = "word " * 400  # About 400 words
        chunks = Processor.split_text_into_large_chunks(text, target_word_count=300)

        assert len(chunks) > 1  # Should split into multiple chunks
        for chunk in chunks:
            word_count = len(chunk.split())
            assert word_count <= 300 or chunk == chunks[-1]  # Last chunk can be smaller
